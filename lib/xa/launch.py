"""Resumable multi-GPU job launcher.  python -m xa.launch --jobs JOBS.jsonl --gpus 0,1,2,3 --per-gpu 4
Each job line: {"arms": path, "arm": name, "fold": f, "seed": s, "stage": st, "out": dir[, "decision_file": p]}.
A job is skipped when <out>/metrics.json exists. stdout/stderr -> <out>/stdout.log. Summary -> <jobs>.status.json
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import LIB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--per-gpu", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    jobs = [json.loads(line) for line in Path(a.jobs).read_text().splitlines() if line.strip()]
    todo = [j for j in jobs if not (Path(j["out"]) / "metrics.json").is_file()]
    lock = threading.Lock()
    status = dict(total=len(jobs), skipped=len(jobs) - len(todo), done=0, failed=[], started=time.strftime("%F %T"))
    print(f"{len(todo)} to run, {status['skipped']} already done", flush=True)

    def worker(gpu):
        while True:
            with lock:
                if not todo:
                    return
                j = todo.pop(0)
            out = Path(j["out"])
            out.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "xa.run_job", "--arms", j["arms"], "--arm", j["arm"], "--fold", str(j["fold"]),
                   "--seed", str(j["seed"]), "--stage", j["stage"], "--out", str(out)]
            if j.get("decision_file"):
                cmd += ["--decision-file", j["decision_file"]]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(LIB), OMP_NUM_THREADS=str(a.threads),
                       MKL_NUM_THREADS=str(a.threads), CUBLAS_WORKSPACE_CONFIG=":4096:8")
            t0 = time.time()
            with open(out / "stdout.log", "w") as log:
                rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            with lock:
                if rc == 0:
                    status["done"] += 1
                else:
                    status["failed"].append(dict(job=j, rc=rc))
                tail = (out / "stdout.log").read_text().strip().splitlines()[-1:] or [""]
                print(f"[gpu{gpu}] rc={rc} {time.time() - t0:6.0f}s {j['arm']} f{j['fold']} s{j['seed']} :: {tail[0][:160]}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in a.gpus.split(",") for _ in range(a.per_gpu)]
    for t in threads:
        t.start()
        time.sleep(2)
    for t in threads:
        t.join()
    status["finished"] = time.strftime("%F %T")
    Path(a.jobs + ".status.json").write_text(json.dumps(status, indent=1))
    print(json.dumps({k: v for k, v in status.items() if k != "failed"}), "failed:", len(status["failed"]), flush=True)


if __name__ == "__main__":
    main()
