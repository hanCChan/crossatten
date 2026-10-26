"""No automatic fit retries; source fitting and sealed target scoring are separate."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import threading
from stage import HERE, ROOT, validate, save
from training import fit_phase, fit_ask, evaluate_phase, evaluate_ask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs")
    p.add_argument("--job")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--per-gpu", type=int, default=2)
    p.add_argument("--evaluate", action="store_true")
    args = p.parse_args()
    if args.job:
        job = json.loads(args.job)
        validate(job)
        fn = (evaluate_ask if job["feature"] == "ask" else evaluate_phase) if args.evaluate else (
             fit_ask if job["feature"] == "ask" else fit_phase)
        fn(job)
        return
    path = HERE / args.jobs
    jobs = json.loads(path.read_text())
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate identities")
    for j in jobs:
        validate(j)
    suffix = "target_metrics.json" if args.evaluate else "metrics.json"
    status = dict(total=len(jobs), done=[], skipped=[], failed=[])
    status_path = path.with_name(path.stem + (".evaluate" if args.evaluate else "") + ".status.json")
    lock, cursor = threading.Lock(), iter(jobs)

    def worker(gpu):
        while True:
            with lock:
                j = next(cursor, None)
            if j is None:
                return
            out = HERE / "runs" / j["phase"] / j["id"]
            out.mkdir(parents=True, exist_ok=True)
            try:
                with (out / "execution.lock").open("a") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    if (out / suffix).exists():
                        with lock:
                            status["skipped"].append(j["id"])
                            save(status_path, status)
                        continue
                    attempt = out / ("target_attempt.json" if args.evaluate else "fit_attempt.json")
                    if attempt.exists():
                        raise ValueError("prior incomplete/failed attempt; automatic retries forbidden")
                    save(attempt, dict(job=j, gpu=gpu, action="evaluate" if args.evaluate else "fit"))
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="2",
                               MKL_NUM_THREADS="2", PYTHONPATH=str(ROOT / "lib"))
                    cmd = [sys.executable, __file__, "--job", json.dumps(j)]
                    if args.evaluate:
                        cmd.append("--evaluate")
                    with (out / ("target_stdout.log" if args.evaluate else "stdout.log")).open("w") as log:
                        rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            except Exception as exc:
                print(j["id"], str(exc), flush=True)
                rc = -1
            with lock:
                status["done" if rc == 0 else "failed"].append(j["id"])
                save(status_path, status)
            print("gpu", gpu, "rc", rc, j["id"], flush=True)
    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus.split(",")
               for _ in range(args.per_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if status["failed"]:
        raise SystemExit("Failed attempts preserved; stage cannot advance.")


if __name__ == "__main__":
    main()
