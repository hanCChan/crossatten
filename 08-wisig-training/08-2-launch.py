#!/usr/bin/env python3
"""Lock-aware bounded S08 launcher retaining interrupted attempts."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
from training import ROOT, fit, validate_job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--per-gpu", type=int, default=4)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--job")
    args = parser.parse_args()
    if args.job:
        fit(json.loads(args.job))
        return
    jobs = json.loads(args.jobs.read_text())
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate run IDs")
    for job in jobs:
        validate_job(job)
    status = dict(total=len(jobs), done=[], skipped=[], failed=[])
    path = args.jobs.with_suffix(".status.json")
    if path.exists():
        path.rename(path.with_name(path.name + f".attempt_{time.time_ns()}"))
    lock, cursor = threading.Lock(), iter(jobs)

    def worker(gpu):
        while True:
            with lock:
                job = next(cursor, None)
            if job is None:
                return
            out = HERE / "runs" / job["phase"] / job["id"]
            out.mkdir(parents=True, exist_ok=True)
            code = -1
            try:
                with (out / "execution.lock").open("a") as stream:
                    fcntl.flock(stream, fcntl.LOCK_EX)
                    if (out / "metrics.json").is_file():
                        result = json.loads((out / "metrics.json").read_text())
                        if result["job"] != job:
                            raise ValueError("completed identity differs")
                        with lock:
                            status["skipped"].append(job["id"])
                            path.write_text(json.dumps(status, indent=1) + "\n")
                        continue
                    old = [p for p in out.iterdir() if p.is_file() and p.name != "execution.lock"]
                    if old:
                        archive = out / f"attempt_{time.time_ns()}"
                        archive.mkdir()
                        for p in old:
                            p.rename(archive / p.name)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(ROOT / "lib"),
                               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                               CUBLAS_WORKSPACE_CONFIG=":4096:8")
                    with (out / "stdout.log").open("w") as log:
                        code = subprocess.call(
                            [sys.executable, __file__, "--job", json.dumps(job)],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            except Exception as exc:
                print(f"{job['id']}: {exc}", flush=True)
            with lock:
                status["done" if code == 0 else "failed"].append(job["id"])
                path.write_text(json.dumps(status, indent=1) + "\n")
            print(f"[gpu{gpu}] rc={code} {job['id']}", flush=True)

    threads = [threading.Thread(target=worker, args=(gpu,))
               for _ in range(args.per_gpu) for gpu in args.gpus.split(",")]
    for thread in threads:
        thread.start()
        time.sleep(.1)
    for thread in threads:
        thread.join()
    if status["failed"]:
        raise SystemExit("failed runs retained; do not advance")
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
