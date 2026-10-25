#!/usr/bin/env python3
"""Stage-07 lock-aware launcher, preserving every interrupted attempt."""
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
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--per-gpu", type=int, default=4)
    parser.add_argument("--job", help="one JSON job, internal child invocation")
    args = parser.parse_args()
    if args.job:
        from xa.explore_train import fit
        job = json.loads(args.job)
        fit(job, HERE / "runs" / job["phase"] / job["id"])
        return
    jobs = json.loads(args.jobs.read_text())
    if len({job["id"] for job in jobs}) != len(jobs):
        raise SystemExit("duplicate job IDs")
    from xa.explore_train import validate_job
    for job in jobs:
        validate_job(job)
    lock, cursor = threading.Lock(), iter(jobs)
    status = dict(total=len(jobs), done=[], skipped=[], failed=[])
    status_path = args.jobs.with_suffix(".status.json")
    if status_path.exists():
        status_path.rename(status_path.with_name(status_path.name + f".attempt_{time.time_ns()}"))

    def worker(gpu):
        while True:
            with lock:
                job = next(cursor, None)
            if job is None:
                return
            code, skipped = -1, False
            try:
                out = HERE / "runs" / job["phase"] / job["id"]
                out.mkdir(parents=True, exist_ok=True)
                with (out / "execution.lock").open("a") as stream:
                    fcntl.flock(stream, fcntl.LOCK_EX)
                    if (out / "metrics.json").exists():
                        result = json.loads((out / "metrics.json").read_text())
                        if result["job"] != job or not (out / "model.pt").is_file():
                            raise ValueError("invalid completed run")
                        with lock:
                            status["skipped"].append(job["id"])
                            status_path.write_text(json.dumps(status, indent=1) + "\n")
                        skipped = True
                        continue
                    old = [p for p in out.iterdir() if p.is_file() and p.name != "execution.lock"]
                    if old:
                        archive = out / f"attempt_{time.time_ns()}"
                        archive.mkdir()
                        for path in old:
                            path.rename(archive / path.name)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(ROOT / "lib"),
                               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
                    with (out / "stdout.log").open("w") as log:
                        code = subprocess.call([sys.executable, str(Path(__file__).resolve()),
                                                "--job", json.dumps(job)], env=env, cwd=ROOT,
                                               stdout=log, stderr=subprocess.STDOUT)
                    if code == 0 and not (out / "metrics.json").is_file():
                        raise ValueError("child exited without completion evidence")
            except Exception as exc:
                code = -1
                print(f"[gpu{gpu}] error {job['id']}: {exc}", flush=True)
            finally:
                if not skipped:
                    with lock:
                        status["done" if code == 0 else "failed"].append(job["id"])
                        status_path.write_text(json.dumps(status, indent=1) + "\n")
            print(f"[gpu{gpu}] rc={code} {job['id']}", flush=True)
    threads = [threading.Thread(target=worker, args=(gpu,)) for _ in range(args.per_gpu)
               for gpu in args.gpus.split(",")]
    for thread in threads:
        thread.start()
        time.sleep(.25)
    for thread in threads:
        thread.join()
    status_path.write_text(json.dumps(status, indent=1) + "\n")
    print(json.dumps(status), flush=True)
    if status["failed"] or len(status["done"]) + len(status["skipped"]) != len(jobs):
        raise SystemExit("failed runs retained; do not advance selection")


if __name__ == "__main__":
    main()
