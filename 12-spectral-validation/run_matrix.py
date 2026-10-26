"""Wait for bounded GPU workers; stop scheduling on failure, never retry."""
import argparse
import json
import os
import subprocess
import sys
import threading
from core import HERE, ROOT, save, validate


def run(phase, gpus, per_gpu):
    jobs = json.loads((HERE / f"jobs_{phase}.json").read_text())
    for j in jobs:
        validate(j)
    status = dict(phase=phase, total=len(jobs), done=[], skipped=[], failed=[])
    pending = iter(enumerate(jobs))
    lock, stop = threading.Lock(), threading.Event()
    status_path = HERE / f"jobs_{phase}.status.json"

    def worker(gpu):
        while not stop.is_set():
            with lock:
                item = next(pending, None)
            if item is None:
                return
            index, j = item
            out = HERE / "runs" / phase / j["id"]
            if (out / "metrics.json").exists():
                with lock:
                    status["skipped"].append(j["id"])
                    save(status_path, status)
                continue
            if (out / "fit_attempt.json").exists():
                stop.set()
                with lock:
                    status["failed"].append(j["id"])
                    save(status_path, status)
                return
            out.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="2",
                       MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            command = [sys.executable, str(HERE / "launch.py"), "--phase", phase,
                       "--index", str(index), "--execute"]
            with (out / "stdout.log").open("w") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT)
            with lock:
                status["done" if result.returncode == 0 else "failed"].append(j["id"])
                save(status_path, status)
            if result.returncode:
                stop.set()
            print(f"{phase} GPU{gpu} rc={result.returncode} {j['id']}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus
               for _ in range(per_gpu)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    save(status_path, status)
    if status["failed"] or len(status["done"]) + len(status["skipped"]) != len(jobs):
        raise RuntimeError("incomplete matrix; no automatic retry or advancement")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "pilot", "confirmation"), required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--per-gpu", type=int, default=2)
    args = parser.parse_args()
    if args.per_gpu < 1:
        parser.error("--per-gpu must be positive")
    run(args.phase, args.gpus.split(","), args.per_gpu)
