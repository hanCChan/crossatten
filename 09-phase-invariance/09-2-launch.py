"""Bounded S09 launcher. Training never implicitly evaluates target receivers."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from training import HERE, ROOT, fit, evaluate_target, validate, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs")
    parser.add_argument("--job")
    parser.add_argument("--per-gpu", type=int, default=6)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.job:
        (evaluate_target if args.evaluate else fit)(json.loads(args.job))
        return
    jobs = json.loads((HERE / args.jobs).read_text())
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate job identity")
    for job in jobs:
        validate(job)
    suffix = "target_metrics.json" if args.evaluate else "metrics.json"
    status = dict(total=len(jobs), done=[], skipped=[], failed=[])
    path = HERE / (args.jobs.replace(".json", "") + (".evaluate" if args.evaluate else "") + ".status.json")
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
                    if (out / suffix).exists():
                        if json.loads((out / suffix).read_text())["job"] != job:
                            raise ValueError("completed identity differs")
                        with lock:
                            status["skipped"].append(job["id"])
                            save(path, status)
                        continue
                    if not args.evaluate:
                        old = [p for p in out.iterdir() if p.is_file() and p.name != "execution.lock"]
                        if old:
                            archive = out / f"attempt_{time.time_ns()}"
                            archive.mkdir()
                            for p in old:
                                p.rename(archive / p.name)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(ROOT / "lib"),
                               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
                    command = [sys.executable, __file__, "--job", json.dumps(job)]
                    if args.evaluate:
                        command.append("--evaluate")
                    with (out / ("target_stdout.log" if args.evaluate else "stdout.log")).open("w") as log:
                        code = subprocess.call(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            except Exception as exc:
                print(f"{job['id']}: {exc}", flush=True)
            with lock:
                status["done" if code == 0 else "failed"].append(job["id"])
                save(path, status)
            print(f"[gpu{gpu}] rc={code} {job['id']}", flush=True)

    threads = [threading.Thread(target=worker, args=(gpu,)) for _ in range(args.per_gpu)
               for gpu in args.gpus.split(",")]
    for thread in threads:
        thread.start()
        time.sleep(.1)
    for thread in threads:
        thread.join()
    if status["failed"]:
        raise SystemExit("failed attempts retained; do not advance")
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
