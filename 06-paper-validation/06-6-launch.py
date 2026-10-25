#!/usr/bin/env python3
"""Paper-only launcher: adopt active fits, preserve attempts, and avoid duplicates."""
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


def active():
    found = {}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            command = (proc / "cmdline").read_bytes().decode().split("\0")
            if "xa.paper_run" not in command or "--out" not in command:
                continue
            out = command[command.index("--out") + 1]
            found[str(Path(out).resolve())] = int(proc.name)
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
    return found


def initialization_ready(job, arm):
    name = arm.get("init_arm")
    if name is None:
        return True
    out = Path(job["out"])
    init = out.parent / f"{name}__f{job['fold']}__s{job['seed']}"
    return (init / "metrics.json").is_file() and (init / "model.pt").is_file()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--per-gpu", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--wait-init", action="store_true")
    args = parser.parse_args()
    jobs = [json.loads(line) for line in Path(args.jobs).read_text().splitlines() if line]
    arms = {}
    if args.wait_init:
        for registration in {job["arms"] for job in jobs}:
            arms[registration] = {arm["name"]: arm for arm in json.loads(Path(registration).read_text())["arms"]}
    attempted, lock = set(), threading.Lock()
    status_path = Path(args.jobs + ".status.json")
    if status_path.exists():
        archive = Path(args.jobs + f".status.{time.time_ns()}.json")
        archive.write_bytes(status_path.read_bytes())
    status = dict(total=len(jobs), skipped=sum((Path(j["out"]) / "metrics.json").is_file() for j in jobs),
                  done=0, failed=[], adopted=[], started=time.strftime("%F %T"))

    def save():
        temp = status_path.with_suffix(".tmp")
        temp.write_text(json.dumps(status, indent=1))
        temp.replace(status_path)

    def worker(gpu):
        while True:
            with lock:
                running = active()
                todo = [j for j in jobs if j["out"] not in attempted and
                        not (Path(j["out"]) / "metrics.json").is_file()]
                if not todo:
                    return
                free = [j for j in todo if str(Path(j["out"]).resolve()) not in running and
                        (not args.wait_init or initialization_ready(j, arms[j["arms"]][j["arm"]]))]
                if not free:
                    job = None
                else:
                    job = free[0]
                    attempted.add(job["out"])
            if job is None:
                time.sleep(5)
                continue
            out = Path(job["out"])
            out.mkdir(parents=True, exist_ok=True)
            with (out / "execution.lock").open("a") as file:
                try:
                    fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    with lock:
                        attempted.remove(job["out"])
                    time.sleep(5)
                    continue
                if (out / "metrics.json").is_file():
                    continue
                if str(out.resolve()) in active():
                    with lock:
                        attempted.remove(job["out"])
                    continue
                if (out / "stdout.log").exists():
                    attempt = out / f"attempt_{time.time_ns()}"
                    attempt.mkdir()
                    for name in ("stdout.log", "history.csv", "config.json"):
                        if (out / name).exists():
                            (out / name).rename(attempt / name)
                command = [sys.executable, "-m", "xa.paper_run", "--arms", job["arms"],
                           "--arm", job["arm"], "--fold", str(job["fold"]), "--seed", str(job["seed"]),
                           "--stage", job["stage"], "--out", str(out)]
                if job.get("decision_file"):
                    command.extend(["--decision-file", job["decision_file"]])
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(ROOT / "lib"),
                           OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                           CUBLAS_WORKSPACE_CONFIG=":4096:8")
                begin = time.time()
                with (out / "stdout.log").open("w") as stream:
                    code = subprocess.call(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                with lock:
                    if code == 0:
                        status["done"] += 1
                    else:
                        status["failed"].append(dict(job=job, rc=code))
                    save()
                print(f"[gpu{gpu}] rc={code} {time.time()-begin:.0f}s {out.name}", flush=True)

    with lock:
        status["adopted"] = [j["out"] for j in jobs if str(Path(j["out"]).resolve()) in active()]
        save()
    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus.split(",")
               for _ in range(args.per_gpu)]
    for thread in threads:
        thread.start()
        time.sleep(1)
    for thread in threads:
        thread.join()
    status["finished"] = time.strftime("%F %T")
    status["complete"] = sum((Path(j["out"]) / "metrics.json").is_file() for j in jobs)
    save()
    print(json.dumps(status), flush=True)
    if status["complete"] != len(jobs):
        raise SystemExit("registered fits incomplete; retained all failed attempts")


if __name__ == "__main__":
    main()
