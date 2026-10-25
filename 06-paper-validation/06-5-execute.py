#!/usr/bin/env python3
"""Resume the bounded registered matrix; freeze/commit before target evaluation."""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENV = dict(os.environ, PYTHONPATH=str(ROOT / "lib"))
WORKFLOW = HERE / "06-2-workflow.py"


def run(args):
    subprocess.run(args, cwd=ROOT, env=ENV, check=True)


def generate(phase, dataset=None, ib=False):
    args = [sys.executable, str(WORKFLOW), phase]
    if dataset:
        args.extend(["--dataset", dataset])
    if ib:
        args.append("--ib")
    run(args)
    suffix = f"_{dataset}" if dataset else ""
    return HERE / "runs" / f"jobs_{phase}{suffix}_{'ib' if ib else 'base'}.jsonl"


def complete(path):
    jobs = [json.loads(line) for line in path.read_text().splitlines() if line]
    return all((Path(job["out"]) / "metrics.json").is_file() for job in jobs)


def wait_existing(paths):
    while not all(complete(path) for path in paths):
        for path in paths:
            status = Path(str(path) + ".status.json")
            if status.exists() and not complete(path):
                result = json.loads(status.read_text())
                if result.get("finished"):
                    raise RuntimeError(f"prerequisite queue ended incomplete: {path.name}: {result.get('failed')}")
        time.sleep(20)


def launch(path, gpus, workers):
    if complete(path):
        return
    run([sys.executable, str(HERE / "06-6-launch.py"), "--jobs", str(path),
         "--gpus", gpus, "--per-gpu", str(workers), "--threads", "2"])
    if not complete(path):
        raise RuntimeError(f"registered queue has failed/incomplete fits: {path}")


def audit_sources(stage, expected):
    spec = importlib.util.spec_from_file_location("paper_report", HERE / "06-4-summary.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    _, results, _, checks = report.collect()
    issues = [p for p in checks["problems"] if f"runs/{stage}/" in p["path"]]
    records = [r for r in results if r["stage"] == stage]
    if issues or checks["missing"][stage] or len(records) != expected:
        raise RuntimeError(f"source audit failed: {stage}, {len(records)}/{expected}, {issues}")


def commit_evidence(stage, decision):
    import fcntl
    with (HERE / "runs/commit.lock").open("a") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        run(["git", "add", str(HERE / "arms.json"), str(HERE / decision),
             str(HERE / "runs" / stage), str(ROOT / "logs/data_access.jsonl"),
             str(ROOT / "logs/paper_data_access.jsonl")])
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
        if staged.returncode:
            run(["git", "commit", "-m", f"Freeze audited {stage} source-only paper-validation selection before target reads"])


def wifi(workers):
    wait_existing([HERE / "runs/jobs_source_ManyRx_base.jsonl", HERE / "runs/jobs_source_ManySig_base.jsonl"])
    paths = {d: generate("source", d, ib=True) for d in ("ManyRx", "ManySig")}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(launch, paths[d], str(gpu), workers) for gpu, d in enumerate(paths)]
        for future in futures:
            future.result()
    audit_sources("source", 176)
    if not (HERE / "decision_wifi.json").exists():
        run([sys.executable, str(WORKFLOW), "freeze_wifi"])
    commit_evidence("source", "decision_wifi.json")
    paths = {d: generate("final", d) for d in ("ManyRx", "ManySig")}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(launch, paths[d], str(gpu), workers) for gpu, d in enumerate(paths)]
        for future in futures:
            future.result()
    paths = {d: generate("final", d, ib=True) for d in ("ManyRx", "ManySig")}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(launch, paths[d], str(gpu), workers) for gpu, d in enumerate(paths)]
        for future in futures:
            future.result()


def shen(workers):
    wait_existing([HERE / "runs/jobs_explore_base.jsonl"])
    launch(generate("explore", ib=True), "2,3", workers)
    audit_sources("explore", 100)
    if not (HERE / "decision_shen.json").exists():
        run([sys.executable, str(WORKFLOW), "freeze_shen"])
    commit_evidence("explore", "decision_shen.json")
    launch(generate("confirm"), "2,3", workers)
    launch(generate("confirm", ib=True), "2,3", workers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--line", choices=("wifi", "shen"), required=True)
    parser.add_argument("--per-gpu", type=int, default=8)
    args = parser.parse_args()
    lock = HERE / "runs" / f"execute_{args.line}.lock"
    lock.parent.mkdir(exist_ok=True)
    import fcntl
    with lock.open("a") as file:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (wifi if args.line == "wifi" else shen)(args.per_gpu)
        run([sys.executable, str(HERE / "06-4-summary.py")])
        print(f"REGISTERED {args.line} LINE FINISHED", flush=True)


if __name__ == "__main__":
    main()
