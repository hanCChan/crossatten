#!/usr/bin/env python3
"""Supervised continuation after exploration: complete, verify, commit, confirm, report."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run(*cmd):
    print("RUN " + " ".join(str(v) for v in cmd), flush=True)
    subprocess.run([str(v) for v in cmd], cwd=ROOT, check=True)


def check_status(stage):
    path = HERE / "runs" / f"jobs_{stage}.jsonl.status.json"
    status = json.loads(path.read_text())
    if status["failed"] or status["done"] + status["skipped"] != status["total"]:
        raise SystemExit(f"{stage}: failures/incomplete launcher; inspect {path}")


def verify_baseline(stage):
    import numpy as np
    import torch

    records = []
    for path in sorted((HERE / "runs" / stage).glob("s03_base__*/metrics.json")):
        r = json.loads(path.read_text())
        ref = ROOT / "02-single-factor-screen/runs" / stage / f"s02_ep200__f{r['fold']}__s{r['seed']}"
        a = torch.load(path.parent / "model.pt", map_location="cpu", weights_only=True)
        b = torch.load(ref / "model.pt", map_location="cpu", weights_only=True)
        weights_equal = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
        predictions_equal = True
        for role in ("dev_val", "confirm") if stage == "confirm" else ("dev_val",):
            with np.load(path.parent / f"preds_{role}.npz") as x, np.load(ref / f"preds_{role}.npz") as y:
                predictions_equal &= set(x.files) == set(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)
        records.append(dict(fold=r["fold"], seed=r["seed"], weights_equal=weights_equal,
                            predictions_equal=bool(predictions_equal)))
        if not weights_equal or not predictions_equal:
            raise SystemExit(f"baseline determinism failed: {stage} fold{r['fold']} seed{r['seed']}")
    expected = 4 if stage == "explore" else 6
    if len(records) != expected:
        raise SystemExit("baseline verification incomplete")
    out = HERE / "summary" / f"{stage}_determinism.json"
    out.write_text(json.dumps(records, indent=1) + "\n")


def commit(message):
    run("git", "add", "03-chirp-structure", "logs/data_access.jsonl")
    # model.pt is ignored; third-party and unrelated edits are never included.
    run("git", "commit", "-m", message)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    status_path = HERE / "runs/jobs_explore.jsonl.status.json"
    while not status_path.exists():
        print("waiting for exploration launcher to finish", flush=True)
        time.sleep(30)
    check_status("explore")
    verify_baseline("explore")
    run(sys.executable, HERE / "03-3-aggregate.py", "explore", "--freeze")
    commit("s03: complete exploration and freeze all eight arms for confirmation")
    run(sys.executable, HERE / "03-1-make-jobs.py", "confirm")
    run(sys.executable, "-m", "xa.launch", "--jobs", HERE / "runs/jobs_confirm.jsonl",
        "--per-gpu", "5", "--threads", str(args.threads))
    check_status("confirm")
    verify_baseline("confirm")
    run(sys.executable, HERE / "03-3-aggregate.py", "confirm", "--freeze")
    run(sys.executable, HERE / "03-5-mechanisms.py")
    run(sys.executable, HERE / "03-6-report.py")
    commit("s03: complete confirmation, mechanism interventions, and round-one results")
    print("M1 complete; STOP before stage04, combinations, or WiSig training", flush=True)


if __name__ == "__main__":
    main()
