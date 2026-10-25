#!/usr/bin/env python3
"""Check complete S08 source/final jobs and prediction metrics before decisions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from training import HERE, RULES, metrics, sha, validate_job


def audit_phase(phase):
    jobs = json.loads((HERE / f"jobs_{phase}.json").read_text())
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate jobs")
    for job in jobs:
        reg, _ = validate_job(job)
        out = HERE / "runs" / phase / job["id"]
        result = json.loads((out / "metrics.json").read_text())
        config = json.loads((out / "config.json").read_text())
        if result["job"] != job or config["job"] != job or result["params"] != config["params"]:
            raise ValueError("run identity/parameter mismatch")
        if result["registration_sha256"] != sha(HERE / "registration.json"):
            raise ValueError("registration mismatch")
        for name, expected in result["weights_sha256"].items():
            if sha(out / name) != expected:
                raise ValueError("changed checkpoint")
        history = list(csv.DictReader((out / "history.csv").open()))
        if len(history) != result["stop_epoch"]:
            raise ValueError("incomplete training history")
        if job["condition"].endswith("full200") and phase not in ("smoke", "final") and len(history) != 200:
            raise ValueError("full200 run stopped early")
        for rule in RULES:
            if rule.endswith("held") and phase == "final":
                continue
            column = rule.replace("_", "_") + "_ce"
            actual_epoch = int(np.argmin([float(r[column]) for r in history])) + 1
            if result["best"][rule]["epoch"] != actual_epoch:
                raise ValueError("checkpoint not source CE minimum")
        if phase not in ("final", "smoke") and set(result["selected"]) != set(RULES):
            raise ValueError("missing source checkpoints or target contamination")
        for filename, expected in result["predictions_sha256"].items():
            if sha(out / filename) != expected:
                raise ValueError("changed predictions")
            label = filename.removeprefix("preds_").removesuffix(".npz")
            entry = result["selected"][label]
            classes = 10 if job["dataset"] == "ManyRx" else 6
            with np.load(out / filename) as archive:
                names = {k for k in archive.files if not k.startswith("y_")}
                allowed = config["roles"]["target" if phase == "final" else "held_rx"]
                if names != set(allowed):
                    raise ValueError("prediction receiver roles differ")
                for rx in names:
                    logits, y = archive[rx], archive[f"y_{rx}"]
                    if logits.dtype != np.float32 or not np.isfinite(logits).all():
                        raise ValueError("nonfinite/non-float32 predictions")
                    actual = metrics(logits, y, classes)
                    for k in ("acc", "macro_f1", "ce", "correct", "total"):
                        if not np.isclose(actual[k], entry["per_rx"][rx][k], atol=1e-7):
                            raise ValueError("prediction-derived metric mismatch")
    print(json.dumps(dict(phase=phase, complete=len(jobs), audit="PASS")), flush=True)
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["screen", "phase", "manyrx", "final", "smoke"])
    audit_phase(parser.parse_args().phase)


if __name__ == "__main__":
    main()
