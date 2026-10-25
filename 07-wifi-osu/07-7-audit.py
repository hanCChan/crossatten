#!/usr/bin/env python3
"""Verify completed run identities, hashes, prediction counts and float32 metrics."""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import numpy as np
from xa.evaluate import summarize
from xa.paper_train import metrics, require_committed, sha
from xa.wisig import receivers


def same_metrics(actual, stored):
    for key in ("acc", "macro_f1", "ce", "tx_equal_acc"):
        if key in actual and not np.isclose(actual[key], stored[key], rtol=1e-6, atol=1e-7):
            raise ValueError(f"prediction metric mismatch: {key}")
    for key in ("confusion", "total", "correct"):
        if actual[key] != stored[key]:
            raise ValueError(f"prediction metric mismatch: {key}")


def same_summary(actual, stored):
    if isinstance(actual, dict):
        if set(actual) != set(stored):
            raise ValueError("prediction summary keys mismatch")
        for key, value in actual.items():
            same_summary(value, stored[key])
    elif isinstance(actual, str):
        if actual != stored:
            raise ValueError("prediction summary receiver mismatch")
    elif not np.isclose(actual, stored, rtol=1e-6, atol=1e-7):
        raise ValueError("prediction summary value mismatch")


def audit_wifi(phase):
    jobs = json.loads((HERE / f"jobs_{phase}.json").read_text())
    registration_sha = sha(HERE / "registration.json")
    decision = None
    if phase == "final":
        decision = json.loads((HERE / "decision_final.json").read_text())
        require_committed(HERE / "decision_final.json")
        if jobs != decision["jobs"] or decision["registration_sha256"] != registration_sha:
            raise ValueError("final queue/freeze mismatch")
        for relative, expected in decision["evidence"].items():
            path = HERE / relative
            require_committed(path)
            if sha(path) != expected:
                raise ValueError("changed source freeze evidence")
    for job in jobs:
        out = HERE / "runs" / phase / job["id"]
        result = json.loads((out / "metrics.json").read_text())
        if result["job"] != job or result["weights_sha256"] != sha(out / "model.pt"):
            raise ValueError("run identity/weights mismatch")
        config = json.loads((out / "config.json").read_text())
        registration = json.loads((HERE / "registration.json").read_text())
        expected = registration["expected_params"][
            f"{job['dataset']}:{job['frontend']}:{job.get('fusion', 'pooled_concat')}"]
        if result["params"] != expected or config["job"] != job:
            raise ValueError("parameter/config mismatch")
        if result["registration_sha256"] != registration_sha or config["registration_sha256"] != registration_sha:
            raise ValueError("registration mismatch")
        if result["code_sha256"] != config["code_sha256"]:
            raise ValueError("config/result code mismatch")
        source, target = receivers(job["dataset"], job["fold"])
        held_count = 6 if job["dataset"] == "ManyRx" else 2
        held = [] if phase == "final" else source[-held_count:]
        fit_rx = source if phase == "final" else source[:-held_count]
        if config["roles"] != dict(source=source, target=target, fit_rx=fit_rx, held_rx=held):
            raise ValueError("source/target role mismatch")
        if phase == "final" and config["decision_sha256"] != sha(HERE / "decision_final.json"):
            raise ValueError("final config differs from frozen decision")
        history = list(csv.DictReader((out / "history.csv").open()))
        if len(history) != result["stop_epoch"] or result["selected_epoch"] not in range(1, len(history) + 1):
            raise ValueError("checkpoint/history mismatch")
        for row in history:
            if any(not np.isfinite(float(row[k])) for k in
                   ("train_loss", "train_acc", "val_ce", "val_acc", "lr")):
                raise ValueError("nonfinite history")
        if phase != "smoke" and job["recipe"] == "unified" and len(history) != job.get("epochs", 200):
            raise ValueError("incomplete unified training budget")
        expected_labels = {"CR", "CRD"} if phase == "final" else {"held_source"}
        if set(result["selected"]) != expected_labels:
            raise ValueError("missing or unexpected evaluation tasks")
        if set(result["predictions_sha256"]) != {f"preds_{label}.npz" for label in expected_labels}:
            raise ValueError("missing prediction artifact")
        for name, expected in result["predictions_sha256"].items():
            if sha(out / name) != expected:
                raise ValueError("predictions changed")
            label = name[len("preds_"):-len(".npz")]
            wanted_rx = target if phase == "final" else held
            if set(result["selected"][label]["per_rx"]) != set(wanted_rx):
                raise ValueError("prediction receiver set mismatch")
            per_rx = {}
            with np.load(out / name) as archive:
                if set(archive.files) != set(wanted_rx) | {f"y_{rx}" for rx in wanted_rx}:
                    raise ValueError("prediction archive receiver set mismatch")
                for rx, value in result["selected"][label]["per_rx"].items():
                    logits, y = archive[rx], archive[f"y_{rx}"]
                    if logits.dtype != np.float32 or not np.isfinite(logits).all():
                        raise ValueError("invalid float32 logits")
                    actual = metrics(logits, y, logits.shape[1])
                    same_metrics(actual, value)
                    per_rx[rx] = actual
            summary = summarize(per_rx)
            summary["packet_acc"] = sum(v["correct"] for v in per_rx.values()) / sum(v["total"] for v in per_rx.values())
            summary["ce_mean"] = float(np.mean([v["ce"] for v in per_rx.values()]))
            stored = result["selected"][label]["summary"]
            same_summary(summary, stored)
        if phase != "final" and any(key in result["selected"] for key in ("CR", "CRD")):
            raise ValueError("source run contains target metrics")
    print(json.dumps(dict(phase=phase, completed=len(jobs), audit="PASS")), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("smoke", "features", "replicate", "training",
                                        "training_replicate", "combine", "final"))
    args = parser.parse_args()
    audit_wifi(args.phase)


if __name__ == "__main__":
    main()
