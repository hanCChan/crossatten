#!/usr/bin/env python3
"""Verify completed run identities, hashes, prediction counts and float32 metrics."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import numpy as np
from xa.paper_train import metrics, sha


def audit_wifi(phase):
    jobs = json.loads((HERE / f"jobs_{phase}.json").read_text())
    for job in jobs:
        out = HERE / "runs" / phase / job["id"]
        result = json.loads((out / "metrics.json").read_text())
        if result["job"] != job or result["weights_sha256"] != sha(out / "model.pt"):
            raise ValueError("run identity/weights mismatch")
        for name, expected in result["predictions_sha256"].items():
            if sha(out / name) != expected:
                raise ValueError("predictions changed")
            label = name[len("preds_"):-len(".npz")]
            with np.load(out / name) as archive:
                for rx, value in result["selected"][label]["per_rx"].items():
                    logits, y = archive[rx], archive[f"y_{rx}"]
                    if logits.dtype != np.float32 or not np.isfinite(logits).all():
                        raise ValueError("invalid float32 logits")
                    actual = metrics(logits, y, logits.shape[1])
                    if actual["confusion"] != value["confusion"] or actual["total"] != value["total"]:
                        raise ValueError("prediction metrics/count mismatch")
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
