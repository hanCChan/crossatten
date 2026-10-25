#!/usr/bin/env python3
"""Explain immutable source-only decisions without reading final target results."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from xa.paper_train import require_committed, sha


def evidence(phase, jobs):
    rows = []
    for job in jobs:
        path = HERE / "runs" / phase / job["id"] / "metrics.json"
        require_committed(path)
        value = json.loads(path.read_text())
        if value["job"] != job or set(value["selected"]) != {"held_source"}:
            raise ValueError("source explanation contains non-source evidence")
        rows.append((job, value["selected"]["held_source"]))
    return rows


def main():
    freeze_path = HERE / "decision_final.json"
    require_committed(freeze_path)
    frozen = json.loads(freeze_path.read_text())
    combined = json.loads((HERE / "decision_combine.json").read_text())["decisions"]
    first = evidence("training", json.loads((HERE / "jobs_training.json").read_text()))
    repeat = evidence("training_replicate", json.loads((HERE / "jobs_training_replicate.json").read_text()))
    lines = [
        "# S07 Source-Only Decision Trace\n\n",
        "All choices were frozen before final target evaluation. This explanation reads only ",
        "registered source fits and final configuration declarations; no CR/CRD metrics.\n\n",
        "## Training Factor Replication\n\n",
        "Two positive seed-matched held-RX deltas, mean >=1 pp and worst individual held-RX ",
        "delta >=-2 pp are required for combination support. A factor with the highest source ",
        "rank can remain the single control even if it fails this stricter gate.\n\n",
        "| Dataset | Fold | Factor | Delta seed3101 pp | Delta seed3102 pp | Mean pp | Worst RX delta pp | Supported |\n",
        "|---|---:|---|---:|---:|---:|---:|---|\n",
    ]
    rows = []
    for key, selected in combined.items():
        dataset, fold_text = key.split(":")
        fold = int(fold_text)
        pairs = [{j["factor"]: e for j, e in phase
                  if j["dataset"] == dataset and j["fold"] == fold} for phase in (first, repeat)]
        for factor in sorted(set(pairs[1]) - {"unified"}):
            deltas, worst = [], []
            for phase in pairs:
                candidate, control = phase[factor], phase["unified"]
                deltas.append(100 * (candidate["summary"]["acc_mean"] - control["summary"]["acc_mean"]))
                worst.extend(100 * (r["acc"] - control["per_rx"][rx]["acc"])
                             for rx, r in candidate["per_rx"].items())
            row = dict(dataset=dataset, fold=fold, factor=factor,
                       delta_seed3101_pp=deltas[0], delta_seed3102_pp=deltas[1],
                       mean_delta_pp=float(np.mean(deltas)), worst_rx_delta_pp=min(worst),
                       supported=factor in selected["supported_factors"])
            rows.append(row)
            lines.append(f"| {dataset} | {fold} | {factor} | {deltas[0]:+.2f} | {deltas[1]:+.2f} | "
                         f"{row['mean_delta_pp']:+.2f} | {min(worst):+.2f} | {row['supported']} |\n")
    lines.extend([
        "\n## Frozen Pipelines\n\n",
        "| Dataset | Fold | Frontend | Supported factors | Strongest single | Final training pipeline | Final combination pipeline |\n",
        "|---|---:|---|---|---|---|---|\n",
    ])
    for key, selected in combined.items():
        dataset, fold_text = key.split(":")
        fold = int(fold_text)
        jobs = {j["pipeline"]: j for j in frozen["jobs"]
                if j["dataset"] == dataset and j["fold"] == fold and j["seed"] == 4101}
        def description(job):
            recipe = job["recipe"]
            if job.get("noise") == "mild":
                recipe += "; AWGN10-50"
            if job.get("fusion") == "gate":
                recipe += "; gate"
            if job.get("checkpoint") == "last":
                recipe += f"; source-frozen {job['epochs']} epochs"
            return recipe
        lines.append(f"| {dataset} | {fold} | {selected['frontend']} | "
                     f"{', '.join(selected['supported_factors']) or 'none'} | "
                     f"{selected['strongest_single']} | {description(jobs['source_training'])} | "
                     f"{description(jobs['source_combination'])} |\n")
    lines.extend([
        "\nNo official/unified recipe mixing, stacked physical augmentations or target-specific ",
        "configuration changes are allowed. Gate is available only for the source-selected split ",
        "frontend. All three pipelines must be reported; identical configurations are not ",
        "independent methods. The first-pass factor table contains additional negative candidates ",
        "that were not selected for replication.\n\n",
        "Frozen final decision SHA256: `", sha(freeze_path), "`.\n",
    ])
    with (HERE / "source_factor_gates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "S07_SOURCE_DECISIONS.md").write_text("".join(lines))
    print("wrote source-only decisions and paired factor gate evidence")


if __name__ == "__main__":
    main()
