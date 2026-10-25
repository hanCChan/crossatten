#!/usr/bin/env python3
"""02-5  POST-HOC diagnostics on stage-02 confirmation fits (NOT pre-registered; descriptive only).

1. source -> unseen gap per arm (source_val acc - dev_val / confirm acc)
2. optimisation take-off: first epoch with source_val acc >= 80% (None = never within the budget);
   relation between take-off epoch and final dev_val accuracy (Spearman over all strict-IB fits)
3. per-fit table for the base and the factors carried into stage-03 candidates
4. confirmation receivers: seed-mean accuracy per RX for every arm
Output: summary/confirm_posthoc.md
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
TAKEOFF = 0.80


def load():
    runs = {}
    dirs = list((HERE / "runs/confirm").glob("*/metrics.json")) + \
        list((HERE.parent / "01-strong-baselines/runs/confirm").glob("cis_flat__*/metrics.json"))
    for m in dirs:
        r = json.loads(m.read_text())
        r["_hist"] = list(csv.DictReader(open(m.parent / "history.csv")))
        r["_takeoff"] = next((int(h["epoch"]) for h in r["_hist"] if float(h["val_acc"]) >= TAKEOFF), None)
        runs[(r["arm"], r["fold"], r["seed"])] = r
    return runs


def main():
    runs = load()
    arms = sorted({k[0] for k in runs}, key=lambda a: -np.mean([runs[k]["selected"]["dev_val"]["summary"]["acc_mean"] for k in runs if k[0] == a]))
    md = ["# Stage 02 post-hoc diagnostics (NOT pre-registered; descriptive)\n\n",
          f"Take-off = first epoch whose source_val accuracy >= {TAKEOFF:.0%} (source RX only; no target data).\n\n",
          "## 1-2. Generalisation gap and optimisation take-off (6 fits per arm)\n\n",
          "| arm | source_val | dev_val | gap dv | confirm | gap cf | take-off epochs (6 fits) | never | final train acc |\n|---|---|---|---|---|---|---|---|---|\n"]
    for a in arms:
        rs = [r for k, r in sorted(runs.items()) if k[0] == a]
        sv = 100 * np.mean([r["selected"]["source_val_acc"] for r in rs])
        dv = 100 * np.mean([r["selected"]["dev_val"]["summary"]["acc_mean"] for r in rs])
        cf = 100 * np.mean([r["selected"]["confirm"]["summary"]["acc_mean"] for r in rs])
        to = [r["_takeoff"] for r in rs]
        ta = 100 * np.mean([float(r["_hist"][-1]["train_acc"]) for r in rs])
        md.append(f"| {a} | {sv:.1f} | {dv:.1f} | {sv - dv:+.1f} | {cf:.1f} | {sv - cf:+.1f} | "
                  f"{', '.join('-' if t is None else str(t) for t in to)} | {sum(t is None for t in to)} | {ta:.1f} |\n")
    ib = [r for k, r in runs.items() if k[0].startswith("s02_") and k[0] != "s02_cis_rank16"]
    budget = {r["run_id"]: int(r["epochs_budget"]) for r in ib}
    x = [r["_takeoff"] if r["_takeoff"] is not None else budget[r["run_id"]] + 1 for r in ib]
    y = [100 * r["selected"]["dev_val"]["summary"]["acc_mean"] for r in ib]
    rho = spearmanr(x, y)[0]
    md.append(f"\nSpearman(take-off epoch, dev_val acc) over {len(ib)} strict-IB fits = {rho:+.2f} "
              "(never-take-off fits ranked last).\n")
    md.append("\n## 3. Per-fit values (source_val / dev_val / confirm, %)\n\n| arm | fold | seed | take-off | best epoch | source_val | dev_val | confirm |\n|---|---|---|---|---|---|---|---|\n")
    for a in arms:
        for (arm, fo, se), r in sorted(runs.items()):
            if arm == a:
                s = r["selected"]
                md.append(f"| {a} | {fo} | {se} | {r['_takeoff'] if r['_takeoff'] is not None else '-'} | {r['best_val_epoch']} | "
                          f"{100 * s['source_val_acc']:.1f} | {100 * s['dev_val']['summary']['acc_mean']:.1f} | {100 * s['confirm']['summary']['acc_mean']:.1f} |\n")
    rxs = ["b200_1", "n210_1", "rtl_4", "rtl_8"]
    md.append("\n## 4. Confirmation receivers: seed-mean accuracy per RX (%), both folds pooled (6 fits)\n\n| arm | " + " | ".join(rxs) + " | mean | worst |\n|" + "---|" * (len(rxs) + 3) + "\n")
    for a in arms:
        rs = [r for k, r in runs.items() if k[0] == a]
        v = [100 * np.mean([r["selected"]["confirm"]["summary"]["per_rx_acc"][rx] for r in rs]) for rx in rxs]
        md.append(f"| {a} | " + " | ".join(f"{t:.1f}" for t in v) + f" | {np.mean(v):.1f} | {min(v):.1f} |\n")
    (HERE / "summary/confirm_posthoc.md").write_text("".join(md))
    print("".join(md))


if __name__ == "__main__":
    main()
