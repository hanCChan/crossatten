#!/usr/bin/env python3
"""01-5  POST-HOC diagnostics on stage-01 confirmation fits (NOT pre-registered; descriptive only).

1. source->unseen generalisation gap per arm: source_val acc - dev_val / confirm acc
2. extra paired comparisons that were not in the registered PAIRS list (AWGN on TrueIB; system-level TrueIB vs CIS)
3. per-seed accuracy of the TrueIB arms (stability) and end-of-training train accuracy (convergence)
4. per-receiver confirm-RX table for the key arms
Output: summary/confirm_posthoc.md
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KEY = ["xa_p1fixed", "xa_cos", "xa_cos_awgn", "xa_lp200_cos_awgn", "xa_nooob_cos_awgn", "xa_trueib_cos", "xa_trueib_cos_awgn",
       "tf_cos_awgn", "cis_flat_noawgn", "cis_flat", "cis_ib_flat", "cis_gap"]
POSTHOC_PAIRS = [("xa_trueib_cos_awgn", "xa_trueib_cos", "AWGN on TrueIB"),
                 ("xa_trueib_cos_awgn", "cis_flat", "SYSTEM: TrueIB+AWGN (0.58M) vs Shen-CIS official (13.4M)"),
                 ("xa_trueib_cos_awgn", "cis_ib_flat", "SYSTEM: TrueIB+AWGN vs strict-IB CIS (13.4M)")]


def main():
    runs = {}
    for m in sorted((HERE / "runs/confirm").glob("*/metrics.json")):
        r = json.loads(m.read_text())
        r["_hist"] = list(csv.DictReader(open(m.parent / "history.csv")))
        runs[(r["arm"], r["fold"], r["seed"])] = r
    arms = sorted({k[0] for k in runs})
    md = ["# Stage 01 post-hoc diagnostics (NOT pre-registered; descriptive)\n\n",
          "## 1. Source -> unseen-RX generalisation gap (selected checkpoint, mean over 6 fits, %)\n\n",
          "| arm | source_val | dev_val | gap dv | confirm | gap cf | final train acc (AWGN-corrupted if AWGN) |\n|---|---|---|---|---|---|---|\n"]
    for a in sorted(arms, key=lambda a: -np.mean([runs[k]["selected"]["dev_val"]["summary"]["acc_mean"] for k in runs if k[0] == a])):
        rs = [r for k, r in runs.items() if k[0] == a]
        sv = 100 * np.mean([r["selected"]["source_val_acc"] for r in rs])
        dv = 100 * np.mean([r["selected"]["dev_val"]["summary"]["acc_mean"] for r in rs])
        cf = 100 * np.mean([r["selected"]["confirm"]["summary"]["acc_mean"] for r in rs])
        ta = 100 * np.mean([float(r["_hist"][-1]["train_acc"]) for r in rs])
        md.append(f"| {a} | {sv:.1f} | {dv:.1f} | {sv - dv:+.1f} | {cf:.1f} | {sv - cf:+.1f} | {ta:.1f} |\n")
    md.append("\n## 2. Extra paired comparisons (delta = B - A, pp; 6 pairs = 2 folds x 3 seeds)\n\n"
              "| B | A | what | role | mean | min | max | +pairs | worst-RX(B) - worst-RX(A) |\n|---|---|---|---|---|---|---|---|---|\n")
    for b, a, what in POSTHOC_PAIRS:
        for role in ("dev_val", "confirm"):
            ds, wb, wa = [], defaultdict(list), defaultdict(list)
            for (arm, fo, se), rb in runs.items():
                if arm == b and (a, fo, se) in runs:
                    ra = runs[(a, fo, se)]
                    ds.append(100 * (rb["selected"][role]["summary"]["acc_mean"] - ra["selected"][role]["summary"]["acc_mean"]))
                    for rx, v in rb["selected"][role]["summary"]["per_rx_acc"].items():
                        wb[(fo, rx)].append(100 * v)
                        wa[(fo, rx)].append(100 * ra["selected"][role]["summary"]["per_rx_acc"][rx])
            wd = min(np.mean(v) for v in wb.values()) - min(np.mean(v) for v in wa.values())
            md.append(f"| {b} | {a} | {what} | {role} | {np.mean(ds):+.2f} | {np.min(ds):+.2f} | {np.max(ds):+.2f} | "
                      f"{int((np.array(ds) > 0).sum())}/{len(ds)} | {wd:+.2f} |\n")
    md.append("\n## 3. Per-fit accuracy of the strict-IB CrossAttn arms (dev_val / confirm, %)\n\n| arm | fold | seed | best epoch | source_val | dev_val | confirm |\n|---|---|---|---|---|---|---|\n")
    for a in ("xa_trueib_cos", "xa_trueib_cos_awgn"):
        for (arm, fo, se), r in sorted(runs.items()):
            if arm == a:
                s = r["selected"]
                md.append(f"| {a} | {fo} | {se} | {r['best_val_epoch']} | {100 * s['source_val_acc']:.1f} | "
                          f"{100 * s['dev_val']['summary']['acc_mean']:.1f} | {100 * s['confirm']['summary']['acc_mean']:.1f} |\n")
    md.append("\n## 4. Confirmation receivers: seed-mean accuracy per RX (%; both folds evaluate the same 4 RX)\n\n")
    rxs = ["b200_1", "n210_1", "rtl_4", "rtl_8"]
    md.append("| arm | fold | " + " | ".join(rxs) + " | mean |\n|" + "---|" * (len(rxs) + 3) + "\n")
    for a in KEY:
        for fo in (0, 1):
            rs = [r for (arm, f, s), r in runs.items() if arm == a and f == fo]
            if not rs:
                continue
            v = [100 * np.mean([r["selected"]["confirm"]["summary"]["per_rx_acc"][rx] for r in rs]) for rx in rxs]
            md.append(f"| {a} | {fo} | " + " | ".join(f"{x:.1f}" for x in v) + f" | {np.mean(v):.1f} |\n")
    (HERE / "summary/confirm_posthoc.md").write_text("".join(md))
    print("".join(md))


if __name__ == "__main__":
    main()
