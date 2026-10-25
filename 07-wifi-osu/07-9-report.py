#!/usr/bin/env python3
"""Incremental reports; incomplete stages are clearly marked, never called final."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def csv_write(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_report(phases, filename):
    rows = []
    expected = 0
    for phase in phases:
        queue = HERE / f"jobs_{phase}.json"
        if not queue.exists():
            continue
        jobs = json.loads(queue.read_text())
        expected += len(jobs)
        for job in jobs:
            path = HERE / "runs" / phase / job["id"] / "metrics.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            if any(k in result["selected"] for k in ("CR", "CRD")):
                raise ValueError("source report contains target metrics")
            held = result["selected"]["held_source"]["summary"]
            rows.append(dict(phase=phase, dataset=job["dataset"], fold=job["fold"], seed=job["seed"],
                             frontend=job["frontend"], factor=job.get("factor", "frontend"), id=job["id"],
                             packet_val_pct=100*result["selected_source_acc"],
                             held_rx_pct=100*held["acc_mean"], held_f1_pct=100*held["f1_mean"],
                             worst_held_rx_pct=100*held["worst_acc"], params=result["params"],
                             selected_epoch=result["selected_epoch"], seconds=result["seconds"]))
    csv_write(HERE / f"{filename}.csv", rows)
    lines = [f"# {filename}\n\nCompleted {len(rows)}/{expected} source fits. ",
             "No new target RX metrics are used in this report.\n\n",
             "Fold-specific selections must only use that fold's legal source RX. ",
             "The following table is descriptive, not a cross-fold selection rule.\n\n",
             "| Phase | Dataset | Fold | Seed | Frontend / factor | Packet val % | Held RX % | Worst RX % | Params |\n",
             "|---|---|---:|---:|---|---:|---:|---:|---:|\n"]
    for r in rows:
        lines.append(f"| {r['phase']} | {r['dataset']} | {r['fold']} | {r['seed']} | "
                     f"{r['frontend']} / {r['factor']} | {r['packet_val_pct']:.2f} | "
                     f"{r['held_rx_pct']:.2f} | {r['worst_held_rx_pct']:.2f} | {r['params']} |\n")
    (HERE / f"{filename}.md").write_text("".join(lines))
    print(filename, len(rows), "/", expected, flush=True)


def osu_report():
    rows, grouped = [], defaultdict(list)
    jobs = json.loads((HERE / "jobs_osu.json").read_text())
    for job in jobs:
        out = HERE / "runs" / "osu" / job["id"]
        path = out / "metrics.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        target = result["target"]
        with np.load(out / "preds_target.npz") as pred:
            logits, y = pred["logits"], pred["y"]
        classes = logits.shape[1]
        cm = np.bincount(y*classes+logits.argmax(1), minlength=classes**2).reshape(classes, classes)
        if not np.array_equal(cm, target["confusion"]):
            raise ValueError("OSU saved predictions disagree with scores")
        row = dict(method=job["method"], source=job["source"], target=job["target"], seed=job["seed"],
                   accuracy_pct=100*target["acc"], tx_equal_pct=100*target["tx_equal_acc"],
                   macro_f1_pct=100*target["macro_f1"], target_packets=target["total"],
                   selected_epoch=result["selected_epoch"], source_val_ce=result["source_val_ce"],
                   params=result["params"], seconds=result["seconds"])
        rows.append(row)
        grouped[job["method"]].append(row)
    csv_write(HERE / "osu_metrics.csv", rows)
    summary = []
    for method, values in grouped.items():
        summary.append(dict(method=method, fits=len(values), accuracy_pct=float(np.mean([r["accuracy_pct"] for r in values])),
                            tx_equal_pct=float(np.mean([r["tx_equal_pct"] for r in values])),
                            macro_f1_pct=float(np.mean([r["macro_f1_pct"] for r in values])),
                            fit_sd_pp=float(np.std([r["accuracy_pct"] for r in values], ddof=1)) if len(values)>1 else 0,
                            params=values[0]["params"]))
    csv_write(HERE / "osu_summary.csv", summary)
    complete = len(rows) == len(jobs)
    lines = ["# S07 OSU Results\n\n",
             f"Date: 2026-10-25. Completed {len(rows)}/{len(jobs)} registered fits. ",
             f"Status: {'complete registered comparison' if complete else 'INCOMPLETE; do not select using partial targets'}.\n\n",
             "Independent 25-class retraining, bidirectional RX1/RX2, seeds5101/5102/5103. ",
             "Source temporal validation chooses the checkpoint; target time blocks are disjoint. ",
             "Only one capture per TX/RX; directions are correlated and do not estimate a broad RX population.\n\n",
             "| Model | Fits | Packet acc % | TX-equal acc % | Macro-F1 % | Fit SD pp | Parameters |\n",
             "|---|---:|---:|---:|---:|---:|---:|\n"]
    for r in summary:
        lines.append(f"| {r['method']} | {r['fits']} | {r['accuracy_pct']:.2f} | {r['tx_equal_pct']:.2f} | "
                     f"{r['macro_f1_pct']:.2f} | {r['fit_sd_pp']:.2f} | {r['params']} |\n")
    lines.extend(["\nAll individual directions/seeds are in `osu_metrics.csv`. ",
                  "Detection failure attempts and full class counts are preserved. ",
                  "This is a controlled comparison, not a verified OSU SOTA claim. ",
                  "There is no local cross-day evidence; official download requests were denied by the proxy.\n\n",
                  "The nominal strict-IB detector pipeline did not validate HStem-HSTU transfer. ",
                  "The post-hoc same-RX future-block control and aliased frequency-offset diagnostics ",
                  "are in `S07_OSU_POSTHOC.md`; they cannot authorize target-based method tuning.\n"])
    (HERE / "S07_OSU_RESULTS.md").write_text("".join(lines))
    print("OSU", len(rows), "/", len(jobs), summary, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=("features", "training", "osu"))
    args = parser.parse_args()
    if args.part == "features":
        source_report(["features", "replicate"], "S07_WIFI_FEATURE_RESULTS")
    elif args.part == "training":
        source_report(["training", "training_replicate", "combine"], "S07_WIFI_TRAINING_RESULTS")
    else:
        osu_report()


if __name__ == "__main__":
    main()
