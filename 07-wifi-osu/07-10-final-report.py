#!/usr/bin/env python3
"""Frozen WiSig CR/CRD comparison with receiver-clustered, seed-unpaired intervals."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    decision = json.loads((HERE / "decision_final.json").read_text())
    jobs = decision["jobs"]
    rows, fits = [], defaultdict(list)
    for job in jobs:
        path = HERE / "runs" / "final" / job["id"] / "metrics.json"
        result = json.loads(path.read_text())
        if result["job"] != job:
            raise ValueError("final run differs from its whitelist")
        for task in ("CR", "CRD"):
            entry = result["selected"][task]
            fits[(job["dataset"], job["pipeline"], task)].append(dict(
                fold=job["fold"], seed=job["seed"], acc=entry["summary"]["acc_mean"],
                packet_acc=entry["summary"]["packet_acc"], f1=entry["summary"]["f1_mean"],
                params=result["params"], seconds=result["seconds"]))
            for rx, value in entry["per_rx"].items():
                rows.append(dict(dataset=job["dataset"], pipeline=job["pipeline"], task=task,
                                 fold=job["fold"], seed=job["seed"], rx=rx,
                                 accuracy_pct=100*value["acc"], macro_f1_pct=100*value["macro_f1"],
                                 correct=value["correct"], total=value["total"], params=result["params"]))
    summary = []
    for (dataset, pipeline, task), values in fits.items():
        seeds = {s: np.mean([v["acc"] for v in values if v["seed"] == s])
                 for s in {v["seed"] for v in values}}
        folds = {f: np.mean([v["acc"] for v in values if v["fold"] == f])
                 for f in {v["fold"] for v in values}}
        selected = [r for r in rows if (r["dataset"], r["pipeline"], r["task"]) == (dataset, pipeline, task)]
        by_rx = defaultdict(list)
        for r in selected:
            by_rx[r["rx"]].append(r["accuracy_pct"])
        summary.append(dict(dataset=dataset, pipeline=pipeline, task=task, fits=len(values),
                            rx_accuracy_pct=100*np.mean([v["acc"] for v in values]),
                            packet_accuracy_pct=100*np.mean([v["packet_acc"] for v in values]),
                            rx_macro_f1_pct=100*np.mean([v["f1"] for v in values]),
                            seed_sd_pp=100*np.std(list(seeds.values()), ddof=1),
                            fold_sd_pp=100*np.std(list(folds.values()), ddof=1),
                            worst_rx_seed_mean_pct=min(np.mean(a) for a in by_rx.values()),
                            parameters=";".join(str(p) for p in sorted({v["params"] for v in values}))))
    save_csv(HERE / "wifi_final_receiver_metrics.csv", rows)
    save_csv(HERE / "wifi_final_metrics.csv", summary)
    frozen = list(csv.DictReader((ROOT / "06-paper-validation/summary/paper/receiver_metrics.csv").open()))
    intervals = []
    rng = np.random.default_rng(7101)
    for row in summary:
        dataset, pipeline, task = row["dataset"], row["pipeline"], row["task"]
        if task != "CRD":
            continue
        ours = defaultdict(list)
        for r in rows:
            if (r["dataset"], r["pipeline"], r["task"]) == (dataset, pipeline, task):
                ours[r["rx"]].append(r["accuracy_pct"])
        for baseline in ("IB-RFF", "WaveMLP", "FAN_source_selected"):
            theirs = {r["rx"]: float(r["seed_mean_accuracy_pct"]) for r in frozen
                      if (r["dataset"], r["model"], r["task"]) == (dataset, baseline, task)}
            if set(ours) != set(theirs):
                raise ValueError("frozen comparison receiver set differs")
            names = sorted(ours)
            delta = np.array([np.mean(ours[rx]) - theirs[rx] for rx in names])
            bootstrap = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(1)
            intervals.append(dict(dataset=dataset, pipeline=pipeline, baseline=baseline,
                                  task=task, receiver_count=len(names), delta_pp=float(delta.mean()),
                                  ci95_low_pp=float(np.quantile(bootstrap, .025)),
                                  ci95_high_pp=float(np.quantile(bootstrap, .975)),
                                  min_rx_delta_pp=float(delta.min()),
                                  improved_receivers=int((delta>0).sum()),
                                  inference="RX-cluster bootstrap of seed-mean deltas; seeds not paired"))
    save_csv(HERE / "wifi_comparison_intervals.csv", intervals)
    lines = ["# S07 WiSig Final Results\n\n",
             f"Completed all {len(jobs)} frozen final fits. Four RX folds x three seeds. ",
             "Source-selected configurations are disclosed in `decision_final.json`.\n\n",
             "CRD is primary. The public targets were previously inspected; this is not a fresh blind test. ",
             "S06 external fair baselines are frozen and use the same normalization and cell budgets. ",
             "S07 seeds4101/4102/4103 are not paired with S06 training seeds. ",
             "Receiver-clustered intervals condition on trained models and do not represent independent datasets.\n\n",
             "| Dataset | Pipeline | Task | RX acc % | Packet acc % | Macro-F1 % | Seed SD pp | Worst RX % | Params |\n",
             "|---|---|---|---:|---:|---:|---:|---:|---:|\n"]
    for r in summary:
        lines.append(f"| {r['dataset']} | {r['pipeline']} | {r['task']} | {r['rx_accuracy_pct']:.2f} | "
                     f"{r['packet_accuracy_pct']:.2f} | {r['rx_macro_f1_pct']:.2f} | "
                     f"{r['seed_sd_pp']:.2f} | {r['worst_rx_seed_mean_pct']:.2f} | {r['parameters']} |\n")
    lines.extend(["\n## CRD Baseline Differences\n\n",
                  "| Dataset | Pipeline | Baseline | Mean delta pp | RX bootstrap 95% interval | Better RX |\n",
                  "|---|---|---|---:|---|---:|\n"])
    for r in intervals:
        lines.append(f"| {r['dataset']} | {r['pipeline']} | {r['baseline']} | {r['delta_pp']:+.2f} | "
                     f"[{r['ci95_low_pp']:+.2f}, {r['ci95_high_pp']:+.2f}] | "
                     f"{r['improved_receivers']}/{r['receiver_count']} |\n")
    lines.append("\nNo method should be selected after seeing this table. All registered pipelines are reported. "
                 "These results support only tested-protocol statements; SOTA requires a separate primary-source review.\n")
    (HERE / "S07_WIFI_FINAL_RESULTS.md").write_text("".join(lines))
    print(json.dumps(dict(final_fits=len(jobs), summary=summary, intervals=intervals)), flush=True)


if __name__ == "__main__":
    main()
