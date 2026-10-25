#!/usr/bin/env python3
"""Frozen WiSig CR/CRD comparison with receiver-clustered, seed-unpaired intervals."""
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
from xa.paper_train import require_committed, sha


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    spec = importlib.util.spec_from_file_location("s07_final_audit", HERE / "07-7-audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    audit.audit_wifi("final")
    decision = json.loads((HERE / "decision_final.json").read_text())
    jobs = decision["jobs"]
    expected = {(dataset, pipeline, fold, seed)
                for dataset in ("ManyRx", "ManySig")
                for pipeline in ("source_frontend", "source_training", "source_combination")
                for fold in range(4) for seed in (4101, 4102, 4103)}
    actual = {(j["dataset"], j["pipeline"], j["fold"], j["seed"]) for j in jobs}
    if actual != expected or len(actual) != len(jobs):
        raise ValueError("missing or duplicate frozen final cell")
    rows, run_rows, fits = [], [], defaultdict(list)
    for job in jobs:
        path = HERE / "runs" / "final" / job["id"] / "metrics.json"
        result = json.loads(path.read_text())
        if result["job"] != job:
            raise ValueError("final run differs from its whitelist")
        run_rows.append(dict(
            id=job["id"], dataset=job["dataset"], fold=job["fold"], seed=job["seed"],
            pipeline=job["pipeline"], frontend=job["frontend"], recipe=job["recipe"],
            augmentation=job.get("augmentation", "none"),
            checkpoint=job.get("checkpoint", "packet"), epoch_budget=job.get("epochs", 200),
            stop_epoch=result["stop_epoch"], selected_epoch=result["selected_epoch"],
            selected_source_accuracy_pct=100 * result["selected_source_acc"],
            selected_source_ce=result["selected_source_ce"],
            CR_rx_accuracy_pct=100 * result["selected"]["CR"]["summary"]["acc_mean"],
            CRD_rx_accuracy_pct=100 * result["selected"]["CRD"]["summary"]["acc_mean"],
            CR_to_CRD_drop_pp=100 * (result["selected"]["CR"]["summary"]["acc_mean"] -
                                    result["selected"]["CRD"]["summary"]["acc_mean"]),
            params=result["params"], training_and_evaluation_seconds=result["seconds"]))
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
                            pooled_packet_accuracy_pct=100*sum(r["correct"] for r in selected) /
                                                       sum(r["total"] for r in selected),
                            rx_macro_f1_pct=100*np.mean([v["f1"] for v in values]),
                            seed_sd_pp=100*np.std(list(seeds.values()), ddof=1),
                            fold_sd_pp=100*np.std(list(folds.values()), ddof=1),
                            worst_rx_seed_mean_pct=min(np.mean(a) for a in by_rx.values()),
                            parameters=";".join(str(p) for p in sorted({v["params"] for v in values}))))
    save_csv(HERE / "wifi_final_receiver_metrics.csv", rows)
    save_csv(HERE / "wifi_final_run_metrics.csv", run_rows)
    save_csv(HERE / "wifi_final_metrics.csv", summary)
    baseline_path = ROOT / "06-paper-validation/summary/paper/receiver_metrics.csv"
    baseline_commit = json.loads((HERE / "registration.json").read_text())["baseline"]
    old = subprocess.check_output(
        ["git", "show", f"{baseline_commit}:{baseline_path.relative_to(ROOT).as_posix()}"], cwd=ROOT
    )
    if hashlib.sha256(old).hexdigest() != sha(baseline_path):
        raise ValueError("historical baseline comparison changed since S07 registration")
    require_committed(baseline_path)
    with baseline_path.open() as stream:
        frozen = list(csv.DictReader(stream))
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
        for baseline in ("W1", "IB-RFF", "WaveMLP", "FAN_source_selected"):
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
    equivalents = {}
    for dataset in ("ManyRx", "ManySig"):
        for pipeline in ("source_frontend", "source_training", "source_combination"):
            specification = [{k: v for k, v in j.items()
                              if k not in ("id", "phase", "pipeline", "factor")}
                             for j in jobs if j["dataset"] == dataset and j["pipeline"] == pipeline]
            equivalents[f"{dataset}:{pipeline}"] = specification
    duplicated = [(a, b) for a in equivalents for b in equivalents if a < b
                  and equivalents[a] == equivalents[b]]
    lines = ["# S07 WiSig Final Results\n\n",
             f"Completed all {len(jobs)} frozen final fits. Four RX folds x three seeds. ",
             "Source-selected configurations are disclosed in `decision_final.json`.\n\n",
             "CRD is primary. The public targets were previously inspected; this is not a fresh blind test. ",
             "S06 external fair baselines are frozen and use the same normalization and cell budgets. ",
             "S07 seeds4101/4102/4103 are not paired with S06 training seeds. ",
             "Receiver-clustered intervals condition on trained models and do not represent independent datasets. ",
             "Intervals are descriptive and not corrected for multiple pipeline/baseline comparisons. ",
             "`Packet acc` is the official four-fold mean of packet-weighted scores; ",
             "the CSV additionally contains accuracy pooled over all test packets/folds/seeds.\n\n",
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
    lines.extend(["\n## Historical Released-Weight References\n\n",
                  "Separate unchanged published-weight verification from S05, not fair S07 multi-seed fits. ",
                  "These CRD aggregates use official packet weighting; do not equate them with RX-equal scores.\n\n",
                  "| Method | ManyRx CRD % | ManySig CRD % |\n",
                  "|---|---:|---:|\n",
                  "| IB-RFF | 89.19 | 93.57 |\n",
                  "| WaveMLP | 88.68 | 93.13 |\n"])
    lines.extend(["\n## Frozen Per-Fold Configurations\n\n",
                  "| Dataset | Fold | Pipeline | Frontend | Recipe | Augmentation | Balanced | Noise | Checkpoint | Epoch budget |\n",
                  "|---|---:|---|---|---|---|---|---|---|---:|\n"])
    for j in jobs:
        if j["seed"] != 4101:
            continue
        lines.append(f"| {j['dataset']} | {j['fold']} | {j['pipeline']} | {j['frontend']} | "
                     f"{j['recipe']} | {j.get('augmentation', 'none')} | {j.get('balanced', False)} | "
                     f"{j.get('noise', 'default')} | {j.get('checkpoint', 'packet')} | {j.get('epochs', 200)} |\n")
    if duplicated:
        lines.append("\nIdentical registered pipelines, not independent methods: " +
                     "; ".join(f"`{a}` = `{b}`" for a, b in duplicated) + ".\n")
    lines.append("\nNo method should be selected after seeing this table. All registered pipelines are reported. "
                 "These results support only tested-protocol statements; SOTA requires a separate primary-source review.\n")
    (HERE / "S07_WIFI_FINAL_RESULTS.md").write_text("".join(lines))
    report_files = ("wifi_final_receiver_metrics.csv", "wifi_final_run_metrics.csv", "wifi_final_metrics.csv",
                    "wifi_comparison_intervals.csv", "S07_WIFI_FINAL_RESULTS.md")
    evidence = dict(
        final_decision_sha256=sha(HERE / "decision_final.json"),
        baseline_commit=baseline_commit, baseline_receiver_metrics_sha256=sha(baseline_path),
        final_fits={j["id"]: sha(HERE / "runs/final" / j["id"] / "metrics.json") for j in jobs},
        files={name: sha(HERE / name) for name in report_files},
        duplicate_pipelines=duplicated,
        uncertainty="RX-cluster seed-mean bootstrap, conditional on trained models; not full seed/dataset uncertainty",
    )
    (HERE / "wifi_final_report_evidence.json").write_text(json.dumps(evidence, indent=1) + "\n")
    print(json.dumps(dict(final_fits=len(jobs), summary=summary, intervals=intervals)), flush=True)


if __name__ == "__main__":
    main()
