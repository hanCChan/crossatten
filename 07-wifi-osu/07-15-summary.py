#!/usr/bin/env python3
"""Summarize the bounded stage; incomplete experiments never become final claims."""
import csv
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHASES = ("features", "replicate", "training", "training_replicate", "combine", "final")


def csv_rows(path):
    return list(csv.DictReader(path.open())) if path.is_file() else []


def main():
    counts = []
    failures = []
    for phase in (*PHASES, "smoke", "osu"):
        path = HERE / f"jobs_{phase}.json"
        jobs = json.loads(path.read_text()) if path.is_file() else []
        completed = sum((HERE / "runs" / phase / j["id"] / "metrics.json").is_file() for j in jobs)
        status_path = HERE / f"jobs_{phase}.status.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        failures.extend((phase, name) for name in status.get("failed", []))
        attempts = list((HERE / "runs" / phase).glob("*/attempt_*"))
        counts.append(dict(phase=phase, registered=len(jobs), completed=completed,
                           failed=len(status.get("failed", [])), archived_attempts=len(attempts)))
    formal_wifi = [r for r in counts if r["phase"] in PHASES]
    wifi_complete = all(r["registered"] and r["registered"] == r["completed"] for r in formal_wifi)
    osu_count = next(r for r in counts if r["phase"] == "osu")
    osu_complete = osu_count["registered"] == osu_count["completed"] == 24
    metrics = csv_rows(HERE / "wifi_final_metrics.csv") if wifi_complete else []
    if wifi_complete and len(metrics) != 12:
        raise ValueError("complete final stage is missing its 12-row report")
    osu = csv_rows(HERE / "osu_summary.csv")
    intervals = csv_rows(HERE / "wifi_comparison_intervals.csv") if wifi_complete else []
    note = "Complete registered exploration; STOP, no extra target-directed search." if wifi_complete else \
        "WiSig INCOMPLETE. Source screening numbers are not CRD results."
    lines = ["# Stage 07 Results\n\n",
             f"Generated {datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')}. {note}\n\n",
             "Baseline `ebe3225`, branch `exp/s07-wifi-osu`. S00-S06 are unchanged. ",
             "No automatic merge, GitHub push, model-weight publication or target-driven extension.\n\n",
             "## Registered Execution\n\n",
             "| Phase | Registered | Completed | Failed | Archived interrupted attempts |\n",
             "|---|---:|---:|---:|---:|\n"]
    for row in counts:
        lines.append(f"| {row['phase']} | {row['registered']} | {row['completed']} | "
                     f"{row['failed']} | {row['archived_attempts']} |\n")
    lines.append("\nInterrupted attempts and detector pilot failure are retained; "
                 "they are not additional successful models. The WiSig formal cap is 280 fits.\n")
    lines.extend(["\n## WiSig\n\n",
                  "Four official RX folds, three final seeds; CRD primary. ",
                  "Selections use each fold's source RX days1-2 only. S06 public targets are not new blind tests. ",
                  "All three pipelines are reported, with fold-specific source decisions disclosed; ",
                  "their configurations may coincide. Reused S06 baseline seeds are unpaired with S07.\n\n"])
    if metrics:
        lines.extend(["| Dataset | Pipeline | CRD RX acc % | Macro-F1 % | Seed SD pp | Worst RX % |\n",
                      "|---|---|---:|---:|---:|---:|\n"])
        for row in metrics:
            if row["task"] != "CRD":
                continue
            lines.append(f"| {row['dataset']} | {row['pipeline']} | {float(row['rx_accuracy_pct']):.2f} | "
                         f"{float(row['rx_macro_f1_pct']):.2f} | {float(row['seed_sd_pp']):.2f} | "
                         f"{float(row['worst_rx_seed_mean_pct']):.2f} |\n")
        lines.append("\nFull CR/CRD, packet/RX weightings, baseline differences and conditional RX bootstrap "
                     "intervals: `S07_WIFI_FINAL_RESULTS.md`. No target-ranked pipeline is designated the method.\n")
    else:
        lines.append("Formal target evaluation is not yet complete. Do not substitute internal held-source scores.\n")
    features = csv_rows(HERE / "S07_WIFI_FEATURE_RESULTS.csv")
    first = defaultdict(list)
    for row in features:
        if row["phase"] == "features":
            first[(row["dataset"], row["frontend"])].append(float(row["held_rx_pct"]))
    if first:
        lines.extend(["\nSingle-seed first-pass source evidence, descriptive only:\n\n",
                      "| Dataset | Legacy held RX % | Frequency-only % | Time/frequency split % |\n",
                      "|---|---:|---:|---:|\n"])
        for dataset in ("ManyRx", "ManySig"):
            values = [np.mean(first[(dataset, f)]) for f in ("legacy", "frequency", "split")]
            lines.append(f"| {dataset} | " + " | ".join(f"{v:.2f}" for v in values) + " |\n")
        lines.append("\nRemoving all frequency information did not solve generalization. "
                     "Pre-LN keeps token RMS near unit scale, but trained sigmoid scores can still saturate. "
                     "Read-only masking is out-of-distribution sensitivity, not causal feature importance.\n")
    lines.extend(["\n## OSU\n\n",
                  f"Registered comparison {'complete' if osu_complete else 'incomplete'}: 25 TX, RX1/RX2 both directions, "
                  "three seeds. 50 local raw records; guarded packet extraction instead of arbitrary windows. ",
                  "Nominal 125 kHz/SF7 is source-pilot supported, not certified by metadata. ",
                  "Only one continuous capture per TX/RX; buffered temporal train/validation/test blocks. ",
                  "Both RX pilot records were checked during detector development, so raw target preparation was not blind.\n\n",
                  "| Model | Cross-RX packet acc % | TX-equal acc % | Parameters |\n",
                  "|---|---:|---:|---:|\n"])
    for row in osu:
        lines.append(f"| {row['method']} | {float(row['accuracy_pct']):.2f} | "
                     f"{float(row['tx_equal_pct']):.2f} | {row['params']} |\n")
    lines.append("\nHStem-HSTU did not validate cross-RX transfer under this frozen pipeline. "
                 "Post-hoc same-RX future-block accuracy is 91.24%, versus 1.42% across RX. "
                 "This is not evidence that CFO compensation would restore transfer. "
                 "No OSU cross-day data are local; official requests were denied by the configured proxy.\n")
    lines.extend(["\n## Decision And Claims\n\n"])
    if metrics:
        strong = [r for r in intervals if r["baseline"] in ("IB-RFF", "WaveMLP")]
        any_dominance = []
        for pipeline in ("source_frontend", "source_training", "source_combination"):
            pairs = [r for r in strong if r["pipeline"] == pipeline]
            if len(pairs) == 4 and all(float(r["ci95_low_pp"]) > 0 for r in pairs):
                any_dominance.append(pipeline)
        if any_dominance:
            lines.append("At least one pre-registered pipeline has positive conditional RX-cluster intervals "
                         "against both fair references on both datasets. This is protocol-specific evidence, "
                         "not permission to select a winner from target results or to claim global SOTA. ")
        else:
            lines.append("The registered comparison does not establish consistent conditional-RX superiority "
                         "over both strong fair baselines on both WiSig datasets. ")
        lines.append("OSU independent cross-RX validation failed. A general LoRa/WiFi SOTA claim is not supported. "
                     "Keep the Shen findings, disclose WiSig/OSU applicability limits, and stop this exploration.\n")
    else:
        lines.append("Wait for all frozen WiSig fits before any WiSig competitiveness decision. "
                     "OSU does not currently support independent LoRa transfer or SOTA. No further OSU "
                     "method search is authorized by its failed target results.\n")
    lines.extend(["\n## Evidence\n\n",
                  "- `registration.json`, `decision_*.json`, `jobs_*.json`: budgets and source lineage.\n",
                  "- `runs/`: complete configs, learning curves, float32 predictions and failure attempts; weights ignored.\n",
                  "- `S07_CODE_AUDIT.md`, `S07_SOURCE_MECHANISMS.md`: code and read-only source diagnostics.\n",
                  "- `S07_WIFI_FEATURE_RESULTS.md`, `S07_WIFI_TRAINING_RESULTS.md`, "
                  "`S07_WIFI_FINAL_RESULTS.md`: sequential outcomes.\n",
                  "- `S07_OSU_DATA_CARD.md`, `S07_OSU_RESULTS.md`, `S07_OSU_POSTHOC.md`: provenance and negative results.\n",
                  "- `S07_COSTS.md`, `S07_EXECUTION.md`: measured inference costs and execution-time accounting.\n",
                  "- `CONCLUSION_ZH.md`: complete audited Chinese research conclusion.\n",
                  "- `acceptance.json`, `wifi_final_report_evidence.json`: acceptance and frozen evidence hashes.\n",
                  "- `S07_SOURCE_DECISIONS.md`: source-only frontend, recipe and combination decisions.\n",
                  "- `S07_NOVELTY_AND_SOTA_REVIEW.md`: primary-source limits and existing ideas.\n"])
    (HERE / "RESULTS.md").write_text("".join(lines))
    status = dict(generated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  wifi_complete=wifi_complete, osu_complete=osu_complete,
                  counts=counts, failures=failures, formal_wifi_completed=sum(r["completed"] for r in formal_wifi),
                  max_formal_wifi_fits=280, note=note)
    (HERE / "stage_status.json").write_text(json.dumps(status, indent=1) + "\n")
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
