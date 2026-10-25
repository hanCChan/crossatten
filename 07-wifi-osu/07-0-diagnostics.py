#!/usr/bin/env python3
"""Read frozen S06 results, source-only WiSig cells, and OSU metadata."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
from legacy_rfhstu.features import torch_rf_views
from xa.wisig import load_cell, receivers


def dump_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    HERE.mkdir(exist_ok=True)
    report = json.loads((ROOT / "06-paper-validation/summary/report.json").read_text())
    rows, per_rx = [], []
    configs = report["configs"]
    for result in report["results"]:
        if result["dataset"] == "Shen":
            continue
        config = configs[f"{result['stage']}:{result['run_id']}"]
        method = config["arm"]["method"]
        if method not in ("W0", "W1", "FAN", "IB-RFF", "WaveMLP"):
            continue
        row = dict(dataset=result["dataset"], method=method, arm=result["arm"],
                   stage=result["stage"], fold=result["fold"], seed=result["seed"],
                   source_packet_acc=100 * result["source_val_acc"],
                   source_ce=result["source_val_ce"], best_epoch=result["best_val_epoch"],
                   stop_epoch=result["stop_epoch"], epochs_budget=result["epochs_budget"],
                   internal_rx_acc="", CR="", CRD="")
        if "source_rx_val" in result:
            row["internal_rx_acc"] = 100 * result["source_rx_val"]["summary"]["acc_mean"]
        for task, entry in result["selected"].items():
            row[task] = 100 * entry["summary"]["acc_mean"]
            for rx, m in entry["per_rx"].items():
                cm = np.array(m["confusion"])
                np.fill_diagonal(cm, 0)
                left, right = np.unravel_index(cm.argmax(), cm.shape)
                per_rx.append(dict(dataset=result["dataset"], method=method, arm=result["arm"],
                                   fold=result["fold"], seed=result["seed"], task=task, rx=rx,
                                   accuracy=100 * m["acc"], macro_f1=100 * m["macro_f1"],
                                   most_confused_true=int(left), most_confused_predicted=int(right),
                                   errors=int(cm[left, right])))
        rows.append(row)
    dump_csv(HERE / "historical_fit_diagnostics.csv", rows)
    dump_csv(HERE / "historical_receiver_diagnostics.csv", per_rx)
    source_stats = []
    torch.set_num_threads(2)
    log_path = HERE / "source_access.jsonl"
    with log_path.open("a") as log:
        for dataset in ("ManyRx", "ManySig"):
            for fold in range(4):
                source, _ = receivers(dataset, fold)
                powers, coherence, cfo, scales, jumps = [], [], [], [], []
                for rx in source:
                    for day in (1, 2):
                        x, y = load_cell(dataset, rx, day)
                        ids = np.concatenate([np.flatnonzero(y == c)[:8] for c in np.unique(y)])
                        x = x[ids]
                        log.write(json.dumps(dict(dataset=dataset, fold=fold, rx=rx, day=day,
                                                  purpose="source_audit", packets=len(x))) + "\n")
                        z = x[:, 0] + 1j * x[:, 1]
                        a = z[:, 20:160] * z[:, :140].conj()
                        denom = np.sqrt((abs(z[:, 20:160]) ** 2).sum(1)
                                        * (abs(z[:, :140]) ** 2).sum(1))
                        coherence.extend((abs(a.sum(1)) / np.maximum(denom, 1e-12)).tolist())
                        cfo.extend((np.angle(a.sum(1)) * 25e6 / (2 * np.pi * 20)).tolist())
                        views = torch_rf_views(torch.from_numpy(x), 25e6, 15.3515625e6, oob_norm="ratio")
                        fused = torch.cat((views[0], views[1], views[3]), 1).numpy()
                        scales.append(np.sqrt((fused ** 2).mean((0, 2))))
                        jumps.append(float((abs(np.diff(np.angle(z), axis=1)) > np.pi).mean()))
                        powers.extend((abs(z) ** 2).mean(1).tolist())
                source_stats.append(dict(dataset=dataset, fold=fold, packets=len(coherence),
                                         stf_lag=20, stf_region=[0, 160],
                                         stf_coherence_median=float(np.median(coherence)),
                                         stf_coherence_p10=float(np.quantile(coherence, .1)),
                                         cfo_p10_hz=float(np.quantile(cfo, .1)),
                                         cfo_p90_hz=float(np.quantile(cfo, .9)),
                                         absolute_phase_jump_fraction=float(np.mean(jumps)),
                                         feature_rms=np.mean(scales, axis=0).tolist(),
                                         normalized_power_mean=float(np.mean(powers))))
    (HERE / "source_signal_audit.json").write_text(json.dumps(source_stats, indent=1) + "\n")
    inventory = []
    base = Path("/workspace/gai/data/raw/osu_lora")
    for path in sorted(base.rglob("*.sigmf-meta")):
        meta = json.loads(path.read_text())["_metadata"]
        data = path.with_suffix(".dat")
        tx = int(meta["annotations"][0]["wines:transmitter"]["ID"]["Transmitter ID"])
        inventory.append(dict(rx=path.parent.name, tx=tx, relative=str(path.relative_to(base)),
                              datatype=meta["global"]["core:datatype"],
                              sample_rate=meta["global"]["core:sample_rate"],
                              sample_count=meta["annotations"][0]["core:sample_count"],
                              date=meta["captures"][0]["core:date"],
                              bytes=data.stat().st_size if data.exists() else 0,
                              metadata_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (HERE / "osu_inventory.json").write_text(json.dumps(inventory, indent=1) + "\n")
    lines = ["# S07 Code And Historical Diagnostics", "", "Date: 2026-10-25. Baseline: `ebe3225`.",
             "Historical target results are descriptive, never new selection inputs.", "",
             "## Code Facts", "",
             "- Legacy channels are IQ, FFT magnitude, log amplitude, absolute phase.",
             "- FFT indices are frequency; the others are time. Shared indices are not physical alignment.",
             "- W1 uses 32x64 tokens, two HSTU blocks, Pre-LN, and minimum source packet CE.",
             "- R0 keeps exact state-dictionary initialization parity with S06 W1.",
             "- S07 separates frequency positions and globally pools spectral tokens before fusion.",
             "", "## Source Audit", "",
             "| Dataset | Fold | STF median coherence | STF p10 | Absolute phase jumps |",
             "|---|---:|---:|---:|---:|"]
    for row in source_stats:
        lines.append(f"| {row['dataset']} | {row['fold']} | {row['stf_coherence_median']:.4f} | "
                     f"{row['stf_coherence_p10']:.4f} | {row['absolute_phase_jump_fraction']:.4f} |")
    lines += ["", "Raw target RX/days are not read by this audit. Source access is logged per fold.",
              "Lag-20 coherence confirms repetition but does not prove a complete LTF or an exact STF boundary.",
              "The auxiliary lag-20 feature uses only [20,160); it is a fixed prefix hypothesis, not synchronization.",
              "", "## Mechanism Hypotheses", "",
              "Removing mixed FFT, phase differencing, and receiver-level selection may reduce source overfit.",
              "These are hypotheses; historical source/target gaps do not identify their cause.",
              "Periodic products retain CFO and receiver noise; they are not receiver-invariant fingerprints.",
              "", "## OSU Inventory", "",
              f"{len(inventory)} metadata files; {sum(r['bytes'] for r in inventory)} IQ bytes.",
              "Only Diff_Receivers/Indoor/SameTx is local; no Diff_Days recordings were found.",
              "Each TX/RX has one continuous capture. Old G1 periodic windows are not aligned preambles.",
              "OSU requires packet/boundary verification and capture/time-block leakage controls.", "",
              "Detailed old scores/confusions: `historical_*_diagnostics.csv`.",
              "Measured source features: `source_signal_audit.json`; OSU metadata: `osu_inventory.json`.", ""]
    (HERE / "DIAGNOSTICS.md").write_text("\n".join(lines))
    (HERE / "S07_CODE_AUDIT.md").write_text("\n".join(lines))
    print(json.dumps(dict(historical_fits=len(rows), receiver_rows=len(per_rx), osu_metadata=len(inventory))))


if __name__ == "__main__":
    main()
