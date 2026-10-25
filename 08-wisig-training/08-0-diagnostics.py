#!/usr/bin/env python3
"""Read-only S07 source evidence and source-only spectral stability audit."""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch
from training import ROOT, frequency_views, guarded_cells, rms_normalize, sha


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(4)
    old = ROOT / "07-wifi-osu"
    rows, evidence = [], {}
    for phase in ("training", "training_replicate"):
        for job in json.loads((old / f"jobs_{phase}.json").read_text()):
            if job["factor"] not in ("unified", "official", "mild_noise", "held_epoch"):
                continue
            out = old / "runs" / phase / job["id"]
            result = json.loads((out / "metrics.json").read_text())
            if set(result["selected"]) != {"held_source"}:
                raise ValueError("historical source artifact contains target results")
            hist = list(csv.DictReader((out / "history.csv").open()))
            selected = hist[result["selected_epoch"] - 1]
            entry = result["selected"]["held_source"]["summary"]
            rows.append(dict(
                dataset=job["dataset"], fold=job["fold"], seed=job["seed"],
                phase=phase, factor=job["factor"], frontend=job["frontend"],
                stop_epoch=result["stop_epoch"], selected_epoch=result["selected_epoch"],
                train_ce=float(selected["train_loss"]), train_acc_pct=100 * float(selected["train_acc"]),
                packet_ce=result["selected_source_ce"], packet_acc_pct=100 * result["selected_source_acc"],
                held_ce=entry["ce_mean"], held_acc_pct=100 * entry["acc_mean"],
                worst_held_pct=100 * entry["worst_acc"]))
            evidence[str((out / "metrics.json").relative_to(ROOT))] = sha(out / "metrics.json")
    write_csv(HERE / "historical_source_diagnostics.csv", rows)
    spectra = []
    from xa.wisig import receivers
    for dataset in ("ManySig", "ManyRx"):
        for fold in range(4):
            source, _ = receivers(dataset, fold)
            cells = guarded_cells(dataset, fold, source, (1, 2), "diagnostic", f"audit_{dataset}_f{fold}")
            x = np.concatenate([cell["x"][np.concatenate([
                np.flatnonzero(cell["y"] == tx)[:4] for tx in np.unique(cell["y"])])]
                for cell in cells.values()])
            views = frequency_views(rms_normalize(torch.from_numpy(x)))
            amplitude = torch.complex(views[:, 0], views[:, 1]).abs()
            reliable = (amplitude[:, 1:] >= .05) & (amplitude[:, :-1] >= .05)
            spectra.append(dict(
                dataset=dataset, fold=fold, packets=len(x),
                real_rms=float(views[:, 0].square().mean().sqrt()),
                imag_rms=float(views[:, 1].square().mean().sqrt()),
                logmag_rms=float(views[:, 2].square().mean().sqrt()),
                amplitude_p01=float(torch.quantile(amplitude.flatten(), .01)),
                low_energy_fraction=float((amplitude < .05).float().mean()),
                median_reliable_pair_fraction=float(reliable.float().mean(-1).median()),
                finite=bool(torch.isfinite(views).all())))
    write_csv(HERE / "source_spectral_diagnostics.csv", spectra)
    old_probe = json.loads((old / "source_mechanisms.json").read_text())
    gates = [dict(dataset=r["dataset"], fold=r["fold"],
                  token_rms=r["trained"]["blocks"][0]["token_rms"],
                  saturation_fraction=r["trained"]["blocks"][0]["saturation_frac"])
             for r in old_probe["rows"] if r["frontend"] == "frequency"]
    write_csv(HERE / "historical_source_gates.csv", gates)
    report = ["# S08 Source-Only Diagnostics\n\n",
              "S07 source training artifacts only; no historical CR/CRD reads or target-derived decisions. "
              "Two disjoint source-held groups are pre-registered by RX-list order, not target weakness.\n\n",
              "| Dataset | Factor | Runs | Median selected epoch | Mean held RX % |\n",
              "|---|---|---:|---:|---:|\n"]
    for dataset in ("ManySig", "ManyRx"):
        for factor in ("official", "unified", "mild_noise", "held_epoch"):
            selected = [r for r in rows if r["dataset"] == dataset and r["factor"] == factor]
            report.append(f"| {dataset} | {factor} | {len(selected)} | "
                          f"{np.median([r['selected_epoch'] for r in selected]):.1f} | "
                          f"{np.mean([r['held_acc_pct'] for r in selected]):.2f} |\n")
    report.extend([
        "\nThese historical rows mix selected frontends (ManySig fold3 uses split) and "
        "selective repeat coverage. They are descriptive, not paired frequency-only ablations. "
        "Raw minimum packet-CE frequently occurs early; that does not establish that EMA "
        "or longer training generalizes better.\n\n",
        "## Spectral Audit\n\n",
        "Each fold uses its own source RX days1-2, first four packets/class/RX, "
        "the frozen normalization and three complex-spectrum channels. Reliability rule is "
        "amplitude/RMS>=0.05 at both adjacent bins; no wrap-around pair.\n\n",
        "| Dataset | Fold | Low-energy bins | Median reliable pairs | Re/Im/Log RMS |\n",
        "|---|---:|---:|---:|---|\n"])
    for r in spectra:
        report.append(f"| {r['dataset']} | {r['fold']} | {r['low_energy_fraction']:.4f} | "
                      f"{r['median_reliable_pair_fraction']:.4f} | "
                      f"{r['real_rms']:.3f}/{r['imag_rms']:.3f}/{r['logmag_rms']:.3f} |\n")
    report.extend([
        "\nThe mask is a conservative fixed numerical/energy threshold, not a noise-floor "
        "estimate or proof of receiver invariance. Phase features are conditional on training "
        "gate failure and median reliable fraction>=0.5 in all four ManySig source folds.\n\n",
        "## Gate Scale\n\n",
        "Frozen S07 frequency-only unified feature-fit probes remain near-unit token RMS "
        "with some abs(score)>6 saturation. They are not the new official-training checkpoints. "
        "Saturation alone does not prove causal failure; no gate redesign is authorized.\n\n",
        "Exact rows: `historical_source_diagnostics.csv`, `source_spectral_diagnostics.csv`, "
        "`historical_source_gates.csv`. Diagnostic inputs are not used to rank new candidates.\n"])
    (HERE / "DIAGNOSTICS.md").write_text("".join(report))
    (HERE / "diagnostic_evidence.json").write_text(json.dumps(evidence, indent=1) + "\n")
    print(json.dumps(dict(source_history_rows=len(rows), spectral_rows=len(spectra), target_reads=0)))


if __name__ == "__main__":
    main()
