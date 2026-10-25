#!/usr/bin/env python3
"""Post-hoc failure diagnostics only; no fitting, method selection or new recipes."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s07_osu_train", HERE / "07-6-osu-train.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def lag_phase(iq):
    z = iq[:, 0] + 1j*iq[:, 1]
    return float(np.angle(np.sum(z[:, 1024:]*z[:, :-1024].conj())))


def main():
    torch.set_num_threads(2)
    reg = json.loads((HERE / "osu_registration.json").read_text())
    manifest = json.loads((HERE / "osu_data_manifest.json").read_text())
    signal = []
    for tx in range(1, 26):
        phases = {}
        for rx in ("RX1", "RX2"):
            with np.load(HERE / "cache" / "osu" / f"{rx}_TX{tx}.npz") as archive:
                phases[rx] = {split: lag_phase(archive[split]) for split in ("train", "target")}
        cross = float(np.angle(np.exp(1j*(phases["RX2"]["train"]-phases["RX1"]["train"]))))
        signal.append(dict(tx=tx, rx1_train_phase=phases["RX1"]["train"],
                           rx2_train_phase=phases["RX2"]["train"],
                           cross_rx_phase_delta=cross,
                           alias_period_hz=1e6/1024,
                           cross_rx_modulo_cfo_hz=cross*1e6/(2*np.pi*1024),
                           rx1_temporal_phase_delta=float(np.angle(np.exp(
                               1j*(phases["RX1"]["target"]-phases["RX1"]["train"])))),
                           rx2_temporal_phase_delta=float(np.angle(np.exp(
                               1j*(phases["RX2"]["target"]-phases["RX2"]["train"]))))))
    rows = []
    for job in json.loads((HERE / "jobs_osu.json").read_text()):
        out = HERE / "runs" / "osu" / job["id"]
        result = json.loads((out / "metrics.json").read_text())
        model = trainer.build(job["method"]).cuda().eval()
        model.load_state_dict(torch.load(out / "model.pt", map_location="cuda", weights_only=True))
        same = trainer.load(job["source"], "target", reg, manifest, f"posthoc:{job['id']}")
        logits = trainer.predict(model, torch.from_numpy(same["x"]).cuda(), batch=64).cpu().numpy()
        value = trainer.metrics(logits, same["y"], 25)
        rows.append(dict(method=job["method"], source=job["source"], seed=job["seed"],
                         same_rx_future_acc_pct=100*value["acc"],
                         cross_rx_future_acc_pct=100*result["target"]["acc"]))
        del model
    for name, values in (("osu_posthoc_signal.csv", signal), ("osu_posthoc_same_rx.csv", rows)):
        with (HERE / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    summary = {method: dict(
        same_rx=float(np.mean([r["same_rx_future_acc_pct"] for r in rows if r["method"]==method])),
        cross_rx=float(np.mean([r["cross_rx_future_acc_pct"] for r in rows if r["method"]==method])))
               for method in reg["methods"]}
    phase = np.array([r["cross_rx_modulo_cfo_hz"] for r in signal])
    lines = ["# OSU Post-Hoc Failure Diagnostics\n\n",
             "This analysis occurs after all 24 registered scores. It is descriptive only, ",
             "cannot select methods, and authorizes no additional fits or compensation.\n\n",
             "All metadata transmitter IDs agree with filenames for both RXs. Original crop ",
             "parity, labels and saved confusion matrices have passed checks.\n\n",
             "| Model | Same RX future time block % | Other RX future time block % |\n",
             "|---|---:|---:|\n"]
    for method, value in summary.items():
        lines.append(f"| {method} | {value['same_rx']:.2f} | {value['cross_rx']:.2f} |\n")
    lines.append(f"\nMedian RX2-minus-RX1 lag-phase implied modulo-CFO difference: {np.median(phase):.1f} Hz. "
                 "This statistic is aliased modulo 976.5625 Hz; it is not an unambiguous oscillator estimate. "
                 "A systematic RX-dependent phase/relative-frequency offset is observed. It does not prove "
                 "that correcting it will recover accuracy or that the detector is unbiased.\n\n"
                 "Unlike Shen's published preprocessed inputs, this pipeline does not compensate CFO. "
                 "Negative OSU results apply to the registered detector and nominal strict-IB pipeline, "
                 "not all possible OSU methods. Sparse per-class packets, a single capture, synchronization "
                 "uncertainty and receiver transfer remain limitations. No OSU SOTA is established.\n")
    (HERE / "S07_OSU_POSTHOC.md").write_text("".join(lines))
    print(json.dumps(dict(summary=summary, modulo_cfo_median_hz=float(np.median(phase)))), flush=True)


if __name__ == "__main__":
    main()
