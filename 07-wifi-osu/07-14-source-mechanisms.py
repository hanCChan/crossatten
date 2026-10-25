#!/usr/bin/env python3
"""Bounded read-only source diagnostics, never a candidate-selection input."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))

import numpy as np
import torch

from xa.explore_models import build_explore_model
from xa.paper_train import sha
from xa.wisig import load_cell, receivers


def sampled_source(job, per_class=2):
    source, target = receivers(job["dataset"], job["fold"])
    count = 6 if job["dataset"] == "ManyRx" else 2
    held = source[-count:]
    if set(held) & set(target):
        raise ValueError("source diagnostic attempted target access")
    xs, ys, provenance = [], [], []
    with (HERE / "source_diagnostic_access.jsonl").open("a") as log:
        for rx in held:
            cells = [load_cell(job["dataset"], rx, day) for day in (1, 2)]
            x, y = (np.concatenate([c[k] for c in cells]) for k in (0, 1))
            ids = np.concatenate([np.flatnonzero(y == tx)[:per_class] for tx in np.unique(y)])
            xs.append(x[ids])
            ys.append(y[ids])
            provenance.append(dict(rx=rx, days=[1, 2], concatenated_rows=ids.tolist()))
            log.write(json.dumps(dict(dataset=job["dataset"], fold=job["fold"], rx=rx,
                                      days=[1, 2], purpose="posthoc_source_diagnostic",
                                      run=job["id"], rows=ids.tolist())) + "\n")
    return torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ys)), provenance


def encoder_of(model):
    return model.net.encoder if hasattr(model, "net") else model.encoder


def stems_of(model):
    if hasattr(model, "net"):
        return {"mixed": model.net.embedder.cnn_stem}
    return {k: getattr(model, k) for k in ("time", "frequency") if getattr(model, k) is not None}


@torch.no_grad()
def probe(model, x):
    captured, handles = {}, []
    encoder = encoder_of(model)
    for index, block in enumerate(encoder.blocks):
        handles.append(block.register_forward_pre_hook(
            lambda module, inputs, i=index: captured.__setitem__(i, inputs[0].detach())))
    features = {}
    for name, stem in stems_of(model).items():
        handles.append(stem.register_forward_pre_hook(
            lambda module, inputs, n=name: features.__setitem__(n, inputs[0].detach())))
    try:
        logits = model(x)
        blocks = []
        for index, block in enumerate(encoder.blocks):
            h = captured[index]
            q, k, _, _ = block.qkvu(h).chunk(4, dim=-1)
            scores = q @ k.transpose(-2, -1) / math.sqrt(block.dim)
            scores += block.relative_bias[block.relative_index].unsqueeze(0)
            blocks.append(dict(block=index, token_rms=float(h.square().mean().sqrt()),
                               score_abs_median=float(scores.abs().median()),
                               saturation_frac=float((scores.abs() > 6).float().mean())))
        scales = {name: value.square().mean((0, 2)).sqrt().tolist()
                  for name, value in features.items()}
        return logits, dict(blocks=blocks, channel_rms=scales)
    finally:
        for handle in handles:
            handle.remove()


def masks_of(model):
    stems = stems_of(model)
    masks = []
    for name, stem in stems.items():
        if name == "mixed":
            groups = dict(IQ=(0, 1), FFT=(2,), amplitude=(3,), absolute_phase=(4,))
        elif name == "frequency":
            groups = dict(spectrum_complex=(0, 1), spectrum_magnitude=(2,),
                          frequency_all=(0, 1, 2))
        else:
            groups = dict(IQ=(0, 1), amplitude=(2,))
            if model.kind in ("diff", "diff_stf", "diff_split"):
                groups["phase_product"] = (3, 4)
            else:
                groups["absolute_phase"] = (3,)
            if model.kind in ("stf", "diff_stf"):
                groups["STF_product"] = tuple(range(5 if "diff" in model.kind else 4,
                                                  7 if "diff" in model.kind else 6))
        for group, channels in groups.items():
            masks.append((f"{name}:{group}", stem, channels))
    return masks


@torch.no_grad()
def ablations(model, x, y):
    baseline = float((model(x).argmax(1) == y).float().mean())
    rows = []
    for name, stem, channels in masks_of(model):
        def hide(module, inputs):
            hidden = inputs[0].clone()
            hidden[:, list(channels)] = 0
            return (hidden, *inputs[1:])
        handle = stem.register_forward_pre_hook(hide)
        try:
            acc = float((model(x).argmax(1) == y).float().mean())
            rows.append(dict(group=name, accuracy_pct=100 * acc,
                             delta_pp=100 * (acc - baseline)))
        finally:
            handle.remove()
    return 100 * baseline, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    jobs = json.loads((HERE / "jobs_features.json").read_text())
    rows = []
    samples = {}
    for job in jobs:
        out = HERE / "runs/features" / job["id"]
        result = json.loads((out / "metrics.json").read_text())
        config = json.loads((out / "config.json").read_text())
        if result["weights_sha256"] != sha(out / "model.pt"):
            raise ValueError("source checkpoint changed")
        key = f"{job['dataset']}:{job['fold']}"
        if key not in samples:
            samples[key] = sampled_source(job)
        x, y, provenance = samples[key]
        torch.manual_seed(config["data_seed"])
        model = build_explore_model(config["arm"]).cpu().eval()
        _, initial = probe(model, x)
        model.load_state_dict(torch.load(out / "model.pt", map_location="cpu", weights_only=True))
        _, trained = probe(model, x)
        baseline, masked = ablations(model, x, y)
        row = dict(id=job["id"], dataset=job["dataset"], fold=job["fold"],
                   frontend=job["frontend"], packets=len(y), baseline_accuracy_pct=baseline,
                   initial=initial, trained=trained, masking=masked,
                   checkpoint_sha256=result["weights_sha256"])
        rows.append(row)
        print(json.dumps(dict(id=job["id"], baseline_pct=round(baseline, 2),
                              block0=trained["blocks"][0])), flush=True)
    value = dict(post_hoc=True, selection_use="none", device="cpu",
                 threshold_abs_score=6, rows=rows,
                 samples={key: v[2] for key, v in samples.items()},
                 policy="first two held-source day1-2 packets per TX/RX; seed3101 feature fits only",
                 masking_caution="Zeroed channels are out-of-distribution; not causal feature importance.")
    (HERE / "source_mechanisms.json").write_text(json.dumps(value, indent=1) + "\n")
    table = ["# S07 Source-Only Mechanism Diagnostics\n\n",
             "Read-only, after feature fits. No extra training or target reads; never used for selection. ",
             "Two packets per TX per held-source RX, fixed across frontends. ",
             "Scores above |6| indicate near-saturated sigmoid entries, not a proven failure cause. ",
             "Feature masking is out-of-distribution sensitivity, not causal feature importance.\n\n",
             "| Dataset | Fold | Frontend | Sample acc % | Block0 init RMS | Trained RMS | Trained saturation |\n",
             "|---|---:|---|---:|---:|---:|---:|\n"]
    for r in rows:
        initial, trained = r["initial"]["blocks"][0], r["trained"]["blocks"][0]
        table.append(f"| {r['dataset']} | {r['fold']} | {r['frontend']} | "
                     f"{r['baseline_accuracy_pct']:.2f} | {initial['token_rms']:.3f} | "
                     f"{trained['token_rms']:.3f} | {trained['saturation_frac']:.4f} |\n")
    (HERE / "S07_SOURCE_MECHANISMS.md").write_text("".join(table))
    masked_rows = [dict(dataset=r["dataset"], fold=r["fold"], frontend=r["frontend"],
                        baseline_accuracy_pct=r["baseline_accuracy_pct"], **a)
                   for r in rows for a in r["masking"]]
    with (HERE / "source_feature_masking.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(masked_rows[0]))
        writer.writeheader()
        writer.writerows(masked_rows)


if __name__ == "__main__":
    main()
