#!/usr/bin/env python3
"""Post-hoc source-only scale diagnostic; never used to select arms or checkpoints."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from xa.data import fold_roles, load_rx, source_split  # noqa: E402
from xa.models import build_pipeline  # noqa: E402
from xa.train import stable_seed  # noqa: E402

SAT = 6.0


def source_packets(fold):
    packets, provenance = [], []
    for rx in fold_roles(fold)["source"]:
        data = load_rx(rx, "source", "03-8-token-scale", mmap=True)
        _, validation = source_split(data["y"], data["rows"])
        selected = np.concatenate(
            [validation[data["y"][validation] == tx][:10] for tx in range(10)]
        )
        if len(selected) != 100:
            raise ValueError(f"{rx}: expected ten source-validation packets per TX")
        packets.append(np.asarray(data["iq"][selected]))
        provenance.append(dict(rx=rx, rows=data["rows"][selected].tolist()))
    return torch.from_numpy(np.concatenate(packets)), provenance


@torch.no_grad()
def probe(model, x, batch=32):
    if model.training:
        raise ValueError("probe requires eval mode")
    block = model.net.encoder.blocks[0]
    captured = {}
    handle = block.register_forward_pre_hook(
        lambda module, inputs: captured.__setitem__("tokens", inputs[0].detach())
    )
    packet_powers, scores = [], []
    try:
        for xb in x.split(batch):
            model(xb)
            tokens = captured.pop("tokens")
            q, k, _, _ = block.qkvu(tokens).chunk(4, dim=-1)
            score = q @ k.transpose(-2, -1) / math.sqrt(block.dim)
            score = score + block.relative_bias[block.relative_index].unsqueeze(0)
            packet_powers.append(tokens.square().mean((1, 2)).cpu())
            scores.append(score.abs().flatten().cpu())
    finally:
        handle.remove()
    powers = torch.cat(packet_powers)
    abs_scores = torch.cat(scores)
    rms = powers.sqrt()
    return dict(
        packets=len(powers),
        token_rms=float(powers.mean().sqrt()),
        packet_token_rms_mean=float(rms.mean()),
        packet_token_rms_min=float(rms.min()),
        packet_token_rms_max=float(rms.max()),
        score_abs_median=float(abs_scores.median()),
        score_saturated_frac=float((abs_scores > SAT).float().mean()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    arms = json.loads((HERE / "arms.json").read_text())["arms"]
    rows, samples = [], {}
    for fold in (0, 1):
        x, provenance = source_packets(fold)
        samples[str(fold)] = provenance
        for arm in arms:
            for seed in (101, 102):
                checkpoint = HERE / f"runs/explore/{arm['name']}__f{fold}__s{seed}/model.pt"
                torch.manual_seed(stable_seed("P-H5x160", fold, seed))
                model = build_pipeline(arm).cpu().eval()
                initial = probe(model, x)
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                trained = probe(model, x)
                row = dict(
                    arm=arm["name"], fold=fold, seed=seed,
                    checkpoint=str(checkpoint.relative_to(HERE)),
                    preln=model.pre_ln is not None, init=initial, trained=trained,
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
    output = dict(
        post_hoc=True, device="cpu", threads=args.threads, threshold=SAT,
        selection_use="none",
        sample_policy="first ten source-validation packets per TX and source RX",
        samples=samples, rows=rows,
    )
    summary = HERE / "summary"
    (summary / "token_scale.json").write_text(json.dumps(output, indent=1) + "\n")
    text = [
        "# Post-Hoc Source-Only Token Scale\n\n",
        "All eight arms, two folds, exploration seeds 101/102. Each fold uses 500 "
        "balanced source-validation packets (ten per TX per source RX), unchanged "
        "across arms. CPU evaluation compares the original stable-seed initialization "
        "with the source-CE-selected exploration checkpoint. No target packets, "
        "checkpoint selection, additional fits, or decision changes.\n\n",
        f"First-block saturation diagnostic: |q.k/sqrt(d) + bias| > {SAT}. "
        "RMS is computed across all packet/token/channel elements; score medians "
        "and saturation fractions include every packet and token pair. Finite "
        "scale does not prove the cause of historical depth4 failures.\n\n",
        "| arm | fold | seed | preln | init RMS | trained RMS | trained packet RMS range "
        "| init saturated | trained saturated | trained median abs(score) |\n",
        "|---|---|---|---|---|---|---|---|---|---|\n",
    ]
    for row in rows:
        initial, trained = row["init"], row["trained"]
        text.append(
            f"| {row['arm']} | {row['fold']} | {row['seed']} | {row['preln']} | "
            f"{initial['token_rms']:.3f} | {trained['token_rms']:.3f} | "
            f"{trained['packet_token_rms_min']:.3f}-{trained['packet_token_rms_max']:.3f} | "
            f"{initial['score_saturated_frac']:.4f} | {trained['score_saturated_frac']:.4f} | "
            f"{trained['score_abs_median']:.3f} |\n"
        )
    (summary / "token_scale.md").write_text("".join(text))
    print("wrote source-only token-scale artifacts", flush=True)


if __name__ == "__main__":
    main()
