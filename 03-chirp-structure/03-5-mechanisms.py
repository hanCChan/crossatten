#!/usr/bin/env python3
"""Diagnostic interventions on every confirmed resid/symstack checkpoint, no fitting."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from xa.data import fold_roles, load_rx
from xa.models import build_pipeline
from xa.models3 import symbol_residual


def intervene_symbols(module, inputs, mode):
    x = inputs[0]
    raw = symbol_residual(x, normalized=False)
    mean = x - raw
    return (mean if mode == "mean_chirp" else mean + raw.roll(1, 0),)


def shuffle_cross_tx_fraction(labels, batch=64):
    return float(sum(np.count_nonzero(y != np.roll(y, 1))
                     for y in (labels[i:i + batch] for i in range(0, len(labels), batch))) / len(labels))


def main():
    torch.set_num_threads(2)
    arms = {a["name"]: a for a in json.loads((HERE / "arms.json").read_text())["arms"]}
    rows = []
    for name in ("s03_resid", "s03_symstack"):
        paths = sorted((HERE / "runs/confirm").glob(f"{name}__*/model.pt"))
        if len(paths) != 6:
            raise SystemExit(f"need all six confirmed {name} checkpoints")
        for path in paths:
            cfg = json.loads((path.parent / "config.json").read_text())
            saved = json.loads((path.parent / "metrics.json").read_text())
            model = build_pipeline(arms[name]).cuda().eval()
            model.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
            interventions = (["original", "zero_residual", "shuffle_residual"] if name == "s03_resid"
                             else ["original", "reverse_symbols", "misalign_symbols", "mean_chirp", "shuffle_symbol_residual"])
            target = model.net.embedder if name == "s03_resid" else model.net.embedder.cnn_stem
            roles = fold_roles(cfg["fold"])
            for role in ("dev_val", "confirm"):
                for rx in roles[role]:
                    data = load_rx(rx, role, f"s03-mechanisms:{path.parent.name}",
                                   decision_file=str(HERE / "decision_explore.json"))
                    x = torch.from_numpy(data["iq"]).cuda()
                    y = torch.from_numpy(data["y"]).cuda()
                    shuffled_cross_tx = shuffle_cross_tx_fraction(data["y"])
                    original = None
                    for intervention in interventions:
                        hook = None
                        if intervention in ("mean_chirp", "shuffle_symbol_residual"):
                            hook = model.net.embedder.register_forward_pre_hook(
                                lambda module, inputs: intervene_symbols(module, inputs, intervention))
                            target.intervention = None
                        else:
                            target.intervention = None if intervention == "original" else intervention
                        correct = 0
                        with torch.no_grad():
                            for i in range(0, len(x), 64):
                                pred = model(x[i:i + 64]).argmax(1)
                                correct += int((pred == y[i:i + 64]).sum())
                        acc = 100 * correct / len(x)
                        if original is None:
                            original = acc
                        rows.append(dict(arm=name, fold=cfg["fold"], seed=cfg["seed"], role=role, rx=rx,
                                         intervention=intervention, acc=acc, delta_pp=acc - original,
                                         shuffle_cross_tx_fraction=shuffled_cross_tx,
                                         original_vs_saved_pp=original - 100 * saved["selected"][role]["per_rx"][rx]["acc"]))
                        if hook is not None:
                            hook.remove()
                    del x, y, data
            del model
            torch.cuda.empty_cache()
            print(f"evaluated {path.parent.name}", flush=True)
    out = HERE / "summary"
    (out / "mechanisms.json").write_text(json.dumps(dict(batch=64, shuffle="cyclic packet shift within batch",
                                                       selection="none; diagnostics only", rows=rows), indent=1) + "\n")
    lines = ["# Confirmation interventions\n\nRX-equal mean; all six fits; delta relative to unchanged batch64 evaluation.\n\n",
             "| arm | role | intervention | accuracy (%) | delta (pp) |\n|---|---|---|---:|---:|\n"]
    for name in ("s03_resid", "s03_symstack"):
        for role in ("dev_val", "confirm"):
            variants = sorted({r["intervention"] for r in rows if r["arm"] == name})
            for intervention in variants:
                rs = [r for r in rows if r["arm"] == name and r["role"] == role and r["intervention"] == intervention]
                lines.append(f"| {name} | {role} | {intervention} | {np.mean([r['acc'] for r in rs]):.3f} | "
                             f"{np.mean([r['delta_pp'] for r in rs]):+.3f} |\n")
    cross_tx = [100 * r["shuffle_cross_tx_fraction"] for r in rows if r["intervention"] == "original"]
    original_delta = max(abs(r["original_vs_saved_pp"]) for r in rows)
    lines.append(
        f"\nThe registered cyclic shift changes donor TX for {min(cross_tx):.2f}-"
        f"{max(cross_tx):.2f}% of packets across RX; the remainder keep a same-TX "
        "residual donor. Do not treat a weak shuffle effect as proof that identity "
        "residuals are unused. The maximum unchanged batch64 accuracy difference "
        f"from saved batch500 evaluation is {original_delta:.4f} pp.\n"
    )
    (out / "mechanisms.md").write_text("".join(lines))


if __name__ == "__main__":
    main()
