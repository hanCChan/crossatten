#!/usr/bin/env python3
"""02-6  POST-HOC mechanism check (NOT pre-registered): scale of the tokens entering the first HSTU mixer and the
fraction of saturated sigmoid attention scores, for the strict-IB base (no LayerNorm before the mixers) versus arms
whose main tokens pass through the fusion module's output LayerNorm (rawoob, cisoob).

Uses exploration checkpoints (fold 0) and 512 source_val packets of the fold-0 SOURCE receivers (no target data).
Also evaluates freshly initialised models (same seed) to separate init-time from learned behaviour.
Output: summary/token_scale.md
"""
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

ARMS = ["s02_base", "s02_tfsplit", "s02_rawoob_full", "s02_cisoob"]
SAT = 6.0   # |score| > 6  ->  sigmoid'(score) < 0.0025


def first_block(model):
    net = getattr(model, "net", None)
    if net is not None:
        return net.encoder.blocks[0]
    return model.blocks[0]


def probe(model, x):
    store = {}
    blk = first_block(model)
    h = blk.register_forward_pre_hook(lambda m, inp: store.__setitem__("x", inp[0].detach()))
    with torch.no_grad():
        model(x)
    h.remove()
    t = store["x"]
    q, k, _, _ = blk.qkvu(t).chunk(4, dim=-1)
    if hasattr(blk, "relative_bias"):
        bias = blk.relative_bias[blk.relative_index]
    else:
        bias = blk.bias()
    score = q @ k.transpose(-2, -1) / math.sqrt(blk.dim) + bias
    return dict(token_rms=float(t.pow(2).mean().sqrt()), score_abs_median=float(score.abs().median()),
                score_saturated_frac=float((score.abs() > SAT).float().mean()))


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xs = []
    for rx in fold_roles(0)["source"]:
        d = load_rx(rx, "source", "02-6-token-scale")
        _, va = source_split(d["y"], d["rows"])
        xs.append(np.asarray(d["iq"][va[:: max(1, len(va) // 103)][:103]]))
    x = torch.from_numpy(np.concatenate(xs)[:512]).to(dev)
    arms = {a["name"]: a for a in json.loads((HERE / "arms.json").read_text())["arms"]}
    rows = []
    for name in ARMS:
        for seed in (101, 102):
            m = build_pipeline(arms[name]).to(dev).eval()
            init = probe(m, x)
            ck = HERE / f"runs/explore/{name}__f0__s{seed}/model.pt"
            m.load_state_dict(torch.load(ck, map_location=dev))
            m.eval()
            trained = probe(m, x)
            dv = json.loads((ck.parent / "metrics.json").read_text())["selected"]["dev_val"]["summary"]["acc_mean"]
            rows.append(dict(arm=name, seed=seed, init=init, trained=trained, dev_val_acc=round(100 * dv, 1)))
            print(json.dumps(rows[-1]), flush=True)
    md = ["# Token scale entering the first HSTU mixer (POST-HOC; fold 0; 512 source_val packets of source RX)\n\n",
          f"saturated = |q.k/sqrt(d) + bias| > {SAT} (sigmoid slope < 0.0025)\n\n",
          "| arm | seed | LayerNorm before mixer 1 | init token RMS | init sat. frac | trained token RMS | trained sat. frac | trained median abs(score) | dev_val acc |\n|---|---|---|---|---|---|---|---|---|\n"]
    for r in rows:
        ln = "no" if r["arm"] in ("s02_base", "s02_tfsplit") else "yes (fusion norm_out)"
        md.append(f"| {r['arm']} | {r['seed']} | {ln} | {r['init']['token_rms']:.2f} | {r['init']['score_saturated_frac']:.3f} | "
                  f"{r['trained']['token_rms']:.2f} | {r['trained']['score_saturated_frac']:.3f} | {r['trained']['score_abs_median']:.2f} | {r['dev_val_acc']} |\n")
    (HERE / "summary/token_scale.md").write_text("".join(md))
    print("".join(md))


if __name__ == "__main__":
    main()
