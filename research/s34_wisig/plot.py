"""Static figure for the matched readouts and the frozen physical contrasts."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(path, output):
    a, b = (json.loads((path / part / "SUMMARY.json").read_text(encoding="utf-8")) for part in ("A", "B"))
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(13.0, 3.7), layout="constrained")
    blue, clay, green = "#245D84", "#AC462C", "#50775D"
    scores = a["scores"]["HStem"]
    stages = ("token", "mix1", "mix2", "pre_pool")
    labels = ("Token", "Mixer 1", "Mixer 2", "Pre-pool")
    for mode, name, color in (("mean", "Mean", blue), ("region4", "Four regions", clay), ("dct4", "Four DCT modes", green)):
        values = [scores[f"{s}_{mode}_ridge"]["all"] for s in stages]
        ax.plot(labels, values, marker="o", color=color, label=name)
    ax.set(ylim=(78, 89), title="Matched 64-D readouts", ylabel="Held RX accuracy (%)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    y = np.arange(4)
    metrics = ("phase_residual_rad", "repeat_loss", "rank1_residual", "spectral_distance")
    names = ("Nonlinear phase", "Repeat loss", "Rank-one residual", "Spectral distance")
    vals = [b["natural"][k]["auc"] for k in metrics]
    bx.barh(y, [v-.5 for v in vals], left=.5, height=.48, color=blue)
    for i, v in enumerate(vals):
        bx.text(v+.006, i, f"{v:.3f}", va="center", fontsize=8)
    bx.set(yticks=y, yticklabels=names, xlim=(.5,.85), xlabel="Matched-cell AUC",
           title="Natural shared errors")
    bx.invert_yaxis()
    names = ("White noise", "Phase walk", "Frequency shape", "Constant CFO")
    vals = [v["paired_common_error_delta_pp"] for v in b["intervention"].values()]
    for i, v in enumerate(vals):
        cx.barh(i, v, height=.48, color=blue if i < 2 else clay)
        cx.text(max(v, 0)+.07, i, f"{v:+.2f}", va="center", fontsize=8)
    cx.axvline(0, color="#71808B", lw=1)
    cx.set(yticks=y, yticklabels=names, xlim=(-.4,5.3), xlabel="Change in shared errors (pp)",
           title="Matched IQ interventions")
    cx.invert_yaxis()
    for panel, direction in ((ax,"y"), (bx,"x"), (cx,"x")):
        panel.spines[["top","right"]].set_visible(False)
        panel.set_axisbelow(True)
        panel.grid(axis=direction, color="#DCE5EB")
    fig.savefig(output, dpi=185, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    render(args.results, args.figure)
