"""Static figures for the source-only perturbation audit."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def figure(summary, path):
    blue, rust = "#245A80", "#AD462A"
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.85), layout="constrained")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#DCE4EB", zorder=-1)
        ax.set_axisbelow(True)
    styles = (("HStem", blue, "o"), ("LongPatch", rust, "s"))
    for mode, color, marker in styles:
        for kind, line, label in (("center", "-", "centered CFO"), ("phase", "--", "phase only")):
            series = summary["curves"][kind]
            x = [int(key) / 1000 for key in series]
            y = [v["loss_pp"][mode] for v in series.values()]
            axes[0].plot(x, y, color=color, ls=line, marker=marker, ms=3,
                         label=f"{mode}, {label}")
        series = summary["curves"]["shifted"]
        axes[1].plot([int(key) for key in series], [v["loss_pp"][mode] for v in series.values()],
                     color=color, marker=marker, ms=4, label=mode)
        for intervention, line, label in (("center_10000", "-", "10 kHz"), ("shifted_9", "--", "9 samples")):
            layers = ["token", "pooled", "logits"]
            y = [summary["layers"][f"{intervention}_{l}"][mode]["rel_l2"] for l in layers]
            axes[2].plot(layers, y, color=color, ls=line, marker=marker, ms=4,
                         label=f"{mode}, {label}")
    axes[0].set(title="Frequency offset and phase controls", xlabel="Absolute offset (kHz)",
                ylabel="Accuracy loss (pp)")
    axes[1].set(title="Time shift with matched samples", xlabel="Absolute shift (samples)",
                ylabel="Accuracy loss (pp)")
    axes[2].set(title="Representation drift by readout", xlabel="Readout", ylabel="Relative L2 drift")
    for ax in axes:
        ax.legend(loc="best", frameon=False, fontsize=8)
    fig.savefig(path, dpi=185, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figure(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")), Path(sys.argv[2]))
