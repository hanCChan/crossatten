"""Plot clean paired effects and absolute CFO dose responses."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(summary, destination):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), layout="constrained")
    styles = (("HStem", "#285D85"), ("LongPatch", "#A1452F"))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    primary = ("HStem", "LongPatch", "interaction")
    names = ("HStem gain", "Long-Patch gain", "Interaction")
    for i, key in enumerate(primary):
        record = summary["clean_comparisons"][key]
        value = record["mean_pp"]
        lo, hi = record["interval95_pp"]
        axes[0].errorbar(value, i, xerr=[[value - lo], [hi - value]], marker="o", capsize=4,
                         color=styles[i][1] if i < 2 else "#52745C")
    axes[0].axvline(0, color="#586471", linestyle="--", lw=.9)
    axes[0].set(yticks=range(3), yticklabels=names, ylim=(-.65, 2.65),
                title="Accuracy on unperturbed packets", xlabel="Paired difference (pp)")
    axes[0].invert_yaxis()
    for arch, color in styles:
        for states, linestyle, name in (("raw", "--", "raw training"), ("cfo25", "-", "CFO augmentation")):
            series = summary["results"][arch][states]["dose_accuracy_pct"]
            axes[1].plot([int(d)/1000 for d in series], list(series.values()),
                         color=color, linestyle=linestyle, marker="o", ms=4,
                         label=f"{arch}, {name}")
    axes[1].set(title="Frequency-offset dose response", xlabel="Absolute centered CFO (kHz)",
                ylabel="Accuracy (%)")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    for ax, direction in zip(axes, ("x", "y")):
        ax.set_axisbelow(True)
        ax.grid(axis=direction, color="#DBE3EA")
    fig.savefig(destination, dpi=185, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("plot", type=Path)
    args = parser.parse_args()
    plot(json.loads(args.summary.read_text(encoding="utf-8")), args.plot)
