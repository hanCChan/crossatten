"""Create a static figure from the sealed S29 summary."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from source import read


def plot(summary, destination):
    cold, warm = "#245A80", "#AC4B31"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#242C33", "axes.titleweight": "bold"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained",
                               gridspec_kw={"width_ratios": [1.15, 1]})
    frozen = summary["frozen"]["contrasts"]
    labels = ["Native CFO removal", "Injected ±10 kHz", "Injected ±2.5 kHz",
              "Remove 9 samples", "Shift retained samples by ±9"]
    values = [-frozen["native_cfo_delta"]["mean_pp"], frozen["cfo10k_loss"]["mean_pp"],
              frozen["cfo2p5k_loss"]["mean_pp"], frozen["aligned9_loss"]["mean_pp"],
              frozen["time9_matched_loss"]["mean_pp"]]
    yy = np.arange(len(labels))
    a.barh(yy, values, color=[warm, cold, cold, "#527460", "#527460"], height=.57)
    for y, v in enumerate(values):
        a.text(v + .35, y, f"{v:.2f}", va="center", fontsize=9)
    a.set(yticks=yy, yticklabels=labels, xlim=(0, 35), xlabel="Accuracy loss (percentage points)",
          title="Frozen models")
    a.invert_yaxis()
    a.grid(axis="x", color="#DFE5E9", zorder=-1)
    a.set_axisbelow(True)
    if "raw_control" in summary:
        comp = summary["raw_control"]["comparisons"]
        contrasts = ("native", "cfo10k_loss", "time9_matched_loss")
        labels = ["Native accuracy", "±10 kHz loss", "Matched time-shift loss"]
        for y, key in enumerate(contrasts):
            estimate = comp[key]["mean_pp"]
            lo, hi = comp[key]["interval95_pp"]
            b.errorbar(estimate, y, xerr=[[estimate - lo], [hi - estimate]], color=cold,
                       marker="o", linewidth=1.5, capsize=4)
        b.axvline(0, linestyle="--", color="#65737E", lw=1)
        b.set(yticks=np.arange(3), yticklabels=labels, ylim=(-.6, 2.6),
              xlabel="Difference (pp, positive favors HStem)", title="Matched tokenizer training")
        b.invert_yaxis()
        b.grid(axis="x", color="#DFE5E9")
        b.set_axisbelow(True)
    else:
        b.set_axis_off()
    fig.savefig(destination, dpi=185, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    plot(read(args.summary), args.destination)


if __name__ == "__main__":
    main()
