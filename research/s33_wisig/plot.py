"""Static figures for paired errors, physical indicators, and fixed readouts."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(data, path):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.8, 4.0), layout="constrained",
                                  gridspec_kw={"width_ratios": [1.0, 1.25, 1.15]})
    blue, green, rust = "#235C86", "#567861", "#A64329"
    for ax in (a, b, c):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
    overlap = data["overlap"]
    points = ("both_correct", "hstem_only", "patch_only", "both_wrong")
    labels = ("Both correct", "HStem only", "Long-Patch only", "Both wrong")
    values = [overlap[m]["mean_pp"] for m in points]
    y = np.arange(len(labels))
    a.barh(y, values, color=(green, blue, "#7F94A7", rust), height=.53)
    for i, v in enumerate(values):
        a.text(v+1, i, f"{v:.1f}%", va="center", fontsize=9)
    a.set(yticks=y, yticklabels=labels, xlim=(0, 94), xlabel="RX-equal packet share (%)",
          title="Same-packet errors")
    a.grid(axis="x", color="#DFE6EA")
    a.invert_yaxis()
    physical = data["physical"]["pair"]
    points = ("phase_spread_khz", "repeat_loss", "spectral_distance", "transition_dev_samples",
              "cfo_departure_khz", "cfo_abs_khz", "outband_fraction")
    labels = ("Phase spread", "Repetition loss", "Spectral distance", "Transition deviation",
              "CFO deviation", "Absolute CFO", "Out-of-band power")
    for i, k in enumerate(points):
        item = physical[k]
        lo, hi = item["simultaneous_cutoff_interval"]
        mean = item["auc"]
        b.errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], color=blue if item["eligible"] else "#778493",
                   marker="o", capsize=3, lw=1.5)
    b.axvline(.5, color="#6C737A", ls="--", lw=1)
    b.set(yticks=np.arange(len(points)), yticklabels=labels, ylim=(-.6, len(points)-.4),
          xlim=(.28, .95), xlabel="Matched-cell rank AUC", title="Physical indicators")
    b.grid(axis="x", color="#DFE6EA")
    b.invert_yaxis()
    points = ("token_mean", "token_flat", "pooled", "head")
    labels = ("Token mean", "Full tokens", "Pooled", "Head")
    for arch, color, mark in (("HStem", blue, "o"), ("LongPatch", rust, "s")):
        values = [data["readouts"]["acc"][arch][k]["source_selection"] for k in points]
        c.plot(labels, values, marker=mark, color=color, label=arch)
    c.set(ylim=(44, 94), xlabel="Frozen readout", ylabel="Held RX accuracy (%)",
          title="Source-only prototype readouts")
    c.grid(axis="y", color="#DFE6EA")
    c.legend(frameon=False, loc="lower right", fontsize=8)
    fig.savefig(path, dpi=185, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("plot", type=Path)
    args = parser.parse_args()
    plot(json.loads(args.summary.read_text(encoding="utf-8")), args.plot)
