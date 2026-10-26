"""Render complete S09 attribution and negative mechanism outcomes together."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def read(name):
    with (HERE / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    summary = read("final_metrics.csv")
    comparisons = read("paired_comparisons.csv")
    probes = read("trained_source_perturbations.csv")
    decision = json.loads((HERE / "decision_final.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    colors = ["#187c8b", "#a95b42"]
    ax = axes[0]
    for i, policy in enumerate(("fixed", "packet")):
        row = next(r for r in comparisons if r["task"] == "CRD" and
                   r["contrast"] == f"first:{policy}-null:{policy}")
        mean, lo, hi = [float(row[k]) for k in ("delta_pp", "ci95_low_pp", "ci95_high_pp")]
        ax.errorbar(i, mean, yerr=np.array([[mean - lo], [hi - mean]]),
                    fmt="o", color=colors[i], capsize=5, markersize=8)
    ax.set_xticks([0, 1], ["S08 fixed policy", "Packet early-stop"])
    ax.axhline(0, color="#666666", linewidth=.8)
    ax.set_ylabel("Phase minus null CRD accuracy (pp)")
    ax.set_title("Matched phase-content gain")
    ax.set_ylim(-1, 11)
    ax = axes[1]
    for i, dataset in enumerate(("ManyRx", "ManySig")):
        values = [decision["choices"][dataset][f"{feature}:raw_held"]["mean_gain"]
                  for feature in ("centered", "curvature")]
        ax.bar(np.arange(2) + (i - .5) * .32, values, .32, label=dataset, color=colors[i])
    ax.set_xticks([0, 1], ["Centered", "Curvature"])
    ax.set_ylabel("Source held-RX gain over first (pp)")
    ax.axhline(0, color="#666666", linewidth=.8)
    ax.set_title("New mechanisms did not pass")
    ax.legend(frameon=False)
    ax = axes[2]
    for i, kind in enumerate(("circular_shift9", "crop_shift9")):
        values = [np.mean([float(r["delta_pp"]) for r in probes
                           if r["dataset"] == "ManySig" and r["feature"] == feature and
                           r["perturbation"] == kind and r["scope"] == "extra_only"])
                  for feature in ("first", "centered", "curvature")]
        ax.bar(np.arange(3) + (i - .5) * .32, values, .32,
               label="Circular shift" if i == 0 else "Zero-padded shift", color=colors[i])
    ax.set_xticks([0, 1, 2], ["First", "Centered", "Curvature"])
    ax.set_ylabel("ManySig source probe accuracy change (pp)")
    ax.axhline(0, color="#666666", linewidth=.8)
    ax.set_title("Ideal invariance has limits")
    ax.legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.savefig(HERE / "S09_MECHANISM_SUMMARY.png", dpi=180)
    fig.savefig(HERE / "S09_MECHANISM_SUMMARY.pdf")
    plt.close(fig)
    print("wrote S09_MECHANISM_SUMMARY.png/.pdf")


if __name__ == "__main__":
    main()
