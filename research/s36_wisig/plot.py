"""Compare the actual fusion gains, fixed hard cases, and preamble readouts."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(root, destination):
    a = json.loads((root/"AD/SUMMARY.json").read_text(encoding="utf-8"))
    e = json.loads((root/"E/SUMMARY.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1,3,figsize=(13.3,3.75),layout="constrained",
                             gridspec_kw={"width_ratios":[1.2,1.0,1.05]})
    blue, red, green = "#245E87", "#A8492D", "#4F785D"
    axis = axes[0]
    items = (("HStem", "vs_original"), ("A1","vs_original"), ("A1","vs_capacity"))
    names = ("HStem vs own head","A1 vs own head","A1 vs deep double")
    for i,(model,key) in enumerate(items):
        stats = a["fusion"][model][key]
        point = stats["mean_pp"]
        lo,hi = stats["interval95_pp"]
        axis.errorbar(point,i,xerr=[[point-lo],[hi-point]],capsize=3,marker="o",
                      color=(blue,red,green)[i],linewidth=1.5)
    axis.axvline(0,ls="--",lw=1,color="#697882")
    axis.set(yticks=range(3),yticklabels=names,ylim=(-.5,2.5),xlabel="Paired gain (pp)",
             title="Frozen early fusion")
    axis.invert_yaxis()
    axis = axes[1]
    taps = ("token","mix1","pooled")
    for name,color in (("HStem",blue),("A1",red)):
        for probe,style in (("linear","-"),("nonlinear","--")):
            vals = [a["metrics"][name][f"{key}_{probe}"]["triple"] for key in taps]
            axis.plot(("Token","Mixer 1","Pooled"),vals,marker="o",ms=4,color=color,ls=style,
                      label=f"{name} {probe}")
    axis.set(title="Three-model shared errors",ylabel="Held RX probe accuracy (%)",ylim=(0,23))
    axis.legend(loc="upper right",frameon=False,fontsize=8)
    axis = axes[2]
    names = ("raw","shared","residual","joint")
    values = [e["readout"][k] for k in names]
    x = np.arange(4)
    axis.bar(x-.18,[v["tx_linear"] for v in values],.35,color=blue,label="TX on held RX")
    axis.bar(x+.18,[v["rx_bal_acc_cross_tx_parities"] for v in values],.35,color=red,
             label="Known RX on held TX")
    axis.set(title="Six-block spectrum probes",ylabel="Accuracy (%)",ylim=(0,69),
             xticks=x,xticklabels=("Raw","Shared","Residual","Joint"))
    axis.legend(loc="upper right",frameon=False,fontsize=8)
    for panel, direction in zip(axes,("x","y","y")):
        panel.spines[["top","right"]].set_visible(False)
        panel.set_axisbelow(True)
        panel.grid(axis=direction,color="#DDE5EC")
    fig.savefig(destination,dpi=185,bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("figure",type=Path)
    a = parser.parse_args()
    draw(a.output,a.figure)
