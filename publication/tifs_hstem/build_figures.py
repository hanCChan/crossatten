"""Render the main results figures from the same frozen inputs as the tables."""

from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

from build_tables import HERE, read_inputs


METHODS = (
    ("h0", "HStem", "#145c57", "o"),
    ("ma", "Spec+Siamese", "#a45014", "s"),
    ("dann", "Shen-DANN", "#355e9b", "^"),
)
METADATA = {"CreationDate": datetime(2026, 11, 25, tzinfo=timezone.utc)}


def prepare(ax, xlabel, ylabel=None):
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", color="#e3e7e9", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def efficiency(inputs):
    cost = inputs.get("gul_cost", inputs["native_cost"])["rows"]
    accuracy = {r["method"]: r["acc"] for r in inputs["external"]["summary"] if r["view"] == "native"}
    accuracy["ma"] = inputs["ma"]["methods"]["ma"]["acc"]
    methods = list(METHODS)
    if "gul_report" in inputs:
        methods.append(("gul", "Gul Federated", "#694687", "D"))
        accuracy["gul"] = inputs["gul_report"]["methods"]["gul"]["acc"]
    fig, (params_ax, throughput_ax) = plt.subplots(1, 2, figsize=(7.1, 2.50), layout="constrained",
                                                  sharey=True)
    prepare(params_ax, "Model parameters (log scale)", "Native RX-equal accuracy (%)")
    prepare(throughput_ax, "Batch-128 throughput (packets/s)")
    values = [100 * accuracy[key] for key, *_ in methods]
    spread = max(values) - min(values)
    offsets = {"h0": ((8, -4), (-8, -4)), "ma": ((-8, 7), (8, 4)),
               "dann": ((-8, -7), (8, -4)), "gul": ((-8, 9), (8, 9))}
    for key, label, color, marker in methods:
        row = cost[key]
        y = 100 * accuracy[key]
        x = row["params"], row["batches"]["128"]["packets_per_second"]
        for ax, value, (dx, dy) in zip((params_ax, throughput_ax), x, offsets[key]):
            ax.scatter(value, y, s=38, color=color, marker=marker, zorder=3)
            ax.annotate(label, (value, y), xytext=(dx, dy), textcoords="offset points",
                        ha="left" if dx > 0 else "right", va="center", color=color)
    params_ax.set(xscale="log", xlim=(95_000, 19_000_000),
                  ylim=(min(93.6, min(values) - max(.6, .06 * spread)),
                        max(98.0, max(values) + max(.8, .06 * spread))))
    params_ax.xaxis.set_major_locator(FixedLocator((100_000, 1_000_000, 10_000_000)))
    params_ax.set_xticklabels(("100k", "1M", "10M"))
    rates = [cost[key]["batches"]["128"]["packets_per_second"] for key, *_ in methods]
    throughput_ax.set(xlim=(min(3500, min(rates)-400), max(8950, max(rates)+500)))
    throughput_ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.savefig(HERE / "figures/native_efficiency.pdf", metadata=METADATA)
    plt.close(fig)


def ablations(inputs):
    structure = {r["arm"]: r["accuracy_pct"] for r in inputs["structure"]["summary"]}
    factorial = {r["arm"]: r["accuracy_pct"] for r in inputs["factorial"]["A"]["dev_summary"]}
    fig, (support_ax, factorial_ax) = plt.subplots(1, 2, figsize=(7.1, 2.60), layout="constrained",
                                                   sharey=True)
    prepare(support_ax, "Tokenizer on the shared RF pipeline", "RX-equal accuracy (%)")
    prepare(factorial_ax, "Intermediate activation")
    support_ax.set_title("Same-support comparison", loc="left", fontsize=9)
    factorial_ax.set_title("BN and GELU factorial", loc="left", fontsize=9)
    for x, key, color, marker in ((0, "P0", "#777777", "s"), (1, "LP", "#355e9b", "^"),
                                  (2, "H0", "#145c57", "o")):
        y = structure[key]
        support_ax.scatter(x, y, color=color, marker=marker, s=36, zorder=3)
        support_ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center")
    support_ax.set(xlim=(-0.42, 2.42), ylim=(94.0, 98.0),
                   xticks=[0, 1, 2], xticklabels=["Patch-Lite", "Long-Patch", "HStem"])
    for label, keys, color, style, marker in (
        ("With BN", ("LA", "H0"), "#145c57", "-", "o"),
        ("Without BN", ("H2_LA", "H2"), "#a45014", "--", "s"),
    ):
        values = [factorial[key] for key in keys]
        factorial_ax.plot([0, 1], values, style, color=color, marker=marker, label=label, linewidth=1.2)
    factorial_ax.set(xlim=(-0.25, 1.25), xticks=[0, 1], xticklabels=["No GELU", "GELU"])
    factorial_ax.legend(frameon=False, loc="lower left", ncols=2, handlelength=1.6)
    fig.savefig(HERE / "figures/core_ablations.pdf", metadata=METADATA)
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    data, _ = read_inputs()
    efficiency(data)
    ablations(data)
