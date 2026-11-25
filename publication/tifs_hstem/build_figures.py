"""Build the manuscript's editable diagrams and plots from the evidence ledger."""

from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np

from build_tables import HERE, read_inputs


INK = "#24323A"
GRAY = "#4F5962"
MUTED = "#73808A"
GRID = "#DDE4E8"
TEAL = "#145C57"
BLUE = "#355E9B"
OCHRE = "#A45014"
PALE = "#EDF2F6"
MINT = "#E7F3EF"
CREAM = "#F6EFE8"
WIDTH = 7.10
METADATA = {"CreationDate": datetime(2026, 11, 25, tzinfo=timezone.utc)}

RECEIVERS = (
    ("b200_mini_1", "B200-mini-1"), ("pluto_1", "Pluto-1"),
    ("rtl_2", "RTL-2"), ("rtl_5", "RTL-5"), ("rtl_7", "RTL-7"),
    ("b210_1", "B210-1"), ("rtl_1", "RTL-1"), ("rtl_3", "RTL-3"),
    ("rtl_6", "RTL-6"), ("rtl_9", "RTL-9"),
)
METHODS = (
    ("h0", "HStem", TEAL, "o"), ("ma", "Ma Spec+Siamese", OCHRE, "D"),
    ("dann", "Shen-DANN", BLUE, "^"), ("ce", "Shen-CE", GRAY, "s"),
)
ARMS = (
    ("P0", "Patch-Lite", GRAY, "s", "--"),
    ("LP", "Long-Patch", BLUE, "^", ":"),
    ("H0", "HStem", TEAL, "o", "-"),
)


def export(fig, name):
    path = HERE / "figures" / name
    fig.savefig(path.with_suffix(".pdf"), metadata=METADATA)
    svg = path.with_suffix(".svg")
    fig.savefig(svg, metadata={"Date": "2026-11-25"})
    lines = svg.read_text(encoding="utf-8").splitlines()
    svg.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)


def diagram(height, ymax):
    fig, ax = plt.subplots(figsize=(WIDTH, height))
    fig.subplots_adjust(left=.012, right=.988, top=.985, bottom=.022)
    ax.set(xlim=(0, 100), ylim=(0, ymax))
    ax.axis("off")
    return fig, ax


def rule(ax, x, y, xx, yy, color=MUTED, style="-", width=.75):
    ax.plot([x, xx], [y, yy], color=color, linestyle=style, lw=width, zorder=2)


def arrow(ax, x, y, xx, yy, color=INK, style="-"):
    ax.annotate("", (xx, yy), (x, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=.8, linestyle=style,
                                mutation_scale=8, shrinkA=0, shrinkB=0), zorder=4)


def box(ax, x, y, width, height, label, fill=PALE, edge=GRID, size=8):
    ax.add_patch(Rectangle((x, y), width, height, fc=fill, ec=edge, lw=.75, zorder=3))
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center",
            fontsize=size, linespacing=1.18, zorder=4)


def band(ax, x, y, width, height, title, fill=PALE, color=INK):
    ax.add_patch(Rectangle((x, y), width, height, fc=fill, ec="none", zorder=0))
    ax.text(x+1.15, y+height-1.1, title, ha="left", va="top", weight="bold", color=color)


def label(ax, x, y, value, **kwargs):
    ax.text(x, y, value, ha="center", va="center", **kwargs)


def tensor(ax, x, y, width, height, rows, columns, color=TEAL, fill=MINT, focus=None):
    ax.add_patch(Rectangle((x, y), width, height, fc=fill, ec=color, lw=.8, zorder=3))
    for j in range(1, rows):
        ax.plot([x, x+width], [y+j*height/rows]*2, color=color, alpha=.47, lw=.45, zorder=3)
    for j in range(1, columns):
        ax.plot([x+j*width/columns]*2, [y, y+height], color=color, alpha=.47, lw=.45, zorder=3)
    if focus:
        left, right = focus
        ax.add_patch(Rectangle((x+left*width/columns, y), (right-left)*width/columns,
                               height, fc=color, alpha=.30, lw=0, zorder=3))


def panel(ax, title):
    ax.set_title(title, loc="left", fontweight="bold", pad=8)


def axes_style(ax, xlabel=None, ylabel=None, grid="y"):
    ax.spines[["right", "top"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(length=3, width=.7, color=MUTED)
    ax.grid(axis=grid, color=GRID, lw=.65)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def system():
    fig, ax = diagram(2.47, 34)
    band(ax, 0, 17.2, 100, 16, "Source training and model selection")
    band(ax, 0, 1, 100, 14.5, "Held-receiver inference", MINT, TEAL)
    box(ax, 2, 20, 13, 7, "Enrolled TXs\n10 LoRa radios", "white")
    box(ax, 19, 20, 12, 7, "Wireless\nchannel", "white")
    for y, title in [(26, "Source RX 1"), (22, "Source RX 2–4"), (18, "Source RX 5")]:
        box(ax, 36, y, 14, 3.4, title, "white")
        arrow(ax, 31, 23.5, 36, y+1.7)
        arrow(ax, 50, y+1.7, 55, 23.5)
    box(ax, 55, 20, 18, 7, "Labeled IQ\nand source validation", "white")
    box(ax, 78, 20, 19.5, 7, "Fit HStem model\nTX supervision", MINT, TEAL)
    for x, xx in [(15, 19), (73, 78)]:
        arrow(ax, x, 23.5, xx, 23.5)
    stages = [(2, 13, "Same TX set"), (19, 12, "Wireless\nchannel"), (36, 14, "Held RX"),
              (55, 18, "Frozen HStem model"), (78, 19.5, "Predicted TX ID")]
    for x, width, title in stages:
        box(ax, x, 4, width, 5, title, "white", TEAL if x == 55 else GRID)
    for x, xx in [(15, 19), (31, 36), (50, 55), (73, 78)]:
        arrow(ax, x, 6.5, xx, 6.5)
    rule(ax, 88, 20, 88, 12.7, TEAL, "--")
    rule(ax, 88, 12.7, 64, 12.7, TEAL, "--")
    arrow(ax, 64, 12.7, 64, 9, TEAL, "--")
    label(ax, 76, 14, "Selected weights", color=TEAL, fontsize=8)
    export(fig, "cross_receiver_system")


def architecture(data):
    fig, ax = diagram(3.06, 43)
    band(ax, 0, 20.5, 37, 21, "RF representation and shared CNN")
    band(ax, 38, 20.5, 62, 21, "Hierarchical tokenizer", MINT, TEAL)
    u = np.linspace(0, 1, 270)
    phase = 2*np.pi*(5*u + 17*u**2)
    for trace, color in [(np.cos(phase), BLUE), (np.sin(phase), OCHRE)]:
        ax.plot(2+8*u, 30.7 + 1.35*trace, color=color, lw=.62)
    label(ax, 6, 35.3, "IQ window")
    label(ax, 6, 24.9, "8,192\ncomplex IQ", fontsize=7.6)
    box(ax, 12, 27.3, 9, 6, "RMS +\nTrueIB", "white")
    label(ax, 16.5, 24.9, "8,192\ncomplex IQ", fontsize=7.6)
    tensor(ax, 23, 27.3, 5, 6, 5, 3, BLUE, "#DBE7F1")
    label(ax, 25.5, 35.3, r"I/Q/S/A/$\phi$", fontsize=7.6)
    label(ax, 25.5, 25.25, "8,192×5", fontsize=8)
    tensor(ax, 31, 27.3, 4.5, 6, 5, 5, BLUE, "#DBE7F1")
    label(ax, 33.25, 35.3, "CNN")
    label(ax, 33.25, 25.25, "8,192×32", fontsize=8)
    for x, xx in [(10, 12), (21, 23), (28, 31), (35.5, 41)]:
        arrow(ax, x, 30.3, xx, 30.3)
    stages = data["method_geometry"]["hierarchical_stages"]
    positions = [41, 56, 71, 86]
    for i, (x, stage, width) in enumerate(zip(positions, stages, [11, 9.7, 8.3, 7])):
        tensor(ax, x, 28, width, 5, 4, 8-i, focus=(1, 3))
        label(ax, x+width/2, 35.5, f"Stage {i+1}", color=TEAL, weight="bold")
        label(ax, x+width/2, 25.7, f"{stage['length']:,}×{stage['output_channels']}", color=TEAL)
        operator = f"k={stage['kernel']}, s=4"
        label(ax, x+width/2, 22.95, operator, fontsize=7.45)
        if i < 3:
            label(ax, x+width/2, 21.55, "BN + GELU", fontsize=7.35)
        if i < 3:
            arrow(ax, x+width, 30.5, positions[i+1], 30.5, TEAL)
    band(ax, 0, 9.8, 100, 9.05, "")
    ax.text(22, 17.65, "Attention and classifier", va="top", weight="bold")
    backend = [(2, 18, "Linear + position + LN"), (23, 13, "MHA ×2"),
               (39, 13, "Final LN"), (55, 13, "Mean pool"),
               (71, 13, "Linear head"), (87, 11, "TX logits")]
    for x, width, title in backend:
        box(ax, x, 11.2, width, 3.6, title, "white")
    for x, xx in [(20, 23), (36, 39), (52, 55), (68, 71), (84, 87)]:
        arrow(ax, x, 13, xx, 13)
    for x, text in [(21.5, "32×64"), (37.5, "32×64"), (53.5, "32×64"), (69.5, "64"), (85.5, "10")]:
        label(ax, x, 10.2, text, fontsize=7.4)
    rule(ax, 93, 30.5, 98, 30.5, TEAL)
    rule(ax, 98, 30.5, 98, 19.55, TEAL)
    rule(ax, 98, 19.55, 11, 19.55, TEAL)
    arrow(ax, 11, 19.55, 11, 14.8, TEAL)
    rule(ax, 29.5, 11.2, 29.5, 8.6, BLUE, "--")
    label(ax, 9, 4.5, "One MHA block\nfour heads", color=BLUE, weight="bold")
    for x, width, title in [
        (20, 11, "Q, K, V"), (34, 22, r"softmax($QK^\top/4$)$V$"),
        (59, 15, "Concat + linear"), (77, 9, "+"), (89, 9, "LN")
    ]:
        box(ax, x, 2, width, 3.7, title, "#F1F5FA")
    for x, xx in [(31, 34), (56, 59), (74, 77), (86, 89)]:
        arrow(ax, x, 3.85, xx, 3.85, BLUE)
    label(ax, 47, 1.05, "Q, K, V per head are 32×16", fontsize=7.6)
    rule(ax, 18, 3.85, 18, 7.65, BLUE)
    rule(ax, 18, 7.65, 81.5, 7.65, BLUE)
    arrow(ax, 81.5, 7.65, 81.5, 5.7, BLUE)
    arrow(ax, 18, 3.85, 20, 3.85, BLUE)
    label(ax, 62, 8.25, "Residual · 32×64", color=BLUE, fontsize=7.6)
    export(fig, "identification_pipeline")


def matched_diagram(data):
    fig = plt.figure(figsize=(WIDTH, 3.28), layout="constrained")
    layout = fig.add_gridspec(2, 2, height_ratios=(1.04, 1))
    structure = {r["arm"]: r["accuracy_pct"] for r in data["structure"]["summary"]}
    params = {r["arm"]: r["params"] for r in data["method_geometry"]["accounting"]}
    for i, (key, header, col, fill) in enumerate((
        ("LP", "(a) Direct long projection", BLUE, PALE),
        ("H0", "(b) Progressive local aggregation", TEAL, MINT)
    )):
        ax = fig.add_subplot(layout[0, i])
        ax.set(xlim=(0, 50), ylim=(0, 22))
        ax.axis("off")
        ax.text(0, 21, header, weight="bold", va="top")
        rectangles = (
            [(2, 12, 10), (20, 6, 4), (39, 9, 5)] if key == "LP"
            else [(2, 9, 9), (14, 7, 7), (24, 6, 5), (33, 5, 4), (41, 7, 5)]
        )
        for j, (x, width, columns) in enumerate(rectangles):
            tensor(ax, x, 8, width, 5, 4, columns, col, fill, (1, min(4, columns)))
            if j:
                prior_x, prior_w, _ = rectangles[j-1]
                ax.add_patch(Polygon([(prior_x+prior_w, 8), (x, 9.5), (x, 11.4),
                                      (prior_x+prior_w, 13)], fc=fill, ec=col, lw=.55))
                arrow(ax, prior_x+prior_w+.2, 10.45, x-.2, 10.45, col)
        label(ax, 6, 6.2, "8,192×32", fontsize=8)
        label(ax, 44.5, 6.2, "32×64", fontsize=8)
        if key == "LP":
            label(ax, 16, 15.6, "Grouped k=340", fontsize=8)
            label(ax, 33.3, 15.6, "Pointwise mixing", fontsize=8)
        else:
            label(ax, 24, 15.6, "Four strided stages · BN + GELU in stages 1–3", fontsize=8)
        label(ax, 25, 2, f"{structure[key]:.3f}%  ·  {params[key]:,} full-model parameters", color=col,
              weight="bold")
    ax = fig.add_subplot(layout[1, :])
    axes_style(ax, "Position in the five-view feature array (zero-based)", grid="x")
    panel(ax, "(c) Identical nominal convolution-path support")
    ax.set(xlim=(-63, 815), ylim=(-.54, 1.84), yticks=(0, 1), yticklabels=("Long-Patch", "HStem"),
           xticks=(0, 128, 256, 384, 512, 640, 768))
    ax.axvspan(-63, 0, color=GRID, alpha=.55)
    first_center = data["method_geometry"]["hierarchical_stages"][-1]["center"]
    spacing = data["method_geometry"]["hierarchical_stages"][-1]["token_jump"]
    support = data["method_geometry"]["hierarchical_stages"][-1]["local_support"]
    first_left = first_center - (support-1)/2
    for row, color in [(0, BLUE), (1, TEAL)]:
        for j in range(2):
            center = first_center + j*spacing
            start = first_left + j*spacing
            ax.add_patch(Rectangle((start-.5, row-.2), support, .4, facecolor=color,
                                   alpha=.19, edgecolor="none", zorder=2))
            ax.add_patch(Rectangle((start-.5, row-.2), support, .4, facecolor="none",
                                   edgecolor=color, lw=.8, zorder=3))
            ax.plot(center, row, "|", color=color, ms=8, mew=1.1)
    ax.add_patch(Rectangle((first_left-.5, -.2), -first_left+.5, .4, facecolor="none",
                           edgecolor=MUTED, hatch="////", lw=0))
    ax.add_patch(Rectangle((first_left-.5, .8), -first_left+.5, .4, facecolor="none",
                           edgecolor=MUTED, hatch="////", lw=0))
    ax.annotate("", (first_center+spacing, 1.44), (first_center, 1.44),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=.75, shrinkA=0, shrinkB=0))
    label(ax, first_center+spacing/2, 1.65, f"{spacing} positions between centers", fontsize=8)
    ax.text(574, 1.05, f"{support} positions per window\n{support-spacing} positions of overlap",
            ha="left", va="center", fontsize=8)
    ax.text(574, .05, f"First center = {first_center:.1f}\nHatching indicates padding",
            ha="left", va="center", fontsize=8, color=GRAY)
    export(fig, "matched_tokenizers")


def receiver_performance(data):
    records = data["ma"]["methods"]
    fig, (ax, delta) = plt.subplots(2, 1, figsize=(WIDTH, 2.94), sharex=True,
                                   gridspec_kw={"height_ratios": [1.6, 1]}, layout="constrained")
    axes_style(ax, ylabel="Native accuracy (%)")
    axes_style(delta, ylabel="HStem − Ma (pp)")
    panel(ax, "(a) Accuracy by receiver")
    panel(delta, "(b) Paired receiver difference")
    positions = np.arange(len(RECEIVERS))
    for offset, (key, name, color, marker) in zip(np.linspace(-.26, .26, 4), METHODS):
        y = [100*records[key]["receiver_accuracy"][r] for r, _ in RECEIVERS]
        ax.scatter(positions+offset, y, marker=marker, color=color, s=24, label=name, zorder=3)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, .99), ncol=4,
              handletextpad=.25, columnspacing=1.1)
    ax.set(ylim=(84.4, 102.6), yticks=[85, 90, 95, 100])
    ax.text(2, 101.0, "Held in fold 0", ha="center", color=GRAY, fontsize=8)
    ax.text(7, 101.0, "Held in fold 1", ha="center", color=GRAY, fontsize=8)
    values = [100*(records["h0"]["receiver_accuracy"][k]
                   - records["ma"]["receiver_accuracy"][k]) for k, _ in RECEIVERS]
    delta.axhline(0, color=MUTED, lw=.9)
    delta.vlines(positions, 0, values, color=TEAL, lw=.9)
    delta.scatter(positions, values, color=TEAL, s=18, zorder=3)
    delta.set(xticks=positions, xticklabels=[label for _, label in RECEIVERS],
              yticks=[-4, 0, 4, 8], ylim=(-5.5, 8.3))
    for a in (ax, delta):
        a.axvline(4.5, color=MUTED, linestyle="--", lw=.75)
        a.set_xlim(-.48, 9.48)
    export(fig, "receiver_performance")


def efficiency(data):
    cost = data["gul_cost"]["rows"]
    records = data["ma"]["methods"]
    fig, (ax, rate) = plt.subplots(1, 2, figsize=(WIDTH, 2.22), sharey=True, layout="constrained")
    for axis, title, xlabel in [(ax, "(a) Parameter budget", "Full-model parameters"),
                                (rate, "(b) Shared GPU session", "Batch-128 throughput (packets/s)")]:
        axes_style(axis, xlabel)
        panel(axis, title)
    ax.set_ylabel("Native accuracy (%)")
    ax.set(xscale="log", xlim=(92_000, 25_000_000), ylim=(92.4, 98.3),
           xticks=[1e5, 1e6, 1e7], xticklabels=["100k", "1M", "10M"])
    rate.set(xlim=(3450, 9200), xticks=[4000, 5500, 7000, 8500])
    rate.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    for key, name, color, marker in METHODS:
        value = 100*records[key]["acc"]
        if key in cost:
            values = cost[key]["params"], cost[key]["batches"]["128"]["packets_per_second"]
        else:
            row = next(r for r in data["external_cost"]["rows"] if r["method"] == key and r["view"] == "native")
            values = row["params"], None
        for axis, x in zip((ax, rate), values):
            if x is None:
                continue
            axis.scatter(x, value, color=color, marker=marker, s=32, zorder=3)
            shift = (6, 5) if key in ("h0", "ma") else (-6, -4)
            if axis is ax and key == "ma":
                shift = (-6, 5)
            if axis is rate and key == "h0":
                shift = (-6, 5)
            if axis is rate and key == "dann":
                shift = (6, -4)
            axis.annotate(name, (x, value), xytext=shift, textcoords="offset points",
                          ha="left" if shift[0] > 0 else "right", va="center", color=color)
    export(fig, "native_efficiency")


def paired_tokenizers(data):
    summary = {r["arm"]: r for r in data["structure"]["summary"]}
    params = {r["arm"]: r["params"] for r in data["method_geometry"]["accounting"]}
    contrast = next(r for r in data["structure"]["paired_comparisons"] if r["a"] == "H0")
    fig, (ax, paired) = plt.subplots(1, 2, figsize=(WIDTH, 2.59), layout="constrained",
                                     gridspec_kw={"width_ratios": [1, 1.14]})
    axes_style(ax, ylabel="RX-equal accuracy (%)")
    axes_style(paired, "HStem − Long-Patch (pp)", grid="x")
    panel(ax, "(a) Matched tokenizer means")
    panel(paired, "(b) Differences across paired fits")
    for x, (key, name, c, m, _) in enumerate(ARMS):
        y = summary[key]["accuracy_pct"]
        ax.scatter(x, y, marker=m, color=c, s=39, zorder=3)
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", color=c)
    ax.set(xlim=(-.52, 2.52), ylim=(93.8, 98.1), xticks=range(3),
           xticklabels=[f"{name}\n{params[key]:,}" for key, name, *_ in ARMS])
    paired.axvline(0, color=MUTED, lw=.8)
    paired.set(xlim=(-.2, 5.85), ylim=(7.8, -.7), yticks=[0, 1, 2, 3, 4, 5, 7],
               yticklabels=["F0 / 201", "F0 / 202", "F0 / 203", "F1 / 201", "F1 / 202", "F1 / 203", "RX mean"])
    paired.axhline(5.8, color=GRID, lw=.8)
    paired.scatter(contrast["fit_deltas_pp"], range(6), color=TEAL, s=22, zorder=3)
    low, high = contrast["conditional_rx_ci95_pp"]
    paired.hlines(7, low, high, color=TEAL, lw=1.2)
    paired.scatter(contrast["mean_delta_pp"], 7, color=TEAL, marker="D", s=29, zorder=3)
    export(fig, "paired_tokenizers")


def mechanism(data):
    fig, (ax, factorial) = plt.subplots(1, 2, figsize=(WIDTH, 2.55), layout="constrained",
                                         gridspec_kw={"width_ratios": [1.1, 1]})
    axes_style(ax, "Accuracy change (pp)", grid="x")
    axes_style(factorial, ylabel="RX-equal accuracy (%)")
    panel(ax, "(a) Kernel and normalization contrasts")
    panel(factorial, "(b) BN × GELU")
    rows = {r["contrast"]: r for r in data["kernel_controls"]["contrasts"] if r["role"] == "dev_val"}
    comparisons = [
        ("S1-P0", "Round-trip parameterization", GRAY, "s"),
        ("H1-S1", "Expanded interpolated kernel", BLUE, "^"),
        ("H0-S2", "BN vs. channel affine", TEAL, "o"),
        ("S2-H2", "Restore channel affine", TEAL, "D"),
    ]
    ax.axvline(0, color=MUTED, lw=.8)
    for y, (key, name, color, marker) in enumerate(comparisons):
        row = rows[key]
        ax.hlines(y, *row["ci95_pp"], color=color, lw=1.1)
        ax.scatter(row["mean_delta_pp"], y, marker=marker, color=color, s=26, zorder=3)
    ax.set(ylim=(3.65, -.65), xlim=(-.12, 1.47), yticks=range(4),
           yticklabels=[r[1] for r in comparisons])
    f = data["factorial"]["A"]
    cells = {r["arm"]: r["accuracy_pct"] for r in f["dev_summary"]}
    for text, keys, color, marker, style in [
        ("With BN", ("LA", "H0"), TEAL, "o", "-"),
        ("Without BN", ("H2_LA", "H2"), OCHRE, "s", "--"),
    ]:
        ys = [cells[key] for key in keys]
        factorial.plot([0, 1], ys, color=color, marker=marker, linestyle=style, label=text, lw=1.2)
    factorial.set(xlim=(-.27, 1.27), ylim=(95.9, 97.63), xticks=(0, 1),
                  xticklabels=("No GELU", "GELU"))
    factorial.legend(loc="upper left", frameon=False, handlelength=1.5)
    export(fig, "mechanism_ablations")


def geometry(data):
    rows = data["geometry"]["geometry"]
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 2.45), layout="constrained")
    layers = ["tokens", "pre_mha", "final"]
    metrics = [
        ("tx_to_within_class_ratio", r"(a) $J_w$"),
        ("tx_to_rx_centroid_ratio", r"(b) $J_c$"),
        ("conditional_rx", r"(c) $D_{\rm RX|TX}$"),
    ]
    styles = [
        ("P0", "Patch-Lite", GRAY, "s", "--"), ("S1", "S1", OCHRE, "D", ":"),
        ("H1", "H1", BLUE, "^", "-."), ("H0", "HStem", TEAL, "o", "-"),
    ]
    final_jw = {}
    for ax, (metric, title) in zip(axs, metrics):
        axes_style(ax)
        panel(ax, title)
        for key, name, color, marker, linestyle in styles:
            y = [np.mean([r[metric] for r in rows if r["arm"] == key and r["layer"] == layer])
                 for layer in layers]
            if metric == "tx_to_within_class_ratio":
                final_jw[key] = y[-1]
            ax.plot(range(3), y, color=color, marker=marker, linestyle=linestyle, lw=1.05,
                    ms=3.5, label=name, zorder=3 if key == "H0" else 2)
        ax.set(xlim=(-.16, 2.16), xticks=range(3), xticklabels=["Tokens", "Pre-MHA", "Final"])
    axs[0].set_ylim(0, 6.55)
    for arm, color, dy in [("H0", TEAL, 8), ("P0", GRAY, -13)]:
        value = final_jw[arm]
        axs[0].annotate(f"{value:.3f}", (2, value), xytext=(-14, dy),
                        textcoords="offset points", color=color)
    axs[1].set_ylim(bottom=0)
    lower = min(min(line.get_ydata()) for line in axs[2].lines)
    axs[2].set(yscale="log", ylim=(.65*lower, .055))
    axs[2].legend(loc="lower right", frameon=False, ncol=2, handlelength=1.3,
                  columnspacing=.7, handletextpad=.4, fontsize=7.4)
    export(fig, "representation_geometry")


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelcolor": INK,
        "axes.titlesize": 8.6, "text.color": INK, "xtick.labelcolor": INK,
        "ytick.labelcolor": INK, "pdf.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "hstem-tifs-journal",
    })
    inputs, _ = read_inputs()
    system()
    architecture(inputs)
    matched_diagram(inputs)
    receiver_performance(inputs)
    efficiency(inputs)
    paired_tokenizers(inputs)
    mechanism(inputs)
    geometry(inputs)
