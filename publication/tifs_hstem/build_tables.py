"""Build submission tables from immutable aggregate evidence, without model or IQ I/O."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
S24 = "24-lora-external-audit/phase2/budget_update"
S26 = "26-long-patch-control"
INPUTS = {
    "external": ("dde0ef9", f"{S24}/RESULTS.json"),
    "external_cost": ("dde0ef9", f"{S24}/COSTS.json"),
    "structure": ("ad5443e", f"{S26}/RESULTS.json"),
    "structure_cost": ("ad5443e", f"{S26}/COSTS.json"),
    "factorial": ("bebf7d6", "28ac-mechanism-training/RESULTS.json"),
    "stress": ("b170624", "research/s29_wisig/evidence/SUMMARY.json"),
    "cfo_augmentation": ("f9698e4", "research/s31_wisig/evidence/SUMMARY.json"),
    "ma": ("607efa3", "research/s42_lora_baselines/evidence/RESULTS.json"),
    "gul_report": ("2fe246263f75", "research/s43_gul_federated/evidence/RESULTS.json"),
    "gul_cost": ("2fe246263f75", "research/s43_gul_federated/evidence/NATIVE_COST_COMPARE.json"),
    "native_cost": ("9a20eb0", "research/s42_lora_baselines/evidence/NATIVE_COST_COMPARE.json"),
}
CORE_METHODS = (("h0", "HStem"), ("ce", "Shen CE"), ("dann", "Shen DANN"))
ADAPTED_METHODS = (("fan", "FAN"), ("ib", "IB-RFF"), ("ask", "ASKNet"))
METHODS = CORE_METHODS + ADAPTED_METHODS
STRUCTURE = (("P0", "Patch-Lite", 266, 2), ("LP", "Long-Patch", 350, 2), ("H0", "HStem", 350, 5))


def git_bytes(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def read_inputs():
    values, origins = {}, {}
    for name, (revision, path) in INPUTS.items():
        commit = git_bytes("rev-parse", revision).decode().strip()
        payload = git_bytes("show", f"{commit}:{path}")
        origins[name] = {"commit": commit, "path": path, "sha256": hashlib.sha256(payload).hexdigest()}
        values[name] = json.loads(payload)
    return values, origins


def table(columns, headings, rows):
    row_end = r"\\" + "\n"
    return "".join([
        "% Generated from the submission evidence ledger.\n",
        "\\begin{tabular}{" + columns + "}\n",
        "\\toprule " + " & ".join(headings) + row_end,
        "\\midrule\n",
        *[" & ".join(row) + row_end for row in rows],
        "\\bottomrule\n\\end{tabular}\n",
    ])


def external(data, methods, ma=None, gul=None):
    indexed = {(row["view"], row["method"]): row for row in data["summary"]}
    for method, name in methods:
        native, strict = (indexed[view, method] for view in ("native", "strict_ib"))
        acc, f1, worst = {}, {}, {}
        for row in (native, strict):
            view = row["view"]
            if row["status"] == "DEFERRED_BY_USER_BUDGET":
                acc[view], f1[view], worst[view] = "Deferred", "--", "--"
            else:
                assert row["fits"] == 6
                acc[view] = f'{100 * row["acc"]:.3f} $\\pm$ {100 * row["fit_sd"]:.3f}'
                f1[view] = f'{100 * row["macro_f1"]:.3f}'
                worst[view] = f'{100 * row["worst_rx_acc"]:.3f}'
        yield [name, *acc.values(), *f1.values(), *worst.values()]
        if method == "h0" and ma is not None:
            yield ["Spec+Siamese", f'{100*ma["acc"]:.3f} $\\pm$ {100*ma["fit_sd"]:.3f}',
                   "--", f'{100*ma["macro_f1"]:.3f}', "--", f'{100*ma["worst_rx_acc"]:.3f}', "--"]
    if gul is not None:
        yield ["Gul Federated", f'{100*gul["acc"]:.3f} $\\pm$ {100*gul["fit_sd"]:.3f}',
               "--", f'{100*gul["macro_f1"]:.3f}', "--", f'{100*gul["worst_rx_acc"]:.3f}', "--"]


def structure(data, costs):
    summary = {row["arm"]: row for row in data["summary"]}
    params = {row["arm"]: row["params"] for row in costs["rows"]}
    for arm, name, support, bn in STRUCTURE:
        row = summary[arm]
        yield [name, f"{params[arm]:,}", str(support), str(bn),
               f'{row["accuracy_pct"]:.3f} $\\pm$ {row["fit_sd_pp"]:.3f}',
               f'{row["macro_f1_pct"]:.3f}', f'{row["worst_rx_accuracy_pct"]:.3f}']


def timing(name, result, params):
    first, batch = (result["batches"][key] for key in ("1", "128"))
    rate = 128_000 / batch["p50_ms"]
    assert abs(rate - batch.get("packets_per_second", rate)) < 0.001
    return [name, f"{params:,}", f'{first["p50_ms"]:.3f} / {first["p95_ms"]:.3f}',
            f'{batch["p50_ms"]:.3f} / {batch["p95_ms"]:.3f}', f"{rate:,.0f}"]


def native_cost_rows(results, external, ma, gul=None):
    accuracies = {row["method"]: row["acc"] for row in external["summary"] if row["view"] == "native"}
    accuracies["ma"] = ma["methods"]["ma"]["acc"]
    methods = [("h0", "HStem"), ("ma", "Spec+Siamese"), ("dann", "Shen-DANN")]
    if gul is not None:
        accuracies["gul"] = gul["acc"]
        methods.append(("gul", "Gul Federated"))
    for key, name in methods:
        row = results["rows"][key]
        method, params, b1, b128, rate = timing(name, row, row["params"])
        first, batch = (row["batches"][size] for size in ("1", "128"))
        memory = " / ".join(f'{item["peak_pytorch_allocated_bytes"] / 2**20:.1f}' for item in (first, batch))
        yield [method, f"{100*accuracies[key]:.3f}", params, b1, b128, rate, memory]


def paired_row(name, mean, low, high):
    return [name, "$" + f"{mean:+.3f}$", f"$[{low:+.3f}, {high:+.3f}]$"]


def paired_rows(inputs):
    structure_names = {("H0", "LP"): "HStem -- Long-Patch", ("LP", "P0"): "Long-Patch -- Patch-Lite"}
    for row in inputs["structure"]["paired_comparisons"]:
        name = structure_names[row["a"], row["b"]]
        yield paired_row(name, row["mean_delta_pp"], *row["conditional_rx_ci95_pp"])
    ma = inputs["ma"]["paired_ma_minus"]
    low, high = ma["h0"]["conditional_interval95_pp"]
    yield paired_row("HStem -- Spec+Siamese (Native)", -ma["h0"]["mean_pp"], -high, -low)
    yield paired_row("Spec+Siamese -- Shen-DANN (Native)", ma["dann"]["mean_pp"],
                     *ma["dann"]["conditional_interval95_pp"])
    comparisons = inputs["external"]["paired_comparisons"]
    for view, baseline, name in (
        ("native", "dann", "HStem -- Shen-DANN (Native)"),
        ("strict_ib", "ce", "HStem -- Shen-CE (Strict-IB)"),
    ):
        row = next(x for x in comparisons
                   if (x["view"], x["a"], x["b"], x["metric"]) == (view, "h0", baseline, "acc"))
        yield paired_row(name, 100 * row["delta"], *(100 * x for x in row["conditional_rx_cluster_ci"]))


def gul_paired_rows(report):
    for key, name in (("paired_h0_minus_gul", "HStem -- Gul Federated"),
                      ("paired_gul_minus_ma", "Gul Federated -- Spec+Siamese"),
                      ("paired_gul_minus_dann", "Gul Federated -- Shen-DANN")):
        row = report[key]
        yield paired_row(name, row["mean_pp"], *row["conditional_interval95_pp"])


def stress_rows(data):
    comparisons = data["raw_control"]["comparisons"]
    names = [
        ("native", "Natural-input accuracy"),
        ("cfo10k_loss", r"$\pm$10 kHz smaller loss"),
        ("time9_matched_loss", r"$\pm$9 samples smaller loss"),
    ]
    for key, name in names:
        row = comparisons[key]
        low, high = row["interval95_pp"]
        yield [name, "$" + f'{row["mean_pp"]:+.2f}' + "$", f"$[{low:+.2f}, {high:+.2f}]$"]


def generate():
    inputs, origins = read_inputs()
    result = {}
    gul_report = inputs["gul_report"]
    gul_data = gul_report["methods"]["gul"]
    external_headers = ["Method", "Native Acc.", "Strict-IB Acc.", "Native F1", "Strict F1",
                        "Native Worst", "Strict Worst"]
    result["external.tex"] = table("lrrrrrr", external_headers,
        external(inputs["external"], CORE_METHODS, ma=inputs["ma"]["methods"]["ma"]))
    result["external_adaptations.tex"] = table("lrrrrrr", external_headers,
        external(inputs["external"], ADAPTED_METHODS, gul=gul_data))
    result["long_patch.tex"] = table(
        "lrrrrrr", ["Tokenizer", "Params", "Support", "BN", r"Acc. $\pm$ fit SD", "F1", "Worst RX"],
        structure(inputs["structure"], inputs["structure_cost"]))
    result["paired_accuracy.tex"] = table(
        "lrr", ["Contrast", "Mean (pp)", "95\\% interval (pp)"], paired_rows(inputs))
    result["paired_gul.tex"] = table(
        "lrr", ["Contrast", "Mean (pp)", "95\\% interval (pp)"], gul_paired_rows(gul_report))
    cost_headers = ["Method", "Parameters", "B1 (ms)", "B128 (ms)", "B128 packets/s"]
    native = {r["method"]: r for r in inputs["external_cost"]["rows"] if r["view"] == "native"}
    result["costs.tex"] = table("lrrrr", cost_headers, [
        timing(name, native[key], native[key]["params"]) for key, name in METHODS
    ])
    strict = {r["method"]: r for r in inputs["external_cost"]["rows"] if r["view"] == "strict_ib"}
    result["strict_timing.tex"] = table("lrrrr", cost_headers, [
        timing(name, strict[key], strict[key]["params"]) for key, name in METHODS if key in strict
    ])
    structure_cost = {r["arm"]: r for r in inputs["structure_cost"]["rows"]}
    result["structure_timing.tex"] = table("lrrrr", cost_headers, [
        timing(name, structure_cost[key], structure_cost[key]["params"])
        for key, name, _, _ in STRUCTURE
    ])
    native_cost_headers = ["Method", r"Acc. (\%)", "Parameters", "B1 (ms)", "B128 (ms)",
                           "B128 pkt/s", "B1/B128 MiB"]
    historical = native_cost_rows(inputs["native_cost"], inputs["external"], inputs["ma"])
    result["native_cost_s42.tex"] = table("lrrrrrr", native_cost_headers, historical)
    result["native_cost_full.tex"] = table("lrrrrrr", native_cost_headers,
        native_cost_rows(inputs["gul_cost"], inputs["external"], inputs["ma"], gul_data))
    session = native_cost_rows(inputs["gul_cost"], inputs["external"], inputs["ma"])
    result["native_cost.tex"] = table("lrrrrrr", native_cost_headers, session)
    result["wisig.tex"] = table("lrr", ["Endpoint", "Diff.", "95\\% interval"], stress_rows(inputs["stress"]))
    arms = {x["arm"]: x for x in inputs["factorial"]["A"]["dev_summary"]}
    result["bn_gelu.tex"] = table("ccrr", ["BN", "GELU", "Acc.", "Macro-F1"], [
        [bn, gelu, f'{arms[key]["accuracy_pct"]:.3f}', f'{arms[key]["macro_f1_pct"]:.3f}']
        for key, bn, gelu in (("H0", "Yes", "Yes"), ("LA", "Yes", "No"),
                              ("H2", "No", "Yes"), ("H2_LA", "No", "No"))
    ])
    first = inputs["structure"]["paired_comparisons"][0]
    assert (first["a"], first["b"], first["positive_fit_pairs"]) == ("H0", "LP", 6)
    assert f'{first["mean_delta_pp"]:.3f}' == "2.835"
    p_h0, p_lp = structure_cost["H0"]["params"], structure_cost["LP"]["params"]
    assert f"{100 * (p_lp / p_h0 - 1):.2f}" == "3.48"
    accuracy = {row["arm"]: row["accuracy_pct"] for row in inputs["structure"]["summary"]}
    result["matched_tokenizer_metrics.tex"] = (
        "% Generated from the matched GPU comparison in the submission evidence ledger.\n"
        f"\\def\\MatchedHStemAccuracy{{{accuracy['H0']:.3f}\\%}}\n"
        f"\\def\\MatchedHStemParameters{{{p_h0:,}}}\n"
        f"\\def\\MatchedLongPatchAccuracy{{{accuracy['LP']:.3f}\\%}}\n"
        f"\\def\\MatchedLongPatchParameters{{{p_lp:,}}}\n"
    )
    f = inputs["factorial"]["A"]["primary_interaction"]
    assert f'{f["mean_delta_pp"]:.3f}' == "1.350"
    assert not inputs["factorial"]["C"]["pilot"]["consistent_pilot_signal"]
    a = inputs["cfo_augmentation"]["clean_comparisons"]["HStem"]
    assert f'{a["mean_pp"]:.2f}' == "0.19" and a["interval95_pp"][0] < 0 < a["interval95_pp"][1]
    return result, origins


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    artifacts, origins = generate()
    files = {HERE / "tables" / name: content for name, content in artifacts.items()}
    files[HERE / "EVIDENCE_SOURCES.json"] = json.dumps(origins, indent=2) + "\n"
    for path, content in files.items():
        if args.check:
            if path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"Evidence mismatch: {path.name}")
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
    print(f"{'Verified' if args.check else 'Generated'} {len(artifacts)} LaTeX artifacts "
          f"from {len(origins)} frozen inputs")
