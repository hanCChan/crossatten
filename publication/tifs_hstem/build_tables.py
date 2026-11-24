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
}
METHODS = (
    ("h0", "HStem"), ("ce", "Shen CE"), ("dann", "Shen DANN"),
    ("fan", "FAN"), ("ib", "IB-RFF"), ("ask", "ASKNet"),
)
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


def external(data, ma):
    indexed = {(row["view"], row["method"]): row for row in data["summary"]}
    for method, name in METHODS:
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
        if method == "h0":
            yield ["Spec+Siamese", f'{100*ma["acc"]:.3f} $\\pm$ {100*ma["fit_sd"]:.3f}',
                   "--", f'{100*ma["macro_f1"]:.3f}', "--", f'{100*ma["worst_rx_acc"]:.3f}', "--"]


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
    result["external.tex"] = table(
        "lrrrrrr", ["Method", "Native Acc.", "Strict-IB Acc.", "Native F1", "Strict F1",
                    "Native Worst", "Strict Worst"], external(inputs["external"], inputs["ma"]["methods"]["ma"]))
    result["long_patch.tex"] = table(
        "lrrrrrr", ["Tokenizer", "Params", "Support", "BN", r"Acc. $\pm$ fit SD", "F1", "Worst RX"],
        structure(inputs["structure"], inputs["structure_cost"]))
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
    ma = inputs["ma"]
    assert len(ma["selected_source"]) == 6
    result["siamese_timing.tex"] = table("lrrrr", cost_headers, [
        timing("Spec+Siamese", ma["cost"], ma["selected_source"][0]["params"])
    ])
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
    tables, origins = generate()
    files = {HERE / "tables" / name: content for name, content in tables.items()}
    files[HERE / "EVIDENCE_SOURCES.json"] = json.dumps(origins, indent=2) + "\n"
    for path, content in files.items():
        if args.check:
            if path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"Evidence mismatch: {path.name}")
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
    print(f"Verified {len(tables)} tables against {len(origins)} frozen inputs")
