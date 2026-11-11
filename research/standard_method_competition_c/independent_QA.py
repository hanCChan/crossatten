"""Independent Stage C JSON-only execution and result consistency checks.

No scientific IQ, model weights, NumPy archives or order arrays are decoded or
read. This checks reported metadata/arithmetic; byte sealing and numerical
payload replay are separately performed by the registered execution pipeline.
This verifier does not import the trainer, method kernels or analysis module.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ("C0", "C1", "C2")
VIEWS = ("V0", "V1")
STREAMS = ("S0", "S1", "S2")
WDS = (0., .001)
ENDPOINTS = ("selected", "epoch10", "final40")
METRICS = ("accuracy", "macro_f1", "ce")
PAIR_METRICS = ("accuracy_delta", "macro_f1_delta", "ce_delta", "disagreement", "corrected_fraction", "new_error_fraction")
INITIAL_FIELDS = ("model_sha256", "bn_buffers_sha256", "optimizer_sha256", "scheduler_sha256", "rng_sha256")
ORDER_FIELDS = ("relative_order_sha256", "canonical_order_sha256", "class_day_order_sha256")


def read_json(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    if path.suffix not in (".json", ".jsonl"):
        raise ValueError("Independent QA reads only JSON/text metadata")
    return json.loads(path.read_text(encoding="utf-8"))


def text_sha(path: Path) -> str:
    if path.suffix not in (".json", ".jsonl", ".py", ".md"):
        raise ValueError("No scientific payload reads in independent QA")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual, expected, message: str, atol=1e-12) -> None:
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)) and
            math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=atol), message)


def finite(value, message: str, minimum=None, maximum=None) -> None:
    require(isinstance(value, (int, float)) and math.isfinite(value), message)
    require(minimum is None or value >= minimum, message)
    require(maximum is None or value <= maximum, message)


def independent_lr_path(values: list[float]) -> list[tuple[float, float]]:
    """PyTorch2.5 min/rel(.0001), patience5, factor.1, min1e-6, eps1e-8."""
    best, bad, lr, result = math.inf, 0, .001, []
    for value in values:
        used = lr
        if value < best * (1 - .0001):
            best, bad = value, 0
        else:
            bad += 1
        if bad > 5:
            next_lr = max(lr * .1, 1e-6)
            if lr - next_lr > 1e-8:
                lr = next_lr
            bad = 0
        result.append((used, lr))
    return result


def verify_training(evidence: dict, jobs: list, plan: dict, registration_sha: str, execution_sha: str) -> dict:
    require(len(jobs) == 24 and set(evidence) == {j["id"] for j in jobs}, "Exactly all24 frozen jobs required")
    expected = {(v, s, wd, family) for v in VIEWS for s in STREAMS for wd in WDS for family in ("ERM_EMA", "RX_GDRO")}
    require({(j["view"], j["stream"], j["weight_decay"], j["family"]) for j in jobs} == expected, "Frozen factorial job grid differs")
    views = {v["id"]: v for v in plan["views"]}
    totals = Counter()
    for job in jobs:
        name = job["id"]
        e = evidence[name]
        config, initial, history, complete = (e[k] for k in ("CONFIG", "INITIAL", "HISTORY", "COMPLETE"))
        view, arms = views[job["view"]], job["readouts"]
        stream_index = STREAMS.index(job["stream"])
        require((job["initialization_seed"], job["order_seed"], job["stochastic_seed"]) == (24101 + stream_index, 24201 + stream_index, 24301), name + " frozen random stream")
        require((job["epochs"], job["batch"], job["updates_per_epoch"], job["max_gradient_updates"]) == (40, 128, 127, 5080), name + " frozen job budget")
        require(arms == (["C0", "C1"] if job["family"] == "ERM_EMA" else ["C2"]), name + " readout family")
        require(config["job"] == job and complete["job"] == job, name + " job metadata")
        for record in (config, complete):
            require(record["registration_sha256"] == registration_sha and record["execution_sha256"] == execution_sha, name + " registration/execution identity")
        require(config["view"] == view["id"] and config["train_rows"] == view["fit_train"]["rows"] and config["checkpoint_val_rows"] == view["checkpoint_val"]["rows"], name + " source roles")
        require(set(config["allowed_training_RX"]) == set(view["fit_rx"]) and set(config["forbidden_training_RX"]) == set(view["source_selection_rx"]), name + " receiver roles")
        require(config["source_selection_IQ_read"] is False, name + " premature source-selection read")
        require(complete["status"] == "COMPLETE" and complete["epochs"] == 40 and complete["updates"] == 5080, name + " complete budget")
        require(complete["active_parameters"] == 82426 and len(history) == 40, name + " model/epoch budget")
        require(all(isinstance(initial[f], str) and len(initial[f]) == 64 for f in INITIAL_FIELDS), name + " initial state hashes")
        best = {arm: math.inf for arm in arms}
        best_epochs = {arm: 0 for arm in arms}
        raw_arm = "C0" if job["family"] == "ERM_EMA" else "C2"
        lr_path = independent_lr_path([row["val_rx_ce"][raw_arm] for row in history])
        for index, row in enumerate(history, start=1):
            require(row["epoch"] == index and row["updates"] == 127 * index, name + " per-epoch updates")
            require(set(row["val_rx_ce"]) == set(arms), name + " validation readouts")
            for arm in arms:
                value = row["val_rx_ce"][arm]
                finite(value, name + " finite validation CE", minimum=0.)
                if value < best[arm]:
                    best[arm], best_epochs[arm] = value, index
            require(row["best_epochs"] == best_epochs, name + " strict minimum / earliest-tie checkpoint")
            close(row["lr_used"], lr_path[index - 1][0], name + " LR used ordinary raw RX CE")
            close(row["lr_next"], lr_path[index - 1][1], name + " LR next ordinary raw RX CE")
            require(row["calibration_preserved_training_state"] is True, name + " training state preserved")
            for field in ORDER_FIELDS:
                require(isinstance(row[field], str) and len(row[field]) == 64, name + " complete epoch order hashes")
            finite(row["train_ce"], name + " training CE", minimum=0.)
            finite(row["train_objective"], name + " training objective", minimum=0.)
            finite(row["train_accuracy"], name + " training accuracy", minimum=0., maximum=1.)
            if job["family"] == "RX_GDRO":
                finite(row["q_min"], name + " GroupDRO lower weight", minimum=0., maximum=1.)
                finite(row["q_max"], name + " GroupDRO upper weight", minimum=0., maximum=1.)
                require(row["q_min"] <= row["q_max"] and 12 * row["q_min"] <= 1. + 1e-12 and 12 * row["q_max"] >= 1. - 1e-12, name + " feasible GroupDRO weight range")
        require(complete["best_epochs"] == best_epochs, name + " selected endpoint epochs")
        for arm in arms:
            close(complete["best_val_rx_ce"][arm], best[arm], name + " selected endpoint CE")
        expected_checkpoints = {f"{arm}_{endpoint}.pt" for arm in arms for endpoint in ENDPOINTS}
        require(set(complete["checkpoints"]) == expected_checkpoints, name + " all endpoint checkpoint records")
        require(all(complete["files_sha256"].get(k) == v for k, v in complete["checkpoints"].items()), name + " checkpoint seals consistent")
        require(set(complete["val_inference_replay"]) == {f"{a}_{e}" for a in arms for e in ENDPOINTS}, name + " all deployment replay records")
        require(all(r["exact"] is True for r in complete["val_inference_replay"].values()), name + " deployment replay exact")
        costs = complete["costs"]
        for key, value in costs.items():
            finite(value, name + " nonnegative finite cost " + key, minimum=0.)
        ema = job["family"] == "ERM_EMA"
        require(costs["BN_calibration_passes"] == (40 if ema else 0), name + " fit-only BN pass count")
        require(costs["BN_calibration_samples"] == (40 * config["train_rows"] if ema else 0), name + " fit-only BN sample count")
        require(costs["EMA_updates"] == (5080 if ema else 0), name + " EMA per optimizer update")
        totals.update(gradient_fits=1, gradient_epochs=40, gradient_updates=5080,
                      BN_calibration_passes=costs["BN_calibration_passes"], BN_calibration_samples=costs["BN_calibration_samples"],
                      EMA_updates=costs["EMA_updates"])
    # Family and wd changes preserve initialization and the actual relative,
    # canonical and class/day orders for each fixed source view and stream.
    for view in VIEWS:
        for stream in STREAMS:
            bucket = [j for j in jobs if j["view"] == view and j["stream"] == stream]
            reference = evidence[bucket[0]["id"]]
            for job in bucket[1:]:
                other = evidence[job["id"]]
                for field in INITIAL_FIELDS:
                    if field == "optimizer_sha256" and job["weight_decay"] != bucket[0]["weight_decay"]:
                        continue  # Coupled wd is explicitly part of the optimizer config.
                    require(reference["INITIAL"][field] == other["INITIAL"][field], "Paired initialization " + field)
                for field in ("train_keys_sha256", "val_keys_sha256"):
                    require(reference["CONFIG"][field] == other["CONFIG"][field], "Paired canonical source members")
                for left, right in zip(reference["HISTORY"], other["HISTORY"]):
                    require(all(left[f] == right[f] for f in ORDER_FIELDS), "Paired actual epoch order")
    require(totals["gradient_updates"] == 121920 and totals["gradient_epochs"] == 960 and totals["BN_calibration_passes"] == 480, "Full24 aggregate budget")
    return dict(status="PASS", totals=dict(totals), checked="All24 completion metadata, every epoch, strict checkpoint ranking, independent LR simulation, fit-only EMA/BN counts, shared initialization and orders")


def verify_quantiles(row: dict, field="margin_quantiles") -> None:
    values = row[field]
    require(set(values) == {"0.01", "0.05", "0.25", "0.5", "0.75", "0.95", "0.99"}, "Complete registered margin quantiles")
    ordered = [values[str(q)] for q in (.01, .05, .25, .5, .75, .95, .99)]
    require(all(math.isfinite(v) for v in ordered) and ordered == sorted(ordered), "Finite ordered margin quantiles")


def verify_metric(row: dict) -> np.ndarray:
    cm = np.asarray(row["confusion"])
    require(cm.shape == (10, 10) and cm.dtype.kind in "iu" and np.all(cm >= 0), "Fixed10 confusion matrix counts")
    n = int(cm.sum())
    require(n == row["packets"] and n > 0 and row["errors"] == n - int(np.trace(cm)), "Confusion support/errors")
    close(row["accuracy"], float(np.trace(cm)) / n, "Confusion accuracy")
    denominator = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * np.diag(cm), denominator, out=np.zeros(10), where=denominator != 0)
    close(row["macro_f1"], float(f1.mean()), "Fixed10 macro F1 incl absent zero classes")
    finite(row["ce"], "Finite nonnegative CE", minimum=0.)
    verify_quantiles(row)
    for field in ("negative_margin_fraction", "zero_margin_fraction"):
        finite(row[field], "Margin fraction bounds", minimum=0., maximum=1.)
    require(row["negative_margin_fraction"] <= row["errors"] / n + 1e-12, "Negative margin implies classification error")
    return cm


def verify_candidate(summary: dict, view: dict) -> None:
    role, expected_rx = view["source_selection"], set(view["source_selection_rx"])
    require(summary["packets"] == role["rows"] and summary["classes"] == 10 and summary["rx_count"] == 6 and set(summary["per_rx"]) == expected_rx, "Candidate frozen source-selection role")
    full_cm = verify_metric(summary["packet_weighted"])
    require(np.array_equal(sum(verify_metric(r) for r in summary["per_rx"].values()), full_cm), "PerRX confusion sums to full candidate")
    for metric in METRICS:
        close(summary["rx_equal"][metric], np.mean([r[metric] for r in summary["per_rx"].values()]), "RXequal candidate arithmetic")
    expected_support = {(r["rx"], r["tx"], r["day"]): r["rows"] for r in role["tx_rx_day_support"]}
    observed = {(r["rx"], r["tx_class_index"], r["day"]): r for r in summary["tx_rx_day"]}
    require(set(observed) == {k for k, n in expected_support.items() if n > 0}, "All positive TX/RX/day cells")
    require({(r["rx"], r["tx_class_index"], r["day"]) for r in summary["zero_support_tx_rx_day"]} == {k for k, n in expected_support.items() if n == 0}, "All zero-support TX/RX/day cells")
    for identity, row in observed.items():
        require(row["packets"] == expected_support[identity], "Frozen TX/RX/day support")
        cm = verify_metric(row)
        require(int(cm[identity[1]].sum()) == row["packets"], "Local confusion true class matches TX identity")
    tx_groups = {(r["rx"], r["tx_class_index"]): r for r in summary["tx_rx"]}
    for rx in expected_rx:
        require(np.array_equal(sum(np.asarray(r["confusion"]) for k, r in observed.items() if k[0] == rx), np.asarray(summary["per_rx"][rx]["confusion"])), "Day groups sum to RX confusion")
        for tx in range(10):
            cells = [r for key, r in observed.items() if key[:2] == (rx, tx)]
            if cells:
                row = tx_groups[(rx, tx)]
                verify_metric(row)
                require(np.array_equal(np.asarray(row["confusion"]), sum(np.asarray(r["confusion"]) for r in cells)), "Day groups sum to TX/RX confusion")
    require(set(tx_groups) == {(rx, tx) for rx in expected_rx for tx in range(10) if sum(n for (r, t, d), n in expected_support.items() if r == rx and t == tx) > 0}, "All positive TX/RX groups")
    require({(r["rx"], r["tx_class_index"]) for r in summary["zero_support_tx_rx"]} == {(rx, tx) for rx in expected_rx for tx in range(10)} - set(tx_groups), "All zero-support TX/RX groups")
    class_pairs = {(r["rx"], r["tx_class_index"], r["competitor_class_index"]): r for r in summary["class_pair_counts"]}
    require(len(class_pairs) == len(summary["class_pair_counts"]) == 600, "All10x10 class pair counts per RX")
    for rx in expected_rx:
        cm = np.asarray(summary["per_rx"][rx]["confusion"])
        for tx in range(10):
            rows = [class_pairs[(rx, tx, j)] for j in range(10)]
            support = int(cm[tx].sum())
            require(all(r["true_class_support"] == support and isinstance(r["predicted_count"], int) and isinstance(r["hardest_negative_count"], int) and r["predicted_count"] >= 0 and r["hardest_negative_count"] >= 0 for r in rows), "Class pair nonnegative counts/support")
            require([r["predicted_count"] for r in rows] == cm[tx].tolist(), "Class pair predictions match confusion")
            require(sum(r["hardest_negative_count"] for r in rows) == support and rows[tx]["hardest_negative_count"] == 0, "Hard negative differs from true class")
    worst = min(tx_groups.values(), key=lambda r: (r["accuracy"], r["rx"], r["tx_class_index"]))
    require(all(summary["worst_tx_rx"][k] == worst[k] for k in summary["worst_tx_rx"]), "Reported worst observed TX/RX")
    supported = [r for r in tx_groups.values() if r["packets"] >= 50]
    require(summary["worst_supported_tx_rx"] == (min(supported, key=lambda r: (r["accuracy"], r["rx"], r["tx_class_index"])) if supported else None), "Reported worst supported TX/RX")


def verify_pair_row(row: dict, first: dict, second: dict) -> None:
    n = first["packets"]
    require(row["packets"] == n == second["packets"], "Paired support")
    count_fields = ("prediction_disagreements", "corrected_errors", "new_errors", "both_wrong_changed_class", "first_errors", "second_errors")
    require(all(isinstance(row[k], int) and 0 <= row[k] <= n for k in count_fields), "Paired error count bounds")
    require(row["first_errors"] == first["errors"] and row["second_errors"] == second["errors"], "Paired endpoint error counts")
    require(row["prediction_disagreements"] == row["corrected_errors"] + row["new_errors"] + row["both_wrong_changed_class"], "Error flip partition")
    require(row["second_errors"] - row["first_errors"] == row["new_errors"] - row["corrected_errors"], "Corrected/new error balance")
    require(row["corrected_errors"] <= first["errors"] and row["new_errors"] <= n - first["errors"] and row["both_wrong_changed_class"] <= min(first["errors"], second["errors"]), "Paired error feasible counts")
    for field, count in (("disagreement", "prediction_disagreements"), ("corrected_fraction", "corrected_errors"), ("new_error_fraction", "new_errors")):
        close(row[field], row[count] / n, "Error flip rate arithmetic")
    for metric in METRICS:
        close(row[metric + "_delta"], second[metric] - first[metric], "Paired " + metric + " delta")
    verify_quantiles(row, "margin_delta_quantiles")


def verify_aggregate(aggregate: dict, cells: dict) -> dict:
    per_rx = {}
    for view in VIEWS:
        for rx in cells[(view, "S0")]["per_rx"]:
            per_rx[rx] = {m: float(np.mean([cells[(view, s)]["per_rx"][rx][m] for s in STREAMS])) for m in METRICS}
            for metric in METRICS:
                close(aggregate["per_rx"][rx][metric], per_rx[rx][metric], "Candidate fixed-stream mean per RX")
        for metric in METRICS:
            close(aggregate["per_view"][view][metric], np.mean([cells[(view, s)]["rx_equal"][metric] for s in STREAMS]), "Candidate mean per view")
    require(set(aggregate["per_rx"]) == set(per_rx) and len(per_rx) == 12, "Candidate 12 unique receivers")
    result = {m: float(np.mean([row[m] for row in per_rx.values()])) for m in METRICS}
    for metric in METRICS:
        close(aggregate["rx_equal"][metric], result[metric], "Candidate twelve-RX equal mean")
        for stream in STREAMS:
            close(aggregate["per_stream"][stream][metric], np.mean([row[metric] for v in VIEWS for row in cells[(v, stream)]["per_rx"].values()]), "Candidate mean per stream")
    return result


def verify_comparison(comparison: dict, first: dict, second: dict, draws: np.ndarray) -> dict:
    per_rx = {}
    group_deltas = {}
    for view in VIEWS:
        for stream in STREAMS:
            left, right = first[(view, stream)], second[(view, stream)]
            cell = comparison["cells"][f"{view}/{stream}"]
            verify_pair_row(cell["packet_weighted"], left["packet_weighted"], right["packet_weighted"])
            for rx in left["per_rx"]:
                verify_pair_row(cell["per_rx"][rx], left["per_rx"][rx], right["per_rx"][rx])
            for table in ("tx_rx", "tx_rx_day"):
                dimensions = ("rx", "tx_class_index") + (("day",) if table == "tx_rx_day" else ())
                def indexed(rows):
                    return {tuple(r[k] for k in dimensions): r for r in rows}
                lrows, rrows, prows = indexed(left[table]), indexed(right[table]), indexed(cell[table])
                require(set(lrows) == set(rrows) == set(prows), "All paired local cells retained")
                for identity, row in prows.items():
                    verify_pair_row(row, lrows[identity], rrows[identity])
            for rx in left["per_rx"]:
                per_rx.setdefault(rx, {m: [] for m in PAIR_METRICS})
                for metric in PAIR_METRICS:
                    per_rx[rx][metric].append(cell["per_rx"][rx][metric])
            for row in cell["tx_rx"]:
                key = row["rx"], row["tx_class_index"]
                group_deltas.setdefault(key, []).append(row)
            for metric in PAIR_METRICS:
                close(cell["rx_equal"][metric], np.mean([r[metric] for r in cell["per_rx"].values()]), "Per-cell RXequal paired mean")
    per_rx = {rx: {m: float(np.mean(values)) for m, values in metrics.items()} for rx, metrics in per_rx.items()}
    require(len(per_rx) == 12 and set(comparison["per_rx"]) == set(per_rx), "Twelve observed RX clusters in comparison")
    receivers = sorted(per_rx)
    for rx in receivers:
        for metric in PAIR_METRICS:
            close(comparison["per_rx"][rx][metric], per_rx[rx][metric], "Three-stream paired RX mean")
    for metric in PAIR_METRICS:
        vector = np.array([per_rx[rx][metric] for rx in receivers])
        close(comparison["rx_equal"][metric], vector.mean(), "Twelve-RX paired mean")
        expected = np.quantile(vector[draws].mean(1), [.025, .975])
        for actual, value in zip(comparison["paired_rx_bootstrap_95_intervals"][metric], expected):
            close(actual, value, "Independent10000x12 RX bootstrap interval")
        for view in VIEWS:
            close(comparison["per_view"][view][metric], np.mean([comparison["cells"][f"{view}/{s}"]["rx_equal"][metric] for s in STREAMS]), "Paired view mean")
        for stream in STREAMS:
            close(comparison["per_stream"][stream][metric], np.mean([r[metric] for v in VIEWS for r in comparison["cells"][f"{v}/{stream}"]["per_rx"].values()]), "Paired stream mean")
    groups = {(r["rx"], r["tx_class_index"]): r for r in comparison["tx_rx_stream_mean"]}
    require(set(groups) == set(group_deltas) and len(groups) == len(comparison["tx_rx_stream_mean"]), "Every paired TX/RX group retained")
    for identity, rows in group_deltas.items():
        require(len(rows) == 3 and len({r["packets"] for r in rows}) == 1, "Three streams share local support")
        require(groups[identity]["packets_per_stream"] == rows[0]["packets"], "Support is not multiplied by streams")
        for metric in PAIR_METRICS:
            close(groups[identity][metric], np.mean([r[metric] for r in rows]), "Paired TX/RX stream mean")
        close(groups[identity]["risk_increase"], -np.mean([r["accuracy_delta"] for r in rows]), "Error risk direction")
    worst_rx = min(receivers, key=lambda rx: (per_rx[rx]["accuracy_delta"], rx))
    require(comparison["worst_paired_rx"]["rx"] == worst_rx, "Worst paired RX identity")
    close(comparison["worst_paired_rx"]["accuracy_delta"], per_rx[worst_rx]["accuracy_delta"], "Worst paired RX difference")
    return dict(per_rx=per_rx, groups=groups)


def independent_promotion(comparison: dict) -> dict:
    checks = dict(mean_accuracy_at_least_1pp=comparison["rx_equal"]["accuracy_delta"] >= .01,
        paired_rx_interval_lower_above_zero=comparison["paired_rx_bootstrap_95_intervals"]["accuracy_delta"][0] > 0.,
        both_views_positive=all(comparison["per_view"][v]["accuracy_delta"] > 0. for v in VIEWS),
        at_least_two_streams_positive=sum(comparison["per_stream"][s]["accuracy_delta"] > 0. for s in STREAMS) >= 2,
        worst_paired_rx_at_least_minus_2pp=comparison["worst_paired_rx"]["accuracy_delta"] >= -.02,
        supported_tx_rx_risk_increase_at_most_5pp=all(r["risk_increase"] <= .05 for r in comparison["tx_rx_stream_mean"] if r["packets_per_stream"] >= 50))
    return dict(passes=all(checks.values()), checks=checks)


def verify_analysis(result: dict, plan: dict) -> dict:
    require(set(result["endpoints"]) == set(ENDPOINTS) and result["primary_endpoint"] == "selected" and result["auxiliary_endpoints"] == ["epoch10", "final40"], "Frozen complete primary/auxiliary endpoints")
    require(result["gradient_fit_count"] == 24 and result["arm_readout_candidates"] == 36 and result["independent_fits_are_not_36"] is True, "Shared C0/C1 fit accounting")
    uncertainty = result["uncertainty"]
    views = {v["id"]: v for v in plan["views"]}
    receivers = sorted(rx for v in views.values() for rx in v["source_selection_rx"])
    require(len(set(receivers)) == 12 and uncertainty["receivers"] == receivers, "Twelve disjoint legal source-selection receivers")
    require(uncertainty["seed"] == 24501 and uncertainty["replicates"] == 10000 and uncertainty["paired_across_all_methods_and_streams"] is True and uncertainty["population_significance_claim"] is False, "Frozen bootstrap and uncertainty interpretation")
    draws = np.random.default_rng(24501).integers(0, 12, size=(10000, 12))
    expected_cells = {f"{a}/wd={wd:g}/{v}/{s}" for a in ARMS for wd in WDS for v in VIEWS for s in STREAMS}
    selected_scores, keysets = {}, {}
    for endpoint in ENDPOINTS:
        block = result["endpoints"][endpoint]
        require(set(block["candidates"]) == expected_cells and len(block["candidates"]) == 36, "All36 candidate readouts per endpoint")
        for name, summary in block["candidates"].items():
            view = name.split("/")[2]
            verify_candidate(summary, views[view])
            if view in keysets:
                require(summary["canonical_key_set_sha256"] == keysets[view], "Same canonical inputs across all streams/methods/endpoints")
            keysets[view] = summary["canonical_key_set_sha256"]
        for arm in ARMS:
            for wd in WDS:
                cells = {(v, s): block["candidates"][f"{arm}/wd={wd:g}/{v}/{s}"] for v in VIEWS for s in STREAMS}
                metrics = verify_aggregate(block["aggregates"][f"{arm}/wd={wd:g}"], cells)
                if endpoint == "selected":
                    selected_scores[(arm, wd)] = metrics
        require(set(block["same_weight_decay_comparisons"]) == {f"{a}-C0/wd={wd:g}" for a in ("C1", "C2") for wd in WDS}, "Both controlled same-wd comparisons")
        for wd in WDS:
            baseline = {(v, s): block["candidates"][f"C0/wd={wd:g}/{v}/{s}"] for v in VIEWS for s in STREAMS}
            for arm in ("C1", "C2"):
                candidate = {(v, s): block["candidates"][f"{arm}/wd={wd:g}/{v}/{s}"] for v in VIEWS for s in STREAMS}
                verify_comparison(block["same_weight_decay_comparisons"][f"{arm}-C0/wd={wd:g}"], baseline, candidate, draws)
    chosen = {a: min(WDS, key=lambda wd: (selected_scores[(a, wd)]["ce"], wd)) for a in ARMS}
    require(result["weight_decay_selection"]["chosen"] == chosen, "WD minimum primary source-selection CE, smaller-wd exact tie")
    for endpoint in ENDPOINTS:
        block = result["endpoints"][endpoint]
        for arm in ARMS:
            require(block["tuned_candidates"][arm] == block["aggregates"][f"{arm}/wd={chosen[arm]:g}"], "All endpoints carry primary chosen wd")
        baseline = {(v, s): block["candidates"][f"C0/wd={chosen['C0']:g}/{v}/{s}"] for v in VIEWS for s in STREAMS}
        require(set(block["tuned_comparisons"]) == {"C1", "C2"}, "Both tuned-source development comparisons")
        for arm in ("C1", "C2"):
            candidate = {(v, s): block["candidates"][f"{arm}/wd={chosen[arm]:g}/{v}/{s}"] for v in VIEWS for s in STREAMS}
            verify_comparison(block["tuned_comparisons"][arm], baseline, candidate, draws)
    decision = result["decision"]
    promotions = {a: independent_promotion(result["endpoints"]["selected"]["tuned_comparisons"][a]) for a in ("C1", "C2")}
    for arm in ("C1", "C2"):
        require(decision["promotions"][arm]["passes"] == promotions[arm]["passes"] and decision["promotions"][arm]["checks"] == promotions[arm]["checks"], "All joint frozen promotion constraints")
    priority = sorted((a for a in ("C1", "C2") if promotions[a]["passes"]), key=lambda a: (selected_scores[(a, chosen[a])]["ce"], 0 if a == "C1" else 1))
    require(decision["priority"] == priority, "Promotion CE priority, C1 exact tie")
    require(decision["status"] == ("SOURCE_DEVELOPMENT_PROMOTION_FOR_SEPARATE_CONFIRMATION" if priority else "NO_STANDARD_METHOD_DEVELOPMENT_PROMOTION"), "Frozen development exit")
    require(decision["all_arms_retained"] is True and decision["final_confirmation_admitted"] is False and decision["new_algorithm_validated"] is False and decision["AB_attribution_line_closed"] is True, "Scientific interpretation gates remain closed")
    return dict(status="PASS", candidate_endpoint_tables=108, paired_comparisons=18,
                independently_recomputed_bootstrap_replicates=10000, bootstrap_seed=24501, chosen_weight_decay=chosen,
                checked="Confusion/error/support/class-pair/margin consistency; all36x3 readouts; same-wd and tuned pair arithmetic; fixed-stream RX means and paired bootstrap; WD selection and every promotion constraint")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=HERE)
    parser.add_argument("--analysis", type=Path, help="JSON or JSON.gz full analysis; default stage/ANALYSIS.json")
    parser.add_argument("--training-only", action="store_true", help="Pre-scoring engineering validation, with a distinct output filename")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or args.stage / ("TRAINING_EXECUTION_QA.json" if args.training_only else "EXECUTION_QA.json")
    require(not output.exists(), "Refusing to overwrite original independent QA")
    jobs, plan = read_json(args.stage / "JOBS.json"), read_json(args.stage / "SOURCE_PLAN.json")
    evidence, seal = read_json(args.stage / "RUN_EVIDENCE.json"), read_json(args.stage / "STAGE_C_SEAL.json")
    registration_sha, execution_sha = text_sha(args.stage / "PRE_REGISTRATION.json"), text_sha(args.stage / "EXECUTION_REGISTRATION.json")
    require(seal["status"] == "SEALED_STAGE_C" and seal["registration_sha256"] == registration_sha and seal["execution_sha256"] == execution_sha, "Stage seal registration identity")
    require(set(seal["runs"]) == {j["id"] for j in jobs} and seal["source_selection_scoring_started"] is False, "Complete pre-scoring seal")
    for job in jobs:
        folder = args.stage / "artifacts" / job["id"]
        require(text_sha(folder / "COMPLETE.json") == seal["runs"][job["id"]]["complete_sha256"], "Completion JSON sealed bytes")
        for field in ("CONFIG", "INITIAL", "HISTORY", "COMPLETE"):
            require(read_json(folder / (field + ".json")) == evidence[job["id"]][field], "Consolidated evidence equals original JSON")
        require(seal["runs"][job["id"]]["checkpoints"] == evidence[job["id"]]["COMPLETE"]["checkpoints"], "Checkpoint digest records agree")
    report = dict(status="PASS", scope="JSON/text metadata only, no scientific payload reads or decodes", registration_sha256=registration_sha,
                  execution_sha256=execution_sha, qa_code_sha256=text_sha(Path(__file__)), training=verify_training(evidence, jobs, plan, registration_sha, execution_sha),
                  limitations="JSON consistency checks do not independently reconstruct logits, model weights, IQ or numerical batch contents; byte sealing and payload replay remain separate registered checks.")
    if not args.training_only:
        analysis_path = args.analysis or args.stage / "ANALYSIS.json"
        report["analysis"] = verify_analysis(read_json(analysis_path), plan)
        report["analysis_path"] = str(analysis_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(status=report["status"], output=str(output))))


if __name__ == "__main__":
    main()
