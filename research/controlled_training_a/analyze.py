"""Finite-level Stage A contrasts on sealed predictions, with same-input pairing.

This module never reads IQ or model weights. Selected checkpoints are the primary
endpoint and epoch10 is auxiliary. Receiver bootstrap intervals describe only
the fixed observed receivers; they do not quantify unseen-RX or seed uncertainty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

RUNS = ("A00", "A10", "A01", "A11")
PAIRS = (("A00", "A10"), ("A00", "A01"), ("A01", "A11"),
         ("A10", "A11"), ("A00", "A11"))
ENDPOINTS = ("selected", "epoch10")
METRICS = ("accuracy", "macro_f1", "ce")
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 22401
PRACTICAL_SCALES = {"accuracy": .01, "macro_f1": .01, "ce": .05,
                    "disagreement": .05}


def validate_predictions(data: dict) -> dict:
    """Normalize predictions without accepting ambiguous or duplicate keys."""
    required = ("logits", "y", "rx", "day", "keys")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"Prediction fields missing: {missing}")
    result = {name: np.asarray(data[name]) for name in required}
    logits, y = result["logits"], result["y"]
    if logits.ndim != 2 or len(logits) == 0 or logits.shape[1] < 2:
        raise ValueError("Expected nonempty [packets, classes>=2] logits")
    n, classes = logits.shape
    if any(result[name].shape != (n,) for name in required[1:]):
        raise ValueError("All packet metadata must be one-dimensional and aligned")
    if not np.issubdtype(y.dtype, np.integer) or np.any(y < 0) or np.any(y >= classes):
        raise ValueError("Labels must be integer class indices within logits")
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite logits")
    # Numeric RX codes have no identity outside their source cache.
    if result["rx"].dtype.kind not in "US":
        if "rx_names" not in data:
            raise ValueError("Numeric RX codes require the explicit rx_names mapping")
        names = np.asarray(data["rx_names"])
        codes = result["rx"]
        if not np.issubdtype(codes.dtype, np.integer) or np.any(codes < 0) or np.any(codes >= len(names)):
            raise ValueError("RX codes outside rx_names")
        result["rx"] = names[codes].astype(str)
    else:
        result["rx"] = result["rx"].astype(str)
    result["keys"] = result["keys"].astype(str)
    if len(np.unique(result["keys"])) != n:
        raise ValueError("Canonical sample keys must be unique")
    if any(not key for key in result["keys"]):
        raise ValueError("Canonical sample keys cannot be empty")
    return result


def align_pair(first: dict, second: dict) -> tuple[dict, dict]:
    """Align by full canonical key, then verify RX, TX, and day identities."""
    left, right = validate_predictions(first), validate_predictions(second)
    if left["logits"].shape != right["logits"].shape:
        raise ValueError("Paired predictions must have the same sample and class counts")
    index = {key: i for i, key in enumerate(right["keys"])}
    if set(left["keys"]) != set(index):
        raise ValueError("Paired predictions do not contain the same canonical inputs")
    take = np.array([index[key] for key in left["keys"]], dtype=np.int64)
    right = {name: value[take] for name, value in right.items()}
    for name in ("keys", "rx", "y", "day"):
        if not np.array_equal(left[name], right[name]):
            raise ValueError(f"Canonical key has conflicting {name} identity")
    return left, right


def packet_quantities(logits: np.ndarray, y: np.ndarray) -> dict:
    values = np.asarray(logits, dtype=np.float64)
    rows = np.arange(len(values))
    maximum = values.max(axis=1)
    ce = maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=1)) - values[rows, y]
    negatives = values.copy()
    negatives[rows, y] = -np.inf
    hardest_negative = negatives.argmax(axis=1)
    return {"prediction": values.argmax(axis=1), "ce": ce,
            "hardest_negative": hardest_negative,
            "margin": values[rows, y] - values[rows, hardest_negative]}


def _quantiles(values: np.ndarray) -> dict:
    return {str(q): float(value) for q, value in zip(QUANTILES, np.quantile(values, QUANTILES))}


def _metrics(logits: np.ndarray, y: np.ndarray) -> dict:
    quantities = packet_quantities(logits, y)
    prediction = quantities["prediction"]
    classes = logits.shape[1]
    cm = np.bincount(y * classes + prediction, minlength=classes * classes).reshape(classes, classes)
    denominator = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * np.diag(cm), denominator, out=np.zeros(classes, dtype=float), where=denominator != 0)
    return {"packets": len(y), "errors": int((prediction != y).sum()),
            "accuracy": float((prediction == y).mean()), "macro_f1": float(f1.mean()),
            "ce": float(quantities["ce"].mean()), "margin_quantiles": _quantiles(quantities["margin"]),
            "negative_margin_fraction": float((quantities["margin"] < 0).mean()),
            "zero_margin_fraction": float((quantities["margin"] == 0).mean()),
            "confusion": cm.tolist()}


def _group_table(data: dict, dimensions: tuple[str, ...]) -> list[dict]:
    identities = sorted(set(zip(*(data[name].tolist() for name in dimensions))))
    rows = []
    for identity in identities:
        mask = np.ones(len(data["y"]), dtype=bool)
        for name, value in zip(dimensions, identity):
            mask &= data[name] == value
        row = {name: int(value) if name in ("y", "day") else str(value)
               for name, value in zip(dimensions, identity)}
        row.update(_metrics(data["logits"][mask], data["y"][mask]))
        if "y" in row:
            row["tx_class_index"] = row.pop("y")
        rows.append(row)
    return rows


def summarize_predictions(data: dict) -> dict:
    data = validate_predictions(data)
    per_rx = {str(rx): _metrics(data["logits"][data["rx"] == rx], data["y"][data["rx"] == rx])
              for rx in sorted(np.unique(data["rx"]))}
    groups = _group_table(data, ("rx", "y"))
    worst = min(groups, key=lambda item: (item["accuracy"], item["rx"], item["tx_class_index"]))
    key_sha = hashlib.sha256("\n".join(sorted(data["keys"])).encode("utf-8")).hexdigest()
    observed_cells = {(row["rx"], row["tx_class_index"], row["day"]) for row in _group_table(data, ("rx", "y", "day"))}
    zero_support = [{"rx": str(rx), "tx_class_index": tx, "day": int(day), "packets": 0}
                    for rx in sorted(per_rx) for tx in range(data["logits"].shape[1])
                    for day in sorted(np.unique(data["day"])) if (rx, tx, day) not in observed_cells]
    return {"packets": len(data["y"]), "classes": data["logits"].shape[1],
            "canonical_key_set_sha256": key_sha, "rx_count": len(per_rx),
            "rx_equal": {metric: float(np.mean([row[metric] for row in per_rx.values()])) for metric in METRICS},
            "packet_weighted": _metrics(data["logits"], data["y"]), "per_rx": per_rx,
            "tx_rx": groups, "tx_rx_day": _group_table(data, ("rx", "y", "day")),
            "zero_support_tx_rx_day": zero_support,
            "worst_tx_rx": {name: worst[name] for name in ("rx", "tx_class_index", "packets", "errors", "accuracy")},
            "macro_f1_definition": "mean over all fixed model classes per RX; absent classes use zero_division=0"}


def _paired_row(first: dict, second: dict, mask: np.ndarray) -> dict:
    left = packet_quantities(first["logits"][mask], first["y"][mask])
    right = packet_quantities(second["logits"][mask], second["y"][mask])
    y = first["y"][mask]
    first_correct, second_correct = left["prediction"] == y, right["prediction"] == y
    corrected = (~first_correct) & second_correct
    new_errors = first_correct & (~second_correct)
    disagreement = left["prediction"] != right["prediction"]
    both_wrong_changed = (~first_correct) & (~second_correct) & disagreement
    metrics_left, metrics_right = _metrics(first["logits"][mask], y), _metrics(second["logits"][mask], y)
    return {"packets": len(y), "prediction_disagreements": int(disagreement.sum()),
            "disagreement": float(disagreement.mean()), "corrected_errors": int(corrected.sum()),
            "corrected_fraction": float(corrected.mean()), "new_errors": int(new_errors.sum()),
            "new_error_fraction": float(new_errors.mean()), "both_wrong_changed_class": int(both_wrong_changed.sum()),
            "accuracy_delta": float(second_correct.mean() - first_correct.mean()),
            "macro_f1_delta": metrics_right["macro_f1"] - metrics_left["macro_f1"],
            "ce_delta": metrics_right["ce"] - metrics_left["ce"],
            "margin_delta_quantiles": _quantiles(right["margin"] - left["margin"]),
            "first_errors": int((~first_correct).sum()), "second_errors": int((~second_correct).sum())}


def compare_predictions(first: dict, second: dict) -> dict:
    first, second = align_pair(first, second)
    per_rx = {str(rx): _paired_row(first, second, first["rx"] == rx) for rx in sorted(np.unique(first["rx"]))}
    tables = {}
    for dimensions in (("rx", "y"), ("rx", "y", "day")):
        table = []
        identities = sorted(set(zip(*(first[name].tolist() for name in dimensions))))
        for identity in identities:
            mask = np.ones(len(first["y"]), dtype=bool)
            record = {}
            for name, value in zip(dimensions, identity):
                mask &= first[name] == value
                record["tx_class_index" if name == "y" else name] = int(value) if name in ("y", "day") else str(value)
            record.update(_paired_row(first, second, mask))
            table.append(record)
        tables["tx_rx" if len(dimensions) == 2 else "tx_rx_day"] = table
    rate_names = ("disagreement", "corrected_fraction", "new_error_fraction", "accuracy_delta", "macro_f1_delta", "ce_delta")
    worst = min(tables["tx_rx"], key=lambda row: (row["accuracy_delta"], row["rx"], row["tx_class_index"]))
    return {"direction": "second minus first; corrected/new counts are first to second",
            "packet_weighted": _paired_row(first, second, np.ones(len(first["y"]), dtype=bool)),
            "rx_equal": {name: float(np.mean([row[name] for row in per_rx.values()])) for name in rate_names},
            "per_rx": per_rx, **tables, "worst_accuracy_change_tx_rx": worst}


def factorial_contrasts(values: dict[str, float | np.ndarray]) -> dict:
    """Signed effects for the two prespecified fixed levels, without percentages."""
    m00, m10, m01, m11 = (values[name] for name in RUNS)
    return {"init": ((m10 - m00) + (m11 - m01)) / 2,
            "order": ((m01 - m00) + (m11 - m10)) / 2,
            "interaction": m11 - m10 - m01 + m00}


def _interval(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(values, (.025, .975))]


def analyze_endpoint(predictions: dict[str, dict], replicates: int = BOOTSTRAP_REPLICATES,
                     seed: int = BOOTSTRAP_SEED) -> dict:
    if set(predictions) != set(RUNS):
        raise ValueError("Factorial analysis requires exactly A00,A10,A01,A11; replay is QC only")
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    aligned = {"A00": validate_predictions(predictions["A00"])}
    for run in RUNS[1:]:
        _, aligned[run] = align_pair(aligned["A00"], predictions[run])
    summaries = {name: summarize_predictions(data) for name, data in aligned.items()}
    receivers = sorted(summaries["A00"]["per_rx"])
    draws = np.random.default_rng(seed).integers(0, len(receivers), size=(replicates, len(receivers)))
    contrasts = {}
    for metric in METRICS:
        rx_values = {run: np.array([summaries[run]["per_rx"][rx][metric] for rx in receivers]) for run in RUNS}
        per_rx_contrasts = factorial_contrasts(rx_values)
        contrasts[metric] = {}
        for factor, values in per_rx_contrasts.items():
            point = float(values.mean())
            ci = _interval(values[draws].mean(axis=1))
            contrasts[metric][factor] = {"estimate": point, "paired_rx_bootstrap_95_interval": ci,
                                       "per_rx": dict(zip(receivers, values.tolist())),
                                       "absolute_practical_scale": PRACTICAL_SCALES[metric],
                                       "exceeds_practical_screen": abs(point) >= PRACTICAL_SCALES[metric],
                                       "interval_excludes_zero": ci[0] > 0 or ci[1] < 0}
    pairs = {}
    for first, second in PAIRS:
        paired = compare_predictions(aligned[first], aligned[second])
        paired["paired_rx_bootstrap_95_intervals"] = {}
        for metric in ("disagreement", "accuracy_delta", "macro_f1_delta", "ce_delta", "corrected_fraction", "new_error_fraction"):
            values = np.array([paired["per_rx"][rx][metric] for rx in receivers])
            paired["paired_rx_bootstrap_95_intervals"][metric] = _interval(values[draws].mean(axis=1))
        paired["disagreement_exceeds_practical_screen"] = paired["rx_equal"]["disagreement"] >= PRACTICAL_SCALES["disagreement"]
        pairs[f"{first}_to_{second}"] = paired
    return {"runs": summaries, "factor_contrasts": contrasts, "paired_changes": pairs,
            "uncertainty": {"unit": "observed RX cluster, paired across all four cells",
                            "receivers": receivers, "replicates": replicates, "seed": seed,
                            "interpretation": "conditional descriptive sensitivity to resampling the fixed observed RX; not unseen-RX or seed-distribution uncertainty",
                            "no_formal_significance_claim": True},
            "effect_interpretation": "finite contrasts of the selected two initializations and two orders under the common adaptive training strategy"}


def make_analysis(predictions: dict[str, dict[str, dict]], replicates: int = BOOTSTRAP_REPLICATES,
                  seed: int = BOOTSTRAP_SEED) -> dict:
    if set(predictions) != set(ENDPOINTS):
        raise ValueError("Report both selected (primary) and epoch10 (auxiliary)")
    return {"status": "STAGE_A_FINITE_LEVEL_ANALYSIS_COMPLETE", "primary_endpoint": "selected",
            "auxiliary_endpoint": "epoch10", "endpoints": {endpoint: analyze_endpoint(predictions[endpoint], replicates, seed) for endpoint in ENDPOINTS},
            "practical_screens": PRACTICAL_SCALES,
            "scope": "HStem+MHA fixed fold2 source data; no ASKNet explanation or population variance attribution",
            "stop_rule": "No additional seeds, architecture or loss variants from this attribution line; Stage B requires separate registration",
            "class_index_note": "tx_class_index is the zero-based model label; use the registered TX name mapping for physical identifiers"}


def render_markdown(analyses: dict[str, dict]) -> str:
    lines = ["# 阶段 A：初始化 × 批顺序受控结果", "",
             "主要端点为最终选中 checkpoint，epoch10 为预定辅助端点。所有差值均按后者减前者计算；Accuracy/Macro-F1 为 0–1 比例，CE 单位为 nat。", "",
             "本结果只描述这两组初始化和两种顺序、固定 source 数据与自适应训练策略下的有限敏感性，不给出总体方差贡献比例，也不解释 ASKNet 的历史案例。", ""]
    for split, analysis in analyses.items():
        lines += [f"## {split}", ""]
        if split == "held":
            lines += ["source-held6 是已知历史接收机上的探索性评价；未参与本轮训练或选模，仍不能充当新接收机确认。", ""]
        for endpoint in ENDPOINTS:
            result = analysis["endpoints"][endpoint]
            lines += [f"### {endpoint}（{'主要' if endpoint == 'selected' else '辅助'}）", "",
                      "| 运行 | RX等权 Accuracy | RX等权 Macro-F1 | RX等权 CE |", "|---|---:|---:|---:|"]
            for run in RUNS:
                metrics = result["runs"][run]["rx_equal"]
                lines.append(f"| {run} | {metrics['accuracy']:.6f} | {metrics['macro_f1']:.6f} | {metrics['ce']:.6f} |")
            lines += ["", "| 指标 | 因素对比 | 差值 | 配对RX重采样95%区间 | 达实际效应筛查尺度 |", "|---|---|---:|---|---|"]
            for metric in METRICS:
                for factor, values in result["factor_contrasts"][metric].items():
                    lo, hi = values["paired_rx_bootstrap_95_interval"]
                    lines.append(f"| {metric} | {factor} | {values['estimate']:.6f} | [{lo:.6f}, {hi:.6f}] | {'是' if values['exceeds_practical_screen'] else '否'} |")
            lines += ["", "| 同一输入配对 | RX等权分歧率 | 纠正错误包数 | 新增错误包数 | 最差 TX×RX 的Accuracy变化 |", "|---|---:|---:|---:|---:|"]
            for name, values in result["paired_changes"].items():
                packet = values["packet_weighted"]
                lines.append(f"| {name} | {values['rx_equal']['disagreement']:.6f} | {packet['corrected_errors']} | {packet['new_errors']} | {values['worst_accuracy_change_tx_rx']['accuracy_delta']:.6f} |")
            lines += [""]
    lines += ["区间使用相同的 RX 重采样索引，10,000 次、seed 22401，仅描述固定已观察接收机的条件不确定性；不代表未见接收机总体或训练 seed 分布。实际尺度为 Accuracy/Macro-F1 绝对差 0.01、CE 0.05 nat、逐包分歧率 0.05，均是预定筛查尺度，不是显著性声明。", "",
              "完整 TX×RX×Day 支持、错误、最差组、margin 分位数和配对翻转见 ANALYSIS.json。重放运行仅作工程验收，不纳入因素估计。", "",
              "本阶段完成后停止扩张归因训练；阶段 B 独立登记，不比较 A/B 绝对性能以排名因素，也不根据历史单个坏例追加种子。", ""]
    return "\n".join(lines)


def load_prediction(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    return validate_predictions(data)


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def export_packet_diagnostics(path: Path, data: dict) -> dict:
    """Retain explicit per-input diagnostics locally, with a file integrity hash."""
    if path.exists():
        raise FileExistsError("Refusing to overwrite packet diagnostics")
    data = validate_predictions(data)
    quantities = packet_quantities(data["logits"], data["y"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, y=data["y"], rx=data["rx"], day=data["day"], keys=data["keys"], **quantities)
    return {"path": str(path), "sha256": file_sha256(path), "packets": len(data["y"])}


def verified_prediction_files(stage: Path) -> dict:
    """Verify all 20 files and the full-stage seal before decoding any NPZ."""
    seal_path = stage / "STAGE_A_SEAL.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not (seal.get("replay_pass") is True and set(seal.get("runs", {})) == set(RUNS) | {"A00-replay"}):
        raise ValueError("Analysis requires a five-run seal with explicit replay_pass=true")
    manifest = json.loads((stage / "EVALUATION_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("stage_a_seal_sha256") != file_sha256(seal_path):
        raise ValueError("Evaluation does not match the sealed five-run block")
    registration_sha = file_sha256(stage / "PRE_REGISTRATION.json")
    if manifest.get("registration_sha256") != registration_sha or seal.get("registration_sha256") != registration_sha:
        raise ValueError("Evaluation and seal must reference the unchanged preregistration")
    expected = {f"{run}/{endpoint}_{split}.npz" for run in (*RUNS, "A00-replay") for endpoint in ENDPOINTS for split in ("val", "held")}
    files = manifest.get("files", {})
    if set(files) != expected:
        raise ValueError("Evaluation manifest must contain all five runs and both splits/endpoints")
    paths = {}
    artifacts = (stage / "artifacts").resolve()
    for run in (*RUNS, "A00-replay"):
        complete_path = artifacts / run / "COMPLETE.json"
        if file_sha256(complete_path) != seal["runs"][run]["complete_sha256"]:
            raise ValueError(f"Sealed run completion record changed: {run}")
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        for relative, digest in complete["files_sha256"].items():
            path = artifacts / run / relative
            if not path.resolve().is_relative_to(artifacts / run) or file_sha256(path) != digest:
                raise ValueError(f"Sealed training artifact changed: {run}/{relative}")
    for relative, digest in files.items():
        path = artifacts / relative
        if not path.resolve().is_relative_to(artifacts) or file_sha256(path) != digest:
            raise ValueError(f"Unsealed or altered prediction artifact: {relative}")
        paths[relative] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path, help="Default: stage directory; packet diagnostics always remain in ignored artifacts")
    args = parser.parse_args()
    paths = verified_prediction_files(args.stage)
    output = args.out or args.stage
    target = output / "ANALYSIS.json"
    if target.exists():
        raise FileExistsError("Refusing to overwrite an existing scientific analysis")
    analyses = {}
    derived = {}
    # No separate split option: the preregistered report must display both.
    for split in ("val", "held"):
        predictions = {endpoint: {run: load_prediction(paths[f"{run}/{endpoint}_{split}.npz"]) for run in RUNS} for endpoint in ENDPOINTS}
        for endpoint in ENDPOINTS:
            replay = load_prediction(paths[f"A00-replay/{endpoint}_{split}.npz"])
            first, second = align_pair(predictions[endpoint]["A00"], replay)
            if not np.array_equal(first["logits"], second["logits"]):
                raise ValueError(f"Post-seal replay logits differ: {endpoint}/{split}")
        analyses[split] = make_analysis(predictions)
        analyses[split]["post_seal_replay_logits_exact"] = True
        for endpoint in ENDPOINTS:
            for run in RUNS:
                path = args.stage / "artifacts/analysis/packet_diagnostics" / f"{run}_{endpoint}_{split}.npz"
                derived[f"{run}_{endpoint}_{split}"] = export_packet_diagnostics(path, predictions[endpoint][run])
    analyses["artifacts"] = {"raw_prediction_manifest": str(args.stage / "EVALUATION_MANIFEST.json"),
                             "raw_prediction_manifest_sha256": file_sha256(args.stage / "EVALUATION_MANIFEST.json"),
                             "packet_diagnostics": derived}
    output.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(analyses, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "RESULTS.zh-CN.md").write_text(render_markdown({split: analyses[split] for split in ("val", "held")}), encoding="utf-8")


if __name__ == "__main__":
    main()
