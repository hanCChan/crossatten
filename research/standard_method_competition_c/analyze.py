"""Frozen Stage C source-CV analysis; never reads IQ or model weights.

Only complete, byte-verified sealed predictions may enter the CLI. The public
array functions exist for synthetic engineering tests. Source-selection results
are development findings, including their conditional RX bootstrap intervals.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_stage_c_frozen_a_metrics", HERE.parent / "controlled_training_a/analyze.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
ARMS = ("C0", "C1", "C2")
VIEWS = ("V0", "V1")
STREAMS = ("S0", "S1", "S2")
WEIGHT_DECAYS = (0.0, 0.001)
ENDPOINTS = ("selected", "epoch10", "final40")
METRICS = ("accuracy", "macro_f1", "ce")
PAIR_METRICS = ("accuracy_delta", "macro_f1_delta", "ce_delta", "disagreement", "corrected_fraction", "new_error_fraction")
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 24501
SUPPORT_THRESHOLD = 50
INTERPRETATION = ("Source-CV development/selection findings on 12 observed source-selection RX and three fixed paired streams; "
                  "not independent confirmation, an unseen-RX population estimate, seed-population uncertainty, or a pure objective causal effect.")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return A.file_sha256(path)


def _safe_child(parent: Path, relative: str) -> Path:
    path = (parent / relative).resolve()
    if not path.is_relative_to(parent.resolve()):
        raise ValueError(f"Artifact path escapes its registered directory: {relative}")
    return path


def expected_prediction_files(jobs: list[dict]) -> set[str]:
    return {f"{job['id']}/{arm}_{endpoint}_selection.npz"
            for job in jobs for arm in job["readouts"] for endpoint in ENDPOINTS}


def verify_design(jobs: list[dict], source_plan: dict) -> None:
    expected = {(v, s, wd, family) for v in VIEWS for s in STREAMS
                for wd in WEIGHT_DECAYS for family in ("ERM_EMA", "RX_GDRO")}
    actual = [(j["view"], j["stream"], float(j["weight_decay"]), j["family"]) for j in jobs]
    if len(jobs) != 24 or len(set(j["id"] for j in jobs)) != 24 or set(actual) != expected or len(set(actual)) != 24:
        raise ValueError("Stage C requires exactly the frozen 24-fit factorial grid")
    for j in jobs:
        if j["readouts"] != (["C0", "C1"] if j["family"] == "ERM_EMA" else ["C2"]):
            raise ValueError("C0/C1 must be readouts of one shared ERM trajectory")
        if j.get("epochs") != 40 or j.get("batch") != 128 or j.get("max_gradient_updates") != 5080:
            raise ValueError("Frozen Stage C fit budget differs")
        index = STREAMS.index(j["stream"])
        if (j.get("initialization_seed"), j.get("order_seed"), j.get("stochastic_seed")) != (24101 + index, 24201 + index, 24301):
            raise ValueError("Frozen Stage C random stream differs")
    views = {v["id"]: v for v in source_plan["views"]}
    if set(views) != set(VIEWS):
        raise ValueError("Both frozen source views are required")
    rx = [r for v in VIEWS for r in views[v]["source_selection_rx"]]
    if len(rx) != 12 or len(set(rx)) != 12 or any(len(views[v]["source_selection_rx"]) != 6 for v in VIEWS):
        raise ValueError("Source-selection views require 12 distinct RX, six in each view")


def verified_prediction_files(stage: Path) -> dict[str, Path]:
    """Verify every completion, training artifact, checkpoint and NPZ before decoding any NPZ."""
    registration = read_json(stage / "PRE_REGISTRATION.json")
    for name in ("JOBS.json", "SOURCE_PLAN.json"):
        if registration["artifact_sha256"][name] != file_sha256(stage / name):
            raise ValueError(f"Frozen registration artifact changed: {name}")
    jobs, plan = read_json(stage / "JOBS.json"), read_json(stage / "SOURCE_PLAN.json")
    verify_design(jobs, plan)
    seal_path = stage / "STAGE_C_SEAL.json"
    seal, manifest = read_json(seal_path), read_json(stage / "EVALUATION_MANIFEST.json")
    if seal.get("status") != "SEALED_STAGE_C" or set(seal.get("runs", {})) != {j["id"] for j in jobs}:
        raise ValueError("Analysis requires all 24 completed fits in the Stage C seal")
    for field, name in (("registration_sha256", "PRE_REGISTRATION.json"), ("execution_sha256", "EXECUTION_REGISTRATION.json")):
        digest = file_sha256(stage / name)
        if manifest.get(field) != digest or seal.get(field) != digest:
            raise ValueError(f"Seal/evaluation references changed {name}")
    if manifest.get("seal_sha256") != file_sha256(seal_path):
        raise ValueError("Evaluation does not match the full Stage C seal")
    expected = expected_prediction_files(jobs)
    files = manifest.get("files", {})
    if set(files) != expected or len(files) != 108:
        raise ValueError("Evaluation manifest must contain all 108 source-selection predictions")
    artifacts = (stage / "artifacts").resolve()
    for job in jobs:
        jobdir = _safe_child(artifacts, job["id"])
        record = seal["runs"][job["id"]]
        complete_path = jobdir / "COMPLETE.json"
        if file_sha256(complete_path) != record["complete_sha256"]:
            raise ValueError(f"Sealed completion changed: {job['id']}")
        complete = read_json(complete_path)
        checkpoints = {f"{arm}_{endpoint}.pt" for arm in job["readouts"] for endpoint in ENDPOINTS}
        if set(record.get("checkpoints", {})) != checkpoints:
            raise ValueError("Seal must retain every selected, epoch10 and final40 arm checkpoint")
        training_files = complete.get("files_sha256", {})
        if not checkpoints.issubset(training_files):
            raise ValueError("Completion must seal all readout checkpoints")
        for name, digest in training_files.items():
            if file_sha256(_safe_child(jobdir, name)) != digest:
                raise ValueError(f"Sealed training artifact changed: {job['id']}/{name}")
        for name, digest in record["checkpoints"].items():
            if digest != training_files[name] or file_sha256(_safe_child(jobdir, name)) != digest:
                raise ValueError(f"Sealed checkpoint changed: {job['id']}/{name}")
    paths = {}
    for relative, digest in files.items():
        path = _safe_child(artifacts, relative)
        if file_sha256(path) != digest:
            raise ValueError(f"Unsealed or altered prediction artifact: {relative}")
        paths[relative] = path
    return paths


def validate_source_predictions(data: dict, view: dict, registration: dict) -> dict:
    """Enforce source role, frozen row identities/support and canonical key provenance."""
    if np.asarray(data.get("rx", [])).dtype.kind not in "US":
        raise ValueError("Stage C prediction RX must be explicit receiver names")
    normalized = A.validate_predictions(data)
    if normalized["logits"].shape[1] != 10:
        raise ValueError("Stage C requires all ten fixed model classes")
    role = view["source_selection"]
    if len(normalized["y"]) != role["rows"] or set(normalized["rx"]) != set(view["source_selection_rx"]):
        raise ValueError("Prediction rows/RX differ from the source-selection role")
    forbidden = set(registration["forbidden_historical_held_rx"]) | set(registration["forbidden_target_rx"])
    if set(normalized["rx"]) & forbidden or not set(normalized["day"]).issubset({1, 2}):
        raise ValueError("Forbidden held/target receiver or day in Stage C predictions")
    pool = np.asarray(data.get("pool_ids", []))
    if pool.shape != normalized["y"].shape or str(pool.dtype) != role["dtype"] or len(np.unique(pool)) != len(pool):
        raise ValueError("Frozen source pool row identity/dtype differs")
    if hashlib.sha256(np.sort(pool).tobytes()).hexdigest() != role["indices_array_sha256"]:
        raise ValueError("Source-selection pool row digest differs")
    for row in role["tx_rx_day_support"]:
        count = int(((normalized["rx"] == row["rx"]) & (normalized["y"] == row["tx"]) & (normalized["day"] == row["day"])).sum())
        if count != row["rows"]:
            raise ValueError("Frozen TX/RX/day support differs")
    for rx, day, key in zip(normalized["rx"], normalized["day"], normalized["keys"]):
        pieces = key.split("|")
        if (len(pieces) != 4 or pieces[0] != rx or pieces[1] != str(day)
                or pieces[2] != registration["source_cell_sha256"].get(f"{rx}|{day}")
                or not pieces[3].isdigit()):
            raise ValueError("Canonical key does not match registered source-cell provenance")
    normalized["pool_ids"] = pool
    return normalized


def _align(first: dict, second: dict) -> tuple[dict, dict]:
    left, right = A.align_pair(first, second)
    if "pool_ids" in first or "pool_ids" in second:
        if "pool_ids" not in first or "pool_ids" not in second:
            raise ValueError("Paired source predictions must both retain pool_ids")
        rows = {key: i for i, key in enumerate(second["keys"])}
        take = np.array([rows[k] for k in first["keys"]])
        if not np.array_equal(first["pool_ids"], second["pool_ids"][take]):
            raise ValueError("Canonical key has conflicting source pool row identity")
    return left, right


def _summary(data: dict) -> dict:
    result = A.summarize_predictions(data)
    q = A.packet_quantities(data["logits"], data["y"])
    # Counts of the real class versus predicted and hardest competing classes
    # remain explicit per RX, including pairs with zero observed counts.
    result["class_pair_counts"] = []
    for rx in sorted(result["per_rx"]):
        for tx in range(10):
            mask = (data["rx"] == rx) & (data["y"] == tx)
            for competitor in range(10):
                result["class_pair_counts"].append(dict(rx=rx, tx_class_index=tx, competitor_class_index=competitor,
                    true_class_support=int(mask.sum()), predicted_count=int((mask & (q["prediction"] == competitor)).sum()),
                    hardest_negative_count=int((mask & (q["hardest_negative"] == competitor)).sum())))
    observed = {(r["rx"], r["tx_class_index"]) for r in result["tx_rx"]}
    result["zero_support_tx_rx"] = [dict(rx=rx, tx_class_index=tx, packets=0) for rx in result["per_rx"] for tx in range(10) if (rx, tx) not in observed]
    supported = [r for r in result["tx_rx"] if r["packets"] >= SUPPORT_THRESHOLD]
    result["worst_supported_tx_rx"] = min(supported, key=lambda r: (r["accuracy"], r["rx"], r["tx_class_index"])) if supported else None
    return result


def _mean_rows(rows: list[dict], names: tuple[str, ...]) -> dict:
    return {name: float(np.mean([r[name] for r in rows])) for name in names}


def aggregate_candidate(summaries: dict[tuple[str, str], dict]) -> dict:
    per_rx, per_view, per_stream = {}, {}, {}
    for view in VIEWS:
        receivers = sorted(summaries[(view, STREAMS[0])]["per_rx"])
        for rx in receivers:
            per_rx[rx] = _mean_rows([summaries[(view, stream)]["per_rx"][rx] for stream in STREAMS], METRICS)
            per_rx[rx]["view"] = view
        per_view[view] = _mean_rows([summaries[(view, stream)]["rx_equal"] for stream in STREAMS], METRICS)
    for stream in STREAMS:
        per_stream[stream] = _mean_rows([summaries[(v, stream)]["per_rx"][rx] for v in VIEWS for rx in summaries[(v, stream)]["per_rx"]], METRICS)
    groups = []
    for view in VIEWS:
        lookup = {(s, r["rx"], r["tx_class_index"]): r for s in STREAMS for r in summaries[(view, s)]["tx_rx"]}
        identities = sorted({(rx, tx) for s, rx, tx in lookup})
        for rx, tx in identities:
            rows = [lookup[(s, rx, tx)] for s in STREAMS]
            if len({r["packets"] for r in rows}) != 1:
                raise ValueError("TX/RX support differs across fixed streams")
            groups.append(dict(rx=rx, view=view, tx_class_index=tx, packets_per_stream=rows[0]["packets"],
                               **_mean_rows(rows, METRICS)))
    supported = [r for r in groups if r["packets_per_stream"] >= SUPPORT_THRESHOLD]
    return dict(rx_equal=_mean_rows(list(per_rx.values()), METRICS), per_rx=per_rx, per_view=per_view, per_stream=per_stream,
                tx_rx_stream_mean=groups, worst_rx=min(per_rx, key=lambda rx: (per_rx[rx]["accuracy"], rx)),
                worst_tx_rx=min(groups, key=lambda r: (r["accuracy"], r["rx"], r["tx_class_index"])),
                worst_supported_tx_rx=min(supported, key=lambda r: (r["accuracy"], r["rx"], r["tx_class_index"])) if supported else None,
                averaging="Each RX metric is averaged over the three fixed streams, then RX are equally weighted; support is per stream, never multiplied by three.")


def paired_comparison(first: dict[tuple[str, str], dict], second: dict[tuple[str, str], dict], draws: np.ndarray) -> dict:
    cells = {}
    for view in VIEWS:
        for stream in STREAMS:
            left, right = _align(first[(view, stream)], second[(view, stream)])
            cells[(view, stream)] = A.compare_predictions(left, right)
    per_rx, per_view, per_stream = {}, {}, {}
    for view in VIEWS:
        receivers = sorted(cells[(view, STREAMS[0])]["per_rx"])
        for rx in receivers:
            per_rx[rx] = dict(view=view, **_mean_rows([cells[(view, s)]["per_rx"][rx] for s in STREAMS], PAIR_METRICS))
        per_view[view] = _mean_rows([cells[(view, s)]["rx_equal"] for s in STREAMS], PAIR_METRICS)
    receivers = sorted(per_rx)
    if len(receivers) != 12 or draws.ndim != 2 or draws.shape[1] != 12:
        raise ValueError("Bootstrap must resample the same twelve distinct RX clusters")
    for stream in STREAMS:
        per_stream[stream] = _mean_rows([cells[(v, stream)]["per_rx"][rx] for v in VIEWS for rx in cells[(v, stream)]["per_rx"]], PAIR_METRICS)
    intervals = {}
    for name in PAIR_METRICS:
        values = np.array([per_rx[rx][name] for rx in receivers])
        intervals[name] = [float(x) for x in np.quantile(values[draws].mean(axis=1), (.025, .975))]
    groups = []
    for view in VIEWS:
        lookup = {(s, r["rx"], r["tx_class_index"]): r for s in STREAMS for r in cells[(view, s)]["tx_rx"]}
        for rx, tx in sorted({(rx, tx) for s, rx, tx in lookup}):
            rows = [lookup[(s, rx, tx)] for s in STREAMS]
            if len({r["packets"] for r in rows}) != 1:
                raise ValueError("Paired TX/RX support differs across streams")
            groups.append(dict(rx=rx, view=view, tx_class_index=tx, packets_per_stream=rows[0]["packets"],
                               risk_increase=-float(np.mean([r["accuracy_delta"] for r in rows])), **_mean_rows(rows, PAIR_METRICS)))
    supported = [r for r in groups if r["packets_per_stream"] >= SUPPORT_THRESHOLD]
    return dict(direction="candidate minus baseline; positive accuracy is improvement, positive risk_increase is deterioration",
        rx_equal=_mean_rows(list(per_rx.values()), PAIR_METRICS), per_rx=per_rx, per_view=per_view, per_stream=per_stream,
        paired_rx_bootstrap_95_intervals=intervals, tx_rx_stream_mean=groups,
        worst_paired_rx=dict(rx=min(receivers, key=lambda rx: (per_rx[rx]["accuracy_delta"], rx)),
                             accuracy_delta=min(row["accuracy_delta"] for row in per_rx.values())),
        worst_tx_rx_risk_increase=max(groups, key=lambda r: (r["risk_increase"], r["rx"], r["tx_class_index"])),
        worst_supported_tx_rx_risk_increase=max(supported, key=lambda r: (r["risk_increase"], r["rx"], r["tx_class_index"])) if supported else None,
        cells={f"{v}/{s}": cells[(v, s)] for v in VIEWS for s in STREAMS})


def promotion_check(comparison: dict) -> dict:
    """All frozen gates apply jointly; exact boundaries retain their registered direction."""
    supported = [r for r in comparison["tx_rx_stream_mean"] if r["packets_per_stream"] >= SUPPORT_THRESHOLD]
    checks = dict(mean_accuracy_at_least_1pp=comparison["rx_equal"]["accuracy_delta"] >= .01,
        paired_rx_interval_lower_above_zero=comparison["paired_rx_bootstrap_95_intervals"]["accuracy_delta"][0] > 0,
        both_views_positive=all(comparison["per_view"][v]["accuracy_delta"] > 0 for v in VIEWS),
        at_least_two_streams_positive=sum(comparison["per_stream"][s]["accuracy_delta"] > 0 for s in STREAMS) >= 2,
        worst_paired_rx_at_least_minus_2pp=comparison["worst_paired_rx"]["accuracy_delta"] >= -.02,
        supported_tx_rx_risk_increase_at_most_5pp=all(r["risk_increase"] <= .05 for r in supported))
    return dict(passes=all(checks.values()), checks=checks, supported_tx_rx_groups=len(supported),
                interpretation="Development promotion means worth separately confirming; failure is not proof of universal ineffectiveness. No extra seed or method search is authorized.")


def choose_weight_decays(candidate_aggregates: dict[tuple[str, float], dict]) -> dict[str, float]:
    return {arm: min(WEIGHT_DECAYS, key=lambda wd: (candidate_aggregates[(arm, wd)]["rx_equal"]["ce"], wd)) for arm in ARMS}


def promotion_priority(promotions: dict, chosen_aggregates: dict) -> list[str]:
    return sorted((arm for arm in ("C1", "C2") if promotions[arm]["passes"]),
                  key=lambda arm: (chosen_aggregates[arm]["rx_equal"]["ce"], 0 if arm == "C1" else 1))


def analyze_predictions(predictions: dict[str, dict[tuple[str, float, str, str], dict]],
                        replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED) -> dict:
    """Analyze the entire endpoint/grid block; bootstrap overrides are for synthetic QA only."""
    expected = {(a, wd, v, s) for a in ARMS for wd in WEIGHT_DECAYS for v in VIEWS for s in STREAMS}
    if set(predictions) != set(ENDPOINTS) or any(set(predictions[e]) != expected for e in ENDPOINTS):
        raise ValueError("Report all three endpoints and all 36 arm candidates without selective omission")
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    # Full canonical identities agree within a view across endpoints, methods,
    # regularization and streams. Cross-view inputs are never paired.
    for view in VIEWS:
        reference = predictions["selected"][("C0", 0., view, "S0")]
        if len(set(reference["rx"])) != 6:
            raise ValueError("Each source-selection view must contain six RX")
        for endpoint in ENDPOINTS:
            for key, data in predictions[endpoint].items():
                if key[2] == view:
                    _align(reference, data)
    rx_by_view = {v: set(predictions["selected"][("C0", 0., v, "S0")]["rx"]) for v in VIEWS}
    if rx_by_view["V0"] & rx_by_view["V1"]:
        raise ValueError("Source-selection receiver sets must be disjoint")
    draws = np.random.default_rng(seed).integers(0, 12, size=(replicates, 12))
    endpoints, aggregate_lookup = {}, {}
    for endpoint in ENDPOINTS:
        summaries = {key: _summary(data) for key, data in predictions[endpoint].items()}
        aggregates = {(arm, wd): aggregate_candidate({(v, s): summaries[(arm, wd, v, s)] for v in VIEWS for s in STREAMS})
                      for arm in ARMS for wd in WEIGHT_DECAYS}
        aggregate_lookup[endpoint] = aggregates
        paired = {}
        for wd in WEIGHT_DECAYS:
            baseline = {(v, s): predictions[endpoint][("C0", wd, v, s)] for v in VIEWS for s in STREAMS}
            for arm in ("C1", "C2"):
                candidate = {(v, s): predictions[endpoint][(arm, wd, v, s)] for v in VIEWS for s in STREAMS}
                paired[f"{arm}-C0/wd={wd:g}"] = paired_comparison(baseline, candidate, draws)
        endpoints[endpoint] = dict(
            candidates={f"{a}/wd={wd:g}/{v}/{s}": summaries[(a, wd, v, s)] for a in ARMS for wd in WEIGHT_DECAYS for v in VIEWS for s in STREAMS},
            aggregates={f"{a}/wd={wd:g}": aggregates[(a, wd)] for a in ARMS for wd in WEIGHT_DECAYS},
            same_weight_decay_comparisons=paired)
    chosen = choose_weight_decays(aggregate_lookup["selected"])
    for endpoint in ENDPOINTS:
        baseline = {(v, s): predictions[endpoint][("C0", chosen["C0"], v, s)] for v in VIEWS for s in STREAMS}
        endpoints[endpoint]["tuned_comparisons"] = {arm: paired_comparison(baseline,
            {(v, s): predictions[endpoint][(arm, chosen[arm], v, s)] for v in VIEWS for s in STREAMS}, draws) for arm in ("C1", "C2")}
        endpoints[endpoint]["tuned_candidates"] = {a: aggregate_lookup[endpoint][(a, chosen[a])] for a in ARMS}
    promotions = {a: promotion_check(endpoints["selected"]["tuned_comparisons"][a]) for a in ("C1", "C2")}
    priority = promotion_priority(promotions, endpoints["selected"]["tuned_candidates"])
    return dict(stage="C", primary_endpoint="selected", auxiliary_endpoints=["epoch10", "final40"], endpoints=endpoints,
        weight_decay_selection=dict(rule="Minimum selected-checkpoint source-selection RX-equal CE over 12 RX and three fixed streams; exact tie smaller wd", chosen=chosen,
                                   auxiliary_rule="Carry the primary selected-endpoint wd choices to epoch10/final40; never select an endpoint after seeing results"),
        decision=dict(status="SOURCE_DEVELOPMENT_PROMOTION_FOR_SEPARATE_CONFIRMATION" if priority else "NO_STANDARD_METHOD_DEVELOPMENT_PROMOTION",
                      promotions=promotions, priority=priority, all_arms_retained=True, final_confirmation_admitted=False,
                      new_algorithm_validated=False, AB_attribution_line_closed=True),
        uncertainty=dict(unit="12 observed source-selection RX; three fixed stream values are averaged within RX before cluster resampling",
                         receivers=sorted(rx_by_view["V0"] | rx_by_view["V1"]), replicates=replicates, seed=seed,
                         paired_across_all_methods_and_streams=True, interval="95% percentile", population_significance_claim=False),
        interpretation=INTERPRETATION, gradient_fit_count=24, arm_readout_candidates=36, independent_fits_are_not_36=True)


def render_markdown(result: dict) -> str:
    lines = ["# 阶段 C：标准方法源域竞争结果", "", INTERPRETATION, "",
             "主端点为 selected；epoch10/final40 为全部展示的辅助结果。24 次梯度拟合产生 36 个读出候选，C0/C1 共享学生轨迹。", "",
             "| 端点 | 方法 | weight decay（按主端点 CE 选择） | RX 等权 Accuracy | Macro-F1 | CE |", "|---|---|---:|---:|---:|---:|"]
    for endpoint in ENDPOINTS:
        for arm in ARMS:
            metrics = result["endpoints"][endpoint]["tuned_candidates"][arm]["rx_equal"]
            lines.append(f"| {endpoint} | {arm} | {result['weight_decay_selection']['chosen'][arm]:g} | {metrics['accuracy']*100:.4f}% | {metrics['macro_f1']:.6f} | {metrics['ce']:.6f} |")
    lines += ["", "全部 wd、视图、流、TX×RX×Day 支持与风险、类别对计数、混淆、裕量及同输入错误变化见完整 ANALYSIS.json。", "",
              "| 比较（selected，各自选 wd） | Accuracy 差 / pp | RX 条件区间 / pp | 分歧率 | 开发晋级 |", "|---|---:|---:|---:|---|"]
    for arm in ("C1", "C2"):
        pair = result["endpoints"]["selected"]["tuned_comparisons"][arm]
        ci = pair["paired_rx_bootstrap_95_intervals"]["accuracy_delta"]
        lines.append(f"| {arm}−C0 | {pair['rx_equal']['accuracy_delta']*100:+.4f} | [{ci[0]*100:+.4f}, {ci[1]*100:+.4f}] | {pair['rx_equal']['disagreement']*100:.4f}% | {result['decision']['promotions'][arm]['passes']} |")
    lines += ["", f"开发决策：`{result['decision']['status']}`；候选优先序：{result['decision']['priority']}。", "",
              "该结果用于源域方法筛选，不能作为独立 CRD 确认。未通过不证明方法完全无用；不追加种子或普通因素实验。训练成本、EMA/BN 开销与延迟见单独成本记录。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=HERE)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or args.stage
    targets = [output / name for name in ("ANALYSIS.json", "RESULTS.zh-CN.md", "DECISION.json")]
    if any(p.exists() for p in targets):
        raise FileExistsError("Refusing to overwrite scientific Stage C analysis")
    paths = verified_prediction_files(args.stage)
    registration = read_json(args.stage / "PRE_REGISTRATION.json")
    jobs = read_json(args.stage / "JOBS.json")
    views = {v["id"]: v for v in read_json(args.stage / "SOURCE_PLAN.json")["views"]}
    predictions = {e: {} for e in ENDPOINTS}
    for job in jobs:
        for arm in job["readouts"]:
            for endpoint in ENDPOINTS:
                relative = f"{job['id']}/{arm}_{endpoint}_selection.npz"
                with np.load(paths[relative], allow_pickle=False) as payload:
                    data = {key: payload[key] for key in payload.files}
                predictions[endpoint][(arm, float(job["weight_decay"]), job["view"], job["stream"])] = validate_source_predictions(data, views[job["view"]], registration)
    result = analyze_predictions(predictions)
    diagnostics = {}
    diagdir = args.stage / "artifacts/analysis/packet_diagnostics"
    expected_diags = [diagdir / f"{job['id']}_{arm}_{e}.npz" for job in jobs for arm in job["readouts"] for e in ENDPOINTS]
    if any(p.exists() for p in expected_diags):
        raise FileExistsError("Refusing to overwrite packet diagnostics")
    for job in jobs:
        for arm in job["readouts"]:
            for endpoint in ENDPOINTS:
                name = f"{job['id']}_{arm}_{endpoint}"
                data = predictions[endpoint][(arm, float(job["weight_decay"]), job["view"], job["stream"])]
                diagnostics[name] = A.export_packet_diagnostics(diagdir / f"{name}.npz", data)
    result["artifacts"] = dict(evaluation_manifest_sha256=file_sha256(args.stage / "EVALUATION_MANIFEST.json"),
        registration_sha256=file_sha256(args.stage / "PRE_REGISTRATION.json"), execution_sha256=file_sha256(args.stage / "EXECUTION_REGISTRATION.json"),
        seal_sha256=file_sha256(args.stage / "STAGE_C_SEAL.json"), packet_diagnostics=diagnostics)
    output.mkdir(parents=True, exist_ok=True)
    targets[0].write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    targets[1].write_text(render_markdown(result), encoding="utf-8", newline="\n")
    targets[2].write_text(json.dumps(result["decision"], ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
