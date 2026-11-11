"""Publish already-frozen C outputs; no IQ/model reads, fitting or new analysis rules.

This reporting helper is added after the registered analysis. The scientific
engine, analyzer and decisions remain unchanged. Existing publications are never
overwritten. The full uncompressed analysis stays in the ignored artifact tree.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ARMS = ("C0", "C1", "C2")
ENDPOINTS = ("selected", "epoch10", "final40")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    target = HERE / name
    if target.exists():
        raise FileExistsError(f"Keep original published output: {target}")
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def text(name, value):
    target = HERE / name
    if target.exists():
        raise FileExistsError(f"Keep original published output: {target}")
    target.write_text(value, encoding="utf-8", newline="\n")


def elapsed(end, start):
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


def latency_summary(records):
    result = {}
    for arm in ARMS:
        rows = [row for name, row in records.items() if name.endswith("/" + arm)]
        result[arm] = {}
        for batch in ("1", "128"):
            rounds = [r for row in rows for r in row["batches"][batch]]
            result[arm][batch] = dict(readouts=len(rows), rounds=len(rounds),
                median_round_p50_ms=median(r["p50_ms"] for r in rounds),
                median_round_p95_ms=median(r["p95_ms"] for r in rounds),
                min_round_p50_ms=min(r["p50_ms"] for r in rounds),
                max_round_p50_ms=max(r["p50_ms"] for r in rounds),
                median_round_throughput_samples_per_second=median(r["throughput_samples_per_second"] for r in rounds))
    return result


def main():
    source = HERE / "artifacts/analysis/report"
    raw = (source / "ANALYSIS.json").read_bytes()
    result = json.loads(raw)
    evidence = read(HERE / "RUN_EVIDENCE.json")
    jobs = read(HERE / "JOBS.json")
    latency = read(HERE / "LATENCY.json")
    qa = read(HERE / "EXECUTION_QA.json")
    assert qa["status"] == "PASS" and qa["analysis"]["status"] == "PASS"
    assert len(evidence) == len(jobs) == 24 and latency["status"] == "COMPLETE"
    expected = {j["id"] + "/" + a for j in jobs for a in j["readouts"]}
    assert set(latency["readouts"]) == expected and len(expected) == 36
    for row in latency["readouts"].values():
        assert row["parameters"] == 82426 and set(row["batches"]) == {"1", "128"}
        for batch, rounds in row["batches"].items():
            assert {r["round"] for r in rounds} == {0, 1, 2} and len(rounds) == 3
            assert all(r["measurements"] == (1000 if batch == "1" else 200) for r in rounds)
            assert all(0 < r["p50_ms"] <= r["p95_ms"] and r["throughput_samples_per_second"] > 0 for r in rounds)

    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    assert gzip.decompress(compressed) == raw
    compressed_path = HERE / "ANALYSIS.json.gz"
    if compressed_path.exists():
        raise FileExistsError("Keep original compressed analysis")
    compressed_path.write_bytes(compressed)
    for name in ("RESULTS.zh-CN.md", "DECISION.json"):
        text(name, (source / name).read_text(encoding="utf-8"))

    summary = {k: v for k, v in result.items() if k not in ("endpoints", "artifacts")}
    summary["endpoints"] = {}
    for endpoint in ENDPOINTS:
        block = result["endpoints"][endpoint]
        summary["endpoints"][endpoint] = {k: v for k, v in block.items() if k not in ("candidates", "same_weight_decay_comparisons", "tuned_comparisons")}
        for field in ("same_weight_decay_comparisons", "tuned_comparisons"):
            summary["endpoints"][endpoint][field] = {name: {k: v for k, v in pair.items() if k != "cells"} for name, pair in block[field].items()}
    summary["full_analysis"] = dict(path="ANALYSIS.json.gz", uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
        scope="Full 108 candidate/endpoint tables, including individual cells, confusion, support, class-pair, margin and same-input diagnostics. Summary retains all aggregate and group comparisons.")
    write("SUMMARY.json", summary)

    family = defaultdict(lambda: defaultdict(float))
    attempts = {}
    for job in jobs:
        complete = evidence[job["id"]]["COMPLETE"]
        attempt = read(HERE / "artifacts" / job["id"] / "ATTEMPT.json")
        assert complete["status"] == attempt["status"] == "COMPLETE"
        assert complete["updates"] == 5080 and complete["epochs"] == 40
        attempts[job["id"]] = attempt
        row = family[job["family"]]
        row["gradient_fits"] += 1
        row["wall_seconds_since_attempt"] += complete["wall_seconds_since_attempt"]
        row["source_input_load_seconds"] += complete["source_input_load_seconds"]
        for key, value in complete["costs"].items():
            row[key] += value
    all_latency = latency_summary(latency["readouts"])
    primary_latency = latency_summary({name: row for name, row in latency["readouts"].items()
        if next(j for j in jobs if j["id"] == name.split("/")[0])["weight_decay"] == result["weight_decay_selection"]["chosen"][name.split("/")[1]]})
    costs = dict(status="COMPLETE", family_totals=dict(family), training_budget=qa["training"]["totals"],
        actual_parameters_per_readout=82426, peak_cuda_allocated_gib=max(v["COMPLETE"]["peak_cuda_allocated_gib"] for v in evidence.values()),
        all_fit_wall_seconds=sum(v["COMPLETE"]["wall_seconds_since_attempt"] for v in evidence.values()),
        source_evaluation_seconds_after_attempt_marker=elapsed(read(HERE / "EVALUATION_MANIFEST.json")["time"], read(HERE / "SELECTION_EVALUATION_ATTEMPT.json")["time"]),
        latency_seconds_after_attempt_marker=elapsed(latency["time"], read(HERE / "LATENCY_ATTEMPT.json")["time"]),
        latency_all_36_readouts=all_latency, latency_primary_wd_readouts=primary_latency,
        latency_aggregation="Median of per-readout/per-round p50 and p95; not percentiles recomputed from pooled raw timings. All two-wd results and primary-wd subset both retained.",
        accounting="C0/C1 share 12 ERM+EMA gradient fits. training_with_ema_seconds includes EMA updates; EMA host/CUDA seconds are overlapping subset measurements, not additive costs. Calibration and EMA-validation are separately timed. No standalone C0 fit was added to estimate counterfactual overhead. Fit wall includes loading/serialization/hash work after attempt marker; excludes CLI startup and earlier validation. No end-to-end pipeline wall claim.")
    write("COSTS.json", costs)

    lines = ["# 阶段 C：训练和部署成本", "", "全部方法使用相同 HStem+MHA 骨干，每个部署读出均为 **82,426 个参数**。EMA 仅改变权重与训练阶段校准，未增加推断分支。设备与环境见 LATENCY.json。", "",
        "| 梯度轨迹族 | 拟合数 | 拟合内部 wall 总和 / 秒 | 梯度循环（含 EMA）/ 秒 | 普通验证 / 秒 | EMA BN 校准 / 秒 | EMA 验证 / 秒 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in family.items():
        lines.append(f"| {name} | {int(row['gradient_fits'])} | {row['wall_seconds_since_attempt']:.3f} | {row['training_with_ema_seconds']:.3f} | {row['raw_validation_seconds']:.3f} | {row['ema_calibration_seconds']:.3f} | {row['ema_validation_seconds']:.3f} |")
    lines += ["", f"24 次拟合内部 wall 合计 {costs['all_fit_wall_seconds']:.3f} 秒（{costs['all_fit_wall_seconds']/60:.3f} 分钟）；这不是整个实验流水线耗时。峰值 CUDA 已分配内存 {costs['peak_cuda_allocated_gib']:.6f} GiB，不等于驱动总占用或 reserved memory。",
        "", "EMA 更新的 host/CUDA 计时均包含在梯度循环中，彼此也不能相加。12 条共享轨迹的 EMA host 包围计时共 %.3f 秒，CUDA event 计时共 %.3f 秒；BN 校准 %.3f 秒，EMA 验证 %.3f 秒。没有额外训练独立 ERM，不能把这些局部计时换算成完整 C1/C0 训练成本比。" % (family['ERM_EMA']['ema_update_host_seconds'], family['ERM_EMA']['ema_update_cuda_seconds'], family['ERM_EMA']['ema_calibration_seconds'], family['ERM_EMA']['ema_validation_seconds']),
        "", "共 60,960 次 EMA 更新；480 次 fit_train-only BN 校准、7,780,560 个校准样本访问。C2 没有这些步骤。逐任务计时和状态见 RUN_EVIDENCE.json 的 COMPLETE；模型保存、哈希和其余控制开销包含在 fit wall 中。", "",
        "## 实测推断延迟", "", "输入已驻留 CUDA，完整信号前端和分类器均计时，排除文件读取与 host-device 复制。全部 36 个 selected 读出均测 batch=1/128，每轮每 batch 100 次预热，随后分别 1,000/200 次测量，三轮轮换 C0/C1/C2 顺序。总测量 129,600 次，另有 21,600 次预热；没有使用目标输入。", "",
        "下表覆盖两个 wd 的全部读出；数值是每读出每轮 p50/p95 的中位数，不是合并全部原始测量重新计算的分位数。相同模型结构下的小差异不作算法推断加速声明。", "",
        "| 方法 | batch | 读出数 | 轮数 | 轮 p50 中位数 / ms | 轮 p95 中位数 / ms | 轮 p50 范围 / ms | 轮吞吐中位数 / 样本每秒 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for batch in ("1", "128"):
            r = all_latency[arm][batch]
            lines.append(f"| {arm} | {batch} | {r['readouts']} | {r['rounds']} | {r['median_round_p50_ms']:.6f} | {r['median_round_p95_ms']:.6f} | [{r['min_round_p50_ms']:.6f}, {r['max_round_p50_ms']:.6f}] | {r['median_round_throughput_samples_per_second']:.2f} |")
    lines += ["", "主端点所选 wd 子集的同口径汇总保存在 COSTS.json；全部逐模型、逐轮记录保存在 LATENCY.json。", "",
        f"源域评分起始标记之后耗时 {costs['source_evaluation_seconds_after_attempt_marker']:.0f} 秒，延迟测试起始标记之后耗时 {costs['latency_seconds_after_attempt_marker']:.0f} 秒；均不含标记之前的完整性验证。ATTEMPT 文件保留启动时状态，COMPLETE 结果清单是对应步骤的终态记录。", ""]
    text("COSTS.zh-CN.md", "\n".join(lines))

    lines = ["# 阶段 C：完整网格与配对对照", "", "所有数值来自冻结 analyze.py 的输出。本表只序列化现有结果，不改变端点、选模或统计规则。两个 wd、三个端点均保留。selected 为主端点；辅助端点沿用其 wd 选择。", "",
        "| 端点 | 方法 | wd | Accuracy / % | RX 等权 Macro-F1 | CE | 最差 RX | 该 RX Accuracy / % | 最差受支持 TX×RX / 类索引 | 该组 Accuracy / % |", "|---|---|---:|---:|---:|---:|---|---:|---|---:|"]
    for endpoint in ENDPOINTS:
        for name, row in result['endpoints'][endpoint]['aggregates'].items():
            arm, wd = name.split('/wd=')
            rx = row['worst_rx']; group = row['worst_supported_tx_rx']; m = row['rx_equal']
            lines.append(f"| {endpoint} | {arm} | {wd} | {m['accuracy']*100:.4f} | {m['macro_f1']:.6f} | {m['ce']:.6f} | {rx} | {row['per_rx'][rx]['accuracy']*100:.4f} | {group['rx']} / {group['tx_class_index']} | {group['accuracy']*100:.4f} |")
    lines += ["", "最差组只在每流至少 50 包的组中取值，三流先平均；缺失组不补零。TX 类索引不是设备 ID。", "",
        "| 端点 | 比较类型 | 比较 | Accuracy 差 / pp | 配对 RX 95% 条件区间 / pp | CE 差 | 预测分歧 / % | 纠错 / % | 新增错误 / % | 最差配对 RX 差 / pp | 最大受支持组风险增加 / pp |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for endpoint in ENDPOINTS:
        block = result['endpoints'][endpoint]
        for kind, field in (("同 wd", "same_weight_decay_comparisons"), ("各自选 wd", "tuned_comparisons")):
            for name, pair in block[field].items():
                m = pair['rx_equal']; lo, hi = pair['paired_rx_bootstrap_95_intervals']['accuracy_delta']
                lines.append(f"| {endpoint} | {kind} | {name} | {m['accuracy_delta']*100:+.4f} | [{lo*100:+.4f}, {hi*100:+.4f}] | {m['ce_delta']:+.6f} | {m['disagreement']*100:.4f} | {m['corrected_fraction']*100:.4f} | {m['new_error_fraction']*100:.4f} | {pair['worst_paired_rx']['accuracy_delta']*100:+.4f} | {pair['worst_supported_tx_rx_risk_increase']['risk_increase']*100:+.4f} |")
    lines += ["", "各候选的每 RX、每视图、每流、全部 TX×RX 风险及支持见 SUMMARY.json；全部 108 个端点的逐候选混淆、类别对、margin、TX×RX×Day 支持和各配对 cell 见 ANALYSIS.json.gz。CI 的单位为已观察 12 RX，先平均固定三流，不代表新 RX 或随机种子总体。", ""]
    text("FULL_COMPARISONS.zh-CN.md", "\n".join(lines))

    write("SCIENTIFIC_ATTEMPTS.json", dict(authorized=24, attempts=attempts, completed=24, failed=0, retried=0, additional_scientific_fits=0,
        synthetic_engineering_separate=True, raw_payloads_local="artifacts/; ignored by Git and retained with seals"))
    write("EXECUTION_RESULT.json", dict(status="COMPLETE", scientific_training_complete=True,
        training=qa['training']['totals'], success=24, failed=0, all_pre_registered_endpoints_reported=True,
        source_selection_predictions=108, deployment_readouts_benchmarked=36, decision=result['decision'],
        formal_CRD_breakthrough_claim=False, AB_attribution_closed=True, ordinary_method_competition_closed=True,
        new_trainable_mechanism_candidates=0, M0="STOP_NO_VERIFIABLE_LOCAL_PAIRING", qualified_new_confirmation_rx=0,
        confirmation_access="CLOSED", historical_held6_target8_days3_4_evaluated=False))
    write("ARTIFACT_ARCHIVE.json", dict(status="COMPLETE", reporting_helper_sha256=sha(Path(__file__)),
        full_analysis=dict(file="ANALYSIS.json.gz", sha256=sha(compressed_path), compressed_bytes=len(compressed),
            uncompressed_bytes=len(raw), uncompressed_sha256=hashlib.sha256(raw).hexdigest(), exact_gzip_roundtrip=True),
        scientific_payloads=dict(local_directory="artifacts/", uploaded_to_git=False,
            training_payload_hashes="RUN_EVIDENCE.json: COMPLETE.files_sha256; STAGE_C_SEAL.json: complete/checkpoint digests",
            source_prediction_hashes="EVALUATION_MANIFEST.json", packet_diagnostic_hashes="ANALYSIS.json.gz: artifacts.packet_diagnostics"),
        scope="Only reports, evidence and compressed numerical metrics are published. IQ caches, checkpoint/optimizer payloads and per-packet NPZ arrays remain local with registered hashes. No additional payload access or scoring."))
    print(json.dumps(dict(status="COMPLETE", compressed_analysis_bytes=len(compressed), scientific_fits=24, readouts=36)))


if __name__ == "__main__":
    main()
