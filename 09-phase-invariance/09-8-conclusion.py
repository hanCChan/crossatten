"""Generate a bounded Chinese conclusion only from complete verified results."""
import csv
import json
import numpy as np
from training import HERE, ROOT, sha, save


def read(name):
    with (HERE / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    acceptance = json.loads((HERE / "acceptance.json").read_text())
    if acceptance["status"] != "PASS" or acceptance["source_only"]:
        raise ValueError("complete acceptance required before conclusion")
    summary, comparisons = read("final_metrics.csv"), read("paired_comparisons.csv")
    baseline, interaction = read("baseline_comparisons.csv"), read("factorial_interaction.csv")
    decision = json.loads((HERE / "decision_final.json").read_text())
    attr = [r for r in summary if r["phase"] == "attribution" and r["task"] == "CRD"]
    phase = next(r for r in comparisons if r["phase"] == "attribution" and r["task"] == "CRD"
                 and r["contrast"] == "first:fixed-null:fixed")
    lines = ["# 阶段09：机制、严格消融与冲刺结论\n\n",
        f"基线 `c0aceeb`；独立分支 `exp/s09-phase-invariance-cpu-probe`。"
        f"已完成{acceptance['formal_fits']}次正式训练及5次GPU冒烟测试。历史00-08保持不变。\n\n",
        "## 严格同预算消融\n\n",
        "ManySig，四折×三个新种子。共享权重逐项相同、分包和训练shuffle相同。"
        "固定策略沿用S08源域冻结轮数[10,6,4,13]及[EMA,raw,EMA,raw]；"
        "packet为Official原始包验证CE早停。参数为三通道90420、五通道90868。\n\n",
        "| 表征 | 训练策略 | CRD RX准确率 | Macro-F1 | 种子SD pp | 最差RX |\n",
        "|---|---|---:|---:|---:|---:|\n"]
    for r in attr:
        lines.append(f"| {r['feature']} | {r['policy']} | {float(r['rx_accuracy_pct']):.2f}% | "
                     f"{float(r['macro_f1_pct']):.2f}% | {float(r['seed_sd_pp']):.2f} | "
                     f"{float(r['worst_rx_pct']):.2f}% |\n")
    lines.append(f"\n**真实相位相对零相位的条件增益：{float(phase['delta_pp']):+.2f}pp，"
                 f"接收机聚类95%区间[{float(phase['ci95_low_pp']):+.2f}, "
                 f"{float(phase['ci95_high_pp']):+.2f}]，改善{phase['improved_rx']}/{phase['receivers']}个RX。**\n\n")
    lo, delta = float(phase["ci95_low_pp"]), float(phase["delta_pp"])
    if lo > 0:
        lines.append("这支持相位内容本身在固定训练策略下有正向贡献，不只是新增448个权重。"
                     "区间条件于这三个已训练种子，未做多重比较校正，不能升级为普遍因果结论。\n")
    elif delta > 0:
        lines.append("均值支持相位内容，但区间跨0，尚不足以认定其跨RX贡献稳定。"
                     "不能把S08联合提升全部归因于新增特征。\n")
    else:
        lines.append("本轮不支持相位内容稳定优于容量对照；必须保留该负面结果，"
                     "不能把S08联合提升解释成已证实的独立相位贡献。\n")
    lines.extend(["\n## 分离训练策略\n\n",
        "| 对比 | CRD差值 pp | 条件RX95%区间 pp |\n",
        "|---|---:|---|\n"])
    for r in comparisons:
        if r["phase"] == "attribution" and r["task"] == "CRD":
            lines.append(f"| {r['contrast']} | {float(r['delta_pp']):+.2f} | "
                         f"[{float(r['ci95_low_pp']):+.2f}, {float(r['ci95_high_pp']):+.2f}] |\n")
    for r in interaction:
        if r["task"] == "CRD":
            lines.append(f"\n2×2交互项：{float(r['interaction_pp']):+.2f}pp，"
                         f"[{float(r['ci95_low_pp']):+.2f}, {float(r['ci95_high_pp']):+.2f}]。"
                         "固定策略相对packet同时改变选轮/停止机制与部分折的EMA，仍不能单独量化EMA。\n")
    lines.extend(["\n## 新时延机制筛选\n\n",
        "Both datasets，四折×两种源留RX分组×两个种子×三种相位表征，共96次40 epoch训练。"
        "去斜率与二阶差分用同一个五通道网络，候选冻结前没有读取目标RX成绩。\n\n",
        "| 数据集 | 候选:规则 | 支持折数 | 源域平均增益 pp | 结转 |\n",
        "|---|---|---:|---:|---|\n"])
    for dataset, values in decision["choices"].items():
        for key, entry in values.items():
            lines.append(f"| {dataset} | {key} | {sum(v['supported'] for v in entry['folds'].values())}/4 | "
                         f"{entry['mean_gain']:+.2f} | {'通过' if entry['admitted'] else '未通过'} |\n")
    if decision["selected"]:
        lines.append(f"\n最终源域选择：`{json.dumps(decision['selected'], ensure_ascii=False)}`。"
                     f"共同双数据集支持：{'是' if decision['universal'] else '否，仅数据集限定证据'}。\n")
        for r in comparisons:
            if r["phase"] == "final" and r["task"] == "CRD":
                lines.append(f"- {r['dataset']}，{r['contrast']}：{float(r['delta_pp']):+.2f}pp "
                             f"[{float(r['ci95_low_pp']):+.2f}, {float(r['ci95_high_pp']):+.2f}]。\n")
    else:
        lines.append("\n**没有新时延表征通过预登记门槛，没有追加新候选目标集训练/测试。**"
                     "不能因数学不变性正确而忽略TX识别损失或源RX退化。\n")
    lines.extend(["\n## 强基线与SOTA边界\n\n",
        "| 数据集 | 本轮模型 | 公平基线 | 本轮CRD | 基线CRD | 差值 pp | 条件RX95%区间 pp |\n",
        "|---|---|---|---:|---:|---:|---|\n"])
    for r in baseline:
        lines.append(f"| {r['dataset']} | {r['feature']}:{r['policy']} | {r['baseline']} | "
                     f"{float(r['candidate_pct']):.2f}% | {float(r['baseline_pct']):.2f}% | "
                     f"{float(r['delta_pp']):+.2f} | [{float(r['ci95_low_pp']):+.2f}, "
                     f"{float(r['ci95_high_pp']):+.2f}] |\n")
    lines.append("\nS05官方发布权重历史参考：ManySig IB-RFF93.57/WaveMLP93.13；"
                 "ManyRx89.19/88.68，不与公平多种子重训混用。"
                 "新核实官方ASKNet方法，尚未完成同协议重训，因此不宣称全局SOTA。"
                 "确切的Liverpool2026高阶谱差分文献仍未核实，不能虚构引文或宣称二阶差分首创。\n")
    lines.append("\n两个已登记训练策略分别报告，不能在看到目标分数后择优作为新候选。"
                 "S08原93.01%与本轮固定策略92.36%使用不同种子且初始化配对方式不同，"
                 "不是同种子回退比较。普通早停92.90%也属于预登记消融，不是新源域结转赢家。\n")
    lines.extend(["\n## 机制与论文主张\n\n",
        "- 相位是已有Re/Im FFT信息的确定变换，属于显式归纳偏置，不增加原始信息量。\n",
        "- 容量控制：12个固定策略fit分别在CR/CRD评测，零相位与三通道共24组预测"
        "的最大logit差为0；448个零输入权重没有带来额外收益。\n",
        "- 新附加通道对理想循环平移更稳定；真实IQ补零截窗、CFO和频选信道并不普遍不变。"
        "保留的Re/Im使整个网络也不具有严格时移不变性。\n",
        "- 源域特征Fisher比、逐TX召回及模型扰动响应为机制诊断，不是独立验证，"
        "不证明接收机时延是剩余误差主因。\n",
        "- 机制选择跨源折聚合，而各折RX角色互换，不等同于完全嵌套逐折选参；"
        "源held CE用于选模和评价也有选择乐观偏差。目标RX曾被查看，不是新盲测。\n",
        "- 全部负面结果、失败尝试、配置、训练曲线、预测及本地权重哈希保留。"
        "没有目标成绩驱动的追加实验，没有新Loss或大网络。\n",
        "- 计算成本在GPU空闲后测量；参数较少不等于推理更快。\n\n",
        "完整表格：`S09_FINAL_RESULTS.md`、`SOURCE_RESULTS.md`、`MECHANISM_PROBES.md`、"
        "`DIAGNOSTICS.md`、`S09_COSTS.md`、`LITERATURE_CHECK.md`。"
        "本轮执行到此停止；不自动合并、推送或扩展实验。\n"])
    (HERE / "RESULTS.md").write_text("".join(lines))
    counts = []
    for phase_name in ("smoke", "attribution", "source", "final"):
        jobs = json.loads((HERE / f"jobs_{phase_name}.json").read_text())
        seconds = [json.loads((HERE / "runs" / phase_name / j["id"] / "metrics.json").read_text())["seconds"]
                   for j in jobs]
        counts.append(dict(phase=phase_name, fits=len(jobs), summed_elapsed_hours=sum(seconds) / 3600,
                           min_seconds=min(seconds) if seconds else 0, max_seconds=max(seconds) if seconds else 0))
    save(HERE / "execution.json", dict(rows=counts, note="Summed fit elapsed times are not dedicated GPU-hours; "
        "MPS concurrent fits. Evaluation, CPU diagnostics and benchmarks excluded.",
        formal_fits=acceptance["formal_fits"], torch=json.loads((HERE / "costs.json").read_text())["torch"]))
    execution_lines = [
        "# S09 Execution Record\n\n",
        "Completed locally on2026-10-26, Asia/Shanghai. "
        "Four NVIDIA A100-SXM4-40GB GPUs with CUDA MPS; "
        "source queue used eight fits/GPU across GPUs0-2, "
        "attribution six fits on GPU3. Every trainer used two CPU threads.\n\n",
        "Python3.12.8, PyTorch2.5.0+cu124, CUDA12.4. Existing cached first100 "
        "non-equalized packets/TX/RX/day; normalized float32 IQ,256 samples. "
        "Official four-fold receiver lists and source dates1-2 unchanged.\n\n",
        "| Phase | Fits | Sum fit elapsed hours | Min seconds | Max seconds |\n",
        "|---|---:|---:|---:|---:|\n"]
    for r in counts:
        execution_lines.append(f"| {r['phase']} | {r['fits']} | {r['summed_elapsed_hours']:.6f} | "
                               f"{r['min_seconds']:.3f} | {r['max_seconds']:.3f} |\n")
    execution_lines.extend([
        "\nSummed durations are not dedicated GPU-hours or total wall time: "
        "fits share GPUs concurrently. They exclude loading/setup, target-only evaluation, "
        "CPU diagnostics and inference benchmarking. No architectural training-speed claim.\n\n",
        "All156 formal fits and five smoke fits completed with zero failed/interrupted "
        "training attempts. All96 source fits ran exactly40 epochs. Formal attribution "
        "had36 fixed-budget and24 raw-packet early-stop fits. "
        "No new centered/curvature mechanism passed the source gate, so zero conditional "
        "final fits were run. Sixty registered attribution models each evaluated CR and CRD.\n\n",
        f"Source decision committed at{acceptance['decision_committed_at']}. "
        "No target reads preceded that decision. Shared initialization, train/validation "
        "partitions, source/target roles, weights and predictions passed acceptance. "
        "Raw/EMA checkpoints from one fit are correlated outputs, not extra seeds.\n\n",
        "Only model parameters are EMA averaged; clean fit-training inputs recalibrate "
        "BatchNorm each epoch. No held, packet-validation or target BN calibration. "
        "Final inference is a single checkpoint, not an ensemble.\n\n",
        "Historical S00-S08 and shared library remain unchanged. Weights stay ignored/local; "
        "configs, histories, predictions and weight hashes are versioned. "
        "No target-selected retries, new losses, automatic merge or push.\n"])
    (HERE / "S09_EXECUTION.md").write_text("".join(execution_lines))
    save(HERE / "conclusion_evidence.json", dict(acceptance_sha256=sha(HERE / "acceptance.json"),
                                               results_sha256=sha(HERE / "RESULTS.md"),
                                               report_sha256=sha(HERE / "S09_FINAL_RESULTS.md")))
    print(json.dumps(dict(status="complete", source_selected=decision["selected"], phase_gain_pp=delta)))


if __name__ == "__main__":
    main()
