#!/usr/bin/env python3
"""Generate a Chinese close-out only from complete, audited Stage-07 evidence."""
import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
PIPELINES = {
    "source_frontend": "前端改造",
    "source_training": "源域选训练配方",
    "source_combination": "源域选组合",
}


def read_csv(name):
    with (HERE / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    acceptance = json.loads((HERE / "acceptance.json").read_text())
    status = json.loads((HERE / "stage_status.json").read_text())
    if acceptance["status"] != "PASS" or not status["wifi_complete"] or not status["osu_complete"]:
        raise ValueError("complete audited WiSig and OSU results are required")
    rows = read_csv("wifi_final_metrics.csv")
    contrasts = read_csv("wifi_comparison_intervals.csv")
    if len(rows) != 12 or len(contrasts) != 24:
        raise ValueError("missing final summary/comparison cells")
    osu = read_csv("osu_summary.csv")
    costs = json.loads((HERE / "costs.json").read_text())
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    training = [next(r for r in rows if r["dataset"] == dataset and
                     r["pipeline"] == "source_training" and r["task"] == "CRD")
                for dataset in ("ManyRx", "ManySig")]
    w1_deltas = [next(r for r in contrasts if r["dataset"] == dataset and
                     r["pipeline"] == "source_training" and r["baseline"] == "W1")
                 for dataset in ("ManyRx", "ManySig")]
    strong = [r for r in contrasts if r["pipeline"] == "source_training" and
              r["baseline"] in ("IB-RFF", "WaveMLP")]
    comparison = "两项均值仍低于 IB-RFF 和 WaveMLP。" if all(
        float(r["delta_pp"]) < 0 for r in strong) else \
        "与强基线的差值和接收机级不确定性见完整对照表。"
    osu_hstem = next(r for r in osu if r["method"] == "hstem_hstu")
    lines = [
        "# 阶段07中文结论\n\n",
        f"生成时间：{now}。基线 `ebe3225`，实验分支 `exp/s07-wifi-osu`。\n\n",
        "## 结论\n\n",
        "源域选训练配方的 ManyRx / ManySig CRD 为 "
        f"**{float(training[0]['rx_accuracy_pct']):.2f}% / "
        f"{float(training[1]['rx_accuracy_pct']):.2f}%**，比原 W1 分别 "
        f"**{float(w1_deltas[0]['delta_pp']):+.2f} / "
        f"{float(w1_deltas[1]['delta_pp']):+.2f} pp**。{comparison} ",
        f"OSU HStem-HSTU 跨 RX 包准确率为 **{float(osu_hstem['accuracy_pct']):.2f}%**，",
        "未通过独立泛化验证。**本轮没有达成 WiSig / OSU 双 SOTA。**\n\n",
        "**本轮注册实验完成并停止，不根据测试成绩继续追加搜索。** ",
        "下表全部来自四折 RX、三个训练种子的正式评测；内部留源 RX 的探索分数不替代 CRD。 ",
        "三条流水线均报告，不在测试集上挑出表现最好的流水线作为最终方法。\n\n",
        "| WiSig | 流水线 | CR RX准确率 % | CRD RX准确率 % | CRD Macro-F1 % | 种子SD pp | 最差RX % | 参数 |\n",
        "|---|---|---:|---:|---:|---:|---:|---|\n",
    ]
    for dataset in ("ManyRx", "ManySig"):
        for pipeline, label in PIPELINES.items():
            cells = {r["task"]: r for r in rows if r["dataset"] == dataset and r["pipeline"] == pipeline}
            crd = cells["CRD"]
            lines.append(f"| {dataset} | {label} | {float(cells['CR']['rx_accuracy_pct']):.2f} | "
                         f"{float(crd['rx_accuracy_pct']):.2f} | {float(crd['rx_macro_f1_pct']):.2f} | "
                         f"{float(crd['seed_sd_pp']):.2f} | {float(crd['worst_rx_seed_mean_pct']):.2f} | "
                         f"{crd['parameters']} |\n")
    lines.extend([
        "\n“源域选训练配方”和“源域选组合”的最终配置相同，不能作为两个独立创新或两组独立证据。 ",
        "ManyRx 全部是频域单路；ManySig 前三折为频域单路，第3折为时频分离。 ",
        "Gate 没有通过该折的源域排名，未进入最终配置。\n\n",
        "## 同协议强基线\n\n",
        "沿用冻结的 S06 公平多种子基线，数据归一化、RX/日期和包预算一致。 ",
        "S06 与 S07 的训练种子不同，因此不是配对种子检验。\n\n",
        "| WiSig | 流水线 | 对照 | CRD差值 pp | RX聚类bootstrap 95%区间 pp | 改善RX |\n",
        "|---|---|---|---:|---|---:|\n",
    ])
    names = {"W1": "原W1", "IB-RFF": "IB-RFF", "WaveMLP": "WaveMLP",
             "FAN_source_selected": "FAN"}
    for row in contrasts:
        lines.append(f"| {row['dataset']} | {PIPELINES[row['pipeline']]} | {names[row['baseline']]} | "
                     f"{float(row['delta_pp']):+.2f} | [{float(row['ci95_low_pp']):+.2f}, "
                     f"{float(row['ci95_high_pp']):+.2f}] | "
                     f"{row['improved_receivers']}/{row['receiver_count']} |\n")
    lines.extend([
        "\n区间以接收机为聚类单位，先对种子取均值，条件于本次已训练模型； ",
        "不包含完整训练种子不确定性，也没有多重比较校正。最差 RX 和 fold 波动见完整报告。 ",
        "大量包和重复种子不能当作额外独立接收机。\n\n",
        "## 公开权重参考\n\n",
        "以下是先前核验的官方发布权重，单独列出，不混入公平多种子训练结果。 ",
        "它们按官方包等权口径汇总，不能直接与 RX 等权相减。\n\n",
        "| 方法/流水线 | ManyRx CRD 包等权 % | ManySig CRD 包等权 % |\n",
        "|---|---:|---:|\n",
        "| IB-RFF发布权重 | 89.19 | 93.57 |\n",
        "| WaveMLP发布权重 | 88.68 | 93.13 |\n",
    ])
    for pipeline, label in PIPELINES.items():
        values = [next(r for r in rows if r["dataset"] == dataset and r["pipeline"] == pipeline
                       and r["task"] == "CRD") for dataset in ("ManyRx", "ManySig")]
        lines.append(f"| {label} | {float(values[0]['packet_accuracy_pct']):.2f} | "
                     f"{float(values[1]['packet_accuracy_pct']):.2f} |\n")
    lines.extend([
        "\n超过两个参考值也只支持已测试协议内的比较。最新同期一手文献无法完整访问， ",
        "本轮不认证全球 SOTA，原官方测试 RX 已被查看过，也不称作全新盲测。\n\n",
        "## OSU独立验证\n\n",
        "本地50份原始记录，25个TX、两台RX、双方向、三个种子，共24次正式重新训练。 ",
        "固定真实前导码检测和时间间隔切分，全部原始哈希、裁剪、标签与预测通过审计。 ",
        "每个TX/RX只有一次连续采集；名义125 kHz/SF7来自源pilot，不是元数据认证。 ",
        "检测开发时检查过双RX样例，原始目标准备不是全盲。\n\n",
        "| 模型 | 跨RX包准确率 % | TX等权 % | Macro-F1 % | 参数 |\n",
        "|---|---:|---:|---:|---:|\n",
    ])
    for row in osu:
        lines.append(f"| {row['method']} | {float(row['accuracy_pct']):.2f} | "
                     f"{float(row['tx_equal_pct']):.2f} | {float(row['macro_f1_pct']):.2f} | "
                     f"{row['params']} |\n")
    lines.extend([
        "\n**HStem-HSTU没有通过本管线下的独立跨RX验证。** 同RX未来时间块的事后控制为91.24%， ",
        "跨RX仅1.42%。不补偿CFO、同步不确定性和单次采集是适用限制， ",
        "不能推断“只要补偿CFO就会恢复准确率”。跨日期数据不在本地，官方请求403， ",
        "没有跨日结果，也没有OSU SOTA证据。\n\n",
        "## 实测成本\n\n",
        f"设备：{costs['gpu']}，PyTorch {costs['torch']}，CPU 4线程。 ",
        "所有训练停止后、两次空闲GPU快照通过才运行；随机权重、合成IQ、常驻设备输入到logits。 ",
        "不含原始包检测、传输和同步。FLOPs不完整包含FFT与标量运算，计时包含。\n\n",
        "| 模型 | 参数 | MFLOPs | CPU单包 ms | GPU单包p50 ms | GPU单包p95 ms | batch256包/s |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ])
    for row in costs["rows"]:
        if not ("frequency" in row["model"] or "split" in row["model"] or
                "FAN_IB-RFF" in row["model"] or "WaveMLP" in row["model"] or row["model"].startswith("OSU:")):
            continue
        lines.append(f"| {row['model']} | {row['params']} | {row['mflops']:.2f} | "
                     f"{row['cpu4t_b1_ms']:.3f} | {row['gpu_b1_ms']:.3f} | "
                     f"{row['gpu_b1_p95_ms']:.3f} | {row['gpu_b256_packets_per_second']:.0f} |\n")
    lines.extend([
        "\n参数少不等于推理快，FAN与IB-RFF推理图相同；IB正则属于训练阶段。\n\n",
        "## 研究决策\n\n",
        "保留Shen上已经验证的结果，不覆盖阶段00–06。频域处理、相位差、时频分离与增强 ",
        "均不是因接入HSTU就成为原创。OSU负结果使“跨LoRa/WiFi通用SOTA”的主张不成立。 ",
        "WiSig是否具备局部竞争力应据上表完整比较判断，不自动选测试赢家。 ",
        "下一步需要基于独立协议、机制和新颖性重新确定论文主张，不能继续目标驱动刷分。\n\n",
        "## 证据索引\n\n",
        "- [完整英文结果](RESULTS.md)与[CR/CRD及逐RX对比](S07_WIFI_FINAL_RESULTS.md)。\n",
        "- [源域决策追踪](S07_SOURCE_DECISIONS.md)、[前端全部结果](S07_WIFI_FEATURE_RESULTS.md)",
        "和[训练因素全部结果](S07_WIFI_TRAINING_RESULTS.md)。\n",
        "- [OSU数据卡](S07_OSU_DATA_CARD.md)、[正式结果](S07_OSU_RESULTS.md)",
        "与[事后诊断](S07_OSU_POSTHOC.md)。\n",
        "- [实测成本](S07_COSTS.md)、[新颖性与SOTA证据限制](S07_NOVELTY_AND_SOTA_REVIEW.md)。\n",
        "- [执行记录](S07_EXECUTION.md)：环境、正式运行数量与耗时口径。\n",
        "- `acceptance.json`、`wifi_final_report_evidence.json`、`decision_final.json`、",
        "`runs/`：验收、冻结谱系、完整配置/训练曲线/预测。权重不纳入Git，未自动推送或合并。\n",
    ])
    (HERE / "CONCLUSION_ZH.md").write_text("".join(lines))
    print("wrote complete audited Chinese Stage-07 conclusion")


if __name__ == "__main__":
    main()
