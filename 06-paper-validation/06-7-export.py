#!/usr/bin/env python3
"""Export audited results as paper-ready tables and an accuracy/cost overview."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent


def export_tables(report, out):
    metrics, receivers, ablations, costs, initializations = [], [], [], [], []
    for key, row in sorted(report["groups"].items()):
        dataset, model, task = key.split(":")
        scores, training = row["metrics"], row["training_cost"]
        metrics.append(dict(
            dataset=dataset, model=model, task=task, fits=row["n"],
            rx_accuracy_pct=scores["acc_mean"]["mean"],
            fit_sd_pp=scores["acc_mean"]["sd"],
            packet_accuracy_pct=scores["packet_acc"]["mean"],
            rx_macro_f1_pct=scores["f1_mean"]["mean"],
            packet_macro_f1_pct=scores["packet_macro_f1"]["mean"],
            seed_sd_pp=row["seed_sd"], fold_sd_pp=row["fold_sd"],
            worst_fold_rx_pct=row["worst_fold_rx_mean"],
            parameters=";".join(map(str, row["params"])),
            training_seconds=training["total_seconds"]["mean"],
            initialization_seconds=training["initialization_seconds"]["mean"]))
        for receiver, accuracy in sorted(row["per_fold_rx_mean"].items()):
            fold, rx = receiver.split(":")
            receivers.append(dict(dataset=dataset, model=model, task=task,
                                  fold=int(fold), rx=rx, seed_mean_accuracy_pct=accuracy,
                                  accuracy_seed_sd_pp=row.get("per_fold_rx_seed_sd", {}).get(receiver, ""),
                                  seed_mean_macro_f1_pct=row.get("per_fold_rx_f1_mean", {}).get(receiver, ""),
                                  macro_f1_seed_sd_pp=row.get("per_fold_rx_f1_seed_sd", {}).get(receiver, "")))
    for key, row in sorted(report["comparisons"].items()):
        dataset, pair, task = key.split(":")
        ablations.append(dict(dataset=dataset, pair=pair, task=task, fits=row["n"],
                              status=row["status"], delta_pp=row.get("delta_pp", {}).get("mean", ""),
                              fit_delta_sd_pp=row.get("delta_pp", {}).get("sd", ""),
                              positive_pairs=row.get("positive", ""),
                              negative_pairs=row.get("negative", ""),
                              worst_fold_rx_delta_pp=row.get("worst_fold_rx_delta_pp", "")))
    for arm, row in sorted(report["costs_by_arm"].items()):
        costs.append(dict(registered_arm=arm, measured_representative=row["arm"],
                          **{k: v for k, v in row.items() if k != "arm"}))
    for key, row in sorted(report.get("initialization_comparisons", {}).items()):
        dataset, model, task = key.split(":")
        for pair in row["pairs"]:
            initializations.append(dict(dataset=dataset, model=model, task=task,
                                        complete=row["complete"], **pair))
    for name, rows in (("main_metrics", metrics), ("receiver_metrics", receivers),
                       ("registered_ablations", ablations), ("measured_costs", costs),
                       ("ib_initialization_pairs", initializations)):
        if rows:
            with (out / f"{name}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    return metrics


def accuracy_cost_plot(report, out):
    image = Image.new("RGB", (1680, 840), "white")
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 18)
    small = ImageFont.truetype(font_path, 15)
    title = ImageFont.truetype(font_path, 25)
    colors = {
        "hstem": "#197f65", "base": "#4678b6", "W0": "#4678b6", "W1": "#197f65",
        "W2": "#b67919", "cnnonly": "#a06297", "C0": "#a06297",
        "cnnparam": "#a77758", "CP": "#a77758", "cnnflops": "#777777", "CF": "#777777",
        "FAN": "#bd4951", "IB-RFF": "#bd4951", "WaveMLP": "#4c458d",
    }
    panels = [
        ("Shen", "dev_val", ["S_base", "S_hstem", "S_cnnonly", "S_cnnparam",
                             "S_cnnflops", "IB-RFF_a0.004_ib_unified",
                             "WaveMLP_scaled_full_unified"]),
        ("ManyRx", "CRD", ["W0", "W1", "W2", "C0", "CP", "CF", "IB-RFF", "WaveMLP"]),
        ("ManySig", "CRD", ["W0", "W1", "W2", "C0", "CP", "CF", "IB-RFF", "WaveMLP"]),
    ]
    draw.text((32, 22), "Registered paper validation: accuracy versus counted compute", font=title, fill="#222222")
    status = "COMPLETE" if report["checks"]["complete"] else "IN PROGRESS: incomplete rows are provisional"
    draw.text((32, 60), f"{status} | {report['generated_at']}", font=font, fill="#555555")
    arms = {arm["name"]: arm for arm in report["registration"]["arms"]}
    for panel, (dataset, task, models) in enumerate(panels):
        left, top, width, height = 80 + panel * 550, 145, 435, 385
        draw.text((left, 110), f"{dataset} / {task} / RX-equal accuracy", font=font, fill="#222222")
        ymin, ymax, xmin, xmax = (50, 100, 0, math.log10(250))
        for y in range(ymin, ymax + 1, 10):
            py = top + height * (ymax - y) / (ymax - ymin)
            draw.line((left, py, left + width, py), fill="#e7e7e7")
            draw.text((left - 35, py - 8), str(y), font=small, fill="#666666")
        for x in (1, 10, 100):
            px = left + width * (math.log10(x) - xmin) / (xmax - xmin)
            draw.line((px, top, px, top + height), fill="#eeeeee")
            draw.text((px - 9, top + height + 9), str(x), font=small, fill="#666666")
        draw.line((left, top, left, top + height, left + width, top + height), fill="#777777", width=2)
        draw.text((left + 100, top + height + 36), "MFLOPs / packet (log scale)", font=small, fill="#444444")
        plotted = 0
        for index, model in enumerate(models, 1):
            row = report["groups"].get(f"{dataset}:{model}:{task}")
            if row is None:
                continue
            if dataset == "Shen":
                matches = [model]
                label = model.removeprefix("S_") if model.startswith("S_") else arms[model]["method"]
            else:
                matches = [r["arm"] for r in report["results"] if r["stage"] == "final"
                           and r["dataset"] == dataset and arms[r["arm"]]["method"] == model]
                if not matches:
                    continue
                label = model
            model_costs = [report["costs_by_arm"][name]["mflops_per_packet"] for name in matches]
            cost = sum(model_costs) / len(model_costs)
            acc = row["metrics"]["acc_mean"]["mean"]
            px = left + width * (math.log10(cost) - xmin) / (xmax - xmin)
            py = top + height * (ymax - acc) / (ymax - ymin)
            color = colors.get(label, "#555555")
            low = left + width * (math.log10(min(model_costs)) - xmin) / (xmax - xmin)
            high = left + width * (math.log10(max(model_costs)) - xmin) / (xmax - xmin)
            draw.line((low, py, high, py), fill=color, width=3)
            draw.ellipse((px - 10, py - 10, px + 10, py + 10), fill=color)
            draw.text((px - 5, py - 9), str(index), font=small, fill="white")
            legend_y = 620 + ((index - 1) // 3) * 28
            legend_x = left + ((index - 1) % 3) * 145
            draw.text((legend_x, legend_y), f"{index} {label}", font=small, fill=color)
            plotted += 1
        if not plotted:
            draw.text((left + 70, top + height // 2), "Awaiting completed confirm fits", font=font, fill="#777777")
        draw.text((left, 592), "Numbers identify models; no target-based selection.", font=small, fill="#666666")
    draw.text((32, 769), "Compute excludes some FFT and elementwise operations; it is not a latency proxy.",
              font=font, fill="#555555")
    draw.text((32, 799), "Shen confirm RX were repeatedly inspected. WiSig CRD uses held-out RX on capture dates 3-4.",
              font=font, fill="#555555")
    image.save(out / "accuracy_cost.png")


def conclusion(report):
    checks, groups = report["checks"], report["groups"]
    complete = checks["complete"]
    route = report.get("paper_route", {}).get("route", "pending")
    routes = {
        "pending": "注册矩阵尚未完成，路线结论待定。",
        "cross_LoRa_WiFi_validation_supported": "注册证据支持继续推进跨 LoRa/WiFi 方法验证，不等于已经可投稿。",
        "narrow_to_LoRa_specific_evidence": "收缩为 LoRa 特定证据路线；本轮不支持跨 LoRa/WiFi 通用方法主张。",
        "engineering_or_limited_method_evidence": "目前主要是工程或有限方法证据，不支持直接宣称独立通用方法贡献。",
    }
    lines = [
        "# 论文竞争力验证汇报", "",
        f"证据时间：{report['generated_at']}。预登记：`bc30820`；"
        "WiSig 源域冻结：`4babf3a`；Shen 源域冻结：`88be16b`。",
        f"状态：**{'全部完成' if complete else '进行中，部分行不可作为完整结论'}**；"
        f"已审计 {checks['validated']} 个正式 fit。", "",
        "## 结论", "",
        f"**{routes[route]}**",
        "这是按注册实验得出的证据路线，不是新颖性审查、全领域 SOTA 或投稿就绪判定。",
        "停止新增 Loss、复杂特征和架构搜索；本轮完成后不自动追加实验。", "",
        "## WiSig 主结果", "",
        "下表为四折 RX × 三 seed 的 RX 等权均值及 fit SD，CRD 为主任务。"
        "SD 包含 fold 和 seed 差异；seed/fold/包不是新增独立接收机。",
        "W0/W1/W2 分别报告，没有在测试域选出胜者。公平重训不是发布权重结果，也不是加载 LoRa 权重迁移。", "",
        "| 数据集 | 模型 | fit | CR (%) | CRD (%) | CRD 包准确率 | CRD RX Macro-F1 | 参数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    def score(row):
        metric = row["metrics"]["acc_mean"]
        return f"{metric['mean']:.2f} ± {metric['sd']:.2f}"

    for dataset in ("ManyRx", "ManySig"):
        for model in ("W0", "W1", "W2", "FAN_source_selected", "IB-RFF", "WaveMLP"):
            row = groups.get(f"{dataset}:{model}:CRD")
            if row is None:
                continue
            cr = groups[f"{dataset}:{model}:CR"]
            lines.append(f"| {dataset} | {model} | {row['n']} | {score(cr)} | {score(row)} | "
                         f"{row['metrics']['packet_acc']['mean']:.2f} | "
                         f"{row['metrics']['f1_mean']['mean']:.2f} | "
                         f"{'/'.join(map(str, row['params']))} |")
    lines += ["", "W0 为普通 Stem；W1 为分层 Stem；W2 为 W1 加冻结截止点。"
              "全部固定 32×64 Token 和两层 HSTU。各折 RF 臂共用源域选出的训练配方。",
              "竞争方法按各自内部源 RX 选配置；IB 包含匹配 FAN 初始化与 HSIC 微调两阶段。", "",
              "IB 与其实际 FAN 初始化的收益见 `ib_initialization_pairs.csv`。"
              "配对是描述性补充，不改变冻结决策或增加新的选择标准。", "",
              "## Shen 同协议", "",
              "P-H5x160：10 TX、每折 5 台源 RX、每 TX/RX 160 训练包和 40 源验证包。"
              "每台评估 RX 为 2,000 包。确认训练为两 fold × seed201/202/203。",
              "checkpoint 和方法配置仅按源验证 CE；dev_val 只报告。"
              "四台 confirm RX 已反复查看，不是新盲测；六台 blind RX 未读取。",
              "当前训练器验证 batch 为 128，S03 历史结果保持冻结，不要求训练比特一致。", "",
              "| 模型/冻结配置 | fit | dev_val (%) | 已使用 confirm (%) | dev_val RX Macro-F1 | 参数 |",
              "|---|---:|---:|---:|---:|---:|"]
    shen_models = sorted({key.split(":")[1] for key in groups if key.startswith("Shen:")})
    for model in shen_models:
        row, confirm = groups[f"Shen:{model}:dev_val"], groups[f"Shen:{model}:confirm"]
        lines.append(f"| {model} | {row['n']} | {score(row)} | {score(confirm)} | "
                     f"{row['metrics']['f1_mean']['mean']:.2f} | {row['params'][0]} |")
    lines += ["", "FAN/IB-RFF 与 WaveMLP 分别保留官方及统一配方，所列 full/ib、"
              "α 和输入适配由源 CE 冻结。没有根据目标成绩重选。"
              "额外 FAN 输入臂用于 IB 初始化，不可冒充另一独立方法。",
              "WaveMLP 的 8192 点适配不等于复现其原论文 LoRa 数字。"
              "完整 CIS、严格 IB CIS、rank-16 CIS 与 S00–S03 结果见仓库总报告，未覆盖。", "",
              "## 机制消融", "",
              "| 数据集 | 任务 | 配对 | fit | Δ pp | 正配对 | 最差 RX Δ pp | 注册判定 |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for key, row in sorted(report["comparisons"].items()):
        dataset, pair, task = key.split(":")
        if dataset != "Shen" and task != "CRD":
            continue
        delta = row.get("delta_pp", {}).get("mean")
        worst = row.get("worst_fold_rx_delta_pp")
        lines.append(f"| {dataset} | {task} | {pair} | {row['n']} | "
                     f"{f'{delta:+.2f}' if delta is not None else '待完成'} | "
                     f"{row.get('positive', '-')}/{row['n']} | "
                     f"{f'{worst:+.2f}' if worst is not None else '待完成'} | {row['status']} |")
    lines += ["", "CNN-only 去 HSTU、保留分层 Stem 和分类头；同参数/同 FLOPs CNN "
              "是两个分别按结构成本匹配的普通 Stem 控制，误差均小于 5%。"
              "它们不是准确率搜索得到的最佳 CNN，也不能证明所有普通 CNN 均不可替代。"
              "均值提升但未满足正配对或最差 RX 门槛仍为 unclear，不解释成确定无效。", "",
              "## 计算成本", "",
              "训练前在空闲 A100 上测量，FP32、设备上 IQ 到 logits。"
              "包含前处理，不含主机传输；FFT 和逐元素计算未完整计入 FLOPs。", "",
              "| 结构 | 参数 | MFLOPs/包 | GPU 单包 p50/p95 ms | batch512 包/s | CPU4t p50 ms |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in ("S_base", "S_hstem", "S_cnnonly", "S_cnnparam", "S_cnnflops",
                 "W0_ManyRx_official", "W1_ManyRx_official", "W2_ManyRx_official"):
        cost = report["costs_by_arm"].get(name)
        if cost is None or "params" not in cost:
            continue
        lines.append(f"| {name} | {cost['params']} | {cost['mflops_per_packet']} | "
                     f"{cost['gpu_b1_ms_p50']}/{cost['gpu_b1_ms_p95']} | "
                     f"{cost['gpu_b512_pkts_per_s']} | {cost['cpu4t_b1_ms_p50']} |")
    lines += ["", "不能因 hstem 参数少便宣称推理更快。完整竞争方法成本、逐 RX、"
              "seed/fold 波动、F1 和两阶段训练时间保存在 CSV 与 `../report.json`。", "",
              "## 负面证据", "",
              "- WiSig W1-W0 的 CRD 稳定改善需满足原注册门槛，不能只看均值。",
              "- W2 截止点来自早期 fold0 源域 PSD，相关 RX 在其他折可能为目标域。"
              "因此 W2 不是每折严格源域独立的预处理证据；W0/W1 不使用该截止点。",
              "- 所有低准确率、早停、负配对以及失败/中断记录保留，不改配置重试以挑好结果。",
              "- FAN CUDA median 有上游非确定性警告，固定 seed 不保证比特一致。",
              "- 未完成 Ma 2025 等其他方法；不能以两个强基线对照直接宣称全领域 SOTA。", "",
              f"文件审计问题：{len(checks['problems'])}；保留失败/中断条目：{len(checks.get('failures', []))}。",
              "指标来自原始 float32 logits。float16 存档可能合并最高分；"
              "权重/预测审计核验非平分结果、标签分母及平分兼容的混淆计数，不覆盖原指标。", "",
              "## 证据索引", "",
              "- [完整验证表](../../RESULTS.md)、[原始审计汇总](../report.json)",
              "- [主指标](main_metrics.csv)、[逐 RX 指标](receiver_metrics.csv)",
              "- [注册消融](registered_ablations.csv)、[成本](measured_costs.csv)",
              "- [精度与计算量图](accuracy_cost.png)、[导出来源 hash](evidence.json)", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=HERE / "summary/report.json")
    parser.add_argument("--out", type=Path, default=HERE / "summary/paper")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report["checks"]["problems"]:
        raise SystemExit("cannot export a report with artifact audit problems")
    if args.require_complete and not report["checks"]["complete"]:
        raise SystemExit("registered matrix is incomplete")
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = export_tables(report, args.out)
    accuracy_cost_plot(report, args.out)
    (args.out / "CONCLUSION.md").write_text(conclusion(report))
    evidence = dict(generated_at=report["generated_at"], complete=report["checks"]["complete"],
                    report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
                    registered_fits=report["checks"]["validated"],
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(args.out.glob("*")) if p.suffix in (".csv", ".png", ".md")},
                    note="Derived exports only; no model selection, training or new data access.")
    (args.out / "evidence.json").write_text(json.dumps(evidence, indent=1) + "\n")
    print(json.dumps(dict(rows=len(metrics), complete=evidence["complete"], out=str(args.out))))


if __name__ == "__main__":
    main()
