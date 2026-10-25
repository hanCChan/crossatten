#!/usr/bin/env python3
"""S08 source and complete-only target reports, preserving negative findings."""
import csv
import importlib.util
import json
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from training import HERE, ROOT, RULES, sha


def save_csv(name, rows):
    if not rows:
        return
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_report():
    rows = []
    counts = {}
    for phase in ("screen", "phase", "manyrx"):
        jobs_path = HERE / f"jobs_{phase}.json"
        if not jobs_path.exists():
            continue
        jobs = json.loads(jobs_path.read_text())
        done = 0
        for job in jobs:
            path = HERE / "runs" / phase / job["id"] / "metrics.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            if result["job"] != job or any(k in result["selected"] for k in ("CR", "CRD")):
                raise ValueError("invalid source artifact")
            done += 1
            for rule in RULES:
                entry = result["selected"][rule]
                rows.append(dict(
                    phase=phase, dataset=job["dataset"], fold=job["fold"], group=job["group"],
                    seed=job["seed"], condition=job["condition"], feature=job["feature"], rule=rule,
                    stop_epoch=result["stop_epoch"], selected_epoch=entry["epoch"],
                    packet_ce=entry["packet_ce"], packet_acc_pct=100 * entry["packet_acc"],
                    held_acc_pct=100 * entry["summary"]["acc_mean"],
                    held_ce=entry["summary"]["ce_mean"],
                    worst_held_pct=100 * entry["summary"]["worst_acc"],
                    params=result["params"], seconds=result["seconds"]))
        counts[phase] = dict(registered=len(jobs), completed=done)
    save_csv("source_results.csv", rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["condition"], row["feature"], row["rule"])].append(row)
    lines = ["# S08 Source Results\n\n",
             "Source RX days1-2 only; two held groups x two seeds. Checkpoints from the same fit "
             "are correlated, not independent models. Aggregates below are descriptive only; "
             "formal candidate selection is fold-local. No target metrics select methods.\n\n",
             "| Dataset | Condition | Feature | Rule | Cells | Mean held RX % | Mean worst RX % | Median epoch |\n",
             "|---|---|---|---|---:|---:|---:|---:|\n"]
    for key, cells in sorted(groups.items()):
        lines.append("| " + " | ".join(key) +
                     f" | {len(cells)} | {np.mean([r['held_acc_pct'] for r in cells]):.2f} | "
                     f"{np.mean([r['worst_held_pct'] for r in cells]):.2f} | "
                     f"{np.median([r['selected_epoch'] for r in cells]):.1f} |\n")
    latest = next((HERE / f"decision_after_{p}.json" for p in ("manyrx", "phase", "screen")
                   if (HERE / f"decision_after_{p}.json").exists()), None)
    if latest:
        decision = json.loads(latest.read_text())
        lines.extend(["\n## Fold-Local Gates\n\n",
                      "| Dataset | Fold | Supported | Source-selected configuration | Mean delta pp | Worst delta pp |\n",
                      "|---|---:|---|---|---:|---:|\n"])
        for dataset, folds in decision["decisions"].items():
            for fold, entry in folds.items():
                gate = entry["candidates"][entry["chosen"]]
                lines.append(f"| {dataset} | {fold} | {entry['supported']} | {entry['chosen']} | "
                             f"{gate['mean_delta_pp']:+.2f} | {gate['worst_rx_delta_pp']:+.2f} |\n")
        lines.append("\nAdmission requires supported candidates on at least3/4 folds. "
                     "Unsupported folds fall back to official/raw packet-CE. "
                     "Held-CE rules refit all source RX for the source-frozen upper median epoch.\n")
    lines.append("\nFull200 truly ran all200 epochs, but its selected checkpoint may be early. "
                 "No result licenses target-driven additional features/losses. "
                 "All exact cells are in `source_results.csv` and the retained `runs/`.\n")
    (HERE / "S08_SOURCE_RESULTS.md").write_text("".join(lines))
    return counts


def final_report():
    decision_path = HERE / "decision_final.json"
    if not decision_path.exists():
        return None
    decision = json.loads(decision_path.read_text())
    jobs = decision["jobs"]
    if not jobs:
        return dict(registered=0, completed=0, admitted=decision["admitted"])
    if any(not (HERE / "runs/final" / j["id"] / "metrics.json").exists() for j in jobs):
        return dict(registered=len(jobs), completed=sum(
            (HERE / "runs/final" / j["id"] / "metrics.json").exists() for j in jobs))
    spec = importlib.util.spec_from_file_location("s08_report_audit", HERE / "08-4-audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    audit.audit_phase("final")
    receivers, runs = [], []
    for job in jobs:
        result = json.loads((HERE / "runs/final" / job["id"] / "metrics.json").read_text())
        source = result["selected"]["source_packet"]
        runs.append(dict(
            id=job["id"], dataset=job["dataset"], fold=job["fold"], seed=job["seed"],
            variant=job["variant"], condition=job["condition"], feature=job["feature"], rule=job["rule"],
            stop_epoch=result["stop_epoch"], selected_epoch=source["epoch"],
            source_packet_acc_pct=100 * source["acc"], source_packet_ce=source["ce"],
            CR_rx_accuracy_pct=100 * result["selected"]["CR"]["summary"]["acc_mean"],
            CRD_rx_accuracy_pct=100 * result["selected"]["CRD"]["summary"]["acc_mean"],
            params=result["params"], seconds=result["seconds"]))
        for task in ("CR", "CRD"):
            for rx, value in result["selected"][task]["per_rx"].items():
                receivers.append(dict(dataset=job["dataset"], variant=job["variant"], task=task,
                                      fold=job["fold"], seed=job["seed"], rx=rx,
                                      accuracy_pct=100 * value["acc"], macro_f1_pct=100 * value["macro_f1"],
                                      correct=value["correct"], total=value["total"], params=result["params"]))
    summary = []
    groups = defaultdict(list)
    for row in receivers:
        groups[(row["dataset"], row["variant"], row["task"])].append(row)
    for key, cells in sorted(groups.items()):
        dataset, variant, task = key
        by_rx, by_seed, by_fold, packet_folds = defaultdict(list), defaultdict(list), defaultdict(list), {}
        for row in cells:
            by_rx[row["rx"]].append(row["accuracy_pct"])
            by_seed[row["seed"]].append(row["accuracy_pct"])
            by_fold[row["fold"]].append(row["accuracy_pct"])
        for fold in range(4):
            for seed in (7101, 7102, 7103):
                local = [r for r in cells if r["fold"] == fold and r["seed"] == seed]
                packet_folds[(fold, seed)] = 100 * sum(r["correct"] for r in local) / sum(r["total"] for r in local)
        summary.append(dict(
            dataset=dataset, variant=variant, task=task, rx_accuracy_pct=float(np.mean(
                [np.mean(v) for v in by_rx.values()])), packet_accuracy_pct=float(np.mean(
                list(packet_folds.values()))), rx_macro_f1_pct=float(np.mean([r["macro_f1_pct"] for r in cells])),
            worst_rx_pct=min(np.mean(v) for v in by_rx.values()),
            seed_sd_pp=float(np.std([np.mean(v) for v in by_seed.values()], ddof=1)),
            fold_sd_pp=float(np.std([np.mean(v) for v in by_fold.values()], ddof=1)),
            params=";".join(str(p) for p in sorted({r["params"] for r in cells}))))
    save_csv("final_metrics.csv", summary)
    save_csv("final_receiver_metrics.csv", receivers)
    save_csv("final_run_metrics.csv", runs)
    frozen_path = ROOT / "06-paper-validation/summary/paper/receiver_metrics.csv"
    with frozen_path.open() as stream:
        frozen = list(csv.DictReader(stream))
    s07_path = ROOT / "07-wifi-osu/wifi_final_receiver_metrics.csv"
    with s07_path.open() as stream:
        s07 = list(csv.DictReader(stream))
    intervals = []
    rng = np.random.default_rng(8101)
    for dataset in sorted({j["dataset"] for j in jobs}):
        ours = defaultdict(list)
        control = defaultdict(list)
        for row in receivers:
            if row["dataset"] == dataset and row["task"] == "CRD":
                (ours if row["variant"] == "candidate" else control)[row["rx"]].append(row["accuracy_pct"])
        baselines = {"S08-paired-control": {rx: np.mean(v) for rx, v in control.items()}}
        for name in ("IB-RFF", "WaveMLP", "FAN_source_selected", "W1"):
            baselines[name] = {r["rx"]: float(r["seed_mean_accuracy_pct"]) for r in frozen
                               if r["dataset"] == dataset and r["model"] == name and r["task"] == "CRD"}
        previous = defaultdict(list)
        for row in s07:
            if row["dataset"] == dataset and row["task"] == "CRD" and row["pipeline"] == "source_training":
                previous[row["rx"]].append(float(row["accuracy_pct"]))
        baselines["S07-source-training"] = {rx: np.mean(v) for rx, v in previous.items()}
        for baseline, theirs in baselines.items():
            if set(ours) != set(theirs):
                raise ValueError("comparison receiver identities differ")
            names = sorted(ours)
            delta = np.array([np.mean(ours[rx]) - theirs[rx] for rx in names])
            sampled = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(1)
            intervals.append(dict(dataset=dataset, baseline=baseline, delta_pp=float(delta.mean()),
                                  ci95_low_pp=float(np.quantile(sampled, .025)),
                                  ci95_high_pp=float(np.quantile(sampled, .975)),
                                  improved_receivers=int((delta > 0).sum()), receivers=len(names),
                                  seeds_paired=baseline == "S08-paired-control"))
    save_csv("comparison_intervals.csv", intervals)
    lines = ["# S08 Formal WiSig Results\n\n",
             f"All {len(jobs)} frozen final fits complete. Four official folds x three fresh seeds "
             "for candidate and paired fixed-frequency official control. CRD primary; "
             "all algorithms selected from source RX days1-2. Public targets are not new blind tests.\n\n",
             "| Dataset | Variant | Task | RX acc % | Packet acc % | Macro-F1 % | Seed SD pp | Worst RX % | Params |\n",
             "|---|---|---|---:|---:|---:|---:|---:|---|\n"]
    for row in summary:
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['task']} | {row['rx_accuracy_pct']:.2f} | "
                     f"{row['packet_accuracy_pct']:.2f} | {row['rx_macro_f1_pct']:.2f} | "
                     f"{row['seed_sd_pp']:.2f} | {row['worst_rx_pct']:.2f} | {row['params']} |\n")
    lines.extend(["\n## Receiver-Matched CRD Differences\n\n",
                  "| Dataset | Baseline | Delta pp | Conditional RX95% interval | Improved RX |\n",
                  "|---|---|---:|---|---:|\n"])
    for row in intervals:
        lines.append(f"| {row['dataset']} | {row['baseline']} | {row['delta_pp']:+.2f} | "
                     f"[{row['ci95_low_pp']:+.2f}, {row['ci95_high_pp']:+.2f}] | "
                     f"{row['improved_receivers']}/{row['receivers']} |\n")
    lines.append("\nRX-cluster bootstrap conditions on seed-averaged trained models; "
                 "it does not capture full seed/dataset uncertainty and is not multiplicity corrected. "
                 "S06/S07 seeds are unpaired; only new S08 control seeds are paired. "
                 "Noise/EMA/stopping changes are training and selection effects, not new architecture. "
                 "No target-ranked pipeline or global-SOTA claim.\n")
    (HERE / "S08_WIFI_FINAL_RESULTS.md").write_text("".join(lines))
    report_files = ["final_metrics.csv", "final_receiver_metrics.csv", "final_run_metrics.csv",
                    "comparison_intervals.csv", "S08_WIFI_FINAL_RESULTS.md"]
    evidence = dict(decision_sha256=sha(decision_path), files={n: sha(HERE / n) for n in report_files},
                    baseline_sha256={str(p.relative_to(ROOT)): sha(p) for p in (frozen_path, s07_path)},
                    fits={j["id"]: sha(HERE / "runs/final" / j["id"] / "metrics.json") for j in jobs})
    (HERE / "final_report_evidence.json").write_text(json.dumps(evidence, indent=1) + "\n")
    return dict(registered=len(jobs), completed=len(jobs), admitted=decision["admitted"], summary=summary)


def main():
    counts = source_report()
    final = final_report()
    completed = sum(r["completed"] for r in counts.values()) + (final or {}).get("completed", 0)
    stopped = (HERE / "decision_final.json").exists() and (
        not (final or {}).get("registered") or final["registered"] == final["completed"])
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    status = dict(generated_at=now, source=counts, final=final, formal_completed=completed,
                  stopped=stopped, no_extra_target_search=True)
    (HERE / "stage_status.json").write_text(json.dumps(status, indent=1) + "\n")
    lines = ["# 阶段08结果\n\n", f"更新时间：{now}。基线 `74f5943`，分支 `exp/s08-wisig-training`。\n\n",
             f"已完成正式训练 **{completed}** 次，预算上限160次。历史阶段00-07保持只读。\n\n"]
    if stopped and not (final or {}).get("registered"):
        lines.append("**源域预登记门槛未通过，停止本轮，没有读取新的目标RX。** "
                     "不能把内部留源RX成绩称为CRD，更不能据此宣称SOTA。\n\n")
    elif final and final.get("summary"):
        lines.extend(["## 正式确认\n\n",
                      "| 数据集 | 算法 | CRD RX准确率 | Macro-F1 | 种子SD pp | 最差RX |\n",
                      "|---|---|---:|---:|---:|---:|\n"])
        for row in final["summary"]:
            if row["task"] == "CRD":
                lines.append(f"| {row['dataset']} | {row['variant']} | {row['rx_accuracy_pct']:.2f}% | "
                             f"{row['rx_macro_f1_pct']:.2f}% | {row['seed_sd_pp']:.2f} | "
                             f"{row['worst_rx_pct']:.2f}% |\n")
        lines.append("\n正式测试已完成并停止，不按目标成绩追加Loss、特征或模型搜索。"
                     "与强基线的完整差值及接收机级区间见英文正式报告。\n\n")
    else:
        lines.append("**仍在执行注册实验，暂无完整正式CRD结论。**\n\n")
    lines.append("## 证据\n\n"
                 "- `DIAGNOSTICS.md`：零GPU源域审计。\n"
                 "- `S08_SOURCE_RESULTS.md`、`source_results.csv`：所有源域训练/checkpoint结果。\n"
                 "- `registration.json`、`decision_after_*.json`、`decision_final.json`：预算与源域决策。\n"
                 "- `S08_WIFI_FINAL_RESULTS.md`、`final_*metrics.csv`：仅在完整确认后生成。\n"
                 "- `runs/`：配置、曲线、预测与失败记录；新权重仅本地保存。\n"
                 "- `S08_COSTS.md`：停止训练后实测成本；参数量不等于速度。\n\n"
                 "不自动合并、推送或发布权重；旧测试RX已被查看，本轮不是全新盲测。\n")
    (HERE / "RESULTS.md").write_text("".join(lines))
    print(json.dumps(dict(formal_completed=completed, stopped=stopped, final_complete=bool(
        final and final.get("registered") and final.get("registered") == final.get("completed")))), flush=True)


if __name__ == "__main__":
    main()
