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
    metrics, receivers, ablations, costs = [], [], [], []
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
                                  fold=int(fold), rx=rx, seed_mean_accuracy_pct=accuracy))
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
    for name, rows in (("main_metrics", metrics), ("receiver_metrics", receivers),
                       ("registered_ablations", ablations), ("measured_costs", costs)):
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
    evidence = dict(generated_at=report["generated_at"], complete=report["checks"]["complete"],
                    report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
                    registered_fits=report["checks"]["validated"],
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(args.out.glob("*")) if p.suffix in (".csv", ".png")},
                    note="Derived exports only; no model selection, training or new data access.")
    (args.out / "evidence.json").write_text(json.dumps(evidence, indent=1) + "\n")
    print(json.dumps(dict(rows=len(metrics), complete=evidence["complete"], out=str(args.out))))


if __name__ == "__main__":
    main()
