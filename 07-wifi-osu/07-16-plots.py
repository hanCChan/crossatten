#!/usr/bin/env python3
"""Descriptive source/OSU plots; final targets only after all frozen fits."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent


def read_csv(name):
    with (HERE / name).open() as stream:
        return list(csv.DictReader(stream))


def canvas(width, height, title):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 22), title, font=font(23), fill="#222222")
    return image, draw


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def center(draw, x, y, text, size=15, color="#222222"):
    draw.text((x, y), text, font=font(size), fill=color, anchor="mm")


def heatmap(draw, matrix, columns, title, left, top):
    draw.text((left, top - 35), title, font=font(18), fill="#222222")
    width, height = 77, 47
    for row in range(4):
        draw.text((left - 65, top + height * row + 14), f"Fold {row}",
                  font=font(14), fill="#555555")
        for col in range(len(columns)):
            value = matrix[row, col]
            strength = np.clip((value - 20) / 75, 0, 1) if np.isfinite(value) else 0
            color = tuple(int(c) for c in np.array([235, 242, 245]) * (1 - strength)
                          + np.array([32, 123, 109]) * strength)
            x, y = left + width * col, top + height * row
            draw.rectangle((x, y, x + width - 2, y + height - 2), fill=color)
            center(draw, x + width / 2, y + height / 2,
                   f"{value:.1f}" if np.isfinite(value) else "pending",
                   14, "white" if strength > .6 else "#222222")
    for col, label in enumerate(columns):
        center(draw, left + width * (col + .5), top + height * 4 + 22,
               label.replace("held_epoch", "RX-epoch").replace("mild_noise", "mild-noise"),
               11)


def sources():
    image, draw = canvas(1660, 680, "S07 source-only evidence; available registered seeds, not target CRD")
    for index, (kind, filename, columns) in enumerate((
        ("frontend", "S07_WIFI_FEATURE_RESULTS.csv",
         ["legacy", "time", "diff", "split", "stf", "filter", "frequency"]),
        ("factor", "S07_WIFI_TRAINING_RESULTS.csv",
         ["unified", "official", "balanced", "held_epoch", "phase", "shift", "cfo", "channel", "mild_noise"]),
    )):
        if not (HERE / filename).is_file():
            continue
        rows = read_csv(filename)
        phases = ("features", "replicate") if index == 0 else ("training", "training_replicate")
        for row_index, dataset in enumerate(("ManyRx", "ManySig")):
            matrix = np.full((4, len(columns)), np.nan)
            for fold in range(4):
                for col, name in enumerate(columns):
                    values = [float(r["held_rx_pct"]) for r in rows
                              if r["dataset"] == dataset and int(r["fold"]) == fold and r[kind] == name
                              and r["phase"] in phases]
                    if values:
                        matrix[fold, col] = np.mean(values)
            heatmap(draw, matrix, columns, f"{dataset}: held-source {kind} accuracy (%)",
                    95 + index * 750, 125 + row_index * 265)
    draw.text((30, 642), "Source choices are per fold; pooled display is descriptive only. "
              "Pending cells remain visible. Darker green indicates higher accuracy.",
              font=font(15), fill="#555555")
    image.save(HERE / "source_factor_results.png")


def axes(draw, left, top, width, height, title):
    draw.text((left, top - 35), title, font=font(18), fill="#222222")
    for value in range(0, 101, 20):
        y = top + height * (1 - value / 100)
        draw.line((left, y, left + width, y), fill="#dddddd")
        draw.text((left - 35, y - 9), str(value), font=font(14), fill="#555555")
    draw.line((left, top, left, top + height, left + width, top + height), fill="#555555", width=2)


def bars(draw, values, positions, left, top, height, width, color, errors=None):
    for index, (x, value) in enumerate(zip(positions, values)):
        y = top + height * (1 - value / 100)
        draw.rectangle((left + x - width / 2, y, left + x + width / 2, top + height), fill=color)
        if errors is not None:
            dy = height * errors[index] / 100
            draw.line((left + x, y - dy, left + x, y + dy), fill="#222222", width=2)
            draw.line((left + x - 6, y - dy, left + x + 6, y - dy), fill="#222222", width=2)
        center(draw, left + x, y - 15, f"{value:.2f}", 15)


def osu():
    rows = read_csv("osu_posthoc_same_rx.csv")
    methods = ["patch_hstu", "hstem_hstu", "hstem_cnn", "cis_rank16"]
    image, draw = canvas(1100, 650, "OSU: registered negative RX transfer; temporal control is post-hoc")
    left, top, width, height = 75, 120, 930, 350
    axes(draw, left, top, width, height, "Packet accuracy (%)")
    positions = np.arange(4) * 230 + 120
    for offset, column, label, color in (
        (-.18, "same_rx_future_acc_pct", "Same RX, future block (post-hoc)", "#268277"),
        (.18, "cross_rx_future_acc_pct", "Other RX, future block (registered)", "#c64750"),
    ):
        values = [np.mean([float(r[column]) for r in rows if r["method"] == method]) for method in methods]
        bars(draw, values, positions + offset * 230, left, top, height, 75, color)
        legend_y = 555 if offset < 0 else 585
        draw.rectangle((45, legend_y, 62, legend_y + 17), fill=color)
        draw.text((75, legend_y - 1), label, font=font(16), fill="#333333")
    chance_y = top + height * .96
    for x in range(left, left + width, 10):
        draw.line((x, chance_y, x + 5, chance_y), fill="#666666")
    for x, method in zip(positions, methods):
        center(draw, left + x, 500, method, 16)
    draw.text((590, 555), "25-class balanced chance: 4%", font=font(16), fill="#555555")
    draw.text((590, 585), "Two RX; three seeds; not independent RX populations.",
              font=font(14), fill="#555555")
    image.save(HERE / "osu_transfer_failure.png")


def final():
    jobs = json.loads((HERE / "jobs_final.json").read_text())
    if len(jobs) != 72 or any(not (HERE / "runs/final" / j["id"] / "metrics.json").exists() for j in jobs):
        raise ValueError("all final fits required before target plotting")
    rows = read_csv("wifi_final_metrics.csv")
    intervals = read_csv("wifi_comparison_intervals.csv")
    pipelines = ["source_frontend", "source_training", "source_combination"]
    image, draw = canvas(1440, 690, "All S07 registered pipelines; seed SD bars are not confidence intervals")
    for panel, dataset in enumerate(("ManyRx", "ManySig")):
        left, top, height = 75 + panel * 715, 130, 390
        axes(draw, left, top, 585, height, f"{dataset}: CRD RX-equal accuracy (%)")
        values = [next(r for r in rows if r["dataset"] == dataset and r["pipeline"] == p and r["task"] == "CRD")
                  for p in pipelines]
        accs = [float(r["rx_accuracy_pct"]) for r in values]
        errors = [float(r["seed_sd_pp"]) for r in values]
        positions = np.arange(3) * 180 + 115
        for index, color in enumerate(("#477bba", "#268277", "#c64750")):
            bars(draw, [accs[index]], [positions[index]], left, top, height, 90,
                 color, [errors[index]])
        for baseline, color in (("IB-RFF", "#222222"), ("WaveMLP", "#a46725")):
            row = next(r for r in intervals if r["dataset"] == dataset
                       and r["pipeline"] == pipelines[0] and r["baseline"] == baseline)
            reference = accs[0] - float(row["delta_pp"])
            y = top + height * (1 - reference / 100)
            for x in range(left, left + 585, 14):
                draw.line((x, y, x + 8, y), fill=color, width=2)
            legend_y = 592 if baseline == "IB-RFF" else 620
            draw.text((left, legend_y), f"S06 fair {baseline}: {reference:.2f}%", font=font(17), fill=color)
        for x, name in zip(positions, ("Frontend", "Training", "Combination")):
            center(draw, left + x, 551, name, 17)
    draw.text((30, 660), "Four RX folds x three seeds; every pipeline is reported. "
              "S06/S07 seeds are unpaired; targets were previously inspected.",
              font=font(15), fill="#555555")
    image.save(HERE / "wifi_final_comparison.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    sources()
    osu()
    if args.final:
        final()
    print("wrote descriptive source/OSU" + (" and frozen final" if args.final else "") + " plots")


if __name__ == "__main__":
    main()
