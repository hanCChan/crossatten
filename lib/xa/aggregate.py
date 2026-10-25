"""Stage aggregation shared by stage >= 02 (same statistics as 01-strong-baselines/01-3-aggregate.py).

aggregate(stage_dir, stage, pairs, ref_globs) reads <stage_dir>/runs/<stage>/*/metrics.json plus optional reference runs
(e.g. stage-01 fits with identical (fold, seed) data streams), writes summary/<stage>_{runs,arms,pairs}.csv and
summary/<stage>_tables.md, and returns (arm_rows, pair_rows, run_table_sha256).
verdict: improves = mean delta >= +1 pp and >= n_needed positive pairs and worst-RX delta >= -2 pp;
         harms = mean delta <= -1 pp and >= n_needed negative pairs; otherwise unclear.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_runs(paths: list[str]) -> list[dict]:
    runs, seen = [], set()
    for p in sorted(paths):
        r = json.loads(Path(p).read_text())
        key = (r["arm"], r["fold"], r["seed"])
        if key in seen:
            raise ValueError(f"duplicate run {key}")
        seen.add(key)
        r["_hist"] = list(csv.DictReader(open(Path(p).parent / "history.csv")))
        r["_dir"] = str(Path(p).parent)
        runs.append(r)
    return runs


def verdict(deltas, worst_delta, n_needed):
    d = np.array(deltas)
    if d.mean() >= 1.0 and int((d > 0).sum()) >= n_needed and worst_delta >= -2.0:
        return "improves"
    if d.mean() <= -1.0 and int((d < 0).sum()) >= n_needed:
        return "harms"
    return "unclear"


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def aggregate(stage_dir: Path, stage: str, pairs: list[tuple], ref_globs: list[str] = ()) -> tuple:
    paths = glob.glob(str(stage_dir / "runs" / stage / "*" / "metrics.json"))
    for g in ref_globs:
        paths += glob.glob(g)
    runs = load_runs(paths)
    if not runs:
        raise SystemExit("no runs")
    out = stage_dir / "summary"
    out.mkdir(exist_ok=True)
    roles = ["dev_val"] + (["confirm"] if stage == "confirm" else [])
    rows = []
    for r in runs:
        s = r["selected"]
        alt = [k for k in r if k in ("last", "best_val")][0]
        row = dict(arm=r["arm"], fold=r["fold"], seed=r["seed"], params=r["params"], best_epoch=r["best_val_epoch"],
                   stop_epoch=r["stop_epoch"], train_s=r["train_seconds"], src_val_acc=round(100 * s["source_val_acc"], 2),
                   final_train_acc=round(100 * float(r["_hist"][-1]["train_acc"]), 2), alt_ckpt=alt,
                   source=("reference" if not r["_dir"].startswith(str(stage_dir)) else "this stage"))
        for role in roles:
            sm = s[role]["summary"]
            row.update({f"{role}_acc": round(100 * sm["acc_mean"], 2), f"{role}_f1": round(100 * sm["f1_mean"], 2),
                        f"{role}_worst": round(100 * sm["worst_acc"], 2), f"{role}_worst_rx": sm["worst_rx"],
                        f"{role}_acc_altckpt": round(100 * r[alt][role]["summary"]["acc_mean"], 2)})
        rows.append(row)
    _write_csv(out / f"{stage}_runs.csv", rows)
    sha = hashlib.sha256((out / f"{stage}_runs.csv").read_bytes()).hexdigest()
    by_arm = defaultdict(list)
    for r in runs:
        by_arm[r["arm"]].append(r)
    arm_rows, per_rx = [], {}
    for arm, rs in by_arm.items():
        ar = dict(arm=arm, n_fits=len(rs), params=rs[0]["params"],
                  best_epoch_median=float(np.median([r["best_val_epoch"] for r in rs])),
                  train_s_mean=round(float(np.mean([r["train_seconds"] for r in rs])), 1),
                  src_val_acc=round(100 * float(np.mean([r["selected"]["source_val_acc"] for r in rs])), 2))
        for role in roles:
            accs = [100 * r["selected"][role]["summary"]["acc_mean"] for r in rs]
            f1s = [100 * r["selected"][role]["summary"]["f1_mean"] for r in rs]
            pf = {fo: [100 * r["selected"][role]["summary"]["acc_mean"] for r in rs if r["fold"] == fo] for fo in (0, 1)}
            rx_acc = defaultdict(list)
            for r in rs:
                for rx, a in r["selected"][role]["summary"]["per_rx_acc"].items():
                    rx_acc[(r["fold"], rx)].append(100 * a)
            rxm = {k: float(np.mean(v)) for k, v in rx_acc.items()}
            per_rx[(arm, role)] = rxm
            wsd = [np.std(v, ddof=1) for v in pf.values() if len(v) > 1]
            ar.update({f"{role}_acc_mean": round(float(np.mean(accs)), 2),
                       f"{role}_acc_sd": round(float(np.std(accs, ddof=1)), 2) if len(accs) > 1 else 0.0,
                       f"{role}_f1_mean": round(float(np.mean(f1s)), 2),
                       f"{role}_fold0": round(float(np.mean(pf[0])), 2) if pf[0] else None,
                       f"{role}_fold1": round(float(np.mean(pf[1])), 2) if pf[1] else None,
                       f"{role}_worst_rx_seedmean": round(min(rxm.values()), 2),
                       f"{role}_worst_rx_name": min(rxm, key=rxm.get)[1],
                       f"{role}_within_fold_seed_sd": round(float(np.mean(wsd)), 2) if wsd else 0.0})
        arm_rows.append(ar)
    arm_rows.sort(key=lambda a: -a["dev_val_acc_mean"])
    _write_csv(out / f"{stage}_arms.csv", arm_rows)
    key = {(r["arm"], r["fold"], r["seed"]): r for r in runs}
    n_needed = 3 if stage == "explore" else 5
    pair_rows = []
    for q, b, a, desc in pairs:
        for role in roles:
            ds, rxd = [], defaultdict(list)
            for (arm, fo, se), rb in key.items():
                if arm != b or (a, fo, se) not in key:
                    continue
                ra = key[(a, fo, se)]
                ds.append(100 * (rb["selected"][role]["summary"]["acc_mean"] - ra["selected"][role]["summary"]["acc_mean"]))
                for rx, v in rb["selected"][role]["summary"]["per_rx_acc"].items():
                    rxd[(fo, rx)].append(100 * (v - ra["selected"][role]["summary"]["per_rx_acc"][rx]))
            if not ds:
                continue
            rxm = {k: float(np.mean(v)) for k, v in rxd.items()}
            worst = min(per_rx[(b, role)].values()) - min(per_rx[(a, role)].values())
            pair_rows.append(dict(q=q, role=role, b=b, a=a, desc=desc, n_pairs=len(ds),
                                  delta_mean=round(float(np.mean(ds)), 2), delta_min=round(float(np.min(ds)), 2),
                                  delta_max=round(float(np.max(ds)), 2), pos_pairs=int((np.array(ds) > 0).sum()),
                                  rx_pos=int(sum(v > 0 for v in rxm.values())), rx_n=len(rxm), worst_rx_delta=round(worst, 2),
                                  verdict=("reference (no verdict)" if q.startswith("ref") else verdict(ds, worst, n_needed))))
    if pair_rows:
        _write_csv(out / f"{stage}_pairs.csv", pair_rows)
    md = [f"# {stage_dir.name} {stage}: auto-generated tables\n\nrun table sha256: `{sha}`\n\n"
          "## Arms (selected checkpoint; %, RX-equal mean over the role's receivers)\n\n"]
    cols = ["arm", "n_fits", "params", "dev_val_acc_mean", "dev_val_acc_sd", "dev_val_fold0", "dev_val_fold1", "dev_val_f1_mean",
            "dev_val_worst_rx_seedmean", "dev_val_worst_rx_name", "dev_val_within_fold_seed_sd"]
    if stage == "confirm":
        cols += ["confirm_acc_mean", "confirm_acc_sd", "confirm_f1_mean", "confirm_worst_rx_seedmean", "confirm_worst_rx_name"]
    cols += ["src_val_acc", "best_epoch_median", "train_s_mean"]
    md.append("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
    md += ["| " + " | ".join(str(ar.get(c, "")) for c in cols) + " |\n" for ar in arm_rows]
    md.append("\n## Pre-registered comparisons (delta = B - A, pp)\n\n| Q | role | B | A | factor | pairs | mean | min | max | +pairs | +RX | worst-RX delta | verdict |\n|" + "---|" * 13 + "\n")
    md += [f"| {p['q']} | {p['role']} | {p['b']} | {p['a']} | {p['desc']} | {p['n_pairs']} | {p['delta_mean']} | {p['delta_min']} | "
           f"{p['delta_max']} | {p['pos_pairs']}/{p['n_pairs']} | {p['rx_pos']}/{p['rx_n']} | {p['worst_rx_delta']} | {p['verdict']} |\n" for p in pair_rows]
    md.append("\n## Per-receiver seed-mean accuracy (%)\n")
    for role in roles:
        rxs = sorted({k for (arm, ro), t in per_rx.items() if ro == role for k in t})
        md.append(f"\n### {role}\n\n| arm | " + " | ".join(f"f{f}:{rx}" for f, rx in rxs) + " |\n|" + "---|" * (len(rxs) + 1) + "\n")
        for ar in arm_rows:
            t = per_rx[(ar["arm"], role)]
            md.append(f"| {ar['arm']} | " + " | ".join(f"{t[k]:.1f}" if k in t else "" for k in rxs) + " |\n")
    (out / f"{stage}_tables.md").write_text("".join(md))
    return arm_rows, pair_rows, sha
