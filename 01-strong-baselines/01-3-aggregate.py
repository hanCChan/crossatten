#!/usr/bin/env python3
"""01-3  Aggregate stage-01 fits.   python 01-3-aggregate.py explore [--freeze]  |  python 01-3-aggregate.py confirm

Writes summary/<stage>_{runs,arms,pairs}.csv and summary/<stage>_tables.md.
--freeze (explore only): applies the pre-registered carry-forward rules (README section 4) and writes
decision_explore.json (with the SHA256 of the exploration run table) -- required before confirmation jobs.
"""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SUM = HERE / "summary"

# pre-registered one-factor comparisons: (B, A) -> delta = B - A
PAIRS = [
    ("Q1", "xa_const", "xa_p1fixed", "horizon 100ep + source-val checkpoint"),
    ("Q1", "xa_cos", "xa_const", "warmup+cosine schedule"),
    ("Q1", "xa_cos_awgn", "xa_cos", "AWGN U[0,50) dB"),
    ("Q2", "cis_gap", "cis_flat", "GAP head vs Flatten-Dense (AWGN)"),
    ("Q2", "cis_gap_noawgn", "cis_flat_noawgn", "GAP head vs Flatten-Dense (no AWGN)"),
    ("Q2", "cis_flat", "cis_flat_noawgn", "AWGN on CIS (flatten)"),
    ("Q2", "cis_gap", "cis_gap_noawgn", "AWGN on CIS (GAP)"),
    ("Q2", "cis_ib_flat", "cis_flat", "strict-IB CIS vs Shen crop (-203..+195 kHz)"),
    ("Q2*", "cis_flat", "xa_cos_awgn", "SYSTEM-LEVEL: Shen-CIS vs CrossAttn (different input/model/recipe)"),
    ("Q3", "xa_oobnear_cos", "xa_cos", "OOB pathway near band only (no AWGN)"),
    ("Q3", "xa_oobnear_cos_awgn", "xa_cos_awgn", "OOB pathway near band only (AWGN)"),
    ("Q3", "xa_lp200_cos", "xa_cos", "input band-limited |f|<=200k (no AWGN)"),
    ("Q3", "xa_lp200_cos_awgn", "xa_cos_awgn", "input band-limited |f|<=200k (AWGN)"),
    ("Q3", "xa_nooob_cos", "xa_cos", "remove explicit OOB pathway (no AWGN)"),
    ("Q3", "xa_nooob_cos_awgn", "xa_cos_awgn", "remove explicit OOB pathway (AWGN)"),
    ("Q3", "xa_trueib_cos", "xa_nooob_cos", "strict IB input vs full input, both without OOB pathway (no AWGN)"),
    ("Q3", "xa_trueib_cos_awgn", "xa_nooob_cos_awgn", "strict IB input vs full input, both without OOB pathway (AWGN)"),
    ("Q3", "xa_trueib_cos", "xa_cos", "TrueIB vs full CrossAttn-OOB (no AWGN)"),
    ("Q3", "xa_trueib_cos_awgn", "xa_cos_awgn", "TrueIB vs full CrossAttn-OOB (AWGN)"),
    ("Q4", "tf_cos", "xa_cos", "time/frequency split tokens (no AWGN)"),
    ("Q4", "tf_cos_awgn", "xa_cos_awgn", "time/frequency split tokens (AWGN)"),
]


def load_runs(stage):
    runs = []
    for m in sorted((HERE / "runs" / stage).glob("*/metrics.json")):
        r = json.loads(m.read_text())
        hist = list(csv.DictReader(open(m.parent / "history.csv")))
        r["_hist"] = hist
        runs.append(r)
    return runs


def verdict(deltas, worst_delta, n_needed):
    d = np.array(deltas)
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    if d.mean() >= 1.0 and pos >= n_needed and worst_delta >= -2.0:
        return "improves"
    if d.mean() <= -1.0 and neg >= n_needed:
        return "harms"
    return "unclear"


def main():
    stage = sys.argv[1]
    freeze = "--freeze" in sys.argv
    runs = load_runs(stage)
    if not runs:
        raise SystemExit("no runs")
    SUM.mkdir(exist_ok=True)
    roles = ["dev_val"] + (["confirm"] if stage == "confirm" else [])
    # ---- per run
    rows = []
    for r in runs:
        s = r["selected"]
        row = dict(arm=r["arm"], fold=r["fold"], seed=r["seed"], params=r["params"], best_epoch=r["best_val_epoch"],
                   stop_epoch=r["stop_epoch"], train_s=r["train_seconds"], src_val_acc=round(100 * s["source_val_acc"], 2),
                   final_train_acc=round(100 * float(r["_hist"][-1]["train_acc"]), 2))
        for role in roles:
            sm = s[role]["summary"]
            row[f"{role}_acc"] = round(100 * sm["acc_mean"], 2)
            row[f"{role}_f1"] = round(100 * sm["f1_mean"], 2)
            row[f"{role}_worst"] = round(100 * sm["worst_acc"], 2)
            row[f"{role}_worst_rx"] = sm["worst_rx"]
            alt = [k for k in r if k in ("last", "best_val")][0]
            row[f"{role}_acc_{alt}ckpt"] = round(100 * r[alt][role]["summary"]["acc_mean"], 2)
        rows.append(row)
    with open(SUM / f"{stage}_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    run_table_sha = hashlib.sha256((SUM / f"{stage}_runs.csv").read_bytes()).hexdigest()
    # ---- per arm
    by_arm = defaultdict(list)
    for r in runs:
        by_arm[r["arm"]].append(r)
    arm_rows, per_rx_tab = [], {}
    for arm, rs in by_arm.items():
        ar = dict(arm=arm, n_fits=len(rs), params=rs[0]["params"],
                  best_epoch_median=float(np.median([r["best_val_epoch"] for r in rs])),
                  train_s_mean=round(float(np.mean([r["train_seconds"] for r in rs])), 1),
                  src_val_acc=round(100 * float(np.mean([r["selected"]["source_val_acc"] for r in rs])), 2))
        for role in roles:
            accs = [100 * r["selected"][role]["summary"]["acc_mean"] for r in rs]
            f1s = [100 * r["selected"][role]["summary"]["f1_mean"] for r in rs]
            per_fold = {fo: [100 * r["selected"][role]["summary"]["acc_mean"] for r in rs if r["fold"] == fo] for fo in (0, 1)}
            # receiver-level: seed-mean accuracy per (fold, rx); worst = min over receivers
            rx_acc = defaultdict(list)
            for r in rs:
                for rx, a in r["selected"][role]["summary"]["per_rx_acc"].items():
                    rx_acc[(r["fold"], rx)].append(100 * a)
            rx_mean = {k: float(np.mean(v)) for k, v in rx_acc.items()}
            per_rx_tab[(arm, role)] = rx_mean
            ar.update({f"{role}_acc_mean": round(float(np.mean(accs)), 2), f"{role}_acc_sd": round(float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0, 2),
                       f"{role}_f1_mean": round(float(np.mean(f1s)), 2),
                       f"{role}_fold0": round(float(np.mean(per_fold[0])), 2) if per_fold[0] else None,
                       f"{role}_fold1": round(float(np.mean(per_fold[1])), 2) if per_fold[1] else None,
                       f"{role}_worst_rx_seedmean": round(min(rx_mean.values()), 2),
                       f"{role}_worst_rx_name": min(rx_mean, key=rx_mean.get)[1],
                       f"{role}_within_fold_seed_sd": round(float(np.mean([np.std(v, ddof=1) for v in per_fold.values() if len(v) > 1])) if any(len(v) > 1 for v in per_fold.values()) else 0.0, 2)})
        arm_rows.append(ar)
    arm_rows.sort(key=lambda a: -a["dev_val_acc_mean"])
    with open(SUM / f"{stage}_arms.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(arm_rows[0].keys()))
        w.writeheader()
        w.writerows(arm_rows)
    # ---- pairs
    key = {(r["arm"], r["fold"], r["seed"]): r for r in runs}
    pair_rows = []
    n_fits_needed = 3 if stage == "explore" else 5
    for q, b, a, desc in PAIRS:
        for role in roles:
            ds = []
            rx_d = defaultdict(list)
            for (arm, fo, se), rb in key.items():
                if arm != b or (a, fo, se) not in key:
                    continue
                ra = key[(a, fo, se)]
                ds.append(100 * (rb["selected"][role]["summary"]["acc_mean"] - ra["selected"][role]["summary"]["acc_mean"]))
                for rx, v in rb["selected"][role]["summary"]["per_rx_acc"].items():
                    rx_d[(fo, rx)].append(100 * (v - ra["selected"][role]["summary"]["per_rx_acc"][rx]))
            if not ds:
                continue
            rxm = {k: float(np.mean(v)) for k, v in rx_d.items()}
            wa, wb = per_rx_tab[(a, role)], per_rx_tab[(b, role)]
            worst_delta = min(wb.values()) - min(wa.values())
            pair_rows.append(dict(q=q, role=role, b=b, a=a, desc=desc, n_pairs=len(ds), delta_mean=round(float(np.mean(ds)), 2),
                                  delta_min=round(float(np.min(ds)), 2), delta_max=round(float(np.max(ds)), 2),
                                  pos_pairs=int((np.array(ds) > 0).sum()), rx_pos=int(sum(v > 0 for v in rxm.values())), rx_n=len(rxm),
                                  worst_rx_delta=round(worst_delta, 2),
                                  verdict=verdict(ds, worst_delta, n_fits_needed) if not q.endswith("*") else "system-level (no verdict)"))
    with open(SUM / f"{stage}_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)
    # ---- markdown
    md = [f"# Stage 01 {stage}: auto-generated tables\n", f"run table sha256: `{run_table_sha}`\n",
          "\n## Arms (selected checkpoint; accuracy in %, RX-equal mean over the role's receivers)\n"]
    cols = ["arm", "n_fits", "params", "dev_val_acc_mean", "dev_val_acc_sd", "dev_val_fold0", "dev_val_fold1", "dev_val_f1_mean",
            "dev_val_worst_rx_seedmean", "dev_val_worst_rx_name", "dev_val_within_fold_seed_sd"]
    if stage == "confirm":
        cols += ["confirm_acc_mean", "confirm_acc_sd", "confirm_f1_mean", "confirm_worst_rx_seedmean", "confirm_worst_rx_name"]
    cols += ["src_val_acc", "best_epoch_median", "train_s_mean"]
    md.append("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
    for ar in arm_rows:
        md.append("| " + " | ".join(str(ar.get(c, "")) for c in cols) + " |\n")
    md.append("\n## Pre-registered one-factor comparisons (delta = B - A, pp)\n\n| Q | role | B | A | factor | pairs | mean | min | max | +pairs | +RX | worst-RX delta | verdict |\n|" + "---|" * 13 + "\n")
    for p in pair_rows:
        md.append(f"| {p['q']} | {p['role']} | {p['b']} | {p['a']} | {p['desc']} | {p['n_pairs']} | {p['delta_mean']} | {p['delta_min']} | {p['delta_max']} | {p['pos_pairs']}/{p['n_pairs']} | {p['rx_pos']}/{p['rx_n']} | {p['worst_rx_delta']} | {p['verdict']} |\n")
    md.append("\n## Per-receiver seed-mean accuracy (%)\n")
    for role in roles:
        rxs = sorted({k for (arm, ro), t in per_rx_tab.items() if ro == role for k in t})
        md.append(f"\n### {role}\n\n| arm | " + " | ".join(f"f{f}:{rx}" for f, rx in rxs) + " |\n|" + "---|" * (len(rxs) + 1) + "\n")
        for ar in arm_rows:
            t = per_rx_tab[(ar["arm"], role)]
            md.append(f"| {ar['arm']} | " + " | ".join(f"{t[k]:.1f}" if k in t else "" for k in rxs) + " |\n")
    (SUM / f"{stage}_tables.md").write_text("".join(md))
    print("".join(md[:4 + len(arm_rows)]))
    if freeze:
        if stage != "explore":
            raise SystemExit("--freeze only for explore")
        dec = decide(arm_rows, pair_rows)
        dec.update(run_table_sha256=run_table_sha, n_runs=len(runs))
        (HERE / "decision_explore.json").write_text(json.dumps(dec, indent=1))
        print(json.dumps(dec, indent=1))


def decide(arm_rows, pair_rows):
    """Pre-registered carry-forward rules (README section 4)."""
    acc = {a["arm"]: a["dev_val_acc_mean"] for a in arm_rows}
    ladder = ["xa_const", "xa_cos", "xa_cos_awgn"]   # simplest first
    best = max(ladder, key=lambda k: acc[k])
    recipe = next(k for k in ladder if acc[k] >= acc[best] - 1.0)
    suf = "_cos_awgn" if recipe == "xa_cos_awgn" else "_cos"
    band_cands = {"full": "xa" + suf, "oobnear": "xa_oobnear" + suf, "lp200": "xa_lp200" + suf,
                  "nooob": "xa_nooob" + suf, "trueib": "xa_trueib" + suf}
    bbest = max(band_cands, key=lambda k: acc[band_cands[k]])
    band = bbest if acc[band_cands[bbest]] >= acc[band_cands["full"]] + 1.0 else "full"
    pv = {(p["b"], p["a"]): p["verdict"] for p in pair_rows if p["role"] == "dev_val"}
    tok = "tfsplit" if pv.get(("tf" + suf, "xa" + suf)) == "improves" else "mixed"
    gap_ok = None
    for p in pair_rows:
        if p["role"] == "dev_val" and (p["b"], p["a"]) == ("cis_gap", "cis_flat"):
            gap_ok = p["delta_mean"] >= -1.0 and p["worst_rx_delta"] >= -2.0
    return dict(rules="README.md section 4", crossattn_recipe_arm=recipe, recipe_ladder_acc={k: acc[k] for k in ladder},
                band_choice=band, band_acc={k: acc[v] for k, v in band_cands.items()}, token_design=tok,
                cis_gap_no_material_loss=gap_ok, confirmation="all registered arms, seeds 201-203, dev_val + confirm RX")


if __name__ == "__main__":
    main()
