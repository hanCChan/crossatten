#!/usr/bin/env python3
"""Generate jobs and freeze source-only choices; no target metrics enter selection."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REG = HERE / "arms.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jobs(kind, arms, seeds, folds, dataset=None, decision=None, ib=False):
    jobs = []
    for seed in seeds:
        for fold in folds:
            for arm in arms:
                if dataset and arm.get("dataset") != dataset:
                    continue
                if decision and [arm["name"], fold, seed] not in decision["allowed_final_jobs"]:
                    continue
                job = dict(runner="xa.paper_run", arms=str(REG), arm=arm["name"],
                           fold=fold, seed=seed, stage=kind,
                           out=str(HERE / "runs" / kind / f"{arm['name']}__f{fold}__s{seed}"))
                if decision:
                    job["decision_file"] = str(HERE / ("decision_wifi.json" if kind == "final" else "decision_shen.json"))
                jobs.append(job)
    suffix = f"_{dataset}" if dataset else ""
    suffix += "_ib" if ib else "_base"
    path = HERE / "runs" / f"jobs_{kind}{suffix}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    print(f"{len(jobs)} jobs: {path}")


def freeze_wifi(arms):
    path = HERE / "decision_wifi.json"
    if path.exists():
        raise SystemExit("WiSig decision already frozen; refusing overwrite")
    records, hashes = {}, {}
    for arm in arms:
        if arm.get("dataset") is None:
            continue
        for fold in range(4):
            p = HERE / "runs/source" / f"{arm['name']}__f{fold}__s2023" / "metrics.json"
            r = json.loads(p.read_text())
            if r["selected"] or r["stage"] != "source" or r["seed"] != 2023:
                raise ValueError("source choice contains target results or wrong stage")
            records[(arm["name"], fold)] = r
            hashes[str(p.relative_to(HERE))] = sha(p)
    allowed, choices, fan_init = [], {}, {}
    for dataset in ("ManyRx", "ManySig"):
        candidates = [a for a in arms if a.get("dataset") == dataset]
        for fold in range(4):
            def rank(a):
                r = records[(a["name"], fold)]
                return (-r["source_rx_val"]["summary"]["acc_mean"], r["source_val_ce"], a["name"])
            recipe = min(("official", "unified"), key=lambda rec: (
                -np.mean([records[(a["name"], fold)]["source_rx_val"]["summary"]["acc_mean"]
                          for a in candidates if a["recipe"] == rec and a["name"].startswith(("W0_", "W1_", "W2_"))]),
                rec))
            selected = [a for a in candidates if a["family"] == "paper_rf" and a["recipe"] == recipe]
            fan = min([a for a in candidates if a["method"] == "FAN"], key=rank)
            ib = min([a for a in candidates if a["method"] == "IB-RFF"], key=rank)
            wave = min([a for a in candidates if a["method"] == "WaveMLP"], key=rank)
            selected += [a for a in candidates if a["method"] == "FAN"] + [ib, wave]
            choices[f"{dataset}:{fold}"] = dict(shared_rf_recipe=recipe, fan=fan["name"],
                                              ib=ib["name"], wave=wave["name"])
            fan_init[f"{dataset}:{fold}"] = ib["init_arm"]
            allowed.extend([[a["name"], fold, seed] for a in selected for seed in (2023, 2024, 2025)])
    decision = dict(selection="internal source RX days1,2 only; no target data",
                    registration_sha256=sha(REG), source_result_hashes=hashes,
                    choices=choices, fan_init=fan_init, allowed_final_jobs=allowed)
    path.write_text(json.dumps(decision, indent=1) + "\n")
    print(json.dumps(choices, indent=1))


def freeze_shen(arms):
    path = HERE / "decision_shen.json"
    if path.exists():
        raise SystemExit("Shen decision already frozen; refusing overwrite")
    candidates = [a for a in arms if a.get("dataset") is None]
    records, hashes = {}, {}
    for arm in candidates:
        rs = []
        for fold in range(2):
            for seed in (101, 102):
                p = HERE / "runs/explore" / f"{arm['name']}__f{fold}__s{seed}" / "metrics.json"
                rs.append(json.loads(p.read_text()))
                hashes[str(p.relative_to(HERE))] = sha(p)
        records[arm["name"]] = rs
    chosen = [a for a in candidates if a["family"] == "paper_rf"]
    choices = {}
    for method in ("FAN", "IB-RFF", "WaveMLP"):
        for recipe in ("official", "unified"):
            group = [a for a in candidates if a["method"] == method and a["recipe"] == recipe]
            arm = min(group, key=lambda a: (np.mean([r["source_val_ce"] for r in records[a["name"]]]), a["name"]))
            choices[f"{method}:{recipe}"] = arm["name"]
            chosen.append(arm)
    names = {a["name"] for a in chosen}
    names.update(a["init_arm"] for a in chosen if "init_arm" in a)
    allowed = [[name, fold, seed] for name in sorted(names) for fold in (0, 1) for seed in (201, 202, 203)]
    decision = dict(selection="mean source-validation CE over four fits; dev_val/confirm never used",
                    registration_sha256=sha(REG), source_result_hashes=hashes, choices=choices,
                    allowed_final_jobs=allowed, confirm_role="report only; repeatedly inspected")
    path.write_text(json.dumps(decision, indent=1) + "\n")
    print(json.dumps(choices, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("smoke", "source", "explore", "freeze_wifi", "freeze_shen", "final", "confirm"))
    parser.add_argument("--dataset", choices=("ManyRx", "ManySig"))
    parser.add_argument("--ib", action="store_true")
    args = parser.parse_args()
    arms = json.loads(REG.read_text())["arms"]
    if args.phase == "freeze_wifi":
        freeze_wifi(arms)
        return
    if args.phase == "freeze_shen":
        freeze_shen(arms)
        return
    if args.phase == "smoke":
        # Smoke all registered architectures/configurations, not scientific comparisons.
        chosen = [a for a in arms if bool(a.get("init_arm")) == args.ib]
        write_jobs("smoke", chosen, [0], [0], args.dataset, ib=args.ib)
    elif args.phase == "source":
        chosen = [a for a in arms if a.get("dataset") and bool(a.get("init_arm")) == args.ib]
        write_jobs("source", chosen, [2023], range(4), args.dataset, ib=args.ib)
    elif args.phase == "explore":
        chosen = [a for a in arms if not a.get("dataset") and bool(a.get("init_arm")) == args.ib]
        write_jobs("explore", chosen, [101, 102], range(2), ib=args.ib)
    else:
        decision = json.loads((HERE / ("decision_wifi.json" if args.phase == "final" else "decision_shen.json")).read_text())
        chosen = [a for a in arms if bool(a.get("init_arm")) == args.ib]
        if args.phase == "final":
            chosen = [a for a in chosen if a.get("dataset")]
            write_jobs("final", chosen, [2023, 2024, 2025], range(4), args.dataset, decision, args.ib)
        else:
            chosen = [a for a in chosen if not a.get("dataset")]
            write_jobs("confirm", chosen, [201, 202, 203], range(2), decision=decision, ib=args.ib)


if __name__ == "__main__":
    main()
