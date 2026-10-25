#!/usr/bin/env python3
"""Advance a bounded source-only phase; never load target metrics or raw IQ."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
from xa.paper_train import require_committed, sha


def write_once(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"refusing changed frozen evidence: {path.name}")
        return
    path.write_text(json.dumps(value, indent=1) + "\n")


def load_phase(phase):
    jobs = json.loads((HERE / f"jobs_{phase}.json").read_text())
    rows = []
    for job in jobs:
        out = HERE / "runs" / phase / job["id"]
        path = out / "metrics.json"
        if not path.is_file():
            raise ValueError(f"unfinished source run: {job['id']}")
        result = json.loads(path.read_text())
        if result["job"] != job or any(k in result["selected"] for k in ("CR", "CRD")):
            raise ValueError("source identity or target contamination")
        if result["registration_sha256"] != sha(HERE / "registration.json"):
            raise ValueError("source registration mismatch")
        if sha(out / "model.pt") != result["weights_sha256"]:
            raise ValueError("source checkpoint changed")
        entry = result["selected"]["held_source"]
        rows.append(dict(job=job, result=result, entry=entry,
                         acc=entry["summary"]["acc_mean"], ce=entry["summary"]["ce_mean"],
                         evidence=path.relative_to(HERE).as_posix(), sha256=sha(path)))
    return rows


def rank(rows):
    return sorted(rows, key=lambda r: (-r["acc"], r["ce"], r["job"]["id"]))


def job_id(job):
    factor = job.get("factor", job["frontend"])
    return f"{job['dataset']}_{factor}__f{job['fold']}__s{job['seed']}"


def queue_copy(row, phase, seed):
    job = dict(row["job"], phase=phase, seed=seed)
    job["id"] = job_id(job)
    return job


def supported(first, second, controls):
    deltas, worst = [], []
    for row, control in zip((first, second), controls):
        if set(row["entry"]["per_rx"]) != set(control["entry"]["per_rx"]):
            raise ValueError("held receiver mismatch")
        deltas.append(100 * (row["acc"] - control["acc"]))
        worst.extend(100 * (v["acc"] - control["entry"]["per_rx"][rx]["acc"])
                     for rx, v in row["entry"]["per_rx"].items())
    return all(d > 0 for d in deltas) and sum(deltas) / 2 >= 1 and min(worst) >= -2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("features", "replicate", "training", "training_replicate", "combine"))
    args = parser.parse_args()
    require_committed(HERE / "registration.json")
    steps = ["features", "replicate", "training", "training_replicate", "combine"]
    previous = {phase: load_phase(phase) for phase in steps[:steps.index(args.step) + 1]}
    evidence = {r["evidence"]: r["sha256"] for rows in previous.values() for r in rows}
    decisions, jobs = {}, []
    for dataset in ("ManyRx", "ManySig"):
        for fold in range(4):
            key = f"{dataset}:{fold}"
            local = {p: [r for r in rows if r["job"]["dataset"] == dataset and r["job"]["fold"] == fold]
                     for p, rows in previous.items()}
            if args.step == "features":
                top = rank(local["features"])[:2]
                control = next(r for r in local["features"] if r["job"]["frontend"] == "legacy")
                selected = {r["job"]["frontend"]: r for r in top + [control]}
                jobs.extend(queue_copy(r, "replicate", 3102) for r in selected.values())
                decisions[key] = list(selected)
                continue
            repeated = {r["job"]["frontend"]: r for r in local["replicate"]}
            first = {r["job"]["frontend"]: r for r in local["features"]}
            selected_frontend = min(repeated, key=lambda f: (
                -(first[f]["acc"] + repeated[f]["acc"]) / 2,
                (first[f]["ce"] + repeated[f]["ce"]) / 2, f))
            base = dict(dataset=dataset, fold=fold, frontend=selected_frontend, recipe="unified")
            if args.step == "replicate":
                decisions[key] = dict(frontend=selected_frontend, source_only=True)
                for factor in ("unified", "official", "balanced", "held_epoch",
                               "phase", "shift", "cfo", "channel", "mild_noise"):
                    job = dict(base, phase="training", seed=3101, factor=factor)
                    if factor == "official":
                        job["recipe"] = "official"
                    elif factor == "balanced":
                        job["balanced"] = True
                    elif factor == "held_epoch":
                        job["checkpoint"] = "held_epoch"
                    elif factor == "mild_noise":
                        job["noise"] = "mild"
                    elif factor != "unified":
                        job["augmentation"] = factor
                    job["id"] = job_id(job)
                    jobs.append(job)
                continue
            if args.step == "training":
                top = rank(local["training"])[:2]
                control = next(r for r in local["training"] if r["job"]["factor"] == "unified")
                selected = {r["job"]["factor"]: r for r in top + [control]}
                jobs.extend(queue_copy(r, "training_replicate", 3102) for r in selected.values())
                decisions[key] = list(selected)
                continue
            train1 = {r["job"]["factor"]: r for r in local["training"]}
            train2 = {r["job"]["factor"]: r for r in local["training_replicate"]}
            factors = sorted((f for f in train2 if f != "unified"
                              and supported(train1[f], train2[f], [train1["unified"], train2["unified"]])),
                             key=lambda f: -(train1[f]["acc"] + train2[f]["acc"]))[:2]
            strongest = min(train2, key=lambda f: (-(train1[f]["acc"] + train2[f]["acc"]),
                                                   train1[f]["ce"] + train2[f]["ce"], f))
            decisions[key] = dict(frontend=selected_frontend, supported_factors=factors,
                                 strongest_single=strongest)
            if args.step == "training_replicate":
                candidates = [dict(train1[strongest]["job"], factor="single_frozen")]
                if len(factors) == 2:
                    one, two = train1[factors[0]]["job"], train1[factors[1]]["job"]
                    # Two physical augmentations and conflicting recipes are not combined.
                    shared = ("recipe", "augmentation", "checkpoint", "noise", "balanced")
                    conflict = any(k in one and k in two and one[k] != two[k]
                                   for k in shared if k != "recipe")
                    if not conflict and "official" not in factors:
                        combo = dict(base, factor="two_factors")
                        for part in (one, two):
                            combo.update({k: v for k, v in part.items() if k in shared})
                        candidates.append(combo)
                if selected_frontend in ("split", "diff_split"):
                    candidates.append(dict(candidates[0], factor="single_gate", fusion="gate"))
                for candidate in candidates:
                    candidate.update(phase="combine", seed=3101)
                    candidate["id"] = job_id(candidate)
                    jobs.append(candidate)
                continue
            combo = rank(local["combine"])[0]
            pipelines = {
                "source_frontend": dict(base),
                "source_training": {k: v for k, v in train1[strongest]["job"].items()
                                    if k not in ("phase", "seed", "factor", "id")},
                "source_combination": {k: v for k, v in combo["job"].items()
                                       if k not in ("phase", "seed", "factor", "id")},
            }
            for name, pipeline in pipelines.items():
                if pipeline.get("checkpoint") == "held_epoch":
                    epochs = [r["result"]["held_best_epoch"] for p in ("training", "training_replicate", "combine")
                              for r in local[p] if r["job"].get("checkpoint") == "held_epoch"
                              and all(r["job"].get(k) == pipeline.get(k)
                                      for k in ("frontend", "recipe", "augmentation", "noise", "balanced"))]
                    if not epochs:
                        raise ValueError("missing held-RX source budget")
                    pipeline["epochs"] = int(sorted(epochs)[len(epochs) // 2])
                    pipeline["checkpoint"] = "last"
                for seed in (4101, 4102, 4103):
                    job = dict(pipeline, dataset=dataset, fold=fold, phase="final",
                               pipeline=name, seed=seed, factor=name)
                    job["id"] = job_id(job)
                    jobs.append(job)
    target_phase = dict(features="replicate", replicate="training", training="training_replicate",
                        training_replicate="combine", combine="final")[args.step]
    caps = json.loads((HERE / "registration.json").read_text())["phase_caps"]
    if len(jobs) > caps[target_phase]:
        raise ValueError("adaptive budget exceeded")
    value = dict(step=args.step, registration_sha256=sha(HERE / "registration.json"),
                 decisions=decisions, evidence=evidence, jobs=jobs)
    write_once(HERE / f"decision_{target_phase}.json", value)
    write_once(HERE / f"jobs_{target_phase}.json", jobs)
    with (HERE / f"source_{args.step}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "fold", "seed", "id", "held_acc", "held_ce"])
        writer.writeheader()
        for row in previous[args.step]:
            job = row["job"]
            writer.writerow(dict(dataset=job["dataset"], fold=job["fold"], seed=job["seed"],
                                 id=job["id"], held_acc=row["acc"], held_ce=row["ce"]))
    print(json.dumps(dict(next_phase=target_phase, jobs=len(jobs), source_decisions=decisions)), flush=True)


if __name__ == "__main__":
    main()
