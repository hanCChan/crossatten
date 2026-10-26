"""Pre-registered source-only mechanism gate; no attribution/target ranking."""
import json
import numpy as np
from training import HERE, sha, require_committed, write_once, identity


def gate(candidates, controls):
    keys = lambda r: (r["job"]["group"], r["job"]["seed"])
    a, b = {keys(r): r for r in candidates}, {keys(r): r for r in controls}
    if len(a) != 4 or set(a) != set(b):
        raise ValueError("missing paired source groups/seeds")
    seeds, groups, rxs, epochs, scores = {}, {}, {}, [], []
    for key, row in a.items():
        ref = b[key]
        if set(row["entry"]["per_rx"]) != set(ref["entry"]["per_rx"]):
            raise ValueError("held RX differ")
        delta = 100 * (row["entry"]["summary"]["acc_mean"] - ref["entry"]["summary"]["acc_mean"])
        groups.setdefault(key[0], []).append(delta)
        seeds.setdefault(key[1], []).append(delta)
        for rx, entry in row["entry"]["per_rx"].items():
            rxs.setdefault(rx, []).append(100 * (entry["acc"] - ref["entry"]["per_rx"][rx]["acc"]))
        epochs.append(row["entry"]["epoch"])
        scores.append(100 * row["entry"]["summary"]["acc_mean"])
    mean = float(np.mean([v for values in seeds.values() for v in values]))
    worst = min(float(np.mean(v)) for v in rxs.values())
    return dict(supported=mean >= .5 and worst >= -3 and
                all(np.mean(v) > 0 for v in seeds.values()) and all(np.mean(v) >= 0 for v in groups.values()),
                mean_delta_pp=mean, worst_rx_delta_pp=worst, epochs=epochs,
                mean_acc_pct=float(np.mean(scores)),
                seed_deltas={str(k): float(np.mean(v)) for k, v in seeds.items()},
                group_deltas={k: float(np.mean(v)) for k, v in groups.items()})


def main():
    reg = json.loads((HERE / "registration.json").read_text())
    require_committed(HERE / "registration.json")
    source_jobs = json.loads((HERE / "jobs_source.json").read_text())
    rows, evidence = [], {}
    for job in source_jobs:
        path = HERE / "runs/source" / job["id"] / "metrics.json"
        require_committed(path)
        result = json.loads(path.read_text())
        if any(k in result["selected"] for k in ("CR", "CRD")):
            raise ValueError("target outcome in source decision")
        evidence[str(path.relative_to(HERE))] = sha(path)
        for rule in reg["source_rules"]:
            rows.append(dict(job=job, rule=rule, entry=result["selected"][rule]))
    choices = {}
    for dataset in ("ManyRx", "ManySig"):
        choices[dataset] = {}
        for feature in ("centered", "curvature"):
            for rule in reg["source_rules"]:
                local = {}
                for fold in range(4):
                    cells = [r for r in rows if r["job"]["dataset"] == dataset and
                             r["job"]["fold"] == fold and r["rule"] == rule]
                    local[str(fold)] = gate([r for r in cells if r["job"]["feature"] == feature],
                                           [r for r in cells if r["job"]["feature"] == "first"])
                choices[dataset][f"{feature}:{rule}"] = dict(
                    admitted=sum(r["supported"] for r in local.values()) >= 3, folds=local,
                    mean_gain=float(np.mean([r["mean_delta_pp"] for r in local.values()])),
                    worst=min(r["worst_rx_delta_pp"] for r in local.values()))
    universal = [key for key in choices["ManyRx"]
                 if all(choices[d][key]["admitted"] for d in choices)]
    selected = {}
    if universal:
        key = min(universal, key=lambda k: (-np.mean([choices[d][k]["mean_gain"] for d in choices]),
                                           -min(choices[d][k]["worst"] for d in choices), k))
        selected = {d: key for d in choices}
    else:
        for dataset, values in choices.items():
            accepted = [k for k, v in values.items() if v["admitted"]]
            if accepted:
                selected[dataset] = min(accepted, key=lambda k: (-values[k]["mean_gain"], -values[k]["worst"], k))
    jobs = []
    for dataset, key in selected.items():
        feature, rule = key.split(":")
        for fold in range(4):
            epochs = int(sorted(choices[dataset][key]["folds"][str(fold)]["epochs"])[2])
            for seed in reg["final_seeds"]:
                for f in (feature, "first"):
                    job = dict(phase="final", dataset=dataset, fold=fold, seed=seed,
                               feature=f, policy="fixed", epochs=epochs, rule=rule.split("_")[0])
                    job["id"] = identity(job)
                    jobs.append(job)
    attribution = json.loads((HERE / "jobs_attribution.json").read_text())
    decision = dict(registration_sha256=sha(HERE / "registration.json"), evidence=evidence,
                    choices=choices, universal=bool(universal), selected=selected, jobs=jobs,
                    target_jobs=attribution + jobs, no_target_selection=True)
    write_once(HERE / "decision_final.json", decision)
    write_once(HERE / "jobs_final.json", jobs)
    print(json.dumps(dict(universal=bool(universal), selected=selected, final_fits=len(jobs))))


if __name__ == "__main__":
    main()
