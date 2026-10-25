#!/usr/bin/env python3
"""Frozen fold-local dual-source selection; never read final or target metrics."""
import argparse
import importlib.util
import json

import numpy as np
from training import HERE, ROOT, RULES, job_id, require_committed, sha, write_once


def load_phase(phase):
    spec = importlib.util.spec_from_file_location("s08_audit", HERE / "08-4-audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    audit.audit_phase(phase)
    jobs = json.loads((HERE / f"jobs_{phase}.json").read_text())
    rows, evidence = [], {}
    for job in jobs:
        path = HERE / "runs" / phase / job["id"] / "metrics.json"
        result = json.loads(path.read_text())
        if any(k in result["selected"] for k in ("CR", "CRD")):
            raise ValueError("target metric in source decision")
        for rule in RULES:
            rows.append(dict(job=job, rule=rule, entry=result["selected"][rule]))
        evidence[str(path.relative_to(HERE))] = sha(path)
    return rows, evidence


def candidate_key(row):
    job = row["job"]
    return f"{job['condition']}:{job['feature']}:{row['rule']}"


def paired_gate(rows, controls):
    keys = lambda r: (r["job"]["group"], r["job"]["seed"])
    a, b = {keys(r): r for r in rows}, {keys(r): r for r in controls}
    if len(a) != 4 or set(a) != set(b):
        raise ValueError("missing paired seed/source-group cells")
    seed_deltas, group_deltas, rx_deltas = {}, {}, {}
    scores, ces, epochs, worst = [], [], [], []
    for key, row in a.items():
        control = b[key]
        per, ref = row["entry"]["per_rx"], control["entry"]["per_rx"]
        if set(per) != set(ref):
            raise ValueError("paired held RX differ")
        delta = 100 * (row["entry"]["summary"]["acc_mean"] -
                       control["entry"]["summary"]["acc_mean"])
        group, seed = key
        seed_deltas.setdefault(seed, []).append(delta)
        group_deltas.setdefault(group, []).append(delta)
        for rx in per:
            rx_deltas.setdefault(rx, []).append(100 * (per[rx]["acc"] - ref[rx]["acc"]))
            worst.append(per[rx]["acc"])
        scores.append(row["entry"]["summary"]["acc_mean"])
        ces.append(row["entry"]["summary"]["ce_mean"])
        epochs.append(row["entry"]["epoch"])
    mean_delta = float(np.mean([v for x in seed_deltas.values() for v in x]))
    min_rx_delta = min(float(np.mean(v)) for v in rx_deltas.values())
    return dict(
        supported=all(np.mean(v) > 0 for v in seed_deltas.values()) and
        mean_delta >= 1 and all(np.mean(v) >= 0 for v in group_deltas.values()) and min_rx_delta >= -3,
        mean_delta_pp=mean_delta, worst_rx_delta_pp=min_rx_delta,
        seed_delta_pp={str(k): float(np.mean(v)) for k, v in seed_deltas.items()},
        group_delta_pp={str(k): float(np.mean(v)) for k, v in group_deltas.items()},
        mean_acc=float(np.mean(scores)), worst_acc=min(worst), mean_ce=float(np.mean(ces)),
        epochs=epochs)


def decisions_for(dataset, rows):
    decisions = {}
    for fold in range(4):
        local = [r for r in rows if r["job"]["dataset"] == dataset and r["job"]["fold"] == fold]
        controls = [r for r in local if candidate_key(r) == "official:frequency:raw_packet"]
        candidates = {}
        for name in sorted({candidate_key(r) for r in local}):
            selected = [r for r in local if candidate_key(r) == name]
            gate = paired_gate(selected, controls)
            if name.split(":")[1] == "phase":
                null = [r for r in local if candidate_key(r) == name.replace(":phase:", ":phase_null:")]
                capacity = paired_gate(selected, null)
                gate["phase_over_null_pp"] = capacity["mean_delta_pp"]
                gate["supported"] &= capacity["mean_delta_pp"] >= .5
            if ":phase_null:" in name:
                gate["supported"] = False
            candidates[name] = gate
        accepted = [name for name, gate in candidates.items() if gate["supported"]]
        chosen = min(accepted, key=lambda k: (
            -candidates[k]["mean_acc"], -candidates[k]["worst_acc"],
            candidates[k]["mean_ce"], k)) if accepted else "official:frequency:raw_packet"
        condition, feature, rule = chosen.split(":")
        selection = dict(condition=condition, feature=feature, rule=rule)
        if rule.endswith("held"):
            selection["fixed_epochs"] = int(sorted(candidates[chosen]["epochs"])[2])
        decisions[str(fold)] = dict(chosen=chosen, supported=bool(accepted),
                                   candidates=candidates, selection=selection)
    return decisions


def queue(phase, dataset, conditions, features=("frequency",)):
    jobs = []
    for fold in range(4):
        for group in ("first", "last"):
            for seed in (6101, 6102):
                for condition in conditions:
                    for feature in features:
                        job = dict(dataset=dataset, fold=fold, group=group, seed=seed,
                                   condition=condition, feature=feature, phase=phase)
                        job["id"] = job_id(job)
                        jobs.append(job)
    return jobs


def freeze_final(decisions, evidence):
    jobs = []
    admitted = {}
    for dataset, local in decisions.items():
        if sum(v["supported"] for v in local.values()) < 3:
            admitted[dataset] = False
            continue
        admitted[dataset] = True
        for fold in range(4):
            for seed in (7101, 7102, 7103):
                for variant in ("candidate", "control"):
                    selection = local[str(fold)]["selection"] if variant == "candidate" else dict(
                        condition="official", feature="frequency", rule="raw_packet")
                    job = dict(selection, dataset=dataset, fold=fold, seed=seed,
                               phase="final", variant=variant)
                    job["id"] = job_id(job)
                    jobs.append(job)
    decision = dict(registration_sha256=sha(HERE / "registration.json"),
                    decisions=decisions, admitted=admitted, evidence=evidence, jobs=jobs,
                    no_target_selection=True)
    write_once(HERE / "decision_final.json", decision)
    write_once(HERE / "jobs_final.json", jobs)
    return admitted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("screen", "phase", "manyrx"))
    step = parser.parse_args().step
    require_committed(HERE / "registration.json")
    rows, evidence = load_phase("screen")
    decisions = {"ManySig": decisions_for("ManySig", rows)}
    if step in ("phase", "manyrx") and (HERE / "jobs_phase.json").exists():
        phase_rows, phase_evidence = load_phase("phase")
        rows += phase_rows
        evidence.update(phase_evidence)
        decisions["ManySig"] = decisions_for("ManySig", rows)
    supported = sum(v["supported"] for v in decisions["ManySig"].values())
    next_phase = "stop"
    jobs = []
    if step == "screen" and supported < 3:
        import csv
        spectral = list(csv.DictReader((HERE / "source_spectral_diagnostics.csv").open()))
        eligible = all(float(r["median_reliable_pair_fraction"]) >= .5
                       for r in spectral if r["dataset"] == "ManySig")
        if eligible:
            next_phase = "phase"
            jobs = queue("phase", "ManySig", ["official"], ["phase", "phase_null"])
    if next_phase != "phase" and step in ("screen", "phase") and supported >= 3:
        next_phase = "manyrx"
        jobs = queue("manyrx", "ManyRx", ["official", "official_awgn"])
    if step == "manyrx":
        more, hashes = load_phase("manyrx")
        evidence.update(hashes)
        decisions["ManyRx"] = decisions_for("ManyRx", more)
        admitted = freeze_final(decisions, evidence)
        next_phase = "final" if any(admitted.values()) else "stop"
    elif next_phase == "stop":
        freeze_final(decisions, evidence)
    if jobs:
        write_once(HERE / f"jobs_{next_phase}.json", jobs)
    summary = dict(step=step, supported_manysig_folds=supported, decisions=decisions,
                   evidence=evidence, next_phase=next_phase, jobs=jobs, no_target_metrics=True)
    write_once(HERE / f"decision_after_{step}.json", summary)
    print(json.dumps(dict(step=step, supported_manysig_folds=supported, next_phase=next_phase,
                          jobs=len(jobs))), flush=True)


if __name__ == "__main__":
    main()
