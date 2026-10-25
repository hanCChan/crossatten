#!/usr/bin/env python3
"""Create the bounded S08 job plan, before any training or target reads."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from training import ROOT, build_model, job_id, n_params, sha, write_once


def main():
    registration = dict(
        baseline="74f594312949f18613030d59743dba88fc20eca7", date="2026-10-26",
        source_seeds=[6101, 6102], final_seeds=[7101, 7102, 7103],
        conditions=["official", "official_awgn", "official_awgn_full200"],
        rules=["raw_packet", "ema_packet", "raw_held", "ema_held"],
        source_groups=["first", "last"], adjacent_threshold=.05,
        phase_audit_gate=dict(median_reliable_pair_fraction=.5),
        gate=dict(mean_delta_pp=1, both_seeds_positive=True,
                  both_group_means_nonnegative=True, worst_rx_seedmean_delta_pp=-3,
                  min_supported_folds=3),
        max_formal_fits=160, caps=dict(screen=48, phase=32, manyrx=32, final=48, smoke=5),
        selection="same fold-local two-source-group gate/rank; no other fold can choose its candidate",
        ema="one-epoch half-life; parameters only; clean fit-training BN recalibration each epoch",
        expected_params={f"{dataset}:{feature}": n_params(build_model(dataset, feature))
                         for dataset in ("ManyRx", "ManySig")
                         for feature in ("frequency", "phase", "phase_null")})
    code = list((ROOT / "lib").rglob("*.py")) + [
        HERE / name for name in ("training.py", "08-1-register.py", "08-2-launch.py",
                                "08-3-select.py", "08-4-audit.py")]
    registration["code_manifest"] = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(code)}
    write_once(HERE / "registration.json", registration)
    jobs = []
    for fold in range(4):
        for group in registration["source_groups"]:
            for seed in registration["source_seeds"]:
                for condition in registration["conditions"]:
                    job = dict(dataset="ManySig", fold=fold, group=group, seed=seed,
                               condition=condition, feature="frequency", phase="screen")
                    job["id"] = job_id(job)
                    jobs.append(job)
    write_once(HERE / "jobs_screen.json", jobs)
    smoke = []
    for i, (condition, feature) in enumerate([
            ("official", "frequency"), ("official_awgn", "frequency"),
            ("official_awgn_full200", "frequency"),
            ("official", "phase"), ("official", "phase_null")]):
        job = dict(dataset="ManySig", fold=0, group="first", seed=0,
                   condition=condition, feature=feature, phase="smoke", variant=f"smoke{i}")
        job["id"] = job_id(job)
        smoke.append(job)
    write_once(HERE / "jobs_smoke.json", smoke)
    print(json.dumps(dict(screen_jobs=len(jobs), params=registration["expected_params"])))


if __name__ == "__main__":
    main()
