#!/usr/bin/env python3
"""Materialize bounded source jobs from the committed Stage-07 specification."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
from xa.explore_models import build_explore_model
from xa.models import n_params


def main():
    frontends = ["legacy", "time", "diff", "split", "stf", "filter", "frequency"]
    registration = dict(version=1, date="2026-10-25",
                        baseline="ebe322507d9d274265572d4f86545e93540a565d",
                        frontends=frontends + ["diff_stf", "diff_split"],
                        augmentations=["none", "phase", "shift", "cfo", "channel"],
                        bounds=dict(shift_samples=2, cfo_hz=2000, channel_log_gain=.1, channel_phase=.1),
                        source_seeds=[3101, 3102], final_seeds=[4101, 4102, 4103],
                        max_wifi_fits=280, max_combinations_per_fold=4,
                        phase_caps=dict(features=56, replicate=24, training=72,
                                        training_replicate=24, combine=32, smoke=14, final=72),
                        training_factors=["unified", "official", "balanced", "held_epoch",
                                          "phase", "shift", "cfo", "channel", "mild_noise"],
                        factor_gate=dict(mean_delta_pp=1, worst_rx_delta_pp=-2,
                                         positive_source_repeats=2, max_factors=2),
                        stf_guard="source audit median coherence >=0.7",
                        selection="per-dataset/fold source-RX only; no pooled cross-fold selection",
                        final_checkpoint="min source packet CE or source-frozen held-RX epoch; no target selection")
    audit = {(row["dataset"], row["fold"]): row
             for row in json.loads((HERE / "source_signal_audit.json").read_text())}
    jobs, excluded = [], []
    for dataset in ("ManyRx", "ManySig"):
        for fold in range(4):
            for frontend in frontends:
                if frontend == "stf" and audit[(dataset, fold)]["stf_coherence_median"] < .7:
                    excluded.append(dict(dataset=dataset, fold=fold, frontend=frontend,
                                         reason="source STF coherence below preregistered gate"))
                    continue
                job = dict(dataset=dataset, fold=fold, seed=3101, phase="features",
                           frontend=frontend, recipe="unified")
                job["id"] = f"{dataset}_{frontend}__f{fold}__s3101"
                jobs.append(job)
    registration["excluded"] = excluded
    registration["expected_params"] = {
        f"{dataset}:{frontend}:{fusion}": n_params(build_explore_model(dict(
            frontend=frontend, fusion=fusion, num_classes=10 if dataset == "ManyRx" else 6)))
        for dataset in ("ManyRx", "ManySig") for frontend in registration["frontends"]
        for fusion in ("pooled_concat", "gate")}
    registration["source_audit_sha256"] = hashlib.sha256((HERE / "source_signal_audit.json").read_bytes()).hexdigest()
    registration["wifi_code_manifest"] = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "lib").rglob("*.py"))}
    path = HERE / "registration.json"
    if path.exists() and json.loads(path.read_text()) != registration:
        raise SystemExit("refusing to replace a different registration")
    path.write_text(json.dumps(registration, indent=1) + "\n")
    (HERE / "jobs_features.json").write_text(json.dumps(jobs, indent=1) + "\n")
    smoke = []
    for frontend in frontends:
        job = dict(dataset="ManySig", fold=0, seed=0, phase="smoke",
                   frontend=frontend, recipe="unified", id=f"smoke_{frontend}")
        smoke.append(job)
    for option in ("phase", "shift", "cfo", "channel", "balanced", "held_epoch", "official"):
        job = dict(dataset="ManySig", fold=0, seed=0, phase="smoke",
                   frontend="time", recipe="official" if option == "official" else "unified",
                   id=f"smoke_train_{option}")
        if option in registration["augmentations"]:
            job["augmentation"] = option
        elif option == "balanced":
            job["balanced"] = True
        elif option == "held_epoch":
            job["checkpoint"] = "held_epoch"
        smoke.append(job)
    (HERE / "jobs_smoke.json").write_text(json.dumps(smoke, indent=1) + "\n")
    print(json.dumps(dict(source_jobs=len(jobs), excluded=excluded)))


if __name__ == "__main__":
    main()
