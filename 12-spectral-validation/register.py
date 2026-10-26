"""Write-once S12-A/B protocol, without loading any historical checkpoint."""
import argparse
import json
import subprocess
import numpy as np
from core import HERE, ROOT, once, save, identity, manifest
from spectral import paired_model

DATASETS = ("ManyRx", "ManySig")


def job(phase, dataset, fold, feature, seed=8101, group="first"):
    result = dict(phase=phase, dataset=dataset, fold=fold, feature=feature,
                  seed=seed, group=group, epochs=2 if phase == "smoke" else 40,
                  policy="source")
    result["id"] = identity(result)
    return result


def protocol():
    pilot = [job("pilot", d, f, kind) for d in DATASETS
             for f in range(4) for kind in ("F1", "F2")]
    smoke = [job("smoke", d, 0, kind) for d in DATASETS for kind in ("F1", "F2")]
    paths = list(HERE.glob("*.py")) + list((ROOT / "lib").rglob("*.py")) + [
        ROOT / "11-spectral-generalization/core.py",
        ROOT / "11-spectral-generalization/spectral.py",
        ROOT / "09-phase-invariance/training.py",
        ROOT / "09-phase-invariance/phase_features.py",
        ROOT / "08-wisig-training/training.py",
        ROOT / "10-tifs-validation/preparation.json"]
    reg = dict(stage="S12 Spectral Validation", date="2026-10-26",
        base_commit="db126640", historical_evidence="accepted as correct per user",
        historical_weights="do not locate, download, hash, load or replay",
        historical_protocol="S11 remains STOP_MECHANISM_GATE; never relabel it GO",
        code_manifest=manifest(paths),
        expected_params={f"{d}:{kind}": sum(p.numel() for p in paired_model(
            d, kind, 0, np.ones(256, np.float32))[0].parameters())
            for d in DATASETS for kind in ("hard", "F1", "F2")},
        caps=dict(smoke=4, pilot=16, confirmation_candidates=32,
                  confirmation_fresh_hard=32),
        primary_rule="raw_packet", prior_minimum_cell_packets=10,
        prior_minimum_rx_per_tx=3, prior_dates=[1, 2],
        prior_partition="complete 70% fit partition; excludes packet-val and held RX",
        smoke="same full-fit prior as pilot fold0/first/8101; only network samples/epochs reduced",
        smoke_budgets=dict(train=512, val=512, per_held_rx=64),
        smoke_sampling="deterministic round-robin TX/RX/day cells; fail if coverage cannot fit",
        smoke_accuracy="engineering-only, never candidate selection or efficacy evidence",
        mechanism="descriptive J, unit geometry, held centroid and worst RX; "
                  "raw retention kept, no raw-scatter or new proxy veto",
        negative_controls=dict(uniform=.75, shuffled="frequency permutation of F2",
            tx_permuted="packet TX permutation within each fit RX/day", seed=12026),
        negative_control_scope="source diagnostics only; no added network fits",
        pilot=dict(datasets=list(DATASETS), folds=list(range(4)), groups=["first"],
            seeds=[8101], candidates=["F1", "F2"], new_fits=16,
            hard="reuse eight S09 first-feature source metric/prediction records; no old weights",
            pairing="same S09 split, shared initialization, data order, 40 epochs, raw_packet"),
        source_gate=dict(manyrx_min_gain_pp=1., manyrx_positive_folds=3,
            worst_rx_min_delta_pp=-2., manysig_min_gain_pp=-.5,
            manysig_nonnegative_folds=3),
        selection="only candidates passing every frozen source criterion; "
                  "maximize ManyRx RX-equal fold mean gain, tie F1; none passing means STOP",
        confirmation=dict(conditional=True, datasets=list(DATASETS), folds=list(range(4)),
            groups=["first", "last"], seeds=[12801, 12802], selected_candidates=1,
            new_candidate_fits=32, fresh_paired_hard_fits=32,
            hard_note="new seeds require new Hard controls, not historical-seed pseudo-pairing",
            gate="same source thresholds plus positive ManyRx mean in each group/seed stratum",
            inference="pilot excluded from confirmation intervals and significance claims",
            admission="separate write-once source decision and jobs, committed before any fit"),
        target_policy="all target reads disabled in S12-A/B; "
                      "conditional RX/day and CRD stages need separate preregistration",
        retries="no automatic retry or replacement; failure halts progression",
        status="PREPARED_NO_DATA_OR_TRAINING_RUN")
    return reg, smoke, pilot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-code-before-commit", action="store_true")
    args = parser.parse_args()
    reg, smoke, pilot = protocol()
    if args.refresh_code_before_commit:
        tracked = subprocess.run(["git", "cat-file", "-e",
                                  "HEAD:12-spectral-validation/registration.json"],
                                 cwd=ROOT, capture_output=True)
        if tracked.returncode == 0 or (HERE / "runs").exists() or (HERE / "access.jsonl").exists():
            raise ValueError("cannot refresh after commit or data/fit access")
        previous = json.loads((HERE / "registration.json").read_text())
        if {k: v for k, v in previous.items() if k != "code_manifest"} != {
                k: v for k, v in reg.items() if k != "code_manifest"}:
            raise ValueError("only pre-commit code manifest refresh is allowed")
        save(HERE / "registration.json", reg)
    once(HERE / "registration.json", reg)
    once(HERE / "jobs_smoke.json", smoke)
    once(HERE / "jobs_pilot.json", pilot)
    baselines = []
    for dataset in DATASETS:
        for fold in range(4):
            name = f"{dataset}__f{fold}__first__s8101__first__source"
            baselines.append(dict(dataset=dataset, fold=fold, group="first", seed=8101,
                feature="first", rule="raw_packet",
                path=f"09-phase-invariance/runs/source/{name}",
                accepted_without_weight_audit=True))
    once(HERE / "historical_hard.json", baselines)
    print("S12 prepared: 4 smoke, 16 pilot; no data read, no old weights accessed.")


if __name__ == "__main__":
    main()
