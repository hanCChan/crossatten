"""Freeze exactly one candidate, one source rule and finite independent baseline."""
import json
from stage import HERE, ROOT, ASK_DIR, ASK_SHA, save, once, sha, identity
from features import paired_model


def job(phase, dataset, fold, seed, feature, group="all", epochs=200):
    policy = "source" if phase == "source" else "packet"
    j = dict(phase=phase, dataset=dataset, fold=fold, seed=seed, feature=feature,
             group=group, epochs=epochs, policy=policy)
    j["id"] = identity(j)
    return j


def main():
    sources = [job("source", d, f, seed, "soft", group, 40)
        for d in ("ManyRx", "ManySig") for f in range(4)
        for group in ("first", "last") for seed in (8101, 8102)]
    asks = [job("ask", d, f, seed, "ask") for d in ("ManyRx", "ManySig")
            for f in range(4) for seed in (2023, 2024, 2025)]
    smokes = [job("smoke", d, 0, 0, feature, "first", 2)
              for d in ("ManyRx", "ManySig") for feature in ("soft", "ask")]
    paths = list(HERE.glob("*.py")) + [
        ROOT / "09-phase-invariance/training.py", ROOT / "09-phase-invariance/phase_features.py",
        ROOT / "08-wisig-training/training.py"] + list((ROOT / "lib").rglob("*.py"))
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}
    params = {f"{d}:{kind}": sum(p.numel() for p in paired_model(d, kind, 0)[0].parameters())
              for d in ("ManyRx", "ManySig") for kind in ("hard", "soft", "unweighted", "no_hstu")}
    controls = {}
    for j in sources:
        old = dict(j, feature="first")
        old["id"] = identity(old)
        base = ROOT / "09-phase-invariance/runs/source" / old["id"]
        for name in ("config.json", "metrics.json", "history.csv", "preds_raw_packet.npz"):
            p = base / name
            controls[str(p.relative_to(ROOT))] = sha(p)
    reg = dict(stage="S10 Reliability-Aware Cross-Receiver Validation", date="2026-10-26",
        base_commit="1cb9c8f9891bee4973d870b83f156ca06e614125",
        candidate="soft only; tau=0.05; identical first 3 channels, HStem/HSTU and losses",
        code_manifest=manifest, upstream_commit=ASK_SHA,
        upstream_files={str(p.relative_to(ASK_DIR)): sha(p) for p in ASK_DIR.rglob("*.py")},
        preparation_sha256=sha(HERE / "preparation.json"), historical_controls=controls,
        expected_params=params, source_rule="raw_packet",
        gate=dict(manyrx_positive_folds=3, manyrx_mean_gain_pp=1., min_rx_delta_pp=-2.,
                  manyrx_group_and_seed_means_strictly_positive=True,
                  manysig_mean_at_least_zero=True, manysig_nonnegative_folds=3,
                  manysig_group_and_seed_means_at_least_zero=True),
        gate_unit="fold-equal averages of 2 groups x 2 seeds; RX minimum averaged over fold/group/seed",
        final_policy="same raw minimum source packet CE/200 epochs/early stop10 for both datasets",
        manysig_hard_reuse="S09 first:packet seeds9101-9103 only after exact config/hash match",
        final_seeds=[9101, 9102, 9103], ask_seeds=[2023, 2024, 2025],
        caps=dict(source=32, ask=24, final=36, ablation=24, smoke=4),
        total_formal_cap=116, retries="none; any failed fit stops progression; no silent replacement",
        ablation_rule="only if both confirmed soft-v-hard CRD means improve; 2 datasets x4folds x3seeds, "
                      "one No-HSTU/No-reliability arm each; receiver intervals must be reported",
        robustness=dict(awgn_db=[0, 5, 10, 20, 30], shifts=[-4, -2, 2, 4],
            multipath=[[2, .3], [4, .3], [8, .3], [2, .6], [4, .6], [8, .6]],
            reflection_phase_rad=0., packets_per_tx_rx_day=20, seed=10101,
            scope="all surviving S10 checkpoints, CRD only; no training/selection; zero padded not circular"),
        no_target_selection=True, historical_targets_previously_seen=True,
        goals_only=dict(ManyRx_CRD=90., ManySig_CRD=93., Shen_preserved=97.48))
    once(HERE / "registration.json", reg)
    for phase, jobs in (("source", sources), ("ask", asks), ("smoke", smokes)):
        once(HERE / f"jobs_{phase}.json", jobs)
    print("Registered 32 soft source +24 official ASKNet +4 bounded smoke fits.")


if __name__ == "__main__":
    main()
