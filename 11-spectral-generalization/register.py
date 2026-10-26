"""Finite S11 protocol, frozen before any source outcome or new fit."""
from core import HERE, ROOT, manifest, once, identity
from spectral import paired_model,center_weight
import numpy as np


def job(phase, dataset, fold, group, seed, feature="hard", epochs=40):
    j = dict(phase=phase, dataset=dataset, fold=fold, group=group,
             seed=seed, feature=feature, epochs=epochs,
             policy="packet" if phase == "final" else "source")
    j["id"] = identity(j)
    return j


def main():
    restore = [job("restore", d, 0, "last", 8101)
               for d in ("ManyRx", "ManySig")]
    smokes = [job("smoke", d, 0, "first", 0, kind, 2)
              for d in ("ManyRx", "ManySig") for kind in ("F1", "F2")]
    paths = list(HERE.glob("*.py")) + list((ROOT / "lib").rglob("*.py")) + [
        ROOT / "09-phase-invariance/training.py", ROOT / "09-phase-invariance/phase_features.py",
        ROOT / "08-wisig-training/training.py", ROOT / "10-tifs-validation/preparation.json"]
    reg = dict(stage="S11 Identity-Preserving Spectral Generalization", date="2026-10-26",
        base_commit="a4a792f8d672d7c866c19b35bb1b1f6785632ee5",
        code_manifest=manifest(paths),
        expected_params={f"{d}:{kind}":sum(p.numel() for p in paired_model(
            d,kind,0,np.ones(256,dtype=np.float32))[0].parameters())
            for d in ("ManyRx","ManySig") for kind in ("hard","F1","F2")},
        source_seeds=[8101, 8102], groups=["first", "last"], final_seeds=[11101, 11102, 11103],
        candidates=["F1", "F2"], primary_rule="raw_packet",
        F1="k32 fixed center FFT bins97:160; original five views masked, adjacent phase pair both bins inside",
        F2="fit-packet-only five-feature equal-TX/RX/day centroids; normalized Cartesian complex/phase groups; "
           "B=max(between TX scatter - cell mean uncertainty,0); raw R/D and uncertainty denominator; "
           "reflect 5-bin score smoothing; c=median positive score; g=.5+.5*S/(S+c)",
        fit_prior_only=True, minimum_cell_packets=10, minimum_rx_per_tx=3,
        day1_prior="same TX/RX/uncertainty estimator using dates(1,); D identically0. No held day2 statistics allowed.",
        physics_admission=dict(min_half_weight_correlation=.5,
            min_manyrx_positive_folds=3, min_manyrx_centroid_gain_pp=.5,
            min_manysig_centroid_gain_pp=-.5,
            minimum_tx_centroid_scatter_retention=.65,
            max_drx_dtx_ratio_change=1.05),
        route_selection="source-only nearest-centroid probes of F1/F2 before fits, "
                        "admit at most one; prioritize F2 if both have mechanism support",
        source_gate=dict(manyrx_min_gain_pp=1., manyrx_positive_folds=3,
            worst_rx_min_delta_pp=-2., manysig_min_gain_pp=-.5,
            manysig_nonnegative_folds=3, manyrx_group_seed_means_positive=True),
        rx_day_gate="if ordinary source gate passes: all four folds, last group, seed8101, "
                    "paired hard and admitted candidate; train fit RX day1; validate different held RX day2; "
                    "ManyRx mean>=0, >=3positive folds, ManySig mean>=-.5 and worst RX delta>=-2",
        caps=dict(restore=2, smoke=4, source=32, rx_day=16, final=48, ablation=24, bn=16),
        budget_note="Do not run both complete F1/F2 grids (64). Zero-fit preselection limits ordinary screening32. "
                    "RX/day confirmation: fold4 xdataset2 xseed1 xgroup1(last) xarm2=16, "
                    "one grouping intentionally narrower than ordinary source32; "
                    "no claim of replicated two-group/two-seed day-transfer validation.",
        gate_resolution="day-transfer cap16 takes precedence; use only last group, seed8101, after ordinary GO",
        bn_route="diagnostic only until restored hard source checkpoint proves abnormal layers; "
                 "no GN+spectral combined fit; optional candidate remains separately registered",
        retries="no automatic retries or replacements; failed training stops progression",
        restoration="two source-only hard fits at exactly S09 fold0/last/8101; compare state tensors and historical seals; "
                    "never relabel a refit as restored original when hashes differ",
        final_rule="same Official Adam/CE/packet-validation minimum; all source RX days1-2; "
                   "source-selected one candidate + hard, paired fresh seeds; separate sealed target evaluation",
        ablation_policy="conditional24: No-phase/No-HSTU x2datasets x4folds xseed11101; "
                        "No-selection reuses formal hard; one-seed ablation limits explicit",
        goals_only=dict(ManyRx_CRD=90., ManySig_CRD=93.5),
        historical_targets_seen=True, no_target_selection=True)
    once(HERE / "registration.json", reg)
    once(HERE / "jobs_restore.json", restore)
    once(HERE / "jobs_smoke.json", smokes)
    print("S11 frozen: zero-fit preselection;32 source cap; no automatic search or target tuning.")


if __name__ == "__main__":
    main()
