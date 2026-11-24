"""Pair all S31 and frozen S29/S30 endpoint outputs without accessing IQ."""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "s29_wisig"))
from source import array_sha, read, sha, write
from summarize import NAMES, check_access, csv_write, group_scores, packet_scores, pair_summary

VIEWS = ("V0", "V1")
ARCHES = ("HStem", "LongPatch")
POLICIES = ("raw", "cfo25")
DOSES = (0, 1000, 2000, 2500, 5000)
CONDITIONS = ("native", *(f"center_{sign * dose}_0" for dose in DOSES[1:] for sign in (-1, 1)))


def check_models(s29, s30, out):
    reg = read(out / "EXECUTION.json")
    fit = read(out / "FIT_SEAL.json")
    assert reg["parent_fit_sha256"] == sha(s29 / "raw_control" / "FIT_SEAL.json")
    assert reg["s30_summary_sha256"] == sha(s30 / "SUMMARY.json")
    assert reg["protocol_sha256"] == sha(HERE / "PROTOCOL.zh-CN.md")
    assert reg["script_sha256"] == sha(HERE / "run.py")
    base = {tuple(r["job"][:3]): r for r in read(s29 / "raw_control" / "FIT_SEAL.json")["results"]}
    records = {tuple(r["job"][:3]): r for r in fit["results"]}
    assert len(records) == len(base) == 8 and records.keys() == base.keys()
    for key, b in base.items():
        row = records[key]
        assert row["epochs"] == b["epochs"] == 40 and row["updates"] == b["updates"] == 5080
        assert row["initial_sha256"] == b["initial_sha256"] and row["head_sha256"] == b["head_sha256"]
        assert [r["order_sha256"] for r in row["history"]] == [r["order_sha256"] for r in b["history"]]
        assert sha(out / row["selected_path"]) == row["selected_sha256"]
        v, seed, _ = row["job"][:3]
        name = f"DOSES_{v}_{seed}.npz"
        assert sha(out / name) == fit["dose_schedule_hashes"][name]
        with np.load(out / name, allow_pickle=False) as z:
            doses = z["effective_hz"]
        assert array_sha(doses) == row["augmentation"]["draw_sha256"]
        assert int(np.count_nonzero(doses)) == row["augmentation"]["nonzero_draws"]
        assert len(doses) == row["augmentation"]["packet_accesses"]
        assert np.abs(doses).max() <= 2500
    return records, fit["total_updates"]


def conditions(root, baseline, s30, reg, v):
    native_file = baseline / "raw_control" / f"RAW_{v}.npz"
    dose_file = s30 / f"PACKETS_{v}.npz"
    new_file = root / f"S31_{v}.npz"
    assert sha(native_file) == reg["base_prediction_hashes"][v]
    assert sha(dose_file) == reg["dose_prediction_hashes"][v]
    assert sha(new_file) == read(root / "PREDICTION_SEAL.json")[v]
    return native_file, dose_file, new_file


def aggregate(root, s29, s30):
    reg = read(root / "EXECUTION.json")
    rows, per_tx = [], []
    receivers = {}
    acc = {(a, arm, c): np.empty((2, 2, 6)) for a in ARCHES for arm in POLICIES for c in CONDITIONS}
    for vi, v in enumerate(VIEWS):
        paths = conditions(root, s29, s30, reg, v)
        with np.load(paths[0], allow_pickle=False) as raw, np.load(paths[1], allow_pickle=False) as dose, \
             np.load(paths[2], allow_pickle=False) as augmented:
            for name in ("y", "rx", "day", "ids"):
                assert np.array_equal(raw[name], dose[name])
                assert np.array_equal(raw[name], augmented[name])
            y, rx = raw["y"], raw["rx"]
            rx_codes = np.unique(rx)
            assert len(rx_codes) == 6
            receivers[v] = [NAMES[k] for k in rx_codes]
            for seed in (0, 1):
                for arch in ARCHES:
                    for arm in POLICIES:
                        for condition in CONDITIONS:
                            k = f"{seed}_{arch}_{condition}"
                            if arm == "cfo25":
                                pred, ce = packet_scores(augmented[k], y)
                            elif condition == "native":
                                pred, ce = packet_scores(raw[k], y)
                                assert np.array_equal(pred, dose[f"{seed}_{arch}_native_0_0"])
                            else:
                                pred, ce = dose[k], dose[f"ce_{k}"]
                            assert np.isfinite(ce).all() and pred.shape == y.shape == ce.shape
                            for ri, r in enumerate(rx_codes):
                                mask = rx == r
                                result = group_scores(y[mask], pred[mask], ce[mask])
                                acc[arch, arm, condition][vi, seed, ri] = result["accuracy_pct"]
                                common = dict(view=v, seed=seed, arch=arch, arm=arm, rx=NAMES[r])
                                rows.append(dict(**common, condition=condition, packets=int(mask.sum()), **result))
                                if condition == "native":
                                    for tx in range(10):
                                        cell = mask & (y == tx)
                                        per_tx.append(dict(**common, tx=tx, n=int(cell.sum()),
                                                           wrong=int((pred[cell] != y[cell]).sum())))
    assert not set(receivers["V0"]) & set(receivers["V1"])
    csv_write(root / "RX_METRICS.csv", rows)
    csv_write(root / "NATIVE_TX_RX.csv", per_tx)
    return acc, receivers, rows


def analyze(out, acc, names, rows, models, updates, calls):
    results, auc, clean = {}, {}, {}
    for arch in ARCHES:
        results[arch] = {}
        for arm in POLICIES:
            base = acc[arch, arm, "native"]
            clean[arch, arm] = base
            real_rows = [r for r in rows if (r["arch"], r["arm"], r["condition"]) == (arch, arm, "native")]
            curve = {0: base}
            for dose in DOSES[1:]:
                curve[dose] = (acc[arch, arm, f"center_{dose}_0"] + acc[arch, arm, f"center_-{dose}_0"]) / 2
            loss = np.stack([base - curve[d] for d in DOSES])
            auc[arch, arm] = np.trapz(loss, DOSES, axis=0) / DOSES[-1]
            results[arch][arm] = dict(accuracy_pct=float(base.mean()),
                macro_f1_pct=float(np.mean([r["macro_f1_pct"] for r in real_rows])),
                rx_ce=float(np.mean([r["ce"] for r in real_rows])),
                worst_rx_accuracy_pct=float(base.mean(1).min()), worst_seed_rx_accuracy_pct=float(base.min()),
                mean_rx_accuracy_pct=base.mean(1).tolist(), loss_auc_pp=float(auc[arch, arm].mean()),
                dose_accuracy_pct={str(d): float(a.mean()) for d, a in curve.items()},
                dose_loss_pp={str(d): float(dif.mean()) for d, dif in zip(DOSES, loss)})
    gains = {a: clean[a, "cfo25"] - clean[a, "raw"] for a in ARCHES}
    comparisons = {a: pair_summary(a_delta) for a, a_delta in gains.items()}
    comparisons["interaction"] = pair_summary(gains["HStem"] - gains["LongPatch"])
    for arm in POLICIES:
        comparisons[f"structure_{arm}"] = pair_summary(clean["HStem", arm] - clean["LongPatch", arm])
    auc_gains = {a: pair_summary(auc[a, "raw"] - auc[a, "cfo25"]) for a in ARCHES}
    h = comparisons["HStem"]
    raw, aug = (results["HStem"][a] for a in POLICIES)
    checks = dict(clean_accuracy_gain_ge_1_pp=h["mean_pp"] >= 1.,
                  conditional_ci_low_gt_0=h["interval95_pp"][0] > 0.,
                  at_least_3_of_4_paired_fits_positive=h["positive_view_seed"] >= 3,
                  both_views_positive=min(h["view_pp"]) > 0.,
                  worst_paired_rx_ge_minus_1_pp=h["worst_rx_pp"] >= -1.,
                  worst_absolute_rx_nondecreasing=aug["worst_rx_accuracy_pct"] >= raw["worst_rx_accuracy_pct"],
                  macro_f1_nondecreasing=aug["macro_f1_pct"] >= raw["macro_f1_pct"],
                  loss_auc_0_5khz_nonincreasing=aug["loss_auc_pp"] <= raw["loss_auc_pp"],
                  identical_deployment=True)
    selected = []
    for (v, seed, arch), trial in models.items():
        best = min(trial["history"], key=lambda r: r["val_ce"])
        selected.append(dict(view=v, seed=seed, arch=arch, selected_epoch=best["epoch"],
                             selected_val_ce=best["val_ce"], seconds=trial["seconds"],
                             augmentation=trial["augmentation"]))
    write(out / "SUMMARY.json", dict(results=results, clean_comparisons=comparisons,
         loss_auc_reduction_pp=auc_gains, rx_names=names, selected=selected, new_fits=len(models),
         fit_seconds=sum(item["seconds"] for item in selected), optimization_updates=updates,
         rx_metric_rows=len(rows), role_payload_checks=calls,
         decision=dict(checks=checks, status="PASS_INDEPENDENT_CONFIRMATION_CANDIDATE"
                       if all(checks.values()) else "STOP_SOURCE_GATE")))


def main():
    parser = argparse.ArgumentParser()
    for name in ("s29", "s30", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    out = args.output
    models, updates = check_models(args.s29, args.s30, out)
    calls = check_access(out, [("fit", v, role) for v in VIEWS for role in ("fit_train", "checkpoint_val")]
                         + [("new_prediction", v, "source_selection") for v in VIEWS])
    analyze(out, *aggregate(out, args.s29, args.s30), models, updates, calls)


if __name__ == "__main__":
    main()
