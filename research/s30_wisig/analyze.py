"""Summarize frozen prediction, representation and CFO-cell readouts."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "s29_wisig"))
from source import C, read, sha, write
from summarize import group_scores, pair_summary

from drift import analyze as drift_analysis

FREQ = (0, 1000, 2000, 2500, 5000, 10000, 15000, 20000)
TIME = (0, 1, 2, 4, 8, 9, 16)
ARCHES = ("HStem", "LongPatch")
VIEWS = ("V0", "V1")
NAMES = read(C / "PRE_REGISTRATION.json")["rx_names"]


def key(kind, f=0, d=0):
    return f"{kind}_{f}_{d}"


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_metrics(out, receipt):
    rx_rows, tx_rows = [], []
    results = {}
    labels = []
    for view in VIEWS:
        path = out / f"PACKETS_{view}.npz"
        assert sha(path) == receipt["hashes"][view]
        with np.load(path, allow_pickle=False) as z:
            y, rx = z["y"], z["rx"]
            codes = np.unique(rx)
            assert len(codes) == 6
            names = [NAMES[c] for c in codes]
            labels.append(names)
            conditions = [k.removeprefix("0_HStem_") for k in z.files if k.startswith("0_HStem_")]
            by_arch = {a: {} for a in ARCHES}
            for a in ARCHES:
                for c in conditions:
                    matrix = np.zeros((2, 6))
                    for seed in (0, 1):
                        label = f"{seed}_{a}_{c}"
                        pred, ce = z[label], z["ce_" + label]
                        assert pred.shape == y.shape == ce.shape and np.isfinite(ce).all()
                        for i, code in enumerate(codes):
                            keep = rx == code
                            metrics = group_scores(y[keep], pred[keep], ce[keep])
                            matrix[seed, i] = metrics["accuracy_pct"]
                            common = dict(view=view, seed=seed, arch=a, condition=c, rx=names[i])
                            rx_rows.append(dict(**common, rows=int(keep.sum()), **metrics))
                            support = np.bincount(y[keep], minlength=10)
                            fails = np.bincount(y[keep & (pred != y)], minlength=10)
                            tx_rows.extend(dict(**common, tx=tx, rows=int(support[tx]), wrong=int(fails[tx]))
                                           for tx in range(10))
                    by_arch[a][c] = matrix
            results[view] = by_arch
    write_csv(out / "RX_METRICS.csv", rx_rows)
    write_csv(out / "TX_RX_METRICS.csv", tx_rows)
    assert not set(labels[0]) & set(labels[1])
    metrics = {a: {c: np.stack([results[v][a][c] for v in VIEWS])
                   for c in results["V0"][a]} for a in ARCHES}
    return metrics, labels


def describe(loss):
    h, lp = (loss[k] for k in ARCHES)
    return dict(loss_pp={a: float(d.mean()) for a, d in loss.items()}, advantage=pair_summary(lp - h),
                absolute_worst_rx={a: float(d.mean(axis=1).max()) for a, d in loss.items()})


def curves(metrics):
    arrays, report, signed_doses = {}, {}, {}
    for mode, doses in (("edge", FREQ), ("center", FREQ), ("phase", FREQ),
                        ("shifted", TIME), ("cyclic", TIME), ("aligned", TIME)):
        arrays[mode] = {}
        signed_doses[mode] = {}
        for dose in doses:
            losses = []
            for sign in (-1, 1):
                value = dose * sign
                pair = {}
                for a, s in metrics.items():
                    reference = s[key("native")]
                    current = reference
                    if dose > 0 and mode in ("edge", "center", "phase"):
                        current = s[key(mode, value)]
                    elif dose > 0:
                        current = s[key(mode, d=value)]
                        if mode == "shifted":
                            reference = s[key("aligned", d=value)]
                    pair[a] = reference - current
                losses.append(pair)
                if value:
                    signed_doses[mode][str(value)] = describe(pair)
            arrays[mode][dose] = {a: (losses[0][a] + losses[1][a]) / 2 for a in ARCHES}
        report[mode] = {str(value): describe(part) for value, part in arrays[mode].items()}

    aucs = {}
    for mode, limit in (("edge", 5000), ("edge", 20000), ("center", 5000), ("center", 20000),
                        ("phase", 5000), ("phase", 20000), ("shifted", 9), ("shifted", 16),
                        ("cyclic", 9), ("aligned", 9)):
        doses = [d for d in arrays[mode] if d <= limit]
        loss = {a: np.trapz(np.stack([arrays[mode][d][a] for d in doses]), doses, axis=0) / limit
                for a in ARCHES}
        aucs[f"{mode}_{limit}"] = describe(loss)
    return report, aucs, signed_doses


def joint(metrics):
    report = {}
    for f in (2000, 5000, 10000):
        for d in (4, 9):
            total, extra = {}, {}
            for a, values in metrics.items():
                total[a], extra[a] = 0., 0.
                for hz in (-f, f):
                    for dt in (-d, d):
                        aligned = values[key("aligned", d=dt)]
                        shifted = values[key("shifted", d=dt)]
                        ref = values[key("joint_ref", hz, dt)]
                        after = values[key("joint", hz, dt)]
                        total[a] += (aligned - after) / 4
                        extra[a] += ((ref - after) - (aligned - shifted)) / 4
            report[f"{f}_{d}"] = dict(total_loss=describe(total), departure_from_additivity=describe(extra))
    return report


def layer_summary(out, receipt, labels):
    path = out / "LAYER.json"
    assert sha(path) == receipt["layer_sha256"]
    rows = read(path)
    index = {(r["view"], r["seed"], r["arch"], r["condition"], r["layer"], r["rx"]): r for r in rows}
    assert len(index) == len(rows)

    def array(a, c, layer, column):
        return np.asarray([[[index[v, seed, a, c, layer, r][column] for r in names]
                            for seed in (0, 1)] for v, names in zip(VIEWS, labels)])

    def dimensionless_pair(values):
        return {field.removesuffix("_pp"): value for field, value in pair_summary(values).items()}

    result = {}
    for kind, doses in (("center", (5000, 10000)), ("shifted", (4, 9))):
        for dose in doses:
            for layer in ("token", "pooled", "logits", "hstem_conv1", "hstem_conv2",
                          "hstem_conv3", "hstem_conv4"):
                h_only = layer.startswith("hstem")
                record = {}
                distributions = {}
                for a in ("HStem",) if h_only else ARCHES:
                    pairs = []
                    for sign in (-1, 1):
                        d = dose * sign
                        c = key(kind, d) if kind == "center" else key(kind, d=d)
                        ref = key("native") if kind == "center" else key("aligned", d=d)
                        part = {field: array(a, c, layer, field) for field in
                                ("rel_l2", "rms_l2", "tx_separation", "margin_change")}
                        part["reference_separation"] = array(a, ref, layer, "tx_separation")
                        part["separation_retained"] = part["tx_separation"] / part["reference_separation"]
                        pairs.append(part)
                    averages = {f: (pairs[0][f] + pairs[1][f]) / 2 for f in pairs[0]}
                    distributions[a] = averages
                    record[a] = {f: float(p.mean()) for f, p in averages.items()}
                if not h_only:
                    h, lp = (distributions[a] for a in ARCHES)
                    record["rel_l2_advantage"] = dimensionless_pair(lp["rel_l2"] - h["rel_l2"])
                    record["rms_l2_advantage"] = dimensionless_pair(lp["rms_l2"] - h["rms_l2"])
                    record["margin_change_advantage"] = dimensionless_pair(h["margin_change"] - lp["margin_change"])
                result[f"{kind}_{dose}_{layer}"] = record
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("s29", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output
    drift = drift_analysis(args.s29, NAMES)
    write(out / "CFO_DRIFT.json", drift)
    if not (out / "RECEIPT.json").is_file():
        print("CFO drift ready; frozen predictor receipt pending.")
        return
    receipt = read(out / "RECEIPT.json")
    metrics, labels = load_metrics(out, receipt)
    curve, auc, signs = curves(metrics)
    layers = layer_summary(out, receipt, labels)
    primary = auc["center_5000"]["advantage"]
    checks = dict(center_cfo_auc_pp_ge_1=primary["mean_pp"] >= 1.,
                  center_cfo_ci_positive=primary["interval95_pp"][0] > 0.,
                  center_cfo_both_views_positive=min(primary["view_pp"]) > 0.,
                  source_rx_association_stable=drift["stable_rx_gate"])
    write(out / "SUMMARY.json", dict(curves=curve, auc=auc, signed_doses=signs,
                                   joint=joint(metrics), layers=layers, rx_labels=labels,
                                   decision=dict(checks=checks, cfo_candidate=all(checks.values())),
                                   parent_audit_sha256={v: sha(args.s29 / f"AUDIT_{v}.npz") for v in VIEWS}))


if __name__ == "__main__":
    main()
