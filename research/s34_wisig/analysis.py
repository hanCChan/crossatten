"""Read sealed S34 predictions and compute position and physical comparisons."""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "s29_wisig"))
from source import read, sha, write
from summarize import NAMES, check_access, csv_write, pair_summary

VIEWS = ("V0", "V1")
ARCHES = ("HStem", "LongPatch")
STAGES = ("token", "mix1", "mix2", "pre_pool")
KINDS = ("mean", "mean_std", "region2", "region4", "flat", "dct4", "random4",
         "shuffle_region4", "shuffle_dct4")
METHODS = ("ridge", "prototype")
SIGNALS = ("rank1_residual", "phase_residual_rad", "repeat_loss", "spectral_distance",
           "early_loss", "middle_loss", "late_loss", "tail_loss")
INTERVENTIONS = ("white", "phase_walk", "frequency_shape", "constant_cfo")
SUBSETS = ("all", "pair_wrong", "triple_wrong", *(f"q{i}" for i in range(4)))


def pack(directory, view, receipt, original):
    filename = directory / f"PRED_{view}.npz"
    assert sha(filename) == receipt["predictions"][view]
    with np.load(filename, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    for name in ("ids", "y", "rx", "day"):
        assert np.array_equal(data[name], original["source_selection_" + name])
    return data


def historic(s33, view, hashes):
    filename = s33 / f"AUDIT_{view}.npz"
    assert sha(filename) == hashes["packets"][view]
    with np.load(filename, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def masks(old, seed, data):
    y = data["y"]
    preds = [old[f"{seed}_{a}_source_selection_logits"].argmax(1) for a in ARCHES]
    a1 = old[f"{seed}_A1_logits"].argmax(1)
    pair = (preds[0] != y) & (preds[1] != y)
    return dict(all=np.ones(len(y), dtype=bool), pair_wrong=pair,
                triple_wrong=pair & (a1 != y))


def view_metrics(data, old, view):
    y, rx = data["y"], data["rx"]
    rxs = np.unique(rx)
    rows = []
    keys = ["head", "A1_head", "input_ridge", "input_prototype", *(
        f"{stage}_{arm}_{method}" for stage in STAGES for arm in KINDS for method in METHODS),
        "pooled_pooled_ridge", "pooled_pooled_prototype"]
    for seed in (0, 1):
        subsets = masks(old, seed, data)
        subsets.update({f"q{i}": data["quality_quartile"] == i for i in range(4)})
        for arch in ARCHES:
            assert np.array_equal(data[f"{seed}_{arch}_head"], old[f"{seed}_{arch}_source_selection_logits"].argmax(1))
            for key in keys:
                if key.startswith("input_"):
                    p = data[key]
                elif key == "A1_head":
                    p = data[f"{seed}_A1_head"]
                else:
                    p = data[f"{seed}_{arch}_{key}"]
                hits = p == y
                for r in rxs:
                    for subset in SUBSETS:
                        idx = (rx == r) & subsets[subset]
                        size = int(idx.sum())
                        rows.append(dict(view=view, seed=seed, arch=arch, readout=key, rx=NAMES[r],
                                         subset=subset, rows=size, correct=int((hits & idx).sum())))
    return rows, [NAMES[r] for r in rxs]


def mean_accuracy(rows, arch, key, subset):
    values = [r["correct"]/r["rows"] for r in rows if (r["arch"], r["readout"], r["subset"]) ==
              (arch, key, subset) and r["rows"] > 0]
    return 100 * float(np.mean(values))


def cohort(rows, arch, readout, subset, names):
    matches = {(r["view"], r["seed"], r["rx"]): r["correct"]/r["rows"] * 100
               for r in rows if (r["arch"], r["readout"], r["subset"]) == (arch, readout, subset) and r["rows"]}
    return np.asarray([[[matches[v, seed, r] for r in names[v]] for seed in (0, 1)] for v in VIEWS])


def rank_corr(x, y):
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def a_summary(out, s33):
    receipt = read(out / "RECEIPT.json")
    hashes = read(s33 / "RECEIPT.json")
    labels, packet_rows = {}, []
    for view in VIEWS:
        orig = historic(s33, view, hashes)
        fitted = pack(out, view, receipt, orig)
        rows, labels[view] = view_metrics(fitted, orig, view)
        packet_rows.extend(rows)
    calls = check_access(out, [("s34_readout", v, r) for v in VIEWS
                               for r in ("fit_train", "checkpoint_val", "source_selection")])
    csv_write(out / "METRICS.csv", packet_rows)
    summary = {}
    for arch in ARCHES:
        names = sorted({r["readout"] for r in packet_rows if r["arch"] == arch})
        summary[arch] = {key: {subset: mean_accuracy(packet_rows, arch, key, subset)
                               for subset in SUBSETS} for key in names}
    hstem = "HStem"
    base = "pre_pool_mean_ridge"
    gate = {}
    for candidate in ("dct4", "region4"):
        name = f"pre_pool_{candidate}_ridge"
        candidate_all = cohort(packet_rows, hstem, name, "all", labels)
        exactly = {key: pair_summary(candidate_all - cohort(packet_rows, hstem, control, "all", labels))
                   for key, control in dict(mean=base, random="pre_pool_random4_ridge",
                          shuffle=f"pre_pool_shuffle_{candidate}_ridge", head="head", A1="A1_head").items()}
        hard = pair_summary(cohort(packet_rows, hstem, name, "pair_wrong", labels)
                            - cohort(packet_rows, hstem, base, "pair_wrong", labels))
        m = exactly["mean"]
        checks = dict(clean_gain=m["mean_pp"] >= .5, both_views_positive=min(m["view_pp"]) > 0,
                      worst_rx=m["worst_rx_pp"] >= -1, shared_error_gain=hard["mean_pp"] >= 5,
                      above_shuffled=exactly["shuffle"]["mean_pp"] > 0,
                      above_random=exactly["random"]["mean_pp"] > 0, learned_parameter_budget=True)
        gate[candidate] = dict(checks=checks, clean=exactly, hard=hard, passes=all(checks.values()))
    position = out / "POSITION_GEOMETRY.json"
    choices = out / "PROBES.json"
    assert sha(position) == receipt["position_geometry"] and sha(choices) == receipt["probe_choices"]
    geometry = {(r["view"], r["seed"], r["arch"], r["layer"], r["role"]): r["ratio"] for r in read(position)}
    correlations = {}
    for arch in ARCHES:
        correlations[arch] = {}
        for stage in STAGES:
            groups = {v: {role: np.mean([geometry[v, seed, arch, stage, role] for seed in (0,1)], axis=0)
                          for role in ("fit_train", "source_selection")} for v in VIEWS}
            correlations[arch][stage] = dict(
                views_fit=rank_corr(groups["V0"]["fit_train"], groups["V1"]["fit_train"]),
                views_held=rank_corr(groups["V0"]["source_selection"], groups["V1"]["source_selection"]),
                fit_to_held={v: rank_corr(*groups[v].values()) for v in VIEWS})
    selection = next((c for c in ("dct4", "region4") if gate[c]["passes"]), None)
    write(out / "SUMMARY.json", dict(scores=summary, gates=gate, candidate=selection,
                                     geometry=correlations, role_checks=calls, probes=len(read(choices)),
                                     metric_rows=len(packet_rows), new_backbone_fits=0))


def mean_group_auc(rows, signal):
    chosen = [r for r in rows if r["signal"] == signal and r["auc"] is not None]
    values = []
    for rx in sorted({r["rx"] for r in chosen}):
        by_seed = [[r["auc"] for r in chosen if (r["rx"], r["seed"]) == (rx, seed)] for seed in (0,1)]
        values.append(np.mean([np.mean(a) for a in by_seed if a]))
    return dict(auc=float(np.mean(values)), supported_rx=len(values), cells_seed=len(chosen))


def rank_auc(high, low):
    lo = np.sort(low)
    a = np.searchsorted(lo, high, side="left")
    b = np.searchsorted(lo, high, side="right")
    return float((a+b).sum() / (2 * len(lo) * len(high)))


def b_summary(out, s33):
    receipt = read(out / "RECEIPT.json")
    hashes = read(s33 / "RECEIPT.json")
    grouped, response, prototypes, evm = [], [], [], []
    for view in VIEWS:
        orig = historic(s33, view, hashes)
        raw = pack(out, view, receipt, orig)
        y, rx, day = (raw[k] for k in ("y", "rx", "day"))
        for kind in ("raw", "shared", "residual"):
            for r in np.unique(rx):
                index = rx == r
                score = raw[kind + "_prototype"][index] == y[index]
                prototypes.append(dict(view=view, rx=NAMES[r], readout=kind, accuracy_pct=float(100*score.mean())))
        for condition in INTERVENTIONS:
            e = float(np.abs(raw[condition+"_evm"] - .12).max())
            assert e < 5e-6
            evm.append(dict(view=view, condition=condition, max_abs_error=e))
        for seed in (0, 1):
            native = [raw[f"{seed}_{arch}_native"] == y for arch in ARCHES]
            easy = native[0] & native[1]
            hard = ~native[0] & ~native[1]
            for r, t, d in np.unique(np.column_stack((rx, y, day)), axis=0):
                group = (rx == r) & (y == t) & (day == d)
                large = group & hard
                small = group & easy
                for sig in SIGNALS:
                    arr = raw[f"native_{sig}"]
                    supported = min(large.sum(), small.sum()) >= 8
                    grouped.append(dict(view=view, seed=seed, rx=NAMES[r], tx=int(t), day=int(d),
                                        signal=sig, hard=int(large.sum()), easy=int(small.sum()),
                                        auc=rank_auc(arr[large],arr[small]) if supported else None))
            for kind in INTERVENTIONS:
                correct = [raw[f"{seed}_{arch}_{kind}"] == y for arch in ARCHES]
                fail = ~correct[0] & ~correct[1]
                for r in np.unique(rx):
                    group = rx == r
                    _easy = group & easy
                    delta = {s: float(np.mean(raw[f"{kind}_{s}"][_easy] - raw[f"native_{s}"][_easy]))
                             for s in SIGNALS}
                    response.append(dict(view=view, seed=seed, rx=NAMES[r], condition=kind,
                        paired_common_error_delta_pp=float(100*(fail[group].mean()-hard[group].mean())),
                        newly_common_wrong_pct_on_easy=float(100*fail[_easy].mean()),
                        hstem_accuracy_pct=float(100*correct[0][group].mean()),
                        longpatch_accuracy_pct=float(100*correct[1][group].mean()),
                        easy=int(_easy.sum()), **delta))
    calls = check_access(out, [("s34_preamble", v, role) for v in VIEWS
                               for role in ("fit_train", "source_selection")])
    csv_write(out / "NATURAL_CELLS.csv", grouped)
    csv_write(out / "INTERVENTIONS.csv", response)
    csv_write(out / "PROTOTYPES.csv", prototypes)
    report = {kind: {col: float(np.mean([r[col] for r in response if r["condition"] == kind]))
                      for col in (*SIGNALS, "paired_common_error_delta_pp", "newly_common_wrong_pct_on_easy",
                                  "hstem_accuracy_pct", "longpatch_accuracy_pct")}
              for kind in INTERVENTIONS}
    pro = {k: float(np.mean([r["accuracy_pct"] for r in prototypes if r["readout"] == k]))
           for k in ("raw", "shared", "residual")}
    write(out / "SUMMARY.json", dict(natural={s: mean_group_auc(grouped, s) for s in SIGNALS},
        intervention=report, prototype=pro, role_checks=calls, max_error=max(r["max_abs_error"] for r in evm),
        raw_eq=read(out / "EXECUTION.json")["raw_eq"], new_backbone_fits=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("A", "B"))
    parser.add_argument("output", type=Path)
    parser.add_argument("s33", type=Path)
    args = parser.parse_args()
    (a_summary if args.stage == "A" else b_summary)(args.output, args.s33)
