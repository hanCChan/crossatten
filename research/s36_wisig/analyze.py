"""Reduce S36 transfers, gates, and preamble results from sealed arrays."""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "s29_wisig"))
from source import read, sha, write
from summarize import NAMES, check_access, csv_write, group_scores, pair_summary

VIEWS = ("V0", "V1")
ARCHES = ("HStem", "A1")
TAPS = ("token", "prepared", "pre_ln", "q", "k", "v", "attention", "residual", "mix1", "pooled")
ABLATIONS = ("self_attention", "no_attention", "skip_block_norm")
FUSIONS = ("native", "token_linear", "pooled_linear", "early_fusion", "deep_double", "shuffle_fusion")
SUBSETS = ("all", "double", "triple", "q0", "q1", "q2", "q3")
PREAD = ("raw", "shared", "residual", "joint")


def pack(directory, view, digest):
    path = directory / f"PRED_{view}.npz"
    assert sha(path) == digest
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def prediction(x):
    return x.argmax(1) if x.ndim == 2 else x


def choices(x, y):
    return dict(all=np.ones(len(y), dtype=bool),
                **{f"q{i}": x["quality_quartile"] == i for i in range(4)})


def transition(a, b, take):
    return [(int(np.sum(z & take))) for z in (a & b, a & ~b, ~a & b, ~a & ~b)]


def margin(z):
    top = np.partition(z, -2, axis=1)[:, -2:]
    return np.abs(top[:, 1] - top[:, 0]) / z.std(1).clip(min=1e-8)


def reduce_neural(ad):
    receipt = read(ad / "RECEIPT.json")
    rows, movements, repeats, cells = [], [], [], []
    census = {}
    for view in VIEWS:
        x = pack(ad, view, receipt["predictions"][view])
        y, rx, day = (x[k] for k in ("y", "rx", "day"))
        census[view] = [NAMES[r] for r in np.unique(rx)]
        # Phase B has its own A1 original reference in both historical views.
        flags = {}
        for seed in (0, 1):
            a1 = x[f"{seed}_A1_native"].argmax(1) == y
            h = x[f"{seed}_HStem_native"].argmax(1) == y
            p = x[f"{seed}_LongPatch_native"].argmax(1) == y
            subsets = choices(x, y)
            subsets.update(double=~(h | p), triple=~(h | p | a1))
            for arch in ARCHES:
                pre = f"{seed}_{arch}_"
                first = x[pre+"token_linear"] == y
                norm_margin = margin(x[pre+"token_linear_logits"])
                predictions = (*FUSIONS, *ABLATIONS, *(f"{layer}_{probe}"
                                for layer in TAPS for probe in ("linear", "nonlinear")
                                if f"{layer}_{probe}" not in FUSIONS))
                for now, future in (("first_mixer", "mix1_linear"), ("native_head", "native")):
                    second = prediction(x[pre+future]) == y
                    flags[seed, arch, now] = first & ~second
                    for r in np.unique(rx):
                        for label, selected in subsets.items():
                            take = (rx == r) & selected
                            cases = transition(first, second, take)
                            b = first & ~second & take
                            movements.append(dict(view=view, seed=seed, arch=arch, contrast=now,
                                                  subset=label, rx=NAMES[r], rows=int(take.sum()),
                                                  both_correct=cases[0], early_only=cases[1],
                                                  later_only=cases[2], both_wrong=cases[3],
                                                  a1_correct_when_early_only=int((a1 & b).sum()),
                                                  mean_early_only_margin=float(norm_margin[b].mean()) if b.any() else None,
                                                  mean_early_correct_margin=float(norm_margin[first & take].mean())
                                                  if (first & take).any() else None))
                    for r, t, d in np.unique(np.column_stack((rx,y,day)), axis=0):
                        g = (rx == r) & (y == t) & (day == d)
                        a, b, c, bad = transition(first, second, g)
                        cells.append(dict(view=view, seed=seed, arch=arch, contrast=now, rx=NAMES[r], tx=int(t),
                                          day=int(d), rows=int(g.sum()), both_correct=a,
                                          early_only=b, later_only=c, both_wrong=bad))
                for variant in predictions:
                    pred = prediction(x[pre+variant])
                    for r in np.unique(rx):
                        for subset, keep in subsets.items():
                            take = (rx == r) & keep
                            n = int(take.sum())
                            scores = group_scores(y[take], pred[take], np.zeros(n)) if n else {}
                            rows.append(dict(view=view, seed=seed, arch=arch, variant=variant, subset=subset,
                                             rx=NAMES[r], rows=n, correct=int(np.sum(take & (pred == y))),
                                             macro_f1_pct=scores.get("macro_f1_pct")))
        for key in ((a,c) for a in ARCHES for c in ("first_mixer", "native_head")):
            u, v = [flags[(s,*key)] for s in (0,1)]
            for r in np.unique(rx):
                hit = rx == r
                repeats.append(dict(view=view, arch=key[0], contrast=key[1], rx=NAMES[r], rows=int(hit.sum()),
                                    both=int((u & v & hit).sum()), union=int(((u | v) & hit).sum()),
                                    total_early_only=int((u & hit).sum() + (v & hit).sum())))
    assert not set(census["V0"]) & set(census["V1"])
    for title, data in (("RX_METRICS",rows),("TRANSITIONS", movements),
                        ("TX_RX_DAY_TRANSITIONS",cells),("SEED_REPEAT",repeats)):
        csv_write(ad / (title + ".csv"), data)
    return rows, movements, repeats, census


def selected(rows, arch, variant, subset):
    return [r for r in rows if r["arch"] == arch and r["variant"] == variant and r["subset"] == subset and r["rows"]]


def full_array(rows, arch, variant, census, column="accuracy"):
    values = {(r["view"],r["seed"],r["rx"]): (100*r["correct"]/r["rows"] if column=="accuracy"
              else r["macro_f1_pct"]) for r in selected(rows,arch,variant,"all")}
    return np.asarray([[[values[v,s,r] for r in census[v]] for s in (0,1)] for v in VIEWS])


def describe_accuracy(rows):
    output = {}
    for arch in ARCHES:
        variants = list(dict.fromkeys(r["variant"] for r in rows if r["arch"] == arch))
        output[arch] = {variant: {
            subset: float(np.mean([100*r["correct"]/r["rows"] for r in selected(rows,arch,variant,subset)]))
            for subset in SUBSETS} for variant in variants}
    return output


def describe_movement(movements, repeats):
    output = {}
    for arch in ARCHES:
        output[arch] = {}
        for contrast in ("first_mixer", "native_head"):
            relevant = [r for r in movements if (r["arch"],r["contrast"]) == (arch,contrast)]
            rep = [r for r in repeats if (r["arch"],r["contrast"]) == (arch,contrast)]
            stats = {}
            for subset in SUBSETS:
                cells = [r for r in relevant if r["rows"] and r["subset"] == subset]
                summary = {k: float(np.mean([100*r[k]/r["rows"] for r in cells])) for k in
                           ("both_correct","early_only","later_only","both_wrong","a1_correct_when_early_only")}
                summary["oracle_pct"] = 100 - summary["both_wrong"]
                if summary["early_only"]:
                    summary["a1_rescue_fraction_of_b"] = summary["a1_correct_when_early_only"] / summary["early_only"]
                for field in ("mean_early_only_margin", "mean_early_correct_margin"):
                    vals = [r[field] for r in cells if r[field] is not None]
                    if vals:
                        summary[field] = float(np.mean(vals))
                stats[subset] = summary
            rxx = [r for r in relevant if r["subset"] == "all"]
            by_rx = {rx: float(np.mean([100*r["early_only"]/r["rows"] for r in rxx if r["rx"] == rx]))
                     for rx in sorted({r["rx"] for r in rxx})}
            by_view = {v: float(np.mean([100*r["early_only"]/r["rows"] for r in rxx if r["view"] == v]))
                       for v in VIEWS}
            output[arch][contrast] = dict(rates=stats, repeated_cache_rows=sum(r["both"] for r in rep),
                union_cache_rows=sum(r["union"] for r in rep), rx_early_only_pct=by_rx,
                view_early_only_pct=by_view, seed_repeat_fraction=float(
                    sum(r["both"] for r in rep)/sum(r["union"] for r in rep)))
    return output


def analyze_ad(ad):
    records = read(ad / "EXECUTION.json")
    receipt = read(ad / "RECEIPT.json")
    assert records["protocol"] == sha(HERE / "PROTOCOL.zh-CN.md")
    assert records["engine"] == sha(HERE / "run.py") and records["probes"] == sha(HERE / "heads.py")
    assert sha(ad / "CALIBRATION.json") == receipt["calibrations"]
    assert sha(ad / "LINEAR.json") == receipt["fits"]
    for path, digest in receipt["nonlinear"].items():
        assert sha(ad/path) == digest
    role_checks = check_access(ad, [("s36_neural",v,r) for v in VIEWS
                               for r in ("fit_train","checkpoint_val","source_selection")])
    rows, movement, repeated, census = reduce_neural(ad)
    metric = describe_accuracy(rows)
    switches = describe_movement(movement, repeated)
    a1 = full_array(rows,"A1","native",census)
    effects = {}
    for model in ARCHES:
        native = full_array(rows,model,"native",census)
        early = full_array(rows,model,"early_fusion",census)
        effects[model] = dict(vs_original=pair_summary(early-native), vs_a1=pair_summary(early-a1),
            deep_double_vs_original=pair_summary(full_array(rows,model,"deep_double",census)-native),
            shuffle_vs_original=pair_summary(full_array(rows,model,"shuffle_fusion",census)-native),
            vs_capacity=pair_summary(early-full_array(rows,model,"deep_double",census)),
            vs_shuffled=pair_summary(early-full_array(rows,model,"shuffle_fusion",census)),
            macro_f1_gain=float((full_array(rows,model,"early_fusion",census,"f1")-
                                 full_array(rows,model,"native",census,"f1")).mean()))
    trial = effects["A1"]
    h = trial["vs_original"]
    gate = dict(min_gain=h["mean_pp"] >= 1., both_views_positive=min(h["view_pp"]) > 0.,
                paired_interval=h["interval95_pp"][0] > 0., worst_rx=h["worst_rx_pp"] >= -1.,
                macro_f1=trial["macro_f1_gain"] >= 0., above_capacity=trial["vs_capacity"]["mean_pp"] > 0.,
                above_shuffled=trial["vs_shuffled"]["mean_pp"] > 0., parameter_budget=650/82426 <= .05)
    advanced = all(gate.values())
    out = dict(metrics=metric, transitions=switches, fusion=effects, rx_names=census,
        decision=dict(gate=gate, verdict="CANDIDATE_SEPARATE_CONFIRMATION" if advanced else "STOP_SOURCE_GATE",
                      minimum_exploratory_point_gain=h["mean_pp"] >= .5),
        role_checks=role_checks, backbone_fits=0,
        ridge_heads=len(read(ad/"LINEAR.json")),
        nonlinear_heads=sum(read(ad/p)["heads"] for p in receipt["nonlinear"]),
        probe_optimizer_steps=sum(read(ad/p)["vectorized_optimizer_steps"] for p in receipt["nonlinear"]))
    write(ad/"SUMMARY.json", out)


def balanced(pred, label):
    return float(np.mean([np.mean(pred[label == level] == level) for level in np.unique(label)]))


def analyze_e(out):
    receipt = read(out/"RECEIPT.json")
    assert sha(out/"LINEAR.json") == receipt["selected_readouts"]
    role_checks = check_access(out, [("s36_preamble",v,r) for v in VIEWS
                              for r in ("fit_train","checkpoint_val","source_selection")])
    views, rows = {}, []
    for v in VIEWS:
        data = pack(out,v,receipt["predictions"][v])
        y, rx, day = (data[k] for k in ("y","rx","day"))
        result = {}
        for name in PREAD:
            t = {}
            for r in np.unique(rx):
                idx = rx == r
                cell = dict(tx_linear=float(100*(data[name+"_ridge"][idx] == y[idx]).mean()),
                            tx_prototype=float(100*(data[name+"_prototype"][idx] == y[idx]).mean()),
                            day_bal_acc=100*balanced(data[name+"_day"][idx], day[idx]-1))
                rows.append(dict(view=v,readout=name,rx=NAMES[r],**cell))
            t["rx_bal_acc_cross_tx_parities"] = [
                100 * balanced(data[f"rx_{name}_{p}_pred"], data[f"rx_{name}_{p}_y"]) for p in (0,1)]
            result[name] = t
        votes = (data["shared_ridge"] == y) | (data["residual_ridge"] == y)
        result["shared_or_residual_oracle"] = float(100 * np.mean([votes[rx == r].mean() for r in np.unique(rx)]))
        views[v] = result
    for v in VIEWS:
        for name in PREAD:
            these = [row for row in rows if (row["view"],row["readout"]) == (v,name)]
            for k in ("tx_linear","tx_prototype","day_bal_acc"):
                views[v][name][k] = float(np.mean([row[k] for row in these]))
    summary = {name: {k: float(np.mean([views[v][name][k] for v in VIEWS]))
                       for k in ("tx_linear","tx_prototype","day_bal_acc","rx_bal_acc_cross_tx_parities")}
               for name in PREAD}
    summary["shared_or_residual_oracle"] = float(np.mean([views[v]["shared_or_residual_oracle"] for v in VIEWS]))
    csv_write(out/"RX_METRICS.csv", rows)
    write(out/"SUMMARY.json",dict(readout=summary, by_view=views, role_checks=role_checks,
          raw_eq=read(out/"EXECUTION.json")["raw_eq"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("AD","E"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    (analyze_ad if args.stage == "AD" else analyze_e)(args.output)
