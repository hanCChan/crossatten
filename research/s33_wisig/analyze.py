"""Analyze same-packet mistakes, fixed physical indicators, and feature readouts."""

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
MODELS = (*ARCHES, "A1")
READOUTS = ("token_mean", "token_flat", "pooled", "head")
CASES = ("both_correct", "hstem_only", "patch_only", "both_wrong")
SIGNALS = ("cfo_abs_khz", "cfo_departure_khz", "repeat_loss", "phase_spread_khz",
           "outband_fraction", "spectral_distance", "transition_dev_samples")


def masked_mean(a, selected):
    return float(a[selected].mean()) if selected.any() else None


def rank_auc(high, low):
    low = np.sort(low)
    pairs = np.searchsorted(low, high, "left") + np.searchsorted(low, high, "right")
    return float(pairs.sum() / (2 * len(high) * len(low)))


def margins(logits, labels):
    row = np.arange(len(labels))
    other = logits.copy()
    other[row, labels] = -np.inf
    raw = logits[row, labels] - other.max(1)
    return raw / logits.std(1).clip(min=1e-8)


def comparison_metrics(cases, correct, predictions, labeled, group):
    h, p, a1 = correct[:, group]
    common = ~h & ~p
    triple = common & ~a1
    observed = cases[group]
    expected = 0.
    for y, day in np.unique(labeled[group], axis=0):
        which = (labeled[group, 0] == y) & (labeled[group, 1] == day)
        expected += which.mean() * (~h[which]).mean() * (~p[which]).mean()
    same = predictions[0, group] == predictions[1, group]
    return dict(**{name: float((observed == i).mean()) for i, name in enumerate(CASES)},
                hstem=float(h.mean()), patch=float(p.mean()), a1=float(a1.mean()),
                pair_oracle=float(1 - common.mean()), triple_oracle=float(1 - triple.mean()),
                all_wrong=float(triple.mean()), a1_rescued=float((common & a1).mean()),
                shared_wrong_label=float((common & same).mean()),
                jaccard=float(common.sum() / ((~h | ~p).sum())),
                expected_both_wrong=float(expected), excess_overlap=float(common.mean() - expected))


def condition_rows(data, view, seed, masks, pred, cases, rows, counts, errors, confusion):
    y, rx, day = (data["source_selection_" + k] for k in ("y", "rx", "day"))
    normalized = {}
    for arch in MODELS:
        suffix = "_source_selection" if arch != "A1" else ""
        normalized[arch] = margins(data[f"{seed}_{arch}{suffix}_logits"], y)
    for rx_id, tx, date in np.unique(np.column_stack([rx, y, day]), axis=0):
        group = (rx == rx_id) & (y == tx) & (day == date)
        base = dict(view=view, seed=seed, rx=NAMES[rx_id], tx=int(tx), day=int(date))
        for arch, prediction in zip(MODELS, pred):
            histogram = np.bincount(prediction[group], minlength=10)
            confusion.extend(dict(**base, arch=arch, prediction=k, rows=int(n))
                             for k, n in enumerate(histogram) if n)
        for i, key in enumerate(CASES):
            keep = group & (cases == i)
            counts.append(dict(**base, case=key, rows=int(keep.sum()),
                               **{f"{a}_normalized_margin": masked_mean(m, keep) for a, m in normalized.items()}))
        same = (pred[0] == pred[1]) & ~masks[0] & ~masks[1]
        for wrong in np.unique(pred[0, group & same]):
            which = group & same & (pred[0] == wrong)
            errors.append(dict(**base, wrong_tx=int(wrong), count=int(which.sum()),
                               a1_same_wrong=int((which & (pred[2] == wrong)).sum())))
        for signal in SIGNALS:
            value = data[signal]
            valid = group & np.isfinite(value)
            for match, easy, hard in (("pair", masks[0] & masks[1], ~masks[0] & ~masks[1]),
                                       ("triple", masks.all(0), (~masks).all(0))):
                x, n = value[valid & easy], value[valid & hard]
                rows.append(dict(**base, signal=signal, match=match, easy=len(x), hard=len(n),
                                 auc=rank_auc(n, x) if min(len(x), len(n)) >= 8 else None))


def paired_rates(rows, labels):
    out = {}
    for metric in ("both_correct", "hstem_only", "patch_only", "both_wrong", "pair_oracle",
                   "all_wrong", "triple_oracle", "a1_rescued", "jaccard", "expected_both_wrong",
                   "excess_overlap", "shared_wrong_label", "hstem", "patch", "a1"):
        by_key = {(r["view"], r["seed"], r["rx"]): r[metric] * 100 for r in rows}
        rates = np.array([[[by_key[v, seed, rx] for rx in labels[v]] for seed in (0, 1)] for v in VIEWS])
        out[metric] = pair_summary(rates)
    return out


def subgroup_mean(rows, column):
    levels = sorted({row[column] for row in rows})
    by_key = {}
    for level in levels:
        block = [r for r in rows if r[column] == level]
        by_seed = [[r["auc"] for r in block if r["seed"] == s] for s in (0, 1)]
        by_key[level] = float(np.mean([np.mean(q) for q in by_seed if q]))
    return by_key


def signal_summary(rows):
    generator = np.random.default_rng(24501)
    result = {}
    for match in ("pair", "triple"):
        result[match] = {}
        for signal in SIGNALS:
            selected = [r for r in rows if (r["match"], r["signal"]) == (match, signal) and r["auc"] is not None]
            if not selected:
                result[match][signal] = dict(eligible=False, cells_seed=0)
                continue
            by_rx = subgroup_mean(selected, "rx")
            by_tx = subgroup_mean(selected, "tx")
            by_day = {d: float(np.mean(list(subgroup_mean(
                [r for r in selected if r["day"] == d], "rx").values()))) for d in (1, 2)
                if any(r["day"] == d for r in selected)}
            view_names = {v: {r["rx"] for r in selected if r["view"] == v} for v in VIEWS}
            arrays = [np.array([by_rx[rx] for rx in sorted(view_names[v])]) for v in VIEWS]
            mixed = np.concatenate([a[generator.integers(0, len(a), (10000, len(a)))]
                                    for a in arrays if len(a)], axis=1).mean(1)
            ci = np.quantile(mixed, [.05 / (2 * len(SIGNALS)), 1 - .05 / (2 * len(SIGNALS))]).tolist()
            stable_means = {v: float(a.mean()) for v, a in zip(VIEWS, arrays) if len(a)}
            m = float(np.mean(list(by_rx.values())))
            rx_sign = sum(x > .5 for x in by_rx.values()) / len(by_rx)
            tx_sign = sum(x > .5 for x in by_tx.values()) / len(by_tx)
            eligible = (m >= .55 and ci[0] > .5 and len(by_rx) >= 8 and len(by_tx) >= 6 and
                        len(by_day) == len(stable_means) == 2 and min(by_day.values()) > .5 and
                        min(stable_means.values()) > .5 and min(rx_sign, tx_sign) >= .75)
            result[match][signal] = dict(auc=m, simultaneous_cutoff_interval=ci, eligible=eligible,
                                         cells_seed=len(selected), rx=by_rx, tx=by_tx,
                                         positive_rx_fraction=rx_sign, positive_tx_fraction=tx_sign,
                                         views=stable_means, dates=by_day)
    return result


def readout_analysis(data, view, probes, recovered, held_cases):
    for role in ("checkpoint_val", "source_selection"):
        y, rx = (data[f"{role}_{k}"] for k in ("y", "rx"))
        for seed in (0, 1):
            for arch in ARCHES:
                prefix = f"{seed}_{arch}_{role}"
                prediction = {name: data[prefix + "_" + name + "_pred"] for name in READOUTS[:-1]}
                prediction["head"] = data[prefix + "_logits"].argmax(1)
                if role == "source_selection":
                    cases = held_cases[seed]
                for r in np.unique(rx):
                    group = rx == r
                    for layer, p in prediction.items():
                        probes.append(dict(view=view, role=role, seed=seed, arch=arch, readout=layer,
                                           rx=NAMES[r], rows=int(group.sum()), correct=int((group & (p == y)).sum())))
                        if role == "source_selection":
                            for i, name in enumerate(CASES):
                                g = group & (cases == i)
                                recovered.append(dict(view=view, seed=seed, arch=arch, readout=layer,
                                                      rx=NAMES[r], case=name, rows=int(g.sum()),
                                                      correct=int((g & (p == y)).sum())))


def readout_summary(probes, recovered, shifts):
    reports, offsets, rescue = {}, {}, {}
    for arch in ARCHES:
        reports[arch], offsets[arch], rescue[arch] = {}, {}, {}
        for layer in READOUTS:
            pos = [r for r in probes if (r["arch"], r["readout"]) == (arch, layer)]
            reports[arch][layer] = {
                role: float(np.mean([r["correct"] / r["rows"] for r in pos if r["role"] == role])) * 100
                for role in ("checkpoint_val", "source_selection")}
            supported = [r for r in recovered if r["arch"] == arch and r["readout"] == layer
                         and r["case"] == "both_wrong" and r["rows"]]
            rescue[arch][layer] = float(np.mean([r["correct"] / r["rows"] for r in supported])) * 100
            if layer == "head":
                continue
            offsets[arch][layer] = {}
            for role in ("checkpoint_val", "source_selection"):
                block = [r["relative_shift"] for r in shifts
                         if (r["arch"], r["layer"], r["role"]) == (arch, layer, role)]
                offsets[arch][layer][role] = dict(median=float(np.median(block)), cells_seed=len(block),
                                                 fraction_ge_1=float((np.asarray(block) >= 1).mean()))
    return dict(acc=reports, relative_centroid_shift=offsets, case_d_accuracy=rescue)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output
    receipt = read(out / "RECEIPT.json")
    execution = read(out / "EXECUTION.json")
    calls = check_access(out, [("s33_frozen", v, r) for v in VIEWS
                               for r in ("fit_train", "checkpoint_val", "source_selection")])
    assert execution["neural_fits"] == execution["optimizer_updates"] == 0
    assert execution["protocol_sha256"] == sha(HERE / "PROTOCOL.zh-CN.md")
    assert execution["code_sha256"] == sha(HERE / "extract.py")
    auc_rows, subgroup, wrong_pairs, rx_rows, probes, recovered = [], [], [], [], [], []
    day_rows, quality, repeats, labels, case_rows, confusion = [], [], [], {}, [], []
    for v in VIEWS:
        path = out / f"AUDIT_{v}.npz"
        assert sha(path) == receipt["packets"][v]
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
            y, rx, day = (data["source_selection_" + k] for k in ("y", "rx", "day"))
            assert len(np.unique(data["source_selection_ids"])) == len(y)
            labeled = np.column_stack([y, day])
            labels[v] = [NAMES[r] for r in np.unique(rx)]
            held_cases, held_triple = {}, {}
            for seed in (0, 1):
                keys = [f"{seed}_{a}_source_selection_logits" for a in ARCHES] + [f"{seed}_A1_logits"]
                predictions = np.stack([data[k].argmax(1) for k in keys])
                correct = predictions == y[None]
                cases = 2 * (~correct[0]).astype(np.uint8) + (~correct[1])
                held_cases[seed], held_triple[seed] = cases, (~correct).all(0)
                condition_rows(data, v, seed, correct, predictions, cases, auc_rows, subgroup, wrong_pairs, confusion)
                for r in np.unique(rx):
                    group = rx == r
                    row = dict(view=v, seed=seed, rx=NAMES[r], rows=int(group.sum()),
                               **comparison_metrics(cases, correct, predictions, labeled, group))
                    rx_rows.append(row)
                    for d in (1, 2):
                        keep = group & (day == d)
                        day_rows.append(dict(view=v, seed=seed, rx=NAMES[r], day=d, rows=int(keep.sum()),
                                             **comparison_metrics(cases, correct, predictions, labeled, keep)))
                    for i, name in enumerate(CASES):
                        keep = group & (cases == i)
                        for signal in SIGNALS:
                            value = data[signal]
                            valid = keep & np.isfinite(value)
                            case_rows.append(dict(view=v, seed=seed, rx=NAMES[r], case=name, signal=signal,
                                                  rows=int(keep.sum()), valid=int(valid.sum()),
                                                  median=float(np.median(value[valid])) if valid.any() else None))
                for i, name in enumerate(CASES):
                    keep = cases == i
                    quality.append(dict(view=v, seed=seed, case=name, rows=int(keep.sum()),
                                        cfo_valid=float(data["reliable"][keep].mean()),
                                        transition_valid=float(np.isfinite(data["transition_pos_samples"][keep]).mean())))
            repeat_pair = (held_cases[0] == 3) & (held_cases[1] == 3)
            repeat_triple = held_triple[0] & held_triple[1]
            repeats.extend(dict(view=v, rx=NAMES[r], rows=int((rx == r).sum()),
                                shared_pair_both_seeds=int((repeat_pair & (rx == r)).sum()),
                                shared_triple_both_seeds=int((repeat_triple & (rx == r)).sum())) for r in np.unique(rx))
            readout_analysis(data, v, probes, recovered, held_cases)
    shift_path = out / "CENTROID_SHIFT.json"
    assert sha(shift_path) == receipt["shift_sha256"]
    shifts = read(shift_path)
    for name, rows in (("RX_ERRORS", rx_rows), ("RX_DAY_ERRORS", day_rows),
                       ("TX_RX_DAY_MARGIN", subgroup), ("WRONG_LABELS", wrong_pairs), ("CONFUSION", confusion),
                       ("PHYSICAL_CELL_AUC", auc_rows), ("PHYSICAL_CASE_GROUPS", case_rows),
                       ("CENTROID_PROBE", probes), ("CENTROID_CASES", recovered), ("PERSISTENT_ERRORS", repeats)):
        csv_write(out / f"{name}.csv", rows)
    effects = signal_summary(auc_rows)
    rates = paired_rates(rx_rows, labels)
    summary = dict(overlap=rates, labels=labels, repeated_two_seeds=repeats, quality=quality,
                   physical=effects, readouts=readout_summary(probes, recovered, shifts), role_checks=calls,
                   decision=dict(primary_physical_association=[sig for sig, info in effects["pair"].items()
                                                               if info["eligible"]],
                                 new_candidate_fits=0))
    write(out / "SUMMARY.json", summary)


if __name__ == "__main__":
    main()
