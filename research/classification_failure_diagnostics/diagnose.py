"""Registered frozen-logit diagnostics. No torch, model inference, IQ or pickle access."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from stats import packet_stats, group_stats, paired_stats, macro_f1, calibration, correlation, equal_tail

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(name, data):
    path = HERE/name
    assert path.resolve().is_relative_to(HERE)
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def committed(path):
    rel = path.relative_to(ROOT).as_posix()
    blob = subprocess.check_output(["git", "-C", str(ROOT), "show", "HEAD:"+rel])
    assert blob == path.read_bytes().replace(b"\r\n", b"\n"), "Uncommitted registration/code: "+rel


def validate_registration():
    committed(HERE/"G0_PROTOCOL.json"); committed(HERE/"INPUT_REGISTRY.json")
    protocol = read(HERE/"G0_PROTOCOL.json")
    assert protocol["registry_sha256"] == digest(HERE/"INPUT_REGISTRY.json")
    for name, expected in protocol["code_sha256"].items():
        committed(HERE/name)
        assert digest(HERE/name) == expected, name
    registry = read(HERE/"INPUT_REGISTRY.json")
    for path, expected in registry["metadata_sha256"].items():
        assert digest(path) == expected, "Changed source metadata: "+path
    return registry


def load_inputs(registry):
    access = []
    predictions, metadata = {}, {}
    # Every array open must come from the finite registration whitelist.
    for item in registry["arrays"]:
        path = Path(item["path"])
        assert digest(path) == item["sha256"], "Changed sealed input: "+str(path)
        if item["kind"] == "sealed_predictions":
            assert path.name in {"source_val_predictions.npz", "held_predictions.npz"}
            with np.load(path, allow_pickle=False) as archive:
                assert set(archive.files) == set(item["keys"]) == {"logits", "y", "rx", "day"}
                data = {k:archive[k] for k in archive.files}
            assert data["logits"].shape == (len(data["y"]), 10)
            assert all(data[k].shape == data["y"].shape for k in ("rx", "day"))
            assert data["rx"].dtype.kind in "iu"
            assert data["rx"].min()>=0 and data["rx"].max()<len(item["rx_names"])
            data["rx_name"] = np.asarray(item["rx_names"])[data["rx"]]
            assert set(data["rx_name"]) == set(item["allowed_rx"])
            assert set(data["day"]) == {1, 2}
            data["p"] = packet_stats(data["logits"], data["y"])
            data["registry"] = item
            predictions[item["fold"], item["arm"], item["role"]] = data
            count = len(data["y"])
        else:
            assert item["kind"] == "source_labels_or_split_indices"
            assert item["name"] in {"fit_y", "fit_rx", "fit_day", "train_ids", "val_ids"}
            array = np.load(path, allow_pickle=False)
            assert array.ndim==1 and array.dtype.kind in "iu"
            metadata.setdefault(item["fold"], {})[item["name"]] = array
            count = len(array)
        access.append(dict(kind=item["kind"], fold=item["fold"], path=str(path), sha256=item["sha256"],
                           role=item["role"], arm=item.get("arm"), array_name=item.get("name"), rows=count))
    for fold in registry["folds"]:
        f = fold["fold"]; m = metadata[f]
        tr, va = m["train_ids"], m["val_ids"]
        n = len(m["fit_y"])
        assert len(tr)==fold["train_packets"] and len(va)==fold["val_packets"]
        assert np.array_equal(np.sort(np.r_[tr, va]), np.arange(n))
        assert not np.intersect1d(tr, va).size
        assert set(m["fit_day"]) == {1, 2}
        assert set(np.asarray(fold["rx_names"])[m["fit_rx"]]) == set(fold["fit_rx"])
        assert hashlib.sha256(tr.tobytes()).hexdigest() == fold["train_indices_array_sha256"]
        assert hashlib.sha256(va.tobytes()).hexdigest() == fold["val_indices_array_sha256"]
        for arm in ("B0", "ASK"):
            d = predictions[f, arm, "fit_validation"]
            for k in ("y", "rx", "day"):
                assert np.array_equal(d[k], m["fit_"+k][va]), "Source sample alignment"
        for role in ("fit_validation", "held_source"):
            a, b = predictions[f, "B0", role], predictions[f, "ASK", role]
            for k in ("y", "rx", "day"):
                assert np.array_equal(a[k], b[k]), "Within-input model pairing"
            assert not set(a["rx_name"]) & set(fold["forbidden_target_rx"])
    return predictions, metadata, access


def concentration(rows):
    observed = [r for r in rows if r["n"]]
    errors = sum(r["errors"] for r in observed)
    n = sum(r["n"] for r in observed)
    top_n = max(1, int(np.ceil(.1*len(observed))))
    ranked = sorted(observed, key=lambda r: (-r["errors"]/r["n"], r["rx"], r["tx"], r.get("day", 0)))
    top = ranked[:top_n]
    return dict(observed_groups=len(observed), missing_groups=len(rows)-len(observed),
                group_equal_mean_error=float(np.mean([1-r["accuracy"] for r in observed])),
                group_equal_ce=float(np.mean([r["ce"] for r in observed])),
                upper_20pct_group_error=equal_tail([1-r["accuracy"] for r in observed]),
                upper_20pct_group_ce=equal_tail([r["ce"] for r in observed]),
                top_10pct_group_count=top_n, top_10pct_error_share=sum(r["errors"] for r in top)/errors if errors else None,
                top_10pct_packet_share=sum(r["n"] for r in top)/n,
                observed_support_min=min(r["n"] for r in observed), observed_support_max=max(r["n"] for r in observed),
                worst_groups=[{k:r[k] for k in ("tx", "rx", "n", "errors", "accuracy", "ce", "training_support")} | ({"day":r["day"]} if "day" in r else {}) for r in ranked[:10]])


def risk_decomposition(rows, rx_names):
    lookup = {(r["rx"], r["tx"]):r for r in rows}
    common_tx = [t for t in range(10) if all(lookup[r, t]["n"]>0 for r in rx_names)]
    if len(common_tx)<2:
        return dict(common_tx=common_tx, identifiable=False)
    matrix = np.array([[1-lookup[r, t]["accuracy"] for t in common_tx] for r in rx_names])
    mean = matrix.mean()
    rx_part = matrix.mean(1, keepdims=True)-mean
    tx_part = matrix.mean(0, keepdims=True)-mean
    interaction = matrix-mean-rx_part-tx_part
    total = float(np.mean((matrix-mean)**2))
    parts = [float(np.mean(v*v)) for v in (rx_part, tx_part, interaction)]
    assert np.isclose(sum(parts), total, atol=1e-14)
    return dict(common_tx=common_tx, rx=len(rx_names), weighting="equal observed TX x RX cells on common TX support",
                total_risk_variance=total, rx_main_variance=parts[0], tx_main_variance=parts[1], interaction_variance=parts[2],
                fractions=[p/total for p in parts] if total else None,
                causal_or_feature_variance=False)


def analyze_model(fold, arm, role, data, metadata):
    f = fold["fold"]; y, rx, day, p = (data[k] for k in ("y", "rx_name", "day", "p"))
    allowed = fold["fit_rx"] if role=="fit_validation" else fold["held_rx"]
    m = metadata[f]
    train_rx = np.asarray(fold["rx_names"])[m["fit_rx"]][m["train_ids"]]
    train_y = m["fit_y"][m["train_ids"]]; train_day = m["fit_day"][m["train_ids"]]
    rows, combined, rx_rows = [], [], []
    for r in allowed:
        rid = rx==r
        rstat = group_stats(p, rid, y)
        f1, cm = macro_f1(y[rid], p["pred"][rid])
        rx_rows.append(dict(rx=r, **rstat, macro_f1=f1, confusion=cm))
        old = data["registry"]["expected_metrics"]["per_rx"][r]
        assert abs(rstat["accuracy"]-old["acc"])<1e-12
        assert abs(rstat["ce"]-old["ce"])<2e-6
        assert abs(rstat["margin_mean"]-old["margin"])<2e-6
        assert abs(f1-old["macro_f1"])<5.1e-5  # archived per-RX F1 may be rounded
        for t in range(10):
            ids = rid & (y==t)
            support = int(((train_rx==r)&(train_y==t)).sum())
            combined.append(dict(fold=f, arm=arm, role=role, tx=t, rx=r, training_support=support,
                                 **group_stats(p, ids, y)))
            for d in (1, 2):
                support = int(((train_rx==r)&(train_y==t)&(train_day==d)).sum())
                rows.append(dict(fold=f, arm=arm, role=role, tx=t, rx=r, day=d, training_support=support,
                                 **group_stats(p, ids & (day==d), y)))
    total = group_stats(p, np.ones(len(y), dtype=bool), y)
    assert sum(r["n"] for r in rows)==len(y) and sum(r["errors"] for r in rows)==total["errors"]
    source_summary = data["registry"]["expected_metrics"]["summary"]
    rx_accuracy = float(np.mean([r["accuracy"] for r in rx_rows]))
    assert abs(rx_accuracy-source_summary["acc"])<1e-12
    observed = [r for r in rows if r["n"]]
    lookup = {(r["tx"], r["rx"], r["day"]):r for r in rows}
    pairs = [(lookup[t, r, 1], lookup[t, r, 2]) for r in allowed for t in range(10)
             if lookup[t, r, 1]["n"] and lookup[t, r, 2]["n"]]
    group_support = dict(
        training_support_error_spearman=correlation([r["training_support"] for r in observed], [1-r["accuracy"] for r in observed]) if role=="fit_validation" else None,
        training_support_ce_spearman=correlation([r["training_support"] for r in observed], [r["ce"] for r in observed]) if role=="fit_validation" else None,
        day1_day2_tx_rx_error_spearman=correlation([1-a["accuracy"] for a,b in pairs], [1-b["accuracy"] for a,b in pairs]),
        day_pairs=len(pairs), correlation_is_descriptive=True)
    summary = dict(fold=f, arm=arm, role=role, packet=total,
                   rx_equal_accuracy=rx_accuracy, rx_equal_ce=float(np.mean([r["ce"] for r in rx_rows])),
                   rx_equal_macro_f1=float(np.mean([r["macro_f1"] for r in rx_rows])),
                   rx_equal_raw_margin=float(np.mean([r["margin_mean"] for r in rx_rows])),
                   rx_equal_normalized_margin=float(np.mean([r["normalized_margin_mean"] for r in rx_rows])),
                   tx_rx_day_concentration=concentration(rows), tx_rx_concentration=concentration(combined),
                   support_and_stability=group_support, common_support_risk_decomposition=risk_decomposition(combined, allowed),
                   calibration=calibration(p))
    write(f"groups/{arm}_f{f}_{role}.json", dict(summary=summary, per_tx_rx_day=rows, per_tx_rx=combined, per_rx=rx_rows))
    return summary, rows, combined, rx_rows


def role_comparison(fold, arm, results):
    fit_rows = results[fold, arm, "fit_validation"][1]
    held_rows = results[fold, arm, "held_source"][1]
    strata = []
    for t in range(10):
        for day in (1, 2):
            a = [r for r in fit_rows if r["tx"]==t and r["day"]==day and r["n"]]
            b = [r for r in held_rows if r["tx"]==t and r["day"]==day and r["n"]]
            if not a or not b:
                continue
            row = dict(tx=t, day=day, fit_rx_supported=len(a), held_rx_supported=len(b))
            for key in ("accuracy", "ce", "margin_mean", "normalized_margin_mean"):
                row[key+"_fit"] = float(np.mean([r[key] for r in a]))
                row[key+"_held"] = float(np.mean([r[key] for r in b]))
                row[key+"_held_minus_fit"] = row[key+"_held"]-row[key+"_fit"]
            strata.append(row)
    return dict(fold=fold, arm=arm, weighting="equal supported RX within TX/day, then equal common TX/day strata",
                strata=strata, common_strata=len(strata),
                standardized_gaps={key:float(np.mean([r[key+"_held_minus_fit"] for r in strata]))
                                   for key in ("accuracy", "ce", "margin_mean", "normalized_margin_mean")},
                same_packet_pairing=False, selection_bias="fit validation used for checkpoint selection; held-source previously inspected")


def compare_models(fold, role, predictions):
    f=fold["fold"]
    a, b = predictions[f, "B0", role], predictions[f, "ASK", role]
    rows = []
    rx_rows = []
    for r in sorted(set(a["rx_name"])):
        rid = a["rx_name"]==r
        rx_rows.append(dict(rx=r, **paired_stats(a["p"], b["p"], rid)))
        for t in range(10):
            for day in (1, 2):
                ids = rid & (a["y"]==t) & (a["day"]==day)
                rows.append(dict(tx=t, rx=r, day=day, **paired_stats(a["p"], b["p"], ids)))
    total = paired_stats(a["p"], b["p"], np.ones(len(a["y"]), dtype=bool))
    assert sum(r.get("both_wrong", 0) for r in rows) == total["both_wrong"]
    summary = dict(fold=f, role=role, packet=total,
                   rx_equal_delta_accuracy=float(np.mean([r["delta_accuracy_ask_minus_b0"] for r in rx_rows])),
                   rx_equal_diagnostic_oracle_accuracy=float(np.mean([r["diagnostic_oracle_accuracy"] for r in rx_rows])))
    write(f"pairs/f{f}_{role}.json", dict(summary=summary, per_tx_rx_day=rows, per_rx=rx_rows))
    return summary, rows, rx_rows


def repeated_held(registry, predictions):
    rows = []
    unique_packets = 0
    union_rx = sorted({r for fold in registry["folds"] for r in fold["held_rx"]})
    for r in union_rx:
        folds = [f for f in registry["folds"] if r in f["held_rx"]]
        first = predictions[folds[0]["fold"], "B0", "held_source"]
        unique_packets += int((first["rx_name"]==r).sum())
        if len(folds)<2:
            continue
        sources = [{entry["day"]:entry["sha256"] for entry in f["held_sources"] if entry["rx"]==r} for f in folds]
        assert all(s==sources[0] for s in sources), "Same RX/day input hashes required across folds"
        for arm in ("B0", "ASK"):
            collection=[]
            for fold in folds:
                d = predictions[fold["fold"], arm, "held_source"]
                ids=d["rx_name"]==r
                collection.append(dict(y=d["y"][ids], day=d["day"][ids],
                                       error=d["p"]["error"][ids], pred=d["p"]["pred"][ids]))
            assert all(np.array_equal(c["y"], collection[0]["y"]) and np.array_equal(c["day"], collection[0]["day"]) for c in collection)
            error = np.stack([c["error"] for c in collection])
            pred = np.stack([c["pred"] for c in collection])
            for t in [None, *range(10)]:
                ids=np.ones(error.shape[1], dtype=bool) if t is None else collection[0]["y"]==t
                n=int(ids.sum())
                if not n:
                    continue
                bits=np.sum(error[:, ids].astype(int)*(2**np.arange(len(folds)))[:, None], axis=0)
                rows.append(dict(rx=r, arm=arm, tx=t, folds=[f["fold"] for f in folds], n=n,
                                 fold_errors=[int(e[ids].sum()) for e in error],
                                 all_folds_wrong=int(error[:, ids].all(0).sum()),
                                 some_but_not_all_folds_wrong=int((error[:, ids].any(0)&~error[:, ids].all(0)).sum()),
                                 all_folds_correct=int((~error[:, ids].any(0)).sum()),
                                 prediction_changes_across_folds=int((pred[:, ids]!=pred[0, ids]).any(0).sum()),
                                 error_pattern_counts=np.bincount(bits, minlength=2**len(folds)).tolist()))
    return dict(scope="Same RX input cell hashes and deterministic row order only; no cross-RX pairing",
                unique_held_rx=len(union_rx), unique_held_input_rows=unique_packets,
                repeated_group_folds=[1, 2, 3], rows=rows,
                independence_caveat="Identical input rows across folds are repeated predictions; changing fit RX, split/data seed and training trajectory are confounded.")


def receiver_uncertainty(paired):
    by_rx={}
    for item in paired:
        if item[0]["role"]!="held_source":
            continue
        for r in item[2]:
            by_rx.setdefault(r["rx"], []).append(r["delta_accuracy_ask_minus_b0"])
    values=np.array([np.mean(v) for _,v in sorted(by_rx.items())])
    rng=np.random.default_rng(21111)
    draws=values[rng.integers(len(values), size=(5000, len(values)))].mean(1)
    return dict(unique_rx=len(values), mean_delta=float(values.mean()),
                conditional_rx_bootstrap_interval=np.quantile(draws, [.025, .975]).tolist(),
                rx=[dict(rx=r, folds=len(v), mean_delta=float(np.mean(v))) for r,v in sorted(by_rx.items())],
                seed=21111, draws=5000,
                interpretation="Descriptive conditional receiver interval; no training-seed, shared-fit or adaptive-hypothesis uncertainty.")


def run():
    assert not (HERE/"ANALYSIS_COMPLETE.json").exists(), "Preserve first run; do not overwrite diagnostics"
    started=time.perf_counter()
    registry=validate_registration()
    predictions, metadata, access=load_inputs(registry)
    write("ACCESS_LOG.json", dict(records=access, checkpoint_loads=0, model_forward_calls=0, iq_arrays_opened=0,
                                  source_role_declaration="All 36 array paths pinned before payload reads; metadata-only registration preceded this run"))
    results={}; summaries=[]; role_gaps=[]; paired=[]
    for fold in registry["folds"]:
        f=fold["fold"]
        for arm in ("B0", "ASK"):
            for role in ("fit_validation", "held_source"):
                result=analyze_model(fold, arm, role, predictions[f, arm, role], metadata)
                results[f, arm, role]=result; summaries.append(result[0])
            role_gaps.append(role_comparison(f, arm, results))
        for role in ("fit_validation", "held_source"):
            paired.append(compare_models(fold, role, predictions))
        print("PASS frozen diagnostic fold", f, flush=True)
    write("SUMMARY.json", dict(status="EXPLORATORY_DIAGNOSTICS_COMPLETE", models=summaries, role_comparisons=role_gaps,
                               paired_models=[p[0] for p in paired], receiver_uncertainty=receiver_uncertainty(paired),
                               margin_caveat="Raw margins depend on score scale. Normalized margins are affine-logit-scale invariant, not robustness radii.",
                               interval_caveat="Cell Wilson intervals assume independent packets and are descriptive; RX is the resampling unit for receiver uncertainty."))
    write("REPEATED_INPUT_FOLDS.json", repeated_held(registry, predictions))
    write("ANALYSIS_COMPLETE.json", dict(status="COMPLETE_NO_NEW_TRAINING", registry_sha256=digest(HERE/"INPUT_REGISTRY.json"),
                                         protocol_sha256=digest(HERE/"G0_PROTOCOL.json"),
                                         prediction_files=16, label_index_files=20, scientific_iq_reads=0,
                                         checkpoint_loads=0, model_forward_calls=0, new_model_fits=0,
                                         forbidden_target_reads=0, dates_3_4_reads=0, new_feature_extraction=0,
                                         cross_rx_packet_pairings=0, seconds=time.perf_counter()-started,
                                         historical_stops_modified=False, uncertainty_not_new_confirmatory_evidence=True))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
