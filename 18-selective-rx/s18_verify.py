"""Post-result QA for prospective S18-D0S; no IQ or forward-pass reads.

Only registered source-checkpoint files are hashed. Source metadata and pooled
feature caches are read; raw IQ files and held/target data are never opened.
CE/margin can be checked for aggregation consistency, but the registered cache
does not contain per-packet logits, so they cannot be independently regenerated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from s18_core import HERE, ROOT, HISTORY, git, now, read, sha, validate, write
from s18_data import halves, locations
from s18_decide import decide
from s18_support_run import validate as validate_support, estimation_ids
from s18_sparse_stats import (compress, direction_metrics, fit_directions, fit_rx_probe,
                              overlap, random_basis, scatters, score_rx_probe)
from xa.wisig import receivers


LAYERS = ("stem1", "stem_mid", "tokens", "mixer")
RESULTS = HERE/"d0_support_matched"
CHECKS = []


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def same(actual, expected, path="value", rtol=1e-8, atol=1e-10):
    """Strict structure and tight numeric equality for independent recomputation."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), path+": keys")
        for key in expected:
            same(actual[key], expected[key], path+"."+str(key), rtol, atol)
    elif isinstance(expected, (list, tuple, np.ndarray)):
        a, e = np.asarray(actual), np.asarray(expected)
        require(a.shape == e.shape, path+": shape")
        if e.dtype.kind in "iufcb":
            np.testing.assert_allclose(a, e, rtol=rtol, atol=atol, err_msg=path)
        else:
            require(np.array_equal(a, e), path+": contents")
    elif isinstance(expected, (float, np.floating)):
        require(np.isfinite(actual) and np.isfinite(expected), path+": nonfinite")
        require(np.isclose(actual, expected, rtol=rtol, atol=atol), path+": numeric mismatch")
    else:
        require(actual == expected, path+": mismatch")


def fixed_probe(z, y, rx, day, vz, vy, vrx, vday):
    fitted = fit_rx_probe(z, y, rx, day)
    scored = score_rx_probe(fitted, vz, vy, vrx, vday)
    return dict(
        accuracy_rx_equal=scored["accuracy"], chance=scored["chance"],
        per_rx={str(int(r)): float(v) for r, v in zip(fitted["rx_labels"], scored["per_rx"])},
        method="TX/day-conditioned equal-cell nearest train RX centroid; source-only diagnostic with truth labels",
    )


def verify_layer(layer, record, cache):
    z, vz = cache["train_"+layer], cache["val_"+layer]
    y, rx, day = (cache[k] for k in ("y", "rx", "day"))
    vy, vrx, vday = (cache[k] for k in ("vy", "vrx", "vday"))
    fitted = fit_directions(z, y, rx, day)
    validation = scatters(vz, vy, vrx, vday)
    u = fitted["U"]
    a, b = halves(y, rx, day)

    def basis(ids):
        return fit_directions(z[ids], y[ids], rx[ids], day[ids])["U"]

    stability = dict(
        packet_halves=overlap(basis(a), basis(b)),
        days=overlap(basis(np.flatnonzero(day == 1)), basis(np.flatnonzero(day == 2))),
        tx_halves=overlap(basis(np.flatnonzero(y < 5)), basis(np.flatnonzero(y >= 5))),
        source_validation=overlap(u, fit_directions(vz, vy, vrx, vday)["U"]),
    )
    expected = dict(
        channels=z.shape[1], rank=3, epsilon=fitted["epsilon"],
        train_cell_count_min=int(fitted["cell_counts"].min()),
        train_cell_count_max=int(fitted["cell_counts"].max()),
        train_metrics=direction_metrics(fitted["S_R"], fitted["S_T"], u),
        val_metrics=direction_metrics(validation["S_R"], validation["S_T"], u),
        stability=stability,
        rx_probe=fixed_probe(z, y, rx, day, vz, vy, vrx, vday),
        chance_overlap=3/z.shape[1], U=u, center=fitted["center"], U_tx=fitted["U_tx"],
    )
    same(record, expected, "layer."+layer)
    return fitted, validation


def verify_score(score, labels, rx_ids, names, path):
    """Reconstruct all classification metrics available from per-RX confusions."""
    per = score["per_rx"]
    expected_names = {names[int(r)] for r in np.unique(rx_ids)}
    require(set(per) == expected_names and len(per) == 18, path+": fit18 RX set")
    totals = np.zeros((10, 10), dtype=np.int64)
    for rid in np.unique(rx_ids):
        name = names[int(rid)]
        cm = np.asarray(per[name]["confusion"])
        require(cm.shape == (10, 10) and cm.dtype.kind in "iu" and np.all(cm >= 0), path+": invalid confusion")
        truth_counts = np.bincount(labels[rx_ids == rid], minlength=10)
        same(cm.sum(1), truth_counts, path+"."+name+".truth_counts", rtol=0, atol=0)
        totals += cm
        diag = np.diag(cm).astype(float)
        denominator = cm.sum(0)+cm.sum(1)
        f1 = np.divide(2*diag, denominator, out=np.zeros(10), where=denominator > 0)
        recall = np.divide(diag, cm.sum(1), out=np.zeros(10), where=cm.sum(1) > 0)
        same(per[name]["acc"], float(diag.sum()/cm.sum()), path+"."+name+".acc")
        same(per[name]["macro_f1"], float(f1.mean()), path+"."+name+".macro_f1")
        same(per[name]["recall"], recall.round(4), path+"."+name+".recall")
    per_tx = {str(t): float(totals[t, t]/totals[t].sum()) for t in range(10)}
    same(score["per_tx"], per_tx, path+".per_tx")
    for metric in ("acc", "macro_f1", "ce", "margin"):
        same(score["summary"][metric], float(np.mean([v[metric] for v in per.values()])), path+".summary."+metric)
    same(score["summary"]["worst_rx"], min(v["acc"] for v in per.values()), path+".worst_rx")
    same(score["summary"]["worst_tx"], min(per_tx.values()), path+".worst_tx")
    require(score["summary"]["margin_sd"] >= 0, path+": margin_sd")


def verify_deltas(row, base):
    current = row["score"]
    a, b = current["summary"], base["summary"]
    expected = dict(
        delta_acc_pp=100*(a["acc"]-b["acc"]),
        delta_f1_pp=100*(a["macro_f1"]-b["macro_f1"]),
        delta_ce=a["ce"]-b["ce"], delta_margin=a["margin"]-b["margin"],
        normalized_delta_margin=(a["margin"]-b["margin"])/max(b["margin_sd"], 1e-12),
        worst_rx_paired_pp=100*min(current["per_rx"][r]["acc"]-base["per_rx"][r]["acc"] for r in current["per_rx"]),
        worst_tx_paired_pp=100*min(current["per_tx"][t]-base["per_tx"][t] for t in current["per_tx"]),
    )
    same({key: row[key] for key in expected}, expected, "deltas."+row["arm"])


def verify_fold(fold, reg, checkpoint_records):
    record = read(RESULTS/f"D0_fold{fold}.json")
    feature_path = RESULTS/f"cache/features_f{fold}.npz"
    require(sha(feature_path) == record["feature_cache_sha256"], "pooled feature cache SHA")
    cache_dir, run = locations(fold)
    manifest = read(cache_dir/"fit_manifest.json")
    cfg, original = read(run/"config.json"), read(run/"metrics.json")
    matches = [r for r in checkpoint_records if r["job"]["fold"] == fold]
    require(len(matches) == 1, "unique checkpoint record")
    historical_record = matches[0]
    expected_job = dict(phase="pilot", dataset="ManyRx", arm="M0", fold=fold, group="first", seed=18101, split_seed=16001)
    same(cfg["job"], expected_job, "checkpoint.job")
    same(original["job"], expected_job, "checkpoint.metrics_job")
    for filename in ("config.json", "metrics.json"):
        require(sha(run/filename) == historical_record["files_sha256"][filename], "historical metadata SHA "+filename)
    digest = sha(run/"model.pt")
    require(digest == historical_record["checkpoint_sha256"] == original["model_sha256"] == record["checkpoint_sha256"], "checkpoint SHA")
    source, target = receivers("ManyRx", fold)
    fit_rx, held_rx = source[6:], source[:6]
    require(manifest["fit_rx"] == fit_rx and manifest["held_rx"] == held_rx and manifest["source_only"], "cache source roles")
    require(len(fit_rx) == 18 and not set(fit_rx) & set(target+held_rx), "fit18 disjointness")
    # Read labels and split IDs only. fit_x.npy is deliberately never opened or
    # hashed here; its original registered loader check is retained in access log.
    pool = {}
    for key in ("y", "rx", "day"):
        p = cache_dir/f"fit_{key}.npy"
        require(sha(p) == manifest["sha256"][p.name], "source metadata SHA "+key)
        pool[key] = np.load(p, allow_pickle=False)
    ids = {}
    for part in ("train", "val"):
        p = cache_dir/f"{part}_ids.npy"
        require(sha(p) == manifest["sha256"][p.name], "source split SHA "+part)
        ids[part] = np.load(p, allow_pickle=False)
    require(not np.intersect1d(ids["train"], ids["val"]).size, "estimation/evaluation disjointness")
    require(np.array_equal(np.sort(np.r_[ids["train"], ids["val"]]), np.arange(len(pool["y"]))), "complete source split")
    require({manifest["rx_names"][int(r)] for r in np.unique(pool["rx"])} == set(fit_rx), "metadata fit RXs")
    require(set(np.unique(pool["day"])) == {1, 2} and set(np.unique(pool["y"])) == set(range(10)), "metadata date/TX scope")
    with np.load(feature_path, allow_pickle=False) as cache:
        estimation = estimation_ids(pool, ids["train"], fold, cap=reg["estimation_cap_per_cell"])
        same(cache["train_indices"], estimation, "cached estimation indices", rtol=0, atol=0)
        same(cache["validation_indices"], ids["val"], "cached evaluation indices", rtol=0, atol=0)
        require(__import__("hashlib").sha256(estimation.tobytes()).hexdigest() == record["train_indices_sha256"], "estimation ID SHA")
        for key in ("y", "rx", "day"):
            same(cache[key], pool[key][estimation], "cached train labels "+key, rtol=0, atol=0)
            same(cache["v"+key], pool[key][ids["val"]], "cached val labels "+key, rtol=0, atol=0)
        require(record["fold"] == fold and record["data_seed"] == cfg["data_seed"], "fold/config seed")
        require(record["train_packets"] == len(estimation) and record["validation_packets"] == len(ids["val"]), "packet counts")
        primary = None
        for layer in LAYERS:
            fitted, validation = verify_layer(layer, record["layers"][layer], cache)
            if layer == "tokens":
                primary = fitted, validation
        fitted, validation = primary
        verify_score(record["baseline"], cache["vy"], cache["vrx"], manifest["rx_names"], "baseline")
        same(record["baseline"]["summary"]["acc"], original["source_validation"]["summary"]["acc_mean"], "original baseline accuracy")
        for receiver, previous in original["source_validation"]["per_rx"].items():
            for metric in ("acc", "macro_f1", "recall", "confusion"):
                same(record["baseline"]["per_rx"][receiver][metric], previous[metric], "original baseline "+receiver+" "+metric)
        expected_arms = {"selective", "tx_dominant", "random_0", "random_1", "random_2"}
        require(len(record["interventions"]) == 5 and {r["arm"] for r in record["interventions"]} == expected_arms, "fixed intervention matrix")
        for row in record["interventions"]:
            arm = row["arm"]
            require(row["alpha"] == reg["alpha"] == .5 and row["rank"] == reg["rank"] == 3, "fixed rank/strength")
            if arm == "selective":
                u = fitted["U"]
            elif arm == "tx_dominant":
                u = fitted["U_tx"]
            else:
                draw = int(arm.split("_")[-1])
                u = random_basis(64, 3, 18200+10*fold+draw)
            train_z = compress(cache["train_tokens"], fitted["center"], u, alpha=.5)
            val_z = compress(cache["val_tokens"], fitted["center"], u, alpha=.5)
            transformed = scatters(val_z, cache["vy"], cache["vrx"], cache["vday"])
            for matrix, name in (("S_R", "rx"), ("S_T", "tx")):
                expected_reduction = 1-float(np.trace(transformed[matrix]))/max(float(np.trace(validation[matrix])), 1e-12)
                same(row[name+"_scatter_reduction"], expected_reduction, "scatter reduction "+arm+" "+name)
            same(row["rx_probe"], fixed_probe(train_z, cache["y"], cache["rx"], cache["day"], val_z, cache["vy"], cache["vrx"], cache["vday"]), "intervention probe "+arm)
            verify_score(row["score"], cache["vy"], cache["vrx"], manifest["rx_names"], arm)
            verify_deltas(row, record["baseline"])
    require(record["costs"]["parameters"] == cfg["active_params"] == 82426, "baseline parameter count")
    require(record["costs"]["added_trainable_parameters_static"] == 0 and record["costs"]["proposed_gate_parameters"] == 3, "bottleneck parameter counts")
    CHECKS.append(f"fold{fold}: checkpoint/metadata/cache SHA; disjoint IDs; 4-layer scatter/bases/stability/probes; 6-arm confusion metrics/deltas; fixed rank/strength")
    print(f"PASS fold{fold}: source metadata, checkpoint, four feature layers, six classification scores", flush=True)
    return record, dict(fold=fold, checkpoint_sha256=digest, feature_cache_sha256=record["feature_cache_sha256"], fit_rx=fit_rx)


def verify_access(folds):
    rows = [json.loads(line) for line in (HERE/"access.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    # Original fold0 stopped before checkpoint/feature access (4 fit cache
    # records); support-only audit read five metadata arrays per fold (20);
    # D0S subsequently read 4 fit caches + 1 checkpoint per fold (20).
    require(len(rows) == 44, "44 whitelisted original-guard/support-audit/D0S access entries required")
    for fold_record in folds:
        fold = fold_record["fold"]
        selected = [r for r in rows if r["fold"] == fold]
        require(len(selected) == (14 if fold == 0 else 10), "fixed source access counts per fold")
        cache_dir, run = locations(fold)
        manifest = read(cache_dir/"fit_manifest.json")
        cache_rows = [r for r in selected if r["kind"] == "fit_cache"]
        require(len(cache_rows) == (8 if fold == 0 else 4), "fixed original-guard and D0S source cache entries")
        expected_paths = {str((cache_dir/f"fit_{key}.npy").resolve()) for key in ("x", "y", "rx", "day")}
        actual_paths = {str(Path(r["path"]).resolve()) for r in cache_rows}
        require(actual_paths == expected_paths, "source access path whitelist")
        for row in cache_rows:
            require(row["dataset"] == "ManyRx" and row["rx"] == fold_record["fit_rx"] and row["days"] == [1, 2], "source access role whitelist")
            require(row["sha256"] == manifest["sha256"][Path(row["path"]).name], "access/cache manifest SHA")
        metadata_rows = [r for r in selected if r["kind"] == "source_support_metadata"]
        require(len(metadata_rows) == 5, "five metadata-only support audit entries")
        metadata_paths = {str((cache_dir/f"{name}.npy").resolve()) for name in ("fit_y", "fit_rx", "fit_day", "train_ids", "val_ids")}
        require({str(Path(r["path"]).resolve()) for r in metadata_rows} == metadata_paths, "metadata-only path whitelist")
        for row in metadata_rows:
            require(row["rx"] == fold_record["fit_rx"] and row["days"] == [1, 2], "metadata-only source role whitelist")
            require(row["sha256"] == manifest["sha256"][Path(row["path"]).name], "support metadata manifest SHA")
        ckpts = [r for r in selected if r["kind"] == "frozen_source_checkpoint"]
        require(len(ckpts) == 1, "one source checkpoint entry")
        require(Path(ckpts[0]["path"]).resolve() == (run/"model.pt").resolve() and ckpts[0]["sha256"] == fold_record["checkpoint_sha256"], "checkpoint access whitelist")
    require({r["kind"] for r in rows} == {"fit_cache", "source_support_metadata", "frozen_source_checkpoint"}, "all access kinds whitelisted")
    return dict(entries=len(rows), fit_cache_entries=20, support_metadata_entries=20, source_checkpoint_entries=4,
                held_source_entries=0, real_target_entries=0, many_sig_entries=0)


def run():
    required = [RESULTS/"decision_d0.json"]
    required += [RESULTS/f"D0_fold{fold}.json" for fold in range(4)]
    required += [RESULTS/f"cache/features_f{fold}.npz" for fold in range(4)]
    require(all(path.is_file() for path in required), "all D0 results and pooled caches must exist before QA")
    reg = validate()
    support_reg = validate_support()
    original_attempt = read(HERE/"attempt_fold0.json")
    require(original_attempt["status"] == "STOP_PRE_FEATURE_SUPPORT" and original_attempt["checkpoint_loaded"] is False
            and original_attempt["learned_features_read"] is False and original_attempt["interventions_run"] == 0,
            "original pre-feature support stop must remain preserved")
    require(support_reg["feature_outcomes_before_amendment"] == 0 and support_reg["rank_alpha_controls_gates_changed"] is False,
            "prospective support amendment constraints")
    require(not list(HERE.glob("D0_fold*.json")), "original failed D0 has no scientific result replacement")
    historical_records = read(HISTORY/"17b-phase-intervention/final_evidence/MANIFEST.json")["records"]
    results, folds = [], []
    for fold in range(4):
        result, verified = verify_fold(fold, reg, historical_records)
        results.append(result)
        folds.append(verified)
    actual_decision = read(RESULTS/"decision_d0.json")
    expected_decision = decide(results)
    expected_decision.update(stage="S18-D0S", support_amendment=True, original_d0_status="STOP_PRE_FEATURE_SUPPORT",
                             sparse_estimator="TX-conditional RX scatter and paired-TX common-RX PSD scatter; no imputation")
    actual_decision.pop("time")
    expected_decision.pop("time")
    # The registered deterministic decision must match exactly, including every
    # scalar, receiver-cluster bootstrap output, and boolean. No tolerance here.
    require(actual_decision == expected_decision, "registered decision recomputation must match JSON exactly")
    access = verify_access(folds)
    changed = git("diff", "--name-only", reg["base_commit"], "--").splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    require(all(path.startswith("18-selective-rx/") for path in changed+untracked), "changes must stay within S18")
    CHECKS.extend(["registered decision and RX-cluster interval: exact recomputation",
                   "access log: source fit18/date1-2 and sealed checkpoint whitelist only",
                   "historical registered evidence unchanged; git diff confined to S18"])
    audit = dict(
        time=now(), status="PASS", registration_sha256=sha(HERE/"registration.json"),
        support_registration_sha256=sha(HERE/"registration_support.json"),
        result_files_sha256={path.name: sha(path) for path in required if path.suffix == ".json"},
        historical_base_commit=reg["base_commit"], audit_code_sha256=sha(Path(__file__)),
        folds=folds, access=access, checks=CHECKS,
        decision_status=actual_decision["status"], new_fits=0,
        raw_iq_read_by_this_audit=False, model_weights_loaded_by_this_audit=False,
        ce_margin_check_scope="Per-RX aggregate consistency and all deltas/gates checked; no per-packet logits in registered cache, so CE/margin were not independently regenerated",
        source_iq_hash_scope="Original registered source-loader access hashes checked against immutable manifest; this audit never reopens fit_x.npy",
        changed_tracked_paths=changed, untracked_paths=untracked,
    )
    write(HERE/"FINAL_ACCESS_AUDIT.json", audit)
    print("PASS S18: registered hashes, four checkpoints, feature statistics, confusion metrics, decision, source-only access and historical preservation", flush=True)


if __name__ == "__main__":
    run()
