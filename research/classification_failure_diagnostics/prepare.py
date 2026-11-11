"""Register read-only prediction/label metadata access before diagnostics.

Does not read prediction arrays, IQ, weights, or any scientific data pickle.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BASE = "7ae38a7b27521c9dc45d2ed1a332e5e584c3e8b8"
HISTORY = Path("D:/crossatten/worktrees/s17-rx-distortion/17b-phase-intervention")
S19 = Path("D:/crossatten/worktrees/s19-tx-spectral-evidence/19-tx-spectral-evidence")
S20 = Path("D:/crossatten/worktrees/s20-asknet-source-comparison/20-asknet-source-comparison")


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(name, obj):
    (HERE/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def main():
    assert subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip() == BASE
    assert not (HERE/"INPUT_REGISTRY.json").exists(), "Keep the first access registration"
    module = ast.parse((ROOT/"lib/xa/wisig.py").read_text(encoding="utf-8"))
    rx_order = next(ast.literal_eval(n.value)["ManyRx"] for n in module.body
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RX" for t in n.targets))
    metadata = {}
    arrays = []
    folds = []
    def meta(p):
        metadata[str(p)] = sha(p)
        return read(p)
    s19_registration = meta(S19/"registration.json")
    for f in range(4):
        cache = HISTORY/"cache"/f"ManyRx_f{f}_first_s16001"
        m = meta(cache/"fit_manifest.json")
        target = rx_order[8*f:8*(f+1)]
        source = [r for r in rx_order if r not in target]
        assert m["fit_rx"] == source[6:] and m["held_rx"] == source[:6] and m["source_only"]
        assert m["rx_names"] == sorted(source)
        old = S19/"runs"/f"ManyRx__B0__f{f}__first__s18101"
        ask = S20/"runs"/f"ASK__f{f}__first__s18101"
        bc, ac = meta(old/"config.json"), meta(ask/"config.json")
        bm, am = meta(old/"metrics.json"), meta(ask/"metrics.json")
        bh, ah = meta(old/"held_metrics.json"), meta(ask/"held_metrics.json")
        bref = meta(S20/"runs"/f"B0__f{f}__first__s18101"/"reference.json")
        b20h = meta(S20/"runs"/f"B0__f{f}__first__s18101"/"held_metrics.json")
        assert bc["source_split_seed"] == ac["split_seed"] == 16001
        assert bc["job"] == dict(phase="pilot", dataset="ManyRx", arm="M0", fold=f, group="first", seed=18101, split_seed=16001)
        pinned_manifest = f"17b-phase-intervention/cache/ManyRx_f{f}_first_s16001/fit_manifest.json"
        assert sha(cache/"fit_manifest.json") == s19_registration["historical_sha256"][pinned_manifest]
        assert ac["fit_rx"] == m["fit_rx"] and ac["held_rx"] == m["held_rx"]
        assert bc["data_seed"] == ac["data_seed"]
        assert bh["rx_names"] == ah["rx_names"] == m["rx_names"]
        assert bh["held_sources"] == ah["held_sources"] == b20h["held_sources"]
        assert bm["prediction_sha256"] == bref["source_predictions_sha256"]
        assert bh["prediction_sha256"] == bref["held_predictions_sha256"] == b20h["prediction_sha256"]
        for arm, directory, metrics, held_metrics, val_sha in [
            ("B0", old, bm, bh, bm["prediction_sha256"]),
            ("ASK", ask, am, ah, am["source_predictions_sha256"]),
        ]:
            for role, filename, expected in [
                ("fit_validation", "source_val_predictions.npz", val_sha),
                ("held_source", "held_predictions.npz", held_metrics["prediction_sha256"]),
            ]:
                path = directory/filename
                assert path.is_file()
                arrays.append(dict(kind="sealed_predictions", fold=f, arm=arm, role=role,
                                   path=str(path), sha256=expected, keys=["logits", "y", "rx", "day"],
                                   rx_names=m["rx_names"], allowed_rx=m["fit_rx"] if role=="fit_validation" else m["held_rx"],
                                   allowed_days=[1, 2], expected_metrics=metrics["source_validation"] if role=="fit_validation" else held_metrics["held_source"]))
        for name in ("fit_y", "fit_rx", "fit_day", "train_ids", "val_ids"):
            path = cache/(name+".npy")
            assert path.is_file()
            arrays.append(dict(kind="source_labels_or_split_indices", fold=f, role="fit_source_metadata",
                               name=name, path=str(path), sha256=m["sha256"][path.name], allowed_days=[1, 2], allowed_rx=m["fit_rx"]))
        folds.append(dict(fold=f, fit_rx=m["fit_rx"], held_rx=m["held_rx"], forbidden_target_rx=target,
                          rx_names=m["rx_names"], train_packets=ac["train_packets"], val_packets=ac["val_packets"],
                          data_seed=ac["data_seed"], split_seed=16001,
                          train_indices_array_sha256=ac["train_indices_sha256"], val_indices_array_sha256=ac["val_indices_sha256"],
                          held_sources=bh["held_sources"],
                          pairing_evidence="Same S19 split hashes; deterministic sequential predict(); ordered held cell hashes match; archived labels checked on load. No immutable packet IDs stored.",
                          recipe_confound="HStem max40 RX-equal CE selection vs official ASK max200 packet-equal CE selection; differing architecture/shuffle; not an operator ablation."))
    for rel in ["13-mixer-cfo-mechanisms/data.py", "17-rx-distortion/s17_data.py", "19-tx-spectral-evidence/s19_data.py",
                "19-tx-spectral-evidence/s19_train.py", "19-tx-spectral-evidence/s19_execute.py", "20-asknet-source-comparison/s20_run.py",
                "lib/xa/wisig.py", "lib/xa/explore_models.py", "lib/xa/frontends.py", "13-mixer-cfo-mechanisms/wifi.py",
                "09-phase-invariance/phase_features.py", "05-wisig/out/05-1-ManyRx-manifest.json"]:
        metadata[str(ROOT/rel)] = sha(ROOT/rel)
    write("INPUT_REGISTRY.json", dict(base_commit=BASE, client_date="2026-11-11", arrays=arrays,
                                        metadata_sha256=metadata, folds=folds,
                                        prediction_arrays_opened_during_preparation=0, iq_arrays_opened=0, checkpoint_loads=0))
    write("G0_PROTOCOL.json", dict(
        client_date="2026-11-11", base_commit=BASE, stage="Frozen classification failure diagnostics",
        status="REGISTERED_EXPLORATORY_REANALYSIS", confirmatory_or_novel_algorithm=False,
        new_classifier_fits=0, model_forward_calls=0, source_iq_reads=0, target_reads=0, dates_3_4_reads=0,
        hypotheses=["decision-margin failure", "environment-conditioned statistical change", "source training-group risk imbalance"],
        analyses=["TX x RX x day support, accuracy, CE, raw/scaled margins, hardest negatives and confusion",
                  "fit-validation vs held-source; matching TX/day strata; score-scale and calibration caveats",
                  "same-input B0/ASK disagreement, common errors, diagnostic oracle ceiling",
                  "same-RX/day frozen-input cross-fold variation with identical cell hashes, never across RX",
                  "training label support, missing cells, high-risk concentration and day rank stability",
                  "static audit of non-equalized/equalized provenance and frontends; no new IQ"],
        predefined_descriptive_settings=dict(group_tail_fraction=0.2, group_concentration_fraction=0.1,
                                              high_confidence_probability=0.9, calibration_bins=15,
                                              quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
                                              rx_bootstrap_draws=5000, bootstrap_seed=21111),
        normalized_margin="(true logit - largest wrong logit) / std(all centered logits), zero std -> 0; diagnostic, not physical boundary distance",
        no_new_go_thresholds=True, no_perturbation_search=True,
        forbidden=["IQ, checkpoint or feature-cache loads", "new inputs/equalized arrays", "target CRD/dates3-4/ManySig/Shen",
                   "unmatched cross-RX packet pairing", "retest S17B FIR/S18 subspace/S19 teacher gate",
                   "changing historical STOP or fitting a new classifier"],
        inference="All existing held-source results are exploratory; repeated folds/packets/seeds are not new independent RX.",
        code_sha256={p.name:sha(p) for p in sorted(HERE.glob("*.py"))},
        registry_sha256=sha(HERE/"INPUT_REGISTRY.json"),
    ))
    print(json.dumps(dict(status="G0_REGISTERED", prediction_files=16, label_index_files=20,
                          unique_held_rx=len({r for f in folds for r in f["held_rx"]}), data_arrays_opened=0)))


if __name__ == "__main__":
    main()
