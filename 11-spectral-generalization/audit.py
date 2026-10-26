"""S11 zero-fit spectral mechanisms and actual saved ASKNet bandwidths."""
import json
import numpy as np
import pandas as pd
import torch
from core import HERE, ROOT, s09, sha, once, save, partition, manifest, require_committed
from register import job
from spectral import views, normalize_statistics, estimate, split_stability, center_weight


@torch.no_grad()
def batch_views(x):
    return torch.cat([views(torch.from_numpy(x[i:i + 256]))
                      for i in range(0, len(x), 256)]).numpy()


def centroid_accuracy(fit, held, y, yh, weight, phase_pair=False):
    a = fit * weight[None, None]
    b = held * weight[None, None]
    if phase_pair:
        pair = weight * np.pad(weight[:-1], (1, 0))
        a[:, 3:] = fit[:, 3:] * pair[None, None]
        b[:, 3:] = held[:, 3:] * pair[None, None]
    a, scale = normalize_statistics(a)
    b = b / scale[None, :, None]
    # Unit length retains direction; one common normalization per observed sample.
    a = a.reshape(len(a), -1)
    b = b.reshape(len(b), -1)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-6)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-6)
    labels = np.unique(y)
    means = np.stack([a[y == tx].mean(0) for tx in labels])
    dist = (b ** 2).sum(1, keepdims=True) + (means ** 2).sum(1)[None] - 2 * b @ means.T
    return float((labels[dist.argmin(1)] == yh).mean())


def checkpoint_audit():
    rows, evidence = [], {}
    for path in sorted((ROOT / "10-tifs-validation/runs/ask").glob("*/metrics.json")):
        m = json.loads(path.read_text())
        checkpoint = path.parent / "model.pt"
        if sha(checkpoint) != m["weights_sha256"]["model.pt"]:
            raise ValueError("ASKNet checkpoint hash mismatch")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        logits = state["cutoff.selection_logits"].numpy()
        k = [8, 16, 24, 32, 48, 64][int(logits.argmax())]
        trace = m["k_history"][-1]
        rows.append(dict(dataset=m["job"]["dataset"], fold=m["job"]["fold"], seed=m["job"]["seed"],
            checkpoint_k=k, last_epoch_k=trace["argmax_k"], same=k == trace["argmax_k"],
            selected_epoch=m["selected"]["source_packet"]["epoch"], retained_bins=2*k-1,
            half_bandwidth_mhz=(k-1)*25/256, weight_sha256=sha(checkpoint)))
        evidence[str(path.relative_to(ROOT))] = sha(path)
    if len(rows) != 24:
        raise ValueError("24 ASKNet checkpoints required")
    pd.DataFrame(rows).to_csv(HERE / "ask_checkpoint_bands.csv", index=False)
    missing = {}
    for stage in ("07-wifi-osu", "08-wisig-training", "09-phase-invariance"):
        paths = list((ROOT / stage / "runs").rglob("*.pt"))
        missing[stage] = dict(local_pt_files=len(paths), original_weights_available=bool(paths))
    save(HERE / "checkpoint_audit.json", dict(ask_verified=24, historical=missing,
        recovery_evidence="recovery/MEMORY.md: S00-S06 released, S07-S09 excluded from both releases",
        new_restore_fits=2, no_target_predictions_read=True, evidence=evidence))
    return rows


def main():
    torch.set_num_threads(2)
    require_committed(HERE / "registration.json")
    reg = json.loads((HERE / "registration.json").read_text())
    bands = checkpoint_audit()
    rows, spectra = [], []
    for dataset in ("ManyRx", "ManySig"):
        for fold in range(4):
            for group in ("first", "last"):
                j = job("audit", dataset, fold, group, 8101)
                split, held, roles, seals = partition(j)
                # audit is explicitly a source grouping; no held packets enter weight fitting.
                fit = split["train"]
                value = batch_views(fit["x"])
                prior = estimate(value, fit["y"], fit["rx"], fit["day"])
                stability = split_stability(value, fit["y"], fit["rx"], fit["day"])
                weight = prior["weight"]
                cache = HERE / "cache" / j["id"]
                cache.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache / "prior.npz", **{k:v for k,v in prior.items() if isinstance(v, np.ndarray)})
                seals_path = HERE / "priors" / f"{j['id']}.json"
                save(seals_path, dict(job=j, split_sha256=seals, roles=roles,
                    prior_sha256=sha(cache / "prior.npz"), weight=weight.tolist(),
                    c=prior["c"], eps=prior["eps"], eligible_rx_per_tx=prior["eligible_rx_per_tx"],
                    included_cells=prior["included_cells"], stability=stability,
                    training_packets=len(fit["y"]), forbidden_inputs="packet-val, held RX, target RX/days"))
                for index in range(256):
                    spectra.append(dict(dataset=dataset, fold=fold, group=group, bin=index,
                        mhz=(index-128)*25/256, b=prior["b"][index], r=prior["r"][index],
                        d=prior["d"][index], uncertainty=prior["uncertainty"][index],
                        score=prior["score"][index], weight=weight[index]))
                controls = dict(hard=np.ones(256, np.float32), F1=center_weight(), F2=weight,
                                outer=1-center_weight())
                bsum = max(float(prior["b"].sum()), 1e-12)
                base_ratio = float((prior["r"] + prior["d"]).sum()) / bsum
                for rx, cell in held.items():
                    hv = batch_views(cell["x"])
                    for kind, w in controls.items():
                        acc = centroid_accuracy(value, hv, fit["y"], cell["y"], w,
                                                phase_pair=kind in ("F1", "outer"))
                        retain = float((prior["b"] * w**2).sum()) / bsum
                        ratio = float(((prior["r"]+prior["d"]) * w**2).sum()) / max(float((prior["b"] * w**2).sum()), 1e-12)
                        rows.append(dict(dataset=dataset, fold=fold, group=group, rx=rx, kind=kind,
                            centroid_acc_pct=100*acc, tx_scatter_retention=retain,
                            drx_day_over_dtx_ratio_change=ratio/base_ratio,
                            half_weight_correlation=stability["correlation"],
                            packets=len(cell["y"])))
                print("Source spectral audit", dataset, fold, group, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "source_centroid_probes.csv", index=False)
    pd.DataFrame(spectra).to_csv(HERE / "source_spectral_statistics.csv", index=False)
    choices = {}
    for kind in ("F1", "F2"):
        checks, detail = {}, {}
        for dataset in ("ManyRx", "ManySig"):
            local = frame[(frame.dataset == dataset) & (frame.kind == kind)]
            hard = frame[(frame.dataset == dataset) & (frame.kind == "hard")]
            keys = ["fold", "group", "rx"]
            a, b = local.set_index(keys), hard.set_index(keys)
            delta = a.centroid_acc_pct - b.centroid_acc_pct
            folds = delta.groupby("fold").mean()
            detail[dataset] = dict(mean_centroid_gain_pp=float(folds.mean()),
                positive_folds=int((folds > 0).sum()), tx_retention=float(local.tx_scatter_retention.mean()),
                nuisance_ratio_change=float(local.drx_day_over_dtx_ratio_change.mean()),
                half_weight_correlation=float(local.half_weight_correlation.mean()))
        rx, sig = detail["ManyRx"], detail["ManySig"]
        checks = dict(manyrx_positive_folds=rx["positive_folds"] >= 3,
            manyrx_centroid_gain=rx["mean_centroid_gain_pp"] >= .5,
            manysig_not_degraded=sig["mean_centroid_gain_pp"] >= -.5,
            tx_retention=all(v["tx_retention"] >= .65 for v in detail.values()),
            nuisance_ratio=all(v["nuisance_ratio_change"] <= 1.05 for v in detail.values()),
            prior_reproducibility=kind == "F1" or all(v["half_weight_correlation"] >= .5 for v in detail.values()))
        choices[kind] = dict(admitted=all(checks.values()), checks=checks, details=detail)
    admitted = [kind for kind in ("F2", "F1") if choices[kind]["admitted"]]
    selected = admitted[0] if admitted else None
    jobs = [job("source", d, f, g, seed, selected) for d in ("ManyRx", "ManySig")
            for f in range(4) for g in ("first", "last") for seed in (8101, 8102)] if selected else []
    evidence_paths = [HERE / "source_centroid_probes.csv", HERE / "source_spectral_statistics.csv",
                      HERE / "checkpoint_audit.json", HERE / "ask_checkpoint_bands.csv"] + list((HERE / "priors").glob("*.json"))
    decision = dict(admitted=bool(selected), selected=selected, choices=choices,
        jobs=jobs, verdict="GO_SOURCE" if selected else "STOP",
        evidence=manifest(evidence_paths), zero_training_selection=True,
        no_target_predictions_read=True, source_probe_seed=8101,
        limits="linear nearest centroid is a necessary registered support probe, not final network efficacy; "
               "no unsupported GN/spectral combination; unselected route not trained")
    once(HERE / "decision_physics.json", decision)
    once(HERE / "jobs_source.json", jobs)
    lines = ["# Source Spectral Physics", "", f"Zero-fit route decision: **{decision['verdict']}**, selected={selected}.",
        "All weights use only the 70% fit packets from legal fit RX/day1-2; "
        "packet validation and held RX never enter the prior. Cartesian phase avoids unwrap/branch cuts.",
        "Equal TX/RX/day centroid scatter, complete day pairs, cell uncertainty included, "
        "minimum10 fit packets/cell and3RX/TX. Per-cell split-half reproducibility is descriptive support.",
        "Scores use fixed5-bin reflect smoothing and floor0.5; no thresholds are searched.",
        "Source nearest-centroid diagnostics do use held RX labels solely for route admission.",
        "", "## Route Evidence", "```json", json.dumps(choices, indent=1), "```", "",
        "F1 k32 is a diagnostic control, not novelty. F2 is a Fisher-like variance-ratio prior, "
        "not a calibrated causal separation or new mathematical primitive.",
        "No target-derived per-fold cutoff or receiver-specific model is constructed.",
        "Missing TX/RX cells are excluded, not filled with zero. Both audits and final scores are RX equal.",
        "Raw per-bin/cell evidence: `source_spectral_statistics.csv`, `priors/`, `source_centroid_probes.csv`."]
    (HERE / "SOURCE_SPECTRAL_PHYSICS.md").write_text("\n".join(lines)+"\n")
    lines = ["# S11 Code And Checkpoint Audit", "", "Evidence cutoff:2026-10-26.",
        "24 actual S10 ASKNet checkpoint hashes verified, saved cutoff logits read directly.",
        "Each saved k and last-trace k are separately recorded in `ask_checkpoint_bands.csv`.",
        "k32:63bins, +/-3.02734375MHz; k24:47bins, +/-2.24609375MHz at25MHz/256.",
        "No claim that ASKNet gain is due solely to bandwidth: SKO and backbone also differ.",
        "Original S07-S09 weights were not found locally/in the previously audited two releases; "
        "only S00-S06 were distributed. Two registered source-only S09 hard refits will be stored under S11 "
        "and labeled refits unless all original tensor/checkpoint evidence matches.",
        "S10 soft weights exist but are not silently treated as hard-phase weights for BN attribution.",
        "Historical S00-S10 files remain read-only; S11 imports independently loaded frozen training modules.",
        "Budget ambiguity: two complete F1/F2 grids require64 not32. Implement both and conduct zero-fit "
        "source-only route admission first, train at most one32-fit grid.",
        "Day-transfer16 limit permits only last group/seed8101, not replicated two-group/two-seed confirmation.",
        "Evidence: `registration.json`, `checkpoint_audit.json`, `ask_checkpoint_bands.csv`."]
    (HERE / "S11_CODE_AND_CHECKPOINT_AUDIT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(verdict=decision["verdict"], selected=selected, choices=choices), indent=1))


if __name__ == "__main__":
    main()
