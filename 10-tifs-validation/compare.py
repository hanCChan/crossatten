"""Receiver-cluster intervals; no cross-seed pairing claims for external methods."""
import json
import numpy as np
import pandas as pd
from stage import HERE, ROOT, sha, save


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(10101)
    samples = values[rng.integers(0, len(values), (20000, len(values)))].mean(1)
    lo, hi = np.quantile(samples, [.025, .975])
    return float(values.mean()), float(lo), float(hi)


def main():
    rows, runrows, evidence = [], [], {}
    for phase in ("ask", "final", "ablation"):
        for path in (HERE / "runs" / phase).glob("*/target_metrics.json"):
            m = json.loads(path.read_text())
            j = m["job"]
            evidence[str(path.relative_to(ROOT))] = sha(path)
            for task, entry in m["selected"].items():
                runrows.append(dict(dataset=j["dataset"], model=f"S10_{j['feature']}", phase=phase,
                    fold=j["fold"], seed=j["seed"], task=task,
                    accuracy_pct=100 * entry["summary"]["acc_mean"],
                    macro_f1_pct=100 * entry["summary"]["f1_mean"],
                    worst_rx_pct=100 * entry["summary"]["worst_acc"]))
                for rx, e in entry["per_rx"].items():
                    rows.append(dict(dataset=j["dataset"], model=f"S10_{j['feature']}", phase=phase,
                        fold=j["fold"], seed=j["seed"], rx=rx, task=task,
                        accuracy_pct=100 * e["acc"], macro_f1_pct=100 * e["macro_f1"]))
    current = pd.DataFrame(rows)
    current.to_csv(HERE / "final_receiver_metrics.csv", index=False)
    runs = pd.DataFrame(runrows)
    runs.to_csv(HERE / "final_run_metrics.csv", index=False)
    historic_path = ROOT / "06-paper-validation/summary/paper/receiver_metrics.csv"
    evidence[str(historic_path.relative_to(ROOT))] = sha(historic_path)
    historic = pd.read_csv(historic_path)
    allrx = []
    for (d, model, task), frame in historic.groupby(["dataset", "model", "task"]):
        if model not in ("IB-RFF", "WaveMLP"):
            continue
        for _, r in frame.iterrows():
            allrx.append(dict(dataset=d, model=model, task=task, rx=r.rx, fold=r.fold,
                accuracy_pct=r.seed_mean_accuracy_pct, macro_f1_pct=r.seed_mean_macro_f1_pct,
                evidence_type="historical_fair_reproduction"))
    for stage, filename, key, wanted in (
        ("07-wifi-osu", "wifi_final_receiver_metrics.csv", "pipeline", "source_training"),
        ("08-wisig-training", "final_receiver_metrics.csv", "variant", "candidate")):
        path = ROOT / stage / filename
        evidence[str(path.relative_to(ROOT))] = sha(path)
        frame = pd.read_csv(path)
        frame = frame[(frame[key] == wanted) & (frame.dataset == ("ManyRx" if stage.startswith("07") else "ManySig"))]
        for (d, task, rx), f in frame.groupby(["dataset", "task", "rx"]):
            allrx.append(dict(dataset=d, model="S07_frequency_mild" if stage.startswith("07") else "S08_hard",
                task=task, rx=rx, fold=int(f.fold.iloc[0]), accuracy_pct=f.accuracy_pct.mean(),
                macro_f1_pct=f.macro_f1_pct.mean(), evidence_type="historical_frozen_final"))
    path = ROOT / "09-phase-invariance/final_receiver_metrics.csv"
    evidence[str(path.relative_to(ROOT))] = sha(path)
    s09 = pd.read_csv(path)
    frame = s09[(s09.feature == "first") & (s09.policy == "packet")]
    for (d, task, rx), f in frame.groupby(["dataset", "task", "rx"]):
        allrx.append(dict(dataset=d, model="S09_hard_packet", task=task, rx=rx,
            fold=int(f.fold.iloc[0]), accuracy_pct=f.accuracy_pct.mean(),
            macro_f1_pct=f.macro_f1_pct.mean(), evidence_type="historical_paired_control"))
    for (d, model, task, rx), f in current.groupby(["dataset", "model", "task", "rx"]):
        allrx.append(dict(dataset=d, model=model, task=task, rx=rx, fold=int(f.fold.iloc[0]),
            accuracy_pct=f.accuracy_pct.mean(), macro_f1_pct=f.macro_f1_pct.mean(),
            evidence_type="S10_three_seed"))
    grouped = pd.DataFrame(allrx)
    grouped.to_csv(HERE / "comparison_receiver_means.csv", index=False)
    summary = []
    for (d, model, task), f in grouped.groupby(["dataset", "model", "task"]):
        mean, lo, hi = bootstrap(f.accuracy_pct)
        local = runs[(runs.dataset == d) & (runs.model == model) & (runs.task == task)] if not runs.empty else runs
        summary.append(dict(dataset=d, model=model, task=task, accuracy_pct=mean,
            macro_f1_pct=float(f.macro_f1_pct.mean()), worst_rx_pct=float(f.accuracy_pct.min()),
            ci95_low_pct=lo, ci95_high_pct=hi,
            seed_sd_pp=float(local.groupby("seed").accuracy_pct.mean().std()) if len(local) else None,
            fold_sd_pp=float(f.groupby("fold").accuracy_pct.mean().std()),
            evidence_type=f.evidence_type.iloc[0]))
    pd.DataFrame(summary).to_csv(HERE / "final_comparisons.csv", index=False)
    pairs = []
    for (d, task), f in grouped.groupby(["dataset", "task"]):
        for model in ("S10_soft", "S10_ask", "S07_frequency_mild", "S08_hard", "S09_hard_packet"):
            candidate = f[f.model == model].set_index("rx")
            if candidate.empty:
                continue
            for ref in ("IB-RFF", "WaveMLP", "S10_ask", "S09_hard_packet", "S10_hard"):
                if ref == model:
                    continue
                baseline = f[f.model == ref].set_index("rx")
                if baseline.empty:
                    continue
                if set(candidate.index) != set(baseline.index):
                    raise ValueError("receiver sets differ in comparison")
                delta = candidate.accuracy_pct - baseline.accuracy_pct
                mean, lo, hi = bootstrap(delta.to_numpy())
                pairs.append(dict(dataset=d, task=task, model=model, baseline=ref,
                    delta_pp=mean, ci95_low_pp=lo, ci95_high_pp=hi, receivers=len(delta),
                    improved_rx=int((delta > 0).sum()), same_training_seeds=model == "S10_soft" and ref in ("S10_hard", "S09_hard_packet")))
    pd.DataFrame(pairs).to_csv(HERE / "receiver_cluster_intervals.csv", index=False)
    decision = json.loads((HERE / "decision_final.json").read_text())
    save(HERE / "comparison_evidence.json", dict(evidence=evidence, bootstrap_replicates=20000,
        seed=10101, cluster="receiver, averaged across seeds before resampling",
        limitations="conditions on fixed fold models; receivers from same fold share training; "
                    "not a population guarantee or simultaneous multiplicity-adjusted interval",
        candidate_verdict=decision["verdict"], global_sota_claim=False))
    lines = ["# Final CR/CRD Comparisons", "", f"Reliability candidate decision: **{decision['verdict']}**.",
        "RX-equal means. Historical controls are kept with their original recipes, never target-ranked.",
        "", "| Dataset | Model | Task | Accuracy (%) | Macro-F1 (%) | Worst RX (%) | 95% RX CI |",
        "|---|---|---|---:|---:|---:|---|"]
    for r in summary:
        lines.append(f"| {r['dataset']} | {r['model']} | {r['task']} | {r['accuracy_pct']:.4f} | "
                     f"{r['macro_f1_pct']:.4f} | {r['worst_rx_pct']:.4f} | "
                     f"[{r['ci95_low_pct']:.3f}, {r['ci95_high_pct']:.3f}] |")
    lines += ["", "Seed/fold dispersion: `final_run_metrics.csv`, `final_comparisons.csv`.",
        "Paired receiver differences: `receiver_cluster_intervals.csv`.",
        "Seeds for ASKNet2023-2025 differ from our seeds. Intervals pair RX, not training randomness.",
        "Published/offered upstream logs are not mixed with fair multi-seed accuracy.",
        "A larger mean does not prove stable superiority when the RX difference interval overlaps zero.",
        "This bounded baseline set is not an exhaustive literature leaderboard; no global SOTA claim.",
        "Shen97.48% remains historical evidence under its own LoRa frontend/protocol; no new Shen fit was made."]
    (HERE / "FINAL_CR_CRD_COMPARISONS.md").write_text("\n".join(lines) + "\n")
    print("Comparison reports written.")


if __name__ == "__main__":
    main()
