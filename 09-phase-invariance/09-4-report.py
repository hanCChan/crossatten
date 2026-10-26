"""Matched source, target, factorial and receiver-cluster reporting."""
import argparse
import csv
import json
import numpy as np
from training import HERE, ROOT, sha, save


def write_csv(name, rows):
    if not rows:
        return
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_report():
    jobs = json.loads((HERE / "jobs_source.json").read_text())
    rows, classes = [], []
    for job in jobs:
        result = json.loads((HERE / "runs/source" / job["id"] / "metrics.json").read_text())
        for rule, entry in result["selected"].items():
            for rx, metric in entry["per_rx"].items():
                rows.append(dict(dataset=job["dataset"], fold=job["fold"], group=job["group"],
                    seed=job["seed"], feature=job["feature"], rule=rule, rx=rx, epoch=entry["epoch"],
                    accuracy_pct=100 * metric["acc"], macro_f1_pct=100 * metric["macro_f1"], ce=metric["ce"]))
                cm = np.asarray(metric["confusion"])
                for c in range(len(cm)):
                    classes.append(dict(dataset=job["dataset"], fold=job["fold"], group=job["group"],
                        seed=job["seed"], feature=job["feature"], rule=rule, rx=rx, tx_class=c,
                        recall_pct=100 * cm[c, c] / max(cm[c].sum(), 1)))
    write_csv("source_receiver_metrics.csv", rows)
    write_csv("source_tx_metrics.csv", classes)
    lines = ["# S09 Source Receiver Screening\n\n",
             "All arms trained exactly40 epochs, four folds x two source groups x two seeds, "
             "both datasets. Checkpoint ranking and evaluation use the same held RX, so "
             "source improvements include selection optimism. No target ranking.\n\n",
             "| Dataset | Feature | Rule | Mean held RX % | Macro-F1 % | Worst held RX % |\n",
             "|---|---|---|---:|---:|---:|\n"]
    for dataset in ("ManyRx", "ManySig"):
        for feature in ("first", "centered", "curvature"):
            for rule in ("raw_held", "ema_held"):
                values = [r for r in rows if (r["dataset"], r["feature"], r["rule"]) == (dataset, feature, rule)]
                lines.append(f"| {dataset} | {feature} | {rule} | "
                    f"{np.mean([r['accuracy_pct'] for r in values]):.2f} | "
                    f"{np.mean([r['macro_f1_pct'] for r in values]):.2f} | "
                    f"{min(r['accuracy_pct'] for r in values):.2f} |\n")
    (HERE / "SOURCE_RESULTS.md").write_text("".join(lines))


def bootstrap(rx_deltas):
    # Receiver clusters keep all seed-level differences together. Conditional
    # on fitted seeds, not full training/data uncertainty or multiplicity corrected.
    values = np.asarray(rx_deltas, dtype=float)
    rng = np.random.default_rng(9917)
    means = values[rng.integers(0, len(values), (20000, len(values)))].mean(1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def target_report():
    decision = json.loads((HERE / "decision_final.json").read_text())
    rows, raw_rx = [], []
    for job in decision["target_jobs"]:
        path = HERE / "runs" / job["phase"] / job["id"] / "target_metrics.json"
        if not path.exists():
            raise ValueError("cannot report incomplete formal target evaluation")
        result = json.loads(path.read_text())
        for task, entry in result["selected"].items():
            for rx, metric in entry["per_rx"].items():
                raw_rx.append(dict(dataset=job["dataset"], phase=job["phase"], fold=job["fold"],
                    seed=job["seed"], feature=job["feature"], policy=job["policy"], task=task, rx=rx,
                    accuracy_pct=100 * metric["acc"], macro_f1_pct=100 * metric["macro_f1"], ce=metric["ce"],
                    correct=metric["correct"], total=metric["total"]))
    write_csv("final_receiver_metrics.csv", raw_rx)
    keys = sorted({(r["dataset"], r["phase"], r["feature"], r["policy"], r["task"]) for r in raw_rx})
    for dataset, phase, feature, policy, task in keys:
        values = [r for r in raw_rx if (r["dataset"], r["phase"], r["feature"], r["policy"], r["task"]) ==
                  (dataset, phase, feature, policy, task)]
        seed_means = [np.mean([r["accuracy_pct"] for r in values if r["seed"] == seed])
                      for seed in (9101, 9102, 9103)]
        rx_means = [np.mean([r["accuracy_pct"] for r in values if r["rx"] == rx])
                    for rx in sorted({r["rx"] for r in values})]
        rows.append(dict(dataset=dataset, phase=phase, feature=feature, policy=policy, task=task,
            rx_accuracy_pct=float(np.mean(seed_means)), macro_f1_pct=float(np.mean([r["macro_f1_pct"] for r in values])),
            seed_sd_pp=float(np.std(seed_means, ddof=1)), worst_rx_pct=float(min(rx_means))))
    write_csv("final_metrics.csv", rows)
    fold_rows, seed_rows = [], []
    for dataset, phase, feature, policy, task in keys:
        values = [r for r in raw_rx if (r["dataset"], r["phase"], r["feature"], r["policy"], r["task"]) ==
                  (dataset, phase, feature, policy, task)]
        for fold in range(4):
            local = [r for r in values if r["fold"] == fold]
            fold_rows.append(dict(dataset=dataset, phase=phase, feature=feature, policy=policy,
                                  task=task, fold=fold, accuracy_pct=float(np.mean([r["accuracy_pct"] for r in local]))))
        for seed in (9101, 9102, 9103):
            local = [r for r in values if r["seed"] == seed]
            seed_rows.append(dict(dataset=dataset, phase=phase, feature=feature, policy=policy,
                                  task=task, seed=seed, accuracy_pct=float(np.mean([r["accuracy_pct"] for r in local]))))
    write_csv("final_fold_metrics.csv", fold_rows)
    write_csv("final_seed_metrics.csv", seed_rows)
    comparisons = []
    for dataset in ("ManySig", "ManyRx"):
        pairs = [("attribution", "first", "fixed", "null", "fixed"),
                 ("attribution", "first", "fixed", "frequency", "fixed"),
                 ("attribution", "null", "fixed", "frequency", "fixed"),
                 ("attribution", "first", "packet", "null", "packet"),
                 ("attribution", "first", "fixed", "first", "packet"),
                 ("attribution", "null", "fixed", "null", "packet")]
        if dataset in decision["selected"]:
            pairs.append(("final", decision["selected"][dataset].split(":")[0], "fixed", "first", "fixed"))
        for phase, a, pa, b, pb in pairs:
            for task in ("CR", "CRD"):
                left = [r for r in raw_rx if (r["dataset"], r["phase"], r["feature"], r["policy"], r["task"]) ==
                        (dataset, phase, a, pa, task)]
                right = [r for r in raw_rx if (r["dataset"], r["phase"], r["feature"], r["policy"], r["task"]) ==
                         (dataset, phase, b, pb, task)]
                if not left and not right:
                    continue
                li = {(r["rx"], r["seed"]): r for r in left}
                ri = {(r["rx"], r["seed"]): r for r in right}
                if set(li) != set(ri) or not li:
                    raise ValueError("incomplete paired comparison")
                rxs = sorted({k[0] for k in li})
                deltas = [np.mean([li[(rx, s)]["accuracy_pct"] - ri[(rx, s)]["accuracy_pct"]
                                   for s in (9101, 9102, 9103)]) for rx in rxs]
                mean, lo, hi = bootstrap(deltas)
                comparisons.append(dict(dataset=dataset, phase=phase, task=task,
                    contrast=f"{a}:{pa}-{b}:{pb}", delta_pp=mean, ci95_low_pp=lo, ci95_high_pp=hi,
                    improved_rx=int(sum(v > 0 for v in deltas)), receivers=len(rxs)))
    write_csv("paired_comparisons.csv", comparisons)
    factorial = []
    for task in ("CR", "CRD"):
        index = {(r["rx"], r["seed"], r["feature"], r["policy"]): r["accuracy_pct"] for r in raw_rx
                 if r["dataset"] == "ManySig" and r["phase"] == "attribution" and r["task"] == task}
        rxs = sorted({k[0] for k in index})
        deltas = [np.mean([(index[(rx, s, "first", "fixed")] - index[(rx, s, "null", "fixed")]) -
                           (index[(rx, s, "first", "packet")] - index[(rx, s, "null", "packet")])
                           for s in (9101, 9102, 9103)]) for rx in rxs]
        mean, lo, hi = bootstrap(deltas)
        factorial.append(dict(dataset="ManySig", task=task, interaction_pp=mean, ci95_low_pp=lo, ci95_high_pp=hi))
    write_csv("factorial_interaction.csv", factorial)
    with (ROOT / "06-paper-validation/summary/paper/receiver_metrics.csv").open() as stream:
        external = list(csv.DictReader(stream))
    baseline_comparisons = []
    for dataset, phase, feature, policy, task in keys:
        if task != "CRD" or (phase == "attribution" and (feature != "first" or policy != "fixed")) or \
                (phase == "final" and feature == "first"):
            continue
        values = [r for r in raw_rx if (r["dataset"], r["phase"], r["feature"], r["policy"], r["task"]) ==
                  (dataset, phase, feature, policy, task)]
        current = {rx: float(np.mean([r["accuracy_pct"] for r in values if r["rx"] == rx]))
                   for rx in sorted({r["rx"] for r in values})}
        for model in ("IB-RFF", "WaveMLP", "FAN_source_selected"):
            refs = {r["rx"]: float(r["seed_mean_accuracy_pct"]) for r in external
                    if r["dataset"] == dataset and r["task"] == task and r["model"] == model}
            if not refs:
                continue
            if set(refs) != set(current):
                raise ValueError("external baseline RX differ")
            mean, lo, hi = bootstrap([current[rx] - refs[rx] for rx in sorted(refs)])
            baseline_comparisons.append(dict(dataset=dataset, phase=phase, feature=feature, policy=policy,
                baseline=model, candidate_pct=float(np.mean(list(current.values()))),
                baseline_pct=float(np.mean(list(refs.values()))), delta_pp=mean,
                ci95_low_pp=lo, ci95_high_pp=hi,
                improved_rx=int(sum(current[rx] > refs[rx] for rx in refs)), receivers=len(refs)))
    write_csv("baseline_comparisons.csv", baseline_comparisons)
    lines = ["# S09 Formal Matched Ablations\n\n",
             "Fresh seeds9101/9102/9103; four folds; RX-equal means. Shared weights, "
             "packet partitions and training shuffle explicitly matched. All target reads "
             "occurred after source-only mechanism selection was committed. Public RX "
             "are not new blind tests.\n\n",
             "| Dataset | Phase | Feature | Policy | Task | Acc % | Macro-F1 % | Seed SD pp | Worst RX % |\n",
             "|---|---|---|---|---|---:|---:|---:|---:|\n"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['phase']} | {r['feature']} | {r['policy']} | {r['task']} | "
                     f"{r['rx_accuracy_pct']:.2f} | {r['macro_f1_pct']:.2f} | {r['seed_sd_pp']:.2f} | {r['worst_rx_pct']:.2f} |\n")
    lines.extend(["\n## Paired Receiver Contrasts\n\n",
                  "| Dataset | Phase | Task | Contrast | Delta pp | Conditional RX95% interval | Improved RX |\n",
                  "|---|---|---|---|---:|---|---:|\n"])
    for r in comparisons:
        lines.append(f"| {r['dataset']} | {r['phase']} | {r['task']} | {r['contrast']} | {r['delta_pp']:+.2f} | "
                     f"[{r['ci95_low_pp']:+.2f}, {r['ci95_high_pp']:+.2f}] | {r['improved_rx']}/{r['receivers']} |\n")
    lines.append("\nIntervals condition on trained seeds and cluster by receiver. They are "
                 "descriptive, not multiplicity corrected, and exclude full seed/dataset uncertainty. "
                 "Fixed-versus-packet changes joint EMA/stopping policy; it does not isolate EMA alone. "
                 "Null and three-channel models are initially function-equivalent; small GPU "
                 "kernel rounding differences can diverge during optimization.\n\n## Factorial Interaction\n\n")
    for r in factorial:
        lines.append(f"- {r['task']}: difference of phase-content gains, fixed minus packet policy "
                     f"{r['interaction_pp']:+.2f}pp [{r['ci95_low_pp']:+.2f}, {r['ci95_high_pp']:+.2f}].\n")
    lines.extend(["\n## Frozen Fair Baselines\n\n",
        "| Dataset | Variant | Baseline | Candidate % | Baseline % | Delta pp | Conditional RX95% interval |\n",
        "|---|---|---|---:|---:|---:|---|\n"])
    for r in baseline_comparisons:
        lines.append(f"| {r['dataset']} | {r['phase']}:{r['feature']}:{r['policy']} | {r['baseline']} | "
                     f"{r['candidate_pct']:.2f} | {r['baseline_pct']:.2f} | {r['delta_pp']:+.2f} | "
                     f"[{r['ci95_low_pp']:+.2f}, {r['ci95_high_pp']:+.2f}] |\n")
    lines.append("\nExternal baseline seeds are unpaired. S05 released single-seed references "
                 "are separate: ManySig IB-RFF93.57/WaveMLP93.13, ManyRx89.19/88.68. "
                 "No global-SOTA claim: ASKNet is a newly located official competitor not "
                 "fairly retrained here; source verification is in LITERATURE_CHECK.md. "
                 "Mechanism selection aggregates source folds: RX source/target roles overlap "
                 "across different folds, so it is not strictly nested fold-local hyperparameter "
                 "selection. No day3-4 source selection or target outcome ranking occurred.\n")
    (HERE / "S09_FINAL_RESULTS.md").write_text("".join(lines))
    save(HERE / "report_evidence.json", dict(decision_sha256=sha(HERE / "decision_final.json"),
        files={n: sha(HERE / n) for n in ("final_metrics.csv", "final_receiver_metrics.csv",
               "paired_comparisons.csv", "factorial_interaction.csv", "baseline_comparisons.csv",
               "final_fold_metrics.csv", "final_seed_metrics.csv", "S09_FINAL_RESULTS.md")}))
    print(json.dumps(dict(rows=len(raw_rx), summary=rows, comparisons=comparisons, factorial=factorial)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("source", "target"))
    (source_report if parser.parse_args().phase == "source" else target_report)()
