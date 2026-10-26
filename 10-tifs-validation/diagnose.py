"""Read-only historical errors and source signal mechanisms; never selects a model."""
import json
import numpy as np
import pandas as pd
import torch
from stage import HERE, ROOT, DATA, RX, s09, sha
from xa.wisig import load_cell
from xa.explore_models import frequency_views
from xa.frontends import rms_normalize
from features import phase_view


def main():
    torch.set_num_threads(2)
    rows, txrows, curves, evidence = [], [], [], {}
    folders = []
    for path in (ROOT / "07-wifi-osu/runs/final").glob("ManyRx_source_training*/metrics.json"):
        folders.append(("S07_frequency_mild", path))
    for path in (ROOT / "09-phase-invariance/runs/source").glob("ManyRx*__first__source/metrics.json"):
        folders.append(("S09_hard_source", path))
    for name, path in folders:
        m = json.loads(path.read_text())
        j = m["job"]
        evidence[str(path.relative_to(ROOT))] = sha(path)
        if name == "S07_frequency_mild":
            labels = ("CR", "CRD")
        else:
            labels = ("raw_packet", "ema_packet", "raw_held", "ema_held")
        for label in labels:
            predpath = path.parent / f"preds_{label}.npz"
            evidence[str(predpath.relative_to(ROOT))] = sha(predpath)
            with np.load(predpath) as pred:
                for rx in sorted(k for k in pred if not k.startswith("y_")):
                    y = pred[f"y_{rx}"]
                    correct = pred[rx].argmax(1) == y
                    days = (1, 2) if label != "CRD" else (3, 4)
                    offset = 0
                    for day in days:
                        _, ref_y = load_cell("ManyRx", rx, day)
                        stop = offset + len(ref_y)
                        if not np.array_equal(ref_y, y[offset:stop]):
                            raise ValueError("historical prediction/date order mismatch")
                        ok = correct[offset:stop]
                        rows.append(dict(stage=name, label=label, fold=j["fold"], seed=j["seed"],
                            group=j.get("group", "all"), rx=rx, day=day,
                            accuracy_pct=100 * ok.mean(), errors=int((~ok).sum()), packets=len(ok)))
                        for tx in np.unique(ref_y):
                            mask = ref_y == tx
                            txrows.append(dict(stage=name, label=label, fold=j["fold"], seed=j["seed"],
                                rx=rx, day=day, tx=int(tx), packets=int(mask.sum()),
                                errors=int((~ok[mask]).sum()), accuracy_pct=100 * ok[mask].mean()))
                        offset = stop
                    if offset != len(y):
                        raise ValueError("unmatched historical packets")
        histpath = path.parent / "history.csv"
        hist = pd.read_csv(histpath)
        evidence[str(histpath.relative_to(ROOT))] = sha(histpath)
        if name == "S09_hard_source":
            for rule in labels:
                ep = m["selected"][rule]["epoch"]
                curves.append(dict(stage=name, fold=j["fold"], seed=j["seed"], group=j["group"],
                    rule=rule, selected_epoch=ep, selected_acc=100 * m["selected"][rule]["summary"]["acc_mean"],
                    source_ce=m["best"][rule]["ce"]))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "historical_rx_day_errors.csv", index=False)
    pd.DataFrame(txrows).to_csv(HERE / "historical_tx_errors.csv", index=False)
    pd.DataFrame(curves).to_csv(HERE / "historical_checkpoint_selection.csv", index=False)
    spectral = []
    for dataset in RX:
        for rx in RX[dataset]:
            for day in (1, 2, 3, 4):
                x, y = load_cell(dataset, rx, day)
                base = frequency_views(rms_normalize(torch.from_numpy(x)))
                z = torch.complex(base[:, 0], base[:, 1])
                a = z.abs()
                reliable = (a[:, 1:] >= .05) & (a[:, :-1] >= .05)
                soft = phase_view(base)[:, 3:, 1:]
                unit = phase_view(base, "unweighted")[:, 3:, 1:]
                for tx in np.unique(y):
                    mask = torch.from_numpy(y == tx)
                    u = torch.complex(unit[mask, 0], unit[mask, 1])
                    resultant = u.mean(0).abs().mean()
                    spectral.append(dict(dataset=dataset, rx=rx, day=day, tx=int(tx),
                        packets=int(mask.sum()), low_amp_ratio=float((a[mask] < .05).float().mean()),
                        valid_adjacent_ratio=float(reliable[mask].float().mean()),
                        soft_weight_mean=float(soft[mask].square().sum(1).sqrt().mean()),
                        phase_circular_dispersion=float(1 - resultant)))
    pd.DataFrame(spectral).to_csv(HERE / "spectral_rx_day_tx.csv", index=False)
    s07 = df[df.stage == "S07_frequency_mild"]
    by_run = s07.groupby(["label", "fold", "seed", "rx"])[["errors", "packets"]].sum()
    by_run["accuracy_pct"] = 100 * (1 - by_run.errors / by_run.packets)
    taskmeans = by_run.accuracy_pct.groupby("label").mean()
    daymeans = s07.groupby(["day", "rx"]).accuracy_pct.mean().groupby("day").mean()
    worst = by_run.xs("CRD").accuracy_pct.groupby("rx").mean().sort_values().head(6)
    source = df[(df.stage == "S09_hard_source") & (df.label == "raw_packet")]
    source_runs = source.groupby(["fold", "group", "seed", "rx"])[["errors", "packets"]].sum()
    source_mean = float((100 * (1 - source_runs.errors / source_runs.packets)).mean())
    signal = pd.DataFrame(spectral)
    target = by_run.xs("CRD").accuracy_pct.groupby("rx").mean()
    reliability = signal[(signal.dataset == "ManyRx") & signal.day.isin([3, 4])]
    low = reliability.groupby("rx").low_amp_ratio.mean()
    corr = float(target.corr(low))
    save = __import__("stage").save
    save(HERE / "diagnostic_evidence.json", dict(evidence=evidence, cr_pct=float(taskmeans["CR"]),
        crd_pct=float(taskmeans["CRD"]), raw_packet_source_pct=float(source_mean),
        low_amp_vs_crd_acc_rx_pearson=corr, historical_weights_available=False,
        diagnostic_role="descriptive only; candidate and gates preregistered before this analysis"))
    text = ["# ManyRx Error Diagnosis", "",
        f"S07 frozen frequency/mild recipe: CR {taskmeans['CR']:.4f}%, CRD {taskmeans['CRD']:.4f}%; "
        f"gap {taskmeans['CR'] - taskmeans['CRD']:.4f} pp.",
        f"S09 hard raw-packet source-held score: {source_mean:.4f}% (different fit-RX pool/recipe; not a causal gap).",
        "", "| Date Index | RX-Equal Accuracy (%) |", "|---|---:|",
        *[f"| {day} | {acc:.4f} |" for day, acc in daymeans.items()],
        "", "| Worst CRD RX | Seed-Mean Accuracy (%) |", "|---|---:|",
        *[f"| {rx} | {acc:.4f} |" for rx, acc in worst.items()],
        "", f"Descriptive RX correlation of low-amplitude bin ratio with CRD accuracy: {corr:.4f}. "
        "This is observational, not identification of a causal failure mechanism.",
        "Low amplitude and circular phase dispersion are saved per RX/TX/day; absent TX cells remain absent.",
        "", "CR and CRD contain DIFFERENT packets and dates. There is no legitimate "
        "'CR-correct packet becomes CRD-wrong' indicator. Only marginal errors for common TX/RX can be compared.",
        "S07-S09 checkpoint files were not distributed/restored. BN running statistics, "
        "feature activations and checkpoint interventions are therefore unavailable for those models.",
        "Historical raw/EMA selection curves are analyzed, but are not a substitute for BN measurements.",
        "Hypotheses: threshold discontinuity and low-amplitude phase uncertainty are plausible; "
        "the source gate can falsify their utility. RX/date errors alone do not prove either is the main bottleneck.",
        "No fold-specific recipe was changed in response to these diagnostics.",
        "", "Evidence: `historical_rx_day_errors.csv`, `historical_tx_errors.csv`, "
        "`historical_checkpoint_selection.csv`, `spectral_rx_day_tx.csv`, `diagnostic_evidence.json`."]
    (HERE / "MANYRX_ERROR_DIAGNOSIS.md").write_text("\n".join(text) + "\n")
    print("Historical diagnosis written; CRD", taskmeans["CRD"], flush=True)


if __name__ == "__main__":
    main()
