"""Frozen RX-equal source selection and conditional independent confirmation."""
import argparse
import json
import numpy as np
from core import HERE, ROOT, save, once, sha, check_manifest, require_committed
from register import DATASETS, job


def verify_pair(a, b):
    for field in ("params", "recipe", "data_seed", "roles", "split_sha256",
                  "shared_initialization_sha256", "train_packets", "val_packets",
                  "held_packets", "ema", "torch", "gpu"):
        if a[field] != b[field]:
            raise ValueError(f"incompatible Hard pairing: {field}")
    for field in ("dataset", "fold", "group", "seed", "epochs", "policy"):
        if a["job"][field] != b["job"][field]:
            raise ValueError(f"incompatible Hard job: {field}")


def gate(rows, thresholds, confirmation=False):
    details = {}
    for dataset in DATASETS:
        local = [r for r in rows if r["dataset"] == dataset]
        runs = {}
        receivers = {}
        for r in local:
            runs.setdefault((r["fold"], r["group"], r["seed"]), []).append(r["delta_pp"])
            receivers.setdefault(r["rx"], []).append(r["delta_pp"])
        if {key[0] for key in runs} != set(range(4)):
            raise ValueError("source gate requires all four folds")
        means = {key: float(np.mean(values)) for key, values in runs.items()}
        folds = {str(f): float(np.mean([v for k, v in means.items() if k[0] == f]))
                 for f in range(4)}
        strata = {f"{g}:{s}": float(np.mean([v for k, v in means.items()
                                            if k[1:] == (g, s)]))
                  for g, s in sorted({k[1:] for k in means})}
        rx_means = {rx: float(np.mean(v)) for rx, v in receivers.items()}
        worst = min(rx_means, key=rx_means.get)
        details[dataset] = dict(mean_delta_pp=float(np.mean(list(folds.values()))),
            folds=folds, group_seed_means=strata, worst_rx=worst,
            worst_rx_delta_pp=rx_means[worst],
            worst_single_pair_delta_pp=min(r["delta_pp"] for r in local))
    a, b = details["ManyRx"], details["ManySig"]
    checks = dict(
        manyrx_mean=a["mean_delta_pp"] >= thresholds["manyrx_min_gain_pp"],
        manyrx_positive_folds=sum(v > 0 for v in a["folds"].values()) >=
            thresholds["manyrx_positive_folds"],
        worst_rx=all(d["worst_rx_delta_pp"] >= thresholds["worst_rx_min_delta_pp"]
                     for d in details.values()),
        manysig_mean=b["mean_delta_pp"] >= thresholds["manysig_min_gain_pp"],
        manysig_nonnegative_folds=sum(v >= 0 for v in b["folds"].values()) >=
            thresholds["manysig_nonnegative_folds"])
    if confirmation:
        checks["manyrx_each_group_seed_positive"] = all(
            v > 0 for v in a["group_seed_means"].values())
    return dict(admitted=all(checks.values()), checks=checks, details=details)


def collect(phase, candidate):
    jobs = json.loads((HERE / f"jobs_{phase}.json").read_text())
    rows, evidence = [], {}
    for j in [j for j in jobs if j["feature"] == candidate]:
        current = HERE / "runs" / phase / j["id"]
        if phase == "pilot":
            name = f"{j['dataset']}__f{j['fold']}__{j['group']}__s{j['seed']}__first__source"
            hard = ROOT / "09-phase-invariance/runs/source" / name
        else:
            hard = HERE / "runs" / phase / job(
                phase, j["dataset"], j["fold"], "hard", j["seed"], j["group"])["id"]
        configs = [json.loads((folder / "config.json").read_text()) for folder in (current, hard)]
        verify_pair(*configs)
        metrics = [json.loads((folder / "metrics.json").read_text()) for folder in (current, hard)]
        for folder, m in zip((current, hard), metrics):
            if m["stop_epoch"] != 40 or any(k in m["selected"] for k in ("CR", "CRD")):
                raise ValueError("budget mismatch or target result in source evidence")
            for name in ("config.json", "metrics.json", "history.csv"):
                p = folder / name
                evidence[str(p.relative_to(ROOT))] = sha(p)
        a, b = [m["selected"]["raw_packet"]["per_rx"] for m in metrics]
        if set(a) != set(b):
            raise ValueError("held receiver mismatch")
        for rx, result in a.items():
            reference = b[rx]
            if result["total"] != reference["total"]:
                raise ValueError("held packet count mismatch")
            rows.append(dict(dataset=j["dataset"], fold=j["fold"], group=j["group"],
                seed=j["seed"], candidate=candidate, rx=rx, total=result["total"],
                hard_acc_pct=100 * reference["acc"], candidate_acc_pct=100 * result["acc"],
                delta_pp=100 * (result["acc"] - reference["acc"]),
                hard_ce=reference["ce"], candidate_ce=result["ce"],
                ce_delta=result["ce"] - reference["ce"],
                f1_delta_pp=100 * (result["macro_f1"] - reference["macro_f1"])))
    return rows, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "confirmation"), required=True)
    args = parser.parse_args()
    reg = json.loads((HERE / "registration.json").read_text())
    require_committed(HERE / "registration.json")
    check_manifest(reg["code_manifest"])
    candidates = ("F1", "F2") if args.phase == "pilot" else (
        json.loads((HERE / "decision_pilot.json").read_text())["selected"],)
    choices, evidence, rows = {}, {}, []
    for candidate in candidates:
        local, seals = collect(args.phase, candidate)
        rows.extend(local)
        evidence.update(seals)
        choices[candidate] = gate(local, reg["source_gate"], args.phase == "confirmation")
    admitted = [c for c in candidates if choices[c]["admitted"]]
    selected = (max(admitted, key=lambda c: (
        choices[c]["details"]["ManyRx"]["mean_delta_pp"], c == "F1")) if admitted else None)
    jobs = []
    if args.phase == "pilot" and selected:
        jobs = [job("confirmation", d, f, kind, seed, group) for d in DATASETS
                for f in range(4) for group in ("first", "last") for seed in (12801, 12802)
                for kind in ("hard", selected)]
    decision = dict(phase=args.phase, admitted=bool(selected), selected=selected,
        verdict=("GO_CONFIRMATION" if args.phase == "pilot" else "GO_SOURCE_CONFIRMED")
                if selected else "STOP_SOURCE_GATE",
        choices=choices, jobs=jobs, evidence=evidence, no_target_reads=True,
        primary_rule="raw_packet", pilot_excluded_from_confirmation_inference=True)
    once(HERE / f"decision_{args.phase}.json", decision)
    once(HERE / f"paired_{args.phase}.json", rows)
    if args.phase == "pilot":
        once(HERE / "jobs_confirmation.json", jobs)
    lines = [f"# S12 {args.phase.title()} Source Results", "",
             f"Decision: **{decision['verdict']}**. Selected: {selected}.", "",
             "| Candidate | Dataset | Mean Gain (pp) | Positive Folds | Worst RX Delta (pp) |",
             "|---|---|---:|---:|---:|"]
    for c, values in choices.items():
        for d, detail in values["details"].items():
            positive = sum(v > 0 for v in detail["folds"].values())
            lines.append(f"| {c} | {d} | {detail['mean_delta_pp']:.4f} | "
                         f"{positive}/4 | {detail['worst_rx_delta_pp']:.4f} |")
    lines += ["", "Fixed raw_packet checkpoint rule; RX-equal run means, then fold means.",
              "Worst RX gate uses receiver means across matched runs; single-pair minima are also disclosed.",
              "No target predictions, old weights, alternative checkpoint rescue, or new search.",
              "Pilot is exploratory; confirmation inference excludes pilot data.",
              "Full paired accuracy/F1/CE and checks: paired JSON and decision JSON."]
    (HERE / f"{args.phase.upper()}_SOURCE_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(verdict=decision["verdict"], selected=selected, choices=choices), indent=1))


if __name__ == "__main__":
    main()
