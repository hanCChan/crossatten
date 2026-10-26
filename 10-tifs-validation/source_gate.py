"""Source-only gate. Never imports or reads target predictions."""
import json
import numpy as np
import pandas as pd
from stage import HERE, ROOT, once, sha, require_committed, identity
from register import job


def verify_pair(soft_config, hard_config):
    fields = ("params", "recipe", "data_seed", "roles", "split_sha256",
              "shared_initialization_sha256", "train_packets", "val_packets",
              "held_packets", "ema", "torch", "gpu")
    for field in fields:
        if soft_config[field] != hard_config[field]:
            raise ValueError(f"S09 control is incompatible: {field}")
    a, b = soft_config["job"], hard_config["job"]
    for field in ("dataset", "fold", "seed", "group", "epochs", "policy"):
        if a[field] != b[field]:
            raise ValueError(f"control job mismatch: {field}")


def gate(rows, reg):
    frame = pd.DataFrame(rows)
    details = {}
    for dataset in ("ManyRx", "ManySig"):
        local = frame[frame.dataset == dataset]
        run = local.groupby(["fold", "group", "seed"]).delta_pp.mean()
        folds = run.groupby("fold").mean()
        groups = run.groupby("group").mean()
        seeds = run.groupby("seed").mean()
        rxs = local.groupby("rx").delta_pp.mean()
        details[dataset] = dict(mean_delta_pp=float(folds.mean()),
            folds={str(k): float(v) for k, v in folds.items()},
            groups={str(k): float(v) for k, v in groups.items()},
            seeds={str(k): float(v) for k, v in seeds.items()},
            worst_rx_delta_pp=float(rxs.min()), worst_rx=str(rxs.idxmin()))
    rx, sig = details["ManyRx"], details["ManySig"]
    checks = dict(
        manyrx_three_positive_folds=sum(v > 0 for v in rx["folds"].values()) >= 3,
        manyrx_mean_at_least_1pp=rx["mean_delta_pp"] >= 1.,
        worst_rx_no_more_than_2pp_loss=all(d["worst_rx_delta_pp"] >= -2. for d in details.values()),
        manyrx_group_and_seed_consistent=all(v > 0 for v in list(rx["groups"].values()) + list(rx["seeds"].values())),
        manysig_no_systematic_degradation=sig["mean_delta_pp"] >= 0 and
            sum(v >= 0 for v in sig["folds"].values()) >= 3 and
            all(v >= 0 for v in list(sig["groups"].values()) + list(sig["seeds"].values())))
    return all(checks.values()), checks, details


def main():
    reg = json.loads((HERE / "registration.json").read_text())
    require_committed(HERE / "registration.json")
    rows, evidence = [], {}
    jobs = json.loads((HERE / "jobs_source.json").read_text())
    for j in jobs:
        base = HERE / "runs/source" / j["id"]
        old = dict(j, feature="first")
        old["id"] = identity(old)
        hard = ROOT / "09-phase-invariance/runs/source" / old["id"]
        verify_pair(json.loads((base / "config.json").read_text()),
                    json.loads((hard / "config.json").read_text()))
        for folder in (base, hard):
            for name in ("config.json", "metrics.json", "history.csv", "preds_raw_packet.npz"):
                path = folder / name
                require_committed(path)
                relative = str(path.relative_to(ROOT))
                evidence[relative] = sha(path)
                if folder == hard and reg["historical_controls"][relative] != sha(path):
                    raise ValueError("historical control evidence changed")
        a, b = [json.loads((folder / "metrics.json").read_text()) for folder in (base, hard)]
        if a["stop_epoch"] != 40 or b["stop_epoch"] != 40:
            raise ValueError("unequal source budget")
        if any(k in a["selected"] for k in ("CR", "CRD")):
            raise ValueError("target data leaked into source evidence")
        rule = reg["source_rule"]
        for rx, entry in a["selected"][rule]["per_rx"].items():
            ref = b["selected"][rule]["per_rx"][rx]
            if entry["total"] != ref["total"]:
                raise ValueError("prediction count mismatch")
            rows.append(dict(dataset=j["dataset"], fold=j["fold"], group=j["group"],
                seed=j["seed"], rx=rx, hard_acc_pct=100 * ref["acc"],
                soft_acc_pct=100 * entry["acc"], delta_pp=100 * (entry["acc"] - ref["acc"])))
    admitted, checks, details = gate(rows, reg)
    final = []
    if admitted:
        final = [job("final", d, fold, seed, feature)
            for d in ("ManyRx", "ManySig") for fold in range(4)
            for seed in reg["final_seeds"]
            for feature in (("hard", "soft") if d == "ManyRx" else ("soft",))]
    decision = dict(admitted=admitted, verdict="GO" if admitted else "STOP",
        checks=checks, source_details=details, jobs=final, target_jobs=final,
        evidence=evidence, registration_sha256=sha(HERE / "registration.json"),
        source_rule=reg["source_rule"], no_target_selection=True)
    once(HERE / "decision_final.json", decision)
    once(HERE / "jobs_final.json", final)
    pd.DataFrame(rows).to_csv(HERE / "source_paired_receivers.csv", index=False)
    lines = ["# Source Screening", "", f"Decision: **{decision['verdict']}**.",
        "Primary criterion frozen before fits: raw checkpoint minimizing source packet CE.",
        "EMA/held-CE alternatives are not ranked or used to rescue a failed candidate.",
        "", "| Dataset | Mean Gain (pp) | Positive Folds | Worst RX Delta (pp) |",
        "|---|---:|---:|---:|"]
    for d, values in details.items():
        lines.append(f"| {d} | {values['mean_delta_pp']:.4f} | "
                     f"{sum(v > 0 for v in values['folds'].values())}/4 | {values['worst_rx_delta_pp']:.4f} |")
    lines += ["", "Checks:", *[f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in checks.items()],
              "", "32 new soft fits; 32 exactly compatible historical hard controls reused.",
              "Missing S09 weights do not prevent prediction/config pairing but preclude historical BN diagnosis.",
              "Details: `decision_final.json`, `source_paired_receivers.csv`."]
    (HERE / "SOURCE_SCREENING.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(verdict=decision["verdict"], details=details), indent=1))


if __name__ == "__main__":
    main()
