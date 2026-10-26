"""Freeze matched attribution, source screening and conditional confirmation."""
import json
from pathlib import Path
from training import HERE, ROOT, BASELINE, FEATURES, paired_model, identity, n_params, sha, write_once


def job(phase, dataset, fold, seed, feature, policy, **kwargs):
    value = dict(phase=phase, dataset=dataset, fold=fold, seed=seed,
                 feature=feature, policy=policy, **kwargs)
    value["id"] = identity(value)
    return value


def main():
    reg = dict(baseline=BASELINE, cpu_probe="ab9dd1e4a6fad01331105638169bf2f39ab6c85b",
        source_seeds=[8101, 8102], final_seeds=[9101, 9102, 9103],
        source_epochs=40, source_groups=["first", "last"],
        features=list(FEATURES), threshold=.05, min_resultant=.1,
        attribution=dict(dataset="ManySig", epochs=[10, 6, 4, 13],
                         rules=["ema", "raw", "ema", "raw"],
                         features=["first", "null", "frequency"],
                         factorial_features=["first", "null"]),
        gate=dict(mean_delta_pp=.5, both_seeds_positive=True, both_groups_nonnegative=True,
                  worst_rx_delta_pp=-3, required_folds=3),
        selection="same feature/rule dataset-wide; source only; require 3/4 folds per dataset; "
                  "universal requires both datasets. Rank admitted choices by mean source gain, "
                  "worst RX delta, lexical name; no target retries.",
        source_rules=["raw_held", "ema_held"],
        caps=dict(smoke=5, attribution=60, source=96, final=48),
        max_formal_fits=204, target_reads="only after source decision is committed",
        expected_params={f"{d}:{f}": n_params(paired_model(d, f, 1)[0])
                         for d in ("ManyRx", "ManySig") for f in FEATURES})
    files = list((ROOT / "lib").rglob("*.py")) + [ROOT / "08-wisig-training/training.py"] + \
        [HERE / n for n in ("phase_features.py", "training.py", "09-1-register.py",
                           "09-2-launch.py", "09-3-select.py")]
    reg["code_manifest"] = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}
    write_once(HERE / "registration.json", reg)
    attr, source = [], []
    for fold in range(4):
        for seed in reg["final_seeds"]:
            for feature in reg["attribution"]["features"]:
                attr.append(job("attribution", "ManySig", fold, seed, feature, "fixed",
                                epochs=reg["attribution"]["epochs"][fold],
                                rule=reg["attribution"]["rules"][fold]))
            for feature in reg["attribution"]["factorial_features"]:
                attr.append(job("attribution", "ManySig", fold, seed, feature, "packet", epochs=200))
        for dataset in ("ManyRx", "ManySig"):
            for group in reg["source_groups"]:
                for seed in reg["source_seeds"]:
                    for feature in ("first", "centered", "curvature"):
                        source.append(job("source", dataset, fold, seed, feature, "source",
                                          group=group, epochs=40))
    smoke = [job("smoke", "ManySig", 0, 0, f, "source", group="first", epochs=2) for f in FEATURES]
    for name, jobs in (("attribution", attr), ("source", source), ("smoke", smoke)):
        write_once(HERE / f"jobs_{name}.json", jobs)
    print(json.dumps(dict(attribution=len(attr), source=len(source), smoke=len(smoke))))


if __name__ == "__main__":
    main()
