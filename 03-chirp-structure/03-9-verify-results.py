#!/usr/bin/env python3
"""Verify saved budgets, source-only checkpoint selection, metrics, and frozen tables."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from xa.data import fold_roles  # noqa: E402
from xa.evaluate import summarize  # noqa: E402
from xa.guard import load_protocol  # noqa: E402
from xa.train import RECIPES, stable_seed  # noqa: E402


def verify_role(role, expected_rx):
    if set(role["per_rx"]) != set(expected_rx):
        raise ValueError("receiver set does not match the frozen protocol")
    for rx, metrics in role["per_rx"].items():
        confusion = np.asarray(metrics["confusion"])
        if (confusion.shape != (10, 10) or np.any(confusion < 0)
                or not np.array_equal(confusion.sum(1), np.full(10, 200))):
            raise ValueError(f"{rx}: invalid confusion matrix or packet denominators")
        accuracy = float(np.trace(confusion) / confusion.sum())
        tp = confusion.diagonal()
        precision = tp / np.maximum(confusion.sum(0), 1)
        recall = tp / confusion.sum(1)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        if not np.isclose(metrics["acc"], accuracy, rtol=0, atol=1e-12):
            raise ValueError(f"{rx}: accuracy differs from confusion matrix")
        if not np.isclose(metrics["macro_f1"], f1.mean(), rtol=0, atol=1e-12):
            raise ValueError(f"{rx}: macro-F1 differs from confusion matrix")
    if summarize(role["per_rx"]) != role["summary"]:
        raise ValueError("receiver-equal summary differs from per-receiver metrics")


def verify_stage(stage, registered, protocol_sha):
    seeds = [0] if stage == "smoke" else load_protocol()["seeds"][
        "exploration" if stage == "explore" else "confirmation"
    ]
    folds = [0] if stage == "smoke" else [0, 1]
    expected = {f"{name}__f{fold}__s{seed}" for name in registered
                for fold in folds for seed in seeds}
    paths = list((HERE / "runs" / stage).glob("*/metrics.json"))
    if {p.parent.name for p in paths} != expected:
        raise ValueError(f"{stage}: incomplete/unexpected fit set")
    records = []
    for path in sorted(paths):
        result = json.loads(path.read_text())
        config = json.loads((path.parent / "config.json").read_text())
        history = list(csv.DictReader((path.parent / "history.csv").open()))
        arm = registered[result["arm"]]
        fold, seed = result["fold"], result["seed"]
        roles = fold_roles(fold)
        epochs = 2 if stage == "smoke" else 200
        if config["arm"] != arm or config["recipe"] != dict(RECIPES[arm["recipe"]], epochs=epochs):
            raise ValueError(f"{path.parent.name}: arm/recipe drift")
        if (config["protocol_sha256"] != protocol_sha or config["roles"] != roles
                or config["batch"] != 64 or config["smoke"] != (stage == "smoke")
                or config["fold"] != fold or config["seed"] != seed
                or config["stage"] != ("explore" if stage == "smoke" else stage)
                or config["data_seed"] != stable_seed("P-H5x160", fold, seed)
                or result["params"] != arm["expected_params"]):
            raise ValueError(f"{path.parent.name}: protocol, seed, or parameter mismatch")
        if (result["epochs_budget"] != epochs or result["stop_epoch"] != epochs
                or [int(row["epoch"]) for row in history] != list(range(1, epochs + 1))):
            raise ValueError(f"{path.parent.name}: incomplete epoch budget")
        best = min(history, key=lambda row: float(row["monitor"]))
        if (result["ckpt_rule"] != "best_val_ce"
                or result["best_val_epoch"] != int(best["epoch"])
                or any(row["monitor"] != row["val_ce"] for row in history)
                or not np.isclose(result["selected"]["source_val_ce"], float(best["val_ce"]),
                                  rtol=0, atol=1e-6)):
            raise ValueError(f"{path.parent.name}: selected checkpoint is not source-CE minimum")
        for tag in ("selected", "last"):
            role_names = ["dev_val", "confirm"] if stage == "confirm" else ["dev_val"]
            if set(result[tag]) != {"source_val_acc", "source_val_ce", *role_names}:
                raise ValueError(f"{path.parent.name}: unexpected evaluated roles")
            for role in role_names:
                verify_role(result[tag][role], roles[role])
        if stage != "smoke":
            state = torch.load(path.parent / "model.pt", map_location="cpu", weights_only=True)
            if not all(torch.isfinite(value).all() for value in state.values()):
                raise ValueError(f"{path.parent.name}: nonfinite selected weights")
            decision = config["decision_file"]
            if stage == "confirm" and Path(decision).resolve() != (HERE / "decision_explore.json").resolve():
                raise ValueError(f"{path.parent.name}: wrong confirmation freeze")
            if stage == "explore" and decision is not None:
                raise ValueError(f"{path.parent.name}: unexpected exploration decision")
        records.append(dict(run=path.parent.name, epochs=epochs, best_epoch=int(best["epoch"]),
                            params=result["params"], passed=True))
    if stage != "smoke":
        decision = json.loads((HERE / f"decision_{stage}.json").read_text())
        table_sha = hashlib.sha256((HERE / "summary" / f"{stage}_runs.csv").read_bytes()).hexdigest()
        if decision["run_table_sha256"] != table_sha:
            raise ValueError(f"{stage}: frozen result-table hash mismatch")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", choices=["smoke", "explore", "confirm"],
                        default=["smoke", "explore", "confirm"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    registered = {arm["name"]: arm for arm in json.loads((HERE / "arms.json").read_text())["arms"]}
    protocol_sha = hashlib.sha256((HERE.parent / "configs/protocol.json").read_bytes()).hexdigest()
    stages = {stage: verify_stage(stage, registered, protocol_sha) for stage in args.stages}
    output = HERE / "summary" / ("verification.json" if set(args.stages) == {"smoke", "explore", "confirm"}
                               else "verification_partial.json")
    output.write_text(json.dumps(dict(protocol_sha256=protocol_sha, stages=stages), indent=1) + "\n")
    print(json.dumps({stage: len(rows) for stage, rows in stages.items()}), "saved-result checks passed")


if __name__ == "__main__":
    main()
