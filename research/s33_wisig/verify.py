"""Independently recount the main indicators from sealed S33 arrays."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify(out, registration):
    summary, receipt = read(out / "SUMMARY.json"), read(out / "RECEIPT.json")
    names = read(registration)["rx_names"]
    with (out / "PHYSICAL_CELL_AUC.csv").open(encoding="utf-8") as f:
        recorded = list(csv.DictReader(f))
    metrics = {key: [] for key in ("both_correct", "hstem_only", "patch_only", "both_wrong",
                                    "all_wrong", "expected_both_wrong", "pair_oracle", "triple_oracle")}
    trial_rows, checked_auc = 0, 0
    for view in ("V0", "V1"):
        p = out / f"AUDIT_{view}.npz"
        assert sha(p) == receipt["packets"][view]
        with np.load(p, allow_pickle=False) as z:
            packets = {key: z[key] for key in z.files}
        y, rx, day = (packets["source_selection_" + k] for k in ("y", "rx", "day"))
        for seed in (0, 1):
            get = lambda model: packets[f"{seed}_{model}_logits"].argmax(1) == y
            h, l, a = (get(model) for model in ("HStem_source_selection", "LongPatch_source_selection", "A1"))
            for r in np.unique(rx):
                idx = np.where(rx == r)[0]
                values = (h & l, h & ~l, ~h & l, ~h & ~l, ~h & ~l & ~a, h | l, h | l | a)
                for key, arr in zip(("both_correct", "hstem_only", "patch_only", "both_wrong",
                                     "all_wrong", "pair_oracle", "triple_oracle"), values):
                    metrics[key].append(float(arr[idx].mean()))
                expected = [
                    np.count_nonzero(~h[cell]) * np.count_nonzero(~l[cell]) / len(cell) / len(idx)
                    for t, d in set(zip(y[idx], day[idx]))
                    for cell in [idx[(y[idx] == t) & (day[idx] == d)]]
                ]
                metrics["expected_both_wrong"].append(sum(expected))
                trial_rows += 1
            for row in recorded:
                if row["view"] != view or int(row["seed"]) != seed:
                    continue
                chosen = (rx == names.index(row["rx"])) & (day == int(row["day"])) & (y == int(row["tx"]))
                obs = packets[row["signal"]]
                chosen &= np.isfinite(obs)
                easy, hard = (h & l, ~h & ~l) if row["match"] == "pair" else (h & l & a, ~h & ~l & ~a)
                low, high = obs[chosen & easy], obs[chosen & hard]
                assert (len(low), len(high)) == (int(row["easy"]), int(row["hard"]))
                if min(len(low), len(high)) < 8:
                    assert row["auc"] == ""
                    continue
                explicit = (high[:, None] > low[None]).mean() + .5 * (high[:, None] == low[None]).mean()
                assert np.isclose(explicit, float(row["auc"]), rtol=0, atol=1e-12)
                checked_auc += 1
    for key, values in metrics.items():
        assert np.isclose(np.mean(values) * 100, summary["overlap"][key]["mean_pp"], rtol=0, atol=1e-11)
    shift = out / "CENTROID_SHIFT.json"
    assert sha(shift) == receipt["shift_sha256"]
    qa = dict(status="PASS", receiver_seed_rows=trial_rows, overlap_metrics=len(metrics),
              tested_auc_cells=checked_auc, cell_support_counts=len(recorded),
              role_checks=summary["role_checks"], source_predictions_exact=receipt["source_seed_held_argmax_checks"],
              strong_reference_hash_and_same_packet_checks=len(receipt["strong_reference"]),
              new_neural_fits=read(out / "EXECUTION.json")["neural_fits"])
    (out / "QA.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    print(qa)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--registration", type=Path,
        default=Path(__file__).resolve().parent.parent / "standard_method_competition_c" / "PRE_REGISTRATION.json")
    a = parser.parse_args()
    verify(a.output, a.registration)
