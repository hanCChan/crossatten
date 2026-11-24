"""Recount prespecified S34 comparisons directly from sealed packets."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archive(path, digest):
    assert sha(path) == digest
    with np.load(path, allow_pickle=False) as z:
        return {name: z[name] for name in z.files}


def eq(x, y):
    assert np.allclose(x, y, rtol=0, atol=1e-10)


def main(root, s33, registry):
    a, b = root / "A", root / "B"
    sa, sb = read(a / "SUMMARY.json"), read(b / "SUMMARY.json")
    ra, rb, source = (read(p / "RECEIPT.json") for p in (a, b, s33))
    rx_names = read(registry)["rx_names"]
    with (b / "NATURAL_CELLS.csv").open(encoding="utf-8") as stream:
        natural = list(csv.DictReader(stream))
    candidates = ("dct4", "region4")
    clean = {key: [] for key in candidates}
    hard = {key: [] for key in candidates}
    effects = {key: [] for key in ("white", "phase_walk", "frequency_shape", "constant_cfo")}
    a_counts, natural_counts, bridge, evms = 0, 0, 0, 0
    for view in ("V0", "V1"):
        pa = archive(a / f"PRED_{view}.npz", ra["predictions"][view])
        pb = archive(b / f"PRED_{view}.npz", rb["predictions"][view])
        old = archive(s33 / f"AUDIT_{view}.npz", source["packets"][view])
        for k in ("ids", "y", "rx", "day"):
            assert np.array_equal(pa[k], pb[k]) and np.array_equal(pa[k], old["source_selection_" + k])
        y, rx, day = (pb[k] for k in ("y", "rx", "day"))
        for condition in effects:
            assert np.abs(pb[f"{condition}_evm"] - .12).max() < 5e-6
            evms += len(y)
        per = {mode: {key: [] for key in candidates} for mode in ("clean", "hard")}
        for seed in (0, 1):
            prefixes = [f"{seed}_{key}" for key in ("HStem", "LongPatch")]
            initial = [pa[p + "_head"] == y for p in prefixes]
            double = ~(initial[0] | initial[1])
            for prefix in prefixes:
                assert np.array_equal(pa[prefix + "_head"], pb[prefix + "_native"])
                for p in ("ridge", "prototype"):
                    assert np.array_equal(pa[f"{prefix}_pooled_pooled_{p}"], pa[f"{prefix}_pre_pool_mean_{p}"])
                    bridge += 1
            base = pa[f"{seed}_HStem_pre_pool_mean_ridge"] == y
            for candidate in candidates:
                compared = pa[f"{seed}_HStem_pre_pool_{candidate}_ridge"] == y
                diff = compared.astype(float) - base
                masks = [rx == r for r in np.unique(rx)]
                per["clean"][candidate].append([100 * diff[mask].mean() for mask in masks])
                per["hard"][candidate].append([100 * diff[mask & double].mean() for mask in masks])
                a_counts += len(masks) * 2
            for condition in effects:
                after = (pb[f"{prefixes[0]}_{condition}"] != y) & (pb[f"{prefixes[1]}_{condition}"] != y)
                value = after.astype(float) - double
                effects[condition].extend(100 * value[rx == r].mean() for r in np.unique(rx))
            for row in natural:
                if row["view"] != view or int(row["seed"]) != seed or not row["auc"]:
                    continue
                idx = (rx == rx_names.index(row["rx"])) & (y == int(row["tx"])) & (day == int(row["day"]))
                x = pb["native_" + row["signal"]]
                left, right = x[idx & double], x[idx & initial[0] & initial[1]]
                assert len(left) == int(row["hard"]) and len(right) == int(row["easy"])
                observed = left[:, None] - right[None]
                eq((observed > 0).mean() + .5 * (observed == 0).mean(), float(row["auc"]))
                natural_counts += 1
        for c in candidates:
            clean[c].append(per["clean"][c])
            hard[c].append(per["hard"][c])
    for c in candidates:
        gate = sa["gates"][c]
        eq(np.mean(clean[c]), gate["clean"]["mean"]["mean_pp"])
        eq(np.mean(hard[c]), gate["hard"]["mean_pp"])
        checks = gate["checks"]
        assert checks["clean_gain"] == (np.mean(clean[c]) >= .5)
        assert checks["shared_error_gain"] == (np.mean(hard[c]) >= 5.)
        assert gate["passes"] == all(checks.values())
    for key, samples in effects.items():
        eq(np.mean(samples), sb["intervention"][key]["paired_common_error_delta_pp"])
    probes = read(a / "PROBES.json")
    assert sha(a / "PROBES.json") == ra["probe_choices"]
    for p in probes:
        scores = p["checkpoint_val_scores"]
        assert p["factor"] == (.1, .01, .0001)[scores.index(max(scores))]
    qa = dict(status="PASS", checked_fit_probe_choices=len(probes), candidate_receiver_seed_subset_checks=a_counts,
              exact_pooled_mean_bridge_checks=bridge, supported_cell_auc_recounts=natural_counts,
              matched_dose_packet_checks=evms, backbone_fits=0,
              a_source_role_checks=sa["role_checks"], b_source_role_checks=sb["role_checks"])
    (root / "QA.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    print(qa)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("s33", type=Path)
    parser.add_argument("--registry", type=Path,
        default=Path(__file__).resolve().parents[1] / "standard_method_competition_c" / "PRE_REGISTRATION.json")
    args = parser.parse_args()
    main(args.output, args.s33, args.registry)
