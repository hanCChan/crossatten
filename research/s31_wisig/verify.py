"""Independent endpoint calculation from paired packet predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(s29, s30, out):
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    summary = read(out / "SUMMARY.json")
    fit_seal = read(out / "FIT_SEAL.json")
    pseal = read(out / "PREDICTION_SEAL.json")
    architectures = ("HStem", "LongPatch")
    doses = (0, 1000, 2000, 2500, 5000)
    gains = {a: [] for a in architectures}
    auc = {(a, arm): [] for a in architectures for arm in ("raw", "cfo25")}
    matches = 0
    for view in ("V0", "V1"):
        output = out / f"S31_{view}.npz"
        assert sha(output) == pseal[view]
        with np.load(output) as new, np.load(s29 / "raw_control" / f"RAW_{view}.npz") as old, \
             np.load(s30 / f"PACKETS_{view}.npz") as perturb:
            y, rx = old["y"], old["rx"]
            observed = np.unique(rx)
            for arch in architectures:
                rx_gain = []
                loss_area = {"raw": [], "cfo25": []}
                for seed in (0, 1):
                    prefix = f"{seed}_{arch}_"
                    native = {a: data[prefix + "native"].argmax(1) == y for a, data in (("raw", old), ("cfo25", new))}
                    delta = native["cfo25"].astype(float) - native["raw"]
                    rx_gain.append([100 * delta[rx == r].mean() for r in observed])
                    for arm, data in (("raw", perturb), ("cfo25", new)):
                        last = np.zeros(len(y))
                        running = np.zeros(len(y))
                        for low, high in zip(doses, doses[1:]):
                            pair = [data[prefix + f"center_{sign * high}_0"] for sign in (-1, 1)]
                            if arm == "cfo25":
                                pair = [logits.argmax(1) for logits in pair]
                            loss = native[arm].astype(float) - .5 * ((pair[0] == y).astype(float) + (pair[1] == y))
                            running += .5 * (last + loss) * (high - low) / doses[-1]
                            last = loss
                        loss_area[arm].append(np.mean([running[rx == r].mean() * 100 for r in observed]))
                    matches += 1
                gains[arch].append(rx_gain)
                for arm in loss_area:
                    auc[arch, arm].extend(loss_area[arm])
    comparisons = {k: np.array(v) for k, v in gains.items()}
    comparisons["interaction"] = comparisons["HStem"] - comparisons["LongPatch"]
    for name, arr in comparisons.items():
        result = summary["clean_comparisons"][name]
        assert np.isclose(arr.mean(), result["mean_pp"], rtol=0, atol=1e-11)
        assert np.allclose(arr.mean(1), result["rx_pp"], rtol=0, atol=1e-11)
        assert np.allclose(arr.mean(2), result["view_seed_pp"], rtol=0, atol=1e-11)
    for (a, arm), values in auc.items():
        assert np.isclose(np.mean(values), summary["results"][a][arm]["loss_auc_pp"], rtol=0, atol=1e-11)
    fit_jobs = fit_seal["results"]
    assert len(fit_jobs) == matches == 8 and fit_seal["total_updates"] == 40640
    original_fits = s29 / "raw_control"
    original = {tuple(job["job"][:3]): job for job in read(original_fits / "FIT_SEAL.json")["results"]}
    for job in fit_jobs:
        base = original[tuple(job["job"][:3])]
        paths = (original_fits / base["selected_path"], out / job["selected_path"])
        assert [sha(path) for path in paths] == [base["selected_sha256"], job["selected_sha256"]]
        states = (torch.load(path, map_location="cpu", weights_only=True)["model"] for path in paths)
        old, new = ({key: (value.shape, value.dtype) for key, value in state.items()} for state in states)
        assert old == new
    data = dict(status="PASS", paired_prediction_recounts=matches, paired_gain_recounts=len(comparisons),
                auc_recounts=len(auc), rx_metrics=summary["rx_metric_rows"], source_role_checks=summary["role_payload_checks"],
                matching_checkpoint_schemas=len(fit_jobs), new_fits=len(fit_jobs),
                optimization_updates=fit_seal["total_updates"])
    (out / "QA.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("s29", "s30", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    verify(args.s29, args.s30, args.output)
