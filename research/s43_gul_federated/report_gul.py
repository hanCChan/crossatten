"""Verify and assess all six Gul FL fits on the identical held-RX packet rows."""

import argparse
from collections import defaultdict
import csv
import io
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from fit_gul import (
    BATCH, CLIENT_PACKETS, CLIENTS, LOCAL_EPOCHS, ROUNDS, SEEDS,
    Data, GulModel, HERE, atomic, binding, read, sha, shen_cis,
)
from report import REF, reference, score, summary

MA_RESULT = read(HERE.parent / "s42_lora_baselines/evidence/RESULTS.json")
MA_OUT = Path("D:/crossatten/.execution-s42/ma")
METHODS = ("gul", "h0", "ma", "dann")


def verify(out):
    registered = binding()
    plan = read(out / "PLAN.json")
    seal = read(out / "FIT_SEAL.json")
    jobs = {(f, s) for f in (0, 1) for s in SEEDS}
    if plan["files"] != registered or seal["plan"] != sha(out / "PLAN.json"):
        raise ValueError("Wrong fit registration")
    if len(seal["fits"]) != len(jobs) or {(r["fold"], r["seed"]) for r in seal["fits"]} != jobs:
        raise ValueError("Incomplete fold-seed cohort")
    expected_updates = ROUNDS * CLIENTS * LOCAL_EPOCHS * (CLIENT_PACKETS // BATCH)
    for row in seal["fits"]:
        directory = out / f'f{row["fold"]}_s{row["seed"]}'
        history_file = directory / "HISTORY.json"
        history = read(history_file)
        selected = min(history, key=lambda x: x["source_ce"])
        if (row != read(directory / "DONE.json") or row["files"] != registered
                or row["weights"] != sha(directory / "selected.pt") or row["history"] != sha(history_file)
                or len(history) != ROUNDS or row["updates"] != expected_updates
                or selected["round"] != row["selected_round"]):
            raise ValueError(f"Unverified training evidence: {directory}")
    return seal


def ma_reference(fold, seed, rx, ids, labels):
    name = f"f{fold}_s{seed}_{rx}.npz"
    path = MA_OUT / "predictions" / name
    if sha(path) != MA_RESULT["predictions"][name]:
        raise ValueError(f"Ma predictions changed: {name}")
    with np.load(path, allow_pickle=False) as values:
        if not np.array_equal(values["rows"], ids) or not np.array_equal(values["y"], labels):
            raise ValueError(f"Prediction identity differs: {name}")
        return values["logits"]


def paired(rows, a, b):
    cells = {(r["method"], r["fold"], r["seed"], r["rx"]): r["acc"] for r in rows}
    receivers, fits = defaultdict(list), defaultdict(list)
    for (method, fold, seed, rx), acc in cells.items():
        if method == a:
            diff = 100 * (acc - cells[b, fold, seed, rx])
            receivers[rx].append(diff)
            fits[fold, seed].append(diff)
    per_rx = {rx: mean(diffs) for rx, diffs in sorted(receivers.items())}
    array = np.asarray(list(per_rx.values()))
    values = np.random.default_rng(24016).choice(array, size=(10_000, len(array)), replace=True).mean(1)
    return dict(mean_pp=float(array.mean()), conditional_interval95_pp=np.quantile(values, [.025, .975]).tolist(),
                positive_fit_pairs=sum(mean(x) > 0 for x in fits.values()), per_rx_pp=per_rx)


@torch.inference_mode()
def evaluate(out):
    seal = verify(out)
    if (out / "RESULTS.json").exists():
        raise FileExistsError("The cohort already has a report")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    access = Data(out)
    proof = read(REF / "PREDICTION_INTEGRITY.json")
    integrity = {(v["job"], v["rx"]): v["sha256"] for v in proof["predictions"]}
    prediction_folder = out / "predictions"
    prediction_folder.mkdir(exist_ok=True)
    rows, prediction_hashes = [], {}
    for fold in (0, 1):
        models = {}
        for seed in SEEDS:
            models[seed] = GulModel().cuda().eval().to(memory_format=torch.channels_last)
            models[seed].load_state_dict(torch.load(out / f"f{fold}_s{seed}/selected.pt",
                                                    map_location="cpu", weights_only=True))
        for rx in access.views[fold]["dev_val"]:
            iq, labels, ids = access.load(fold, rx, "dev_val")
            all_iq = torch.tensor(iq, device="cuda")
            specs = torch.cat([shen_cis(part) for part in all_iq.split(128)])
            specs = specs.contiguous(memory_format=torch.channels_last)
            for seed, model in models.items():
                gul = torch.cat([model.logits(part) for part in specs.split(128)]).cpu().numpy()
                if gul.shape != (2000, 10) or not np.isfinite(gul).all():
                    raise ValueError("Invalid Gul TX logits")
                path = prediction_folder / f"f{fold}_s{seed}_{rx}.npz"
                np.savez_compressed(path, logits=gul, rows=ids, y=labels)
                prediction_hashes[path.name] = sha(path)
                logits = {
                    "gul": gul,
                    "ma": ma_reference(fold, seed, rx, ids, labels),
                    **{key: reference(key, fold, seed, rx, integrity, ids, labels) for key in ("h0", "dann")},
                }
                for key, value in logits.items():
                    rows.append(dict(method=key, fold=fold, seed=seed, rx=rx, **score(value, labels)))
            print(f"Paired packet rows: fold {fold} {rx}", flush=True)
    if len(rows) != 120 or len(prediction_hashes) != 30:
        raise ValueError("Incomplete predictions")

    summaries = {m: summary([r for r in rows if r["method"] == m]) for m in METHODS}
    external = {row["method"]: row for row in read(REF / "RESULTS.json")["summary"] if row["view"] == "native"}
    external["ma"] = MA_RESULT["methods"]["ma"]
    for method in ("h0", "ma", "dann"):
        for metric in ("acc", "macro_f1", "worst_rx_acc", "fit_sd"):
            if not np.isclose(summaries[method][metric], external[method][metric], atol=1e-10, rtol=0.):
                raise ValueError(f"Historical comparator {method}/{metric} differs")
    result = dict(protocol="P-H5x160 Native",
                  inference="FP32 networks with TF32 off, CUDA Gul, historical comparison logits reused",
                  methods=summaries, paired_h0_minus_gul=paired(rows, "h0", "gul"),
                  paired_gul_minus_ma=paired(rows, "gul", "ma"),
                  paired_gul_minus_dann=paired(rows, "gul", "dann"),
                  fits=seal["fits"], fit_seal_sha256=sha(out / "FIT_SEAL.json"), predictions=prediction_hashes)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=("method", "fold", "seed", "rx", "acc", "macro_f1", "ce"),
                            extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (out / "RX_METRICS.csv").write_text(buffer.getvalue(), encoding="utf-8", newline="\n")
    atomic(out / "RESULTS.json", result)
    print("Gul", summaries["gul"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    evaluate(parser.parse_args().output)
