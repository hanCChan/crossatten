"""Matched raw training and predeclared interventions after S29 stops."""

import argparse
from pathlib import Path

import numpy as np

from experiment import (ARCHITECTURES, DEVICE, DIAGNOSTIC_CFO, SEEDS, VIEWS, FeatureModel,
                        build_original, check_plan, featurize, fit_one, predict, tensor)
from physics import correct, matched_time_shift, rotate
from source import HERE, Source, read, sha, write
import torch


def fit(parent, out):
    if check_plan(parent):
        raise ValueError("S29-R is activated only after the original preflight stops")
    registration = out / "EXECUTION.json"
    manifest = dict(parent_run_sha256=sha(parent / "EXECUTION.json"),
                    parent_preflight_sha256=sha(parent / "PREFLIGHT.json"),
                    protocol_sha256=sha(HERE / "RAW_CONTROL.zh-CN.md"),
                    script_sha256=sha(Path(__file__)))
    if registration.exists():
        assert read(registration) == manifest
    else:
        write(registration, manifest)
    source = Source(out)
    results = []
    for view in VIEWS:
        train = source.load(view, "fit_train", "fit")
        val = source.load(view, "checkpoint_val", "fit")
        original = build_original("ManyRx", 24101)
        train_x, val_x = [featurize(original, tensor(data["x"])) for data in (train, val)]
        for seed in SEEDS:
            paired = []
            for arch in ARCHITECTURES:
                job = (view, seed, arch, "raw")
                done = out / "_".join(map(str, job)) / "COMPLETE.json"
                result = read(done) if done.exists() else fit_one(out, job, train_x, val_x, train, val)
                assert sha(out / result["selected_path"]) == result["selected_sha256"]
                paired.append(result)
            a, b = paired
            assert a["head_sha256"] == b["head_sha256"]
            assert [e["order_sha256"] for e in a["history"]] == [e["order_sha256"] for e in b["history"]]
            results.extend(paired)
        del train_x, val_x
    write(out / "FIT_SEAL.json", dict(results=results, total_updates=sum(r["updates"] for r in results)))
    print("S29-R fits sealed", len(results), flush=True)


def variants(x):
    yield "native", x
    yield "native_cfo", correct(x)
    for hz in DIAGNOSTIC_CFO:
        yield f"cfo_{hz}", rotate(x, hz)
    for offset in (-9, 9):
        yield f"time_{offset}_circular", x.roll(offset, -1)
        aligned, shifted = matched_time_shift(x, offset)
        yield f"time_{offset}_aligned", aligned
        yield f"time_{offset}_shifted", shifted


def evaluate(out):
    seal = read(out / "FIT_SEAL.json")
    assert len(seal["results"]) == 8
    for result in seal["results"]:
        assert sha(out / result["selected_path"]) == result["selected_sha256"]
    source = Source(out)
    for view in VIEWS:
        models = []
        for job in seal["results"]:
            v, seed, arch, _ = job["job"]
            if v == view:
                model = FeatureModel(arch, 24101 + seed)
                model.load_state_dict(torch.load(out / job["selected_path"], weights_only=True,
                                                 map_location="cpu")["model"])
                models.append((seed, arch, model.to(DEVICE)))
        data = source.load(view, "source_selection", "new_prediction")
        payload = {k: data[k] for k in ("ids", "y", "rx", "day")}
        for condition, iq in variants(tensor(data["x"])):
            features = featurize(models[0][2].net, iq)
            for seed, arch, model in models:
                logits = predict(model, features).cpu().numpy()
                assert np.isfinite(logits).all()
                payload[f"{seed}_{arch}_{condition}"] = logits
        np.savez_compressed(out / f"RAW_{view}.npz", **payload)
    write(out / "PREDICTION_SEAL.json", {v: sha(out / f"RAW_{v}.npz") for v in VIEWS})
    print("S29-R predictions sealed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parent", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    out = args.parent / "raw_control"
    out.mkdir(exist_ok=True)
    fit(args.parent, out)
    evaluate(out)


if __name__ == "__main__":
    main()
