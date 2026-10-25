#!/usr/bin/env python3
"""Evaluate inspected upstream released weights only; never train or choose variants."""
import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
from xa.data import sha256_file
from xa.wisig import RX, load_cell, receivers, released_log_result


def activate(repo):
    for name in list(sys.modules):
        if name == "backbones" or name.startswith("backbones.") or name == "utils" or name.startswith("utils."):
            del sys.modules[name]
    for repo_name in ("IB-RFF", "WaveMLP-RFF"):
        path = str(ROOT / "external" / repo_name)
        if path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, str(ROOT / "external" / repo))
    # Upstream only uses timm's initializer. No third-party source is vendored.
    if "timm" not in sys.modules:
        timm, layers = types.ModuleType("timm"), types.ModuleType("timm.layers")
        layers.trunc_normal_ = torch.nn.init.trunc_normal_
        timm.layers = layers
        sys.modules["timm"], sys.modules["timm.layers"] = timm, layers
    return importlib.import_module("utils.load_data")


def verify_loader(dataset, loader):
    largest_delta = 0.0
    for ri, rx in enumerate(RX[dataset]):
        for day in range(1, 5):
            x, y = loader.load_single_dataset(dataset, ri, day, 10 if dataset == "ManyRx" else 6, "non_equalized")
            own_x, own_y = load_cell(dataset, rx, day)
            assert x.shape == own_x.shape and np.array_equal(y, own_y)
            delta = float(np.max(np.abs(x.astype(np.float32) - own_x))) if len(x) else 0.0
            if delta > 1e-6:
                raise ValueError((dataset, rx, day, "upstream loader mismatch", delta))
            largest_delta = max(largest_delta, delta)
    return largest_delta


def upstream_log(dataset, fold, alpha, stage):
    if stage == 1:
        path = ROOT / f"external/IB-RFF/logs/stage1/{dataset}_pre0_a0.0_round{fold}.log"
    else:
        path = ROOT / f"external/IB-RFF/logs/stage2/{dataset}_CRD_pre1_a{alpha}_round{fold}.log"
    return released_log_result(path)


def evaluate(model, dataset, fold, repo, loader, exp):
    _, test = receivers(dataset, fold)
    per_rx = {}
    model = model.cuda().eval()
    with torch.no_grad():
        for rx in test:
            correct = total = 0
            ri = RX[dataset].index(rx)
            for day in ((1, 2) if exp == "CR" else (3, 4)):
                x, y = loader.load_single_dataset(dataset, ri, day, 10 if dataset == "ManyRx" else 6, "non_equalized")
                x = torch.from_numpy(x.astype(np.float32)).cuda()
                y = torch.from_numpy(y).cuda()
                for i in range(0, len(x), 32):
                    out = model(x[i:i + 32])
                    logits = out[1] if repo == "IB-RFF" else out[0]
                    correct += int((logits.argmax(1) == y[i:i + 32]).sum())
                total += len(x)
            per_rx[rx] = dict(correct=correct, total=total, acc=100 * correct / total)
    return dict(packet_acc=100 * sum(d["correct"] for d in per_rx.values()) / sum(d["total"] for d in per_rx.values()),
                rx_equal_acc=float(np.mean([d["acc"] for d in per_rx.values()])), per_rx=per_rx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", choices=["IB-RFF", "WaveMLP-RFF"], required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    loader = activate(args.repo)
    results, validations = [], {}
    for dataset in ("ManyRx", "ManySig"):
        manifest = json.loads((HERE / "out" / f"05-1-{dataset}-manifest.json").read_text())
        if not manifest["counts_match"]:
            raise SystemExit("count gate failed")
        validations[dataset] = verify_loader(dataset, loader)
    out_path = HERE / "out" / f"05-3-released-{args.repo}.json"
    alphas = {"ManyRx": [0.01, 0.008, 0.08, 0.004], "ManySig": [0.04, 0.001, 0.08, 0.1]}
    for dataset in ("ManyRx", "ManySig"):
        for fold in range(4):
            stages = (1, 2) if args.repo == "IB-RFF" else (1,)
            for stage in stages:
                if args.repo == "IB-RFF":
                    cls = importlib.import_module("backbones.FAN").FAN
                    alpha = 0.0 if stage == 1 else alphas[dataset][fold]
                    patch = 16 if dataset == "ManyRx" else 32
                    path = ROOT / "external/IB-RFF/weights" / (
                        f"FAN_DSCNN_{patch}_128_2.0_{alpha}_{int(stage == 2)}_{dataset}_[1, 2]_test_round={fold}_seed=2023.pth")
                    # Upstream publishes whole-module pickles; source and repository have been inspected.
                    old = torch.load(path, map_location="cpu", weights_only=False)
                    # Historical whole-module pickles lack modern GELU attributes.
                    # Rebuild the unchanged official architecture and strictly load all state.
                    model = cls(seq_len=256, in_chans=2, patch_size=patch, embed_dim=128,
                                mlp_ratio=2.0, num_classes=10 if dataset == "ManyRx" else 6,
                                bottleneck_type="DSCNN", use_SAF=True, use_FAF=True)
                    model.load_state_dict(old.state_dict(), strict=True)
                    assert all(torch.equal(v, model.state_dict()[k]) for k, v in old.state_dict().items())
                    del old
                    exp, expected = upstream_log(dataset, fold, alpha, stage)
                    method = "FAN" if stage == 1 else "IB-RFF"
                    if exp != ("CR" if stage == 1 else "CRD"):
                        raise ValueError("unexpected test protocol in the published log")
                else:
                    cls = importlib.import_module("backbones.MyModel").MyModel
                    level, kernel, nc = (5, 32, 10) if dataset == "ManyRx" else (4, 16, 6)
                    model = cls(num_classes=nc, use_wavelet=True, wavelet_levels=level,
                                wavelet_kernel_size=kernel, wavelet_learnable=True, wavelet_init="randn")
                    path = ROOT / "external/WaveMLP-RFF/weights" / (
                        f"{dataset}_CRD_noeq_tx{nc}_wave1_lvl{level}_learn1_randn_k{kernel}_wlrm1.0_date1_2_round{fold}_seed2023.pth")
                    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                    expected = None
                    alpha = None
                    method = "WaveMLP"
                    exp = "CRD"
                metrics = evaluate(model, dataset, fold, args.repo, loader, exp)
                delta = metrics["packet_acc"] - expected if expected is not None else None
                result = dict(dataset=dataset, fold=fold, method=method, exp=exp, alpha=alpha, expected_log_acc=expected,
                              delta_pp=delta, weight_sha256=sha256_file(path), **metrics)
                results.append(result)
                print(json.dumps(result), flush=True)
                out_path.write_text(json.dumps(dict(repo=args.repo, code_state="released weights only",
                                                   compatibility="IB-RFF rebuilds official model; strict identical released state",
                                                   loader_max_abs_delta=validations, rows=results), indent=1) + "\n")
                if delta is not None and abs(delta) > 0.5:
                    raise SystemExit(f"released weight mismatch: {dataset} fold{fold} {method}: {delta:+.3f} pp; STOP")
                del model
                torch.cuda.empty_cache()
    if args.repo == "WaveMLP-RFF":
        averages = {dataset: float(np.mean([r["packet_acc"] for r in results if r["dataset"] == dataset]))
                    for dataset in ("ManyRx", "ManySig")}
        expected = {"ManyRx": 88.68, "ManySig": 93.13}
        # Upstream has README aggregate values, not per-fold logs.
        passed = all(abs(averages[k] - expected[k]) <= 0.5 for k in averages)
        out_path.write_text(json.dumps(dict(repo=args.repo, code_state="released weights only",
                                           loader_max_abs_delta=validations, rows=results, means=averages,
                                           expected_readme=expected, release_gate_passed=passed), indent=1) + "\n")
        if not passed:
            raise SystemExit("WaveMLP released mean differs from README >0.5 pp; STOP")


if __name__ == "__main__":
    main()
