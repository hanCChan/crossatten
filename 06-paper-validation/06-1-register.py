#!/usr/bin/env python3
"""Data-free registration of a bounded paper-validation matrix."""
import json
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from xa.models import build_pipeline, n_params
from xa.paper_models import UPSTREAM


def costs(arm):
    model = build_pipeline(arm).eval()
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(torch.zeros(1, 2, arm["length"]))
    return n_params(model), counter.get_total_flops()


def match_widths(length):
    target = dict(family="paper_rf", stem="hierarchical", length=length, num_classes=10)
    params, flops = costs(target)
    candidates = []
    # Conv FLOPs: k7/k5 at full resolution and one k=L/32 patch projection.
    for width in range(1, 129):
        arm = dict(target, stem="patch", mixer=False, width=width)
        model = build_pipeline(arm)
        counted = (2 * length * (5 * width * 7 + width * width * 5)
                   + 2 * 32 * width * 64 * (length // 32) + 2 * 32 * 64 * 64 + 2 * 64 * 10)
        candidates.append(dict(width=width, params=n_params(model), flops=counted))
    selected = {}
    for key, goal in (("params", params), ("flops", flops)):
        row = min(candidates, key=lambda r: abs(r[key] - goal))
        actual = costs(dict(target, stem="patch", mixer=False, width=row["width"]))
        assert actual == (row["params"], row["flops"])
        selected[key] = dict(row, relative_error=abs(row[key] - goal) / goal)
        assert selected[key]["relative_error"] < 0.05, selected
    return dict(length=length, reference_params=params, reference_flops=flops,
                selected=selected, search="widths 1..128; data-free nearest cost only")


def rf(name, length, stem, recipe, fmax=None, mixer=True, width=32, dataset=None):
    arm = dict(name=name, family="paper_rf", method=name.split("_")[0],
               length=length, stem=stem, recipe=recipe, input_fmax=fmax,
               mixer=mixer, width=width, num_classes=6 if dataset == "ManySig" else 10)
    if dataset:
        arm["dataset"] = dataset
    return arm


def external(name, method, length, recipe, dataset=None, fmax=None, layout="absolute", alpha=0.0):
    arm = dict(name=name, family="paper_upstream", method=method, length=length,
               recipe=recipe, input_fmax=fmax, alpha=alpha,
               num_classes=6 if dataset == "ManySig" else 10,
               sample_rate=25e6 if dataset else 1e6)
    if dataset:
        arm["dataset"] = dataset
    if method in ("FAN", "IB-RFF"):
        arm["patch_size"] = (32 if dataset == "ManySig" else 16) if dataset else 512
    else:
        arm.update(patch_size=2048 if layout == "scaled" else 64,
                   stride=1024 if layout == "scaled" else 32,
                   wavelet_levels=10 if layout == "scaled" else (4 if dataset == "ManySig" else 5),
                   wavelet_kernel_size=16 if dataset == "ManySig" else 32, layout=layout)
    return arm


def main():
    torch.set_num_threads(1)
    for path in (HERE / "arms.json", HERE / "cost_matching.json"):
        if path.exists():
            raise SystemExit("registration already exists; do not overwrite")
    matching = {str(n): match_widths(n) for n in (256, 8192)}
    wifi, shen = [], []
    for dataset in ("ManyRx", "ManySig"):
        for recipe in ("official", "unified"):
            suffix = f"{dataset}_{recipe}"
            wifi.extend([
                rf(f"W0_{suffix}", 256, "patch", recipe, dataset=dataset),
                rf(f"W1_{suffix}", 256, "hierarchical", recipe, dataset=dataset),
                rf(f"W2_{suffix}", 256, "hierarchical", recipe, 7675781.25, dataset=dataset),
                rf(f"C0_{suffix}", 256, "hierarchical", recipe, mixer=False, dataset=dataset),
                rf(f"CP_{suffix}", 256, "patch", recipe, mixer=False,
                   width=matching["256"]["selected"]["params"]["width"], dataset=dataset),
                rf(f"CF_{suffix}", 256, "patch", recipe, mixer=False,
                   width=matching["256"]["selected"]["flops"]["width"], dataset=dataset),
                external(f"FAN_{suffix}", "FAN", 256, recipe, dataset),
            ])
            for variant in ("default", "alternate"):
                a = external(f"WaveMLP_{variant}_{suffix}", "WaveMLP", 256, recipe, dataset)
                if variant == "alternate":
                    a.update(wavelet_levels=5 if dataset == "ManySig" else 4,
                             wavelet_kernel_size=32 if dataset == "ManySig" else 16)
                wifi.append(a)
            for alpha in (0.004, 0.04):
                a = external(f"IB-RFF_a{alpha}_{suffix}", "IB-RFF", 256, recipe, dataset, alpha=alpha)
                a["init_arm"] = f"FAN_{suffix}"
                wifi.append(a)
    for recipe in ("official", "unified"):
        for band, cutoff in (("full", None), ("ib", 62500)):
            suffix = f"{band}_{recipe}"
            shen.append(external(f"FAN_{suffix}", "FAN", 8192, recipe, fmax=cutoff))
            for layout in ("absolute", "scaled"):
                shen.append(external(f"WaveMLP_{layout}_{suffix}", "WaveMLP", 8192, recipe,
                                     fmax=cutoff, layout=layout))
            for alpha in (0.004, 0.04):
                a = external(f"IB-RFF_a{alpha}_{suffix}", "IB-RFF", 8192, recipe, fmax=cutoff, alpha=alpha)
                a["init_arm"] = f"FAN_{suffix}"
                shen.append(a)
    shen.extend([
        rf("S_base", 8192, "patch", "unified", 62500),
        rf("S_hstem", 8192, "hierarchical", "unified", 62500),
        rf("S_cnnonly", 8192, "hierarchical", "unified", 62500, mixer=False),
        rf("S_cnnparam", 8192, "patch", "unified", 62500, mixer=False,
           width=matching["8192"]["selected"]["params"]["width"]),
        rf("S_cnnflops", 8192, "patch", "unified", 62500, mixer=False,
           width=matching["8192"]["selected"]["flops"]["width"]),
    ])
    arms = wifi + shen
    for arm in arms:
        arm["expected_params"] = n_params(build_pipeline(arm))
    metadata = dict(version="paper-validation-v1", date="2026-10-25", upstream=UPSTREAM,
                    wifi_selection_seed=2023, wifi_final_seeds=[2023, 2024, 2025],
                    shen_source_seeds=[101, 102], shen_confirm_seeds=[201, 202, 203],
                    wifi_shared_recipe="per-fold mean internal RX accuracy across W0/W1/W2; all RF arms share it",
                    wifi_external_selection="per-fold internal source RX accuracy; tie-break source CE/name",
                    shen_selection="minimum mean source-validation CE across four fits; separately per method/recipe",
                    stop="report all registered results; do not add losses or feature modules", arms=arms)
    (HERE / "arms.json").write_text(json.dumps(metadata, indent=1) + "\n")
    (HERE / "cost_matching.json").write_text(json.dumps(matching, indent=1) + "\n")
    print(json.dumps(dict(wifi_arms=len(wifi), shen_arms=len(shen), matching=matching), indent=1))


if __name__ == "__main__":
    main()
