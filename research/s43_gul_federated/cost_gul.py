"""Measure four frozen Native deployment pipelines in one synchronized session."""

import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns

import numpy as np
import torch

from fit_gul import GulModel, atomic
from report_gul import verify
from cost_compare import (
    WARMUPS, benchmark_models, gpu_snapshot, load_frozen, settings, synthetic_inputs,
)

MA_CHECKPOINT = Path("D:/crossatten/.execution-s42/ma/f0_s201/selected.pt")
DANN_CHECKPOINT = Path("D:/crossatten/worktrees/s24-lora-external-audit/"
                       "24-lora-external-audit/phase2/runs/dann__native__f0__s201/best.pt")
BLOCKS = 3


def models_for(out):
    settings()
    models, predict, hashes = benchmark_models(MA_CHECKPOINT, DANN_CHECKPOINT)
    row = next(r for r in verify(out)["fits"] if (r["fold"], r["seed"]) == (0, 201))
    gul = GulModel().to(memory_format=torch.channels_last)
    hashes["gul"] = load_frozen(gul, out / "f0_s201/selected.pt", row["weights"])
    models["gul"] = gul
    predict["gul"] = gul
    return models, predict, hashes


def prepare(name, model):
    if name == "dann":
        del model.rx_input, model.rx_classifier
    if name == "gul":
        del model.receiver


@torch.inference_mode()
def isolated_peak(out, name):
    models, predict, _ = models_for(out)
    prepare(name, models[name])
    if torch.cuda.memory_allocated():
        raise RuntimeError("Expected an empty CUDA allocator")
    models[name].cuda()
    result = {}
    for size, batch in synthetic_inputs().items():
        iq = batch.cuda()
        for _ in range(WARMUPS):
            predict[name](iq)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            predict[name](iq)
        torch.cuda.synchronize()
        result[size] = torch.cuda.max_memory_allocated()
        del iq
    return result


@torch.inference_mode()
def run(out):
    file = out / "NATIVE_COST_COMPARE.json"
    if file.exists():
        raise FileExistsError(file)
    models, predict, hashes = models_for(out)
    rows = {name: {"params": sum(p.numel() for p in model.parameters()),
                   "checkpoint_sha256": hashes[name], "batches": {}}
            for name, model in models.items()}
    inputs = synthetic_inputs()
    dann = models["dann"].cuda()
    for batch in inputs.values():
        iq = batch.cuda()
        torch.testing.assert_close(dann(iq), predict["dann"](iq), rtol=0, atol=0)
        del iq
    for name, model in models.items():
        prepare(name, model)
        rows[name]["tx_params"] = sum(p.numel() for p in model.parameters())
        model.cuda()

    before = gpu_snapshot()
    orders = tuple(itertools.permutations(models))
    for size, batch in inputs.items():
        iq = batch.cuda()
        for apply_model in predict.values():
            for _ in range(WARMUPS):
                logits = apply_model(iq)
            if logits.shape != (size, 10) or not torch.isfinite(logits).all():
                raise ValueError("Cost workload produced invalid logits")
        del logits
        torch.cuda.synchronize()
        samples = {name: [] for name in models}
        for _ in range(BLOCKS):
            for order in orders:
                for name in order:
                    start = perf_counter_ns()
                    logits = predict[name](iq)
                    torch.cuda.synchronize()
                    samples[name].append((perf_counter_ns() - start) / 1e6)
                    del logits
        for name, times in samples.items():
            p50, p95 = np.percentile(times, (50, 95)).tolist()
            rows[name]["batches"][size] = dict(p50_ms=p50, p95_ms=p95, samples_ms=times,
                                              packets_per_second=1000 * size / p50)
        del iq
    after = gpu_snapshot()

    for model in models.values():
        model.cpu()
    torch.cuda.empty_cache()
    for name in models:
        peaks = json.loads(subprocess.check_output(
            [sys.executable, __file__, str(out), "--memory-only", name], text=True))
        for size, row in rows[name]["batches"].items():
            row["peak_pytorch_allocated_bytes"] = peaks[str(size)]

    result = dict(protocol="P-H5x160 Native frozen fold 0 seed 201",
                  system=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                              cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                              measured_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                              before=before, after=after),
                  settings=dict(precision="FP32 network, original front-end precision, AMP/TF32 off",
                                input="identical synthetic GPU-resident 8192-sample IQ from S42 cost generator",
                                timing="synchronized wall latency for IQ to TX logits",
                                memory="one process per model, peak live PyTorch allocations",
                                software_orders=orders, order_blocks=BLOCKS, repeats=len(orders) * BLOCKS,
                                warmups_per_model_batch=WARMUPS),
                  rows=rows)
    atomic(file, result)
    for name, row in rows.items():
        print(name, {s: (r["p50_ms"], r["packets_per_second"]) for s, r in row["batches"].items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--memory-only", choices=("h0", "ma", "dann", "gul"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.memory_only:
        print(json.dumps(isolated_peak(args.output, args.memory_only)))
    else:
        run(args.output)
