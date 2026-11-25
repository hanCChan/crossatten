"""Measure frozen Shen Native TX inference from GPU-resident IQ to logits."""

import argparse
from datetime import datetime
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns

import numpy as np
import torch
from torch.nn import functional as F

from siamese import SpecSiamese, spectrum

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
S24 = ROOT / "24-lora-external-audit"
S13 = ROOT / "13-mixer-cfo-mechanisms"
BATCHES = (1, 128)
WARMUPS = 12
ORDER_BLOCKS = 10
INPUT_SEED = 420201


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_frozen(model, path, expected):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"Checkpoint hash mismatch for {path.name}")
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    model.eval()
    return digest


def benchmark_models(ma_checkpoint, dann_checkpoint):
    sys.path.insert(0, str(S13))
    hstem_code = load_module("s42_hstem", S13 / "models.py")
    dann_code = load_module("s42_dann", S24 / "s24_dann.py")
    history = read(S24 / "HISTORY_AUDIT.json")["frozen_hstem_checkpoints"]
    hstem_record = next(row for row in history if row["key"] == "H0_f0_s201")
    external = read(S24 / "phase2/budget_update/seal_baseline48.json")["jobs"]
    ma_fits = read(HERE / "evidence/FIT_SEAL.json")["fits"]
    ma_record = next(row for row in ma_fits if (row["fold"], row["seed"]) == (0, 201))

    hstem = hstem_code.build("A1_mha", hstem_record["data_seed"])
    dann = dann_code.ShenDANN(strict_ib=False)
    ma = SpecSiamese().to(memory_format=torch.channels_last)
    models = {"h0": hstem, "ma": ma, "dann": dann}
    hashes = {
        "h0": load_frozen(hstem, Path(hstem_record["path"]) / "model.pt",
                          hstem_record["artifacts"]["model.pt"]["sha256"]),
        "ma": load_frozen(ma, ma_checkpoint, ma_record["weights"]),
        "dann": load_frozen(dann, dann_checkpoint,
                            external["dann__native__f0__s201"]["checkpoint_sha256"]),
    }

    def ma_tx(iq):
        return ma(spectrum(torch.complex(iq[:, 0], iq[:, 1])))[0]

    def dann_tx(iq):
        feature = dann.features(dann_code.shen_cis(iq))
        return dann.classifier(F.relu(dann.tx_input(feature)))

    return models, {"h0": hstem, "ma": ma_tx, "dann": dann_tx}, hashes


def gpu_snapshot():
    fields = "driver_version,temperature.gpu,clocks.current.graphics,utilization.gpu,memory.used"
    snapshot = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return dict(zip(fields.split(","), snapshot.stdout.splitlines()[0].split(", ")))


def settings():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def synthetic_inputs():
    iq = torch.randn((max(BATCHES), 2, 8192), generator=torch.Generator().manual_seed(INPUT_SEED))
    return {batch: iq[:batch].clone() for batch in BATCHES}


@torch.inference_mode()
def isolated_memory(name, ma_checkpoint, dann_checkpoint):
    """Run in a fresh process with only the measured model and one batch on CUDA."""
    settings()
    models, predict, _ = benchmark_models(ma_checkpoint, dann_checkpoint)
    model = models[name]
    if name == "dann":
        del model.rx_input, model.rx_classifier
    if torch.cuda.memory_allocated():
        raise RuntimeError("Expected an empty PyTorch CUDA allocator before loading model")
    model.cuda()
    rows = {}
    for batch, iq_cpu in synthetic_inputs().items():
        iq = iq_cpu.cuda()
        for _ in range(WARMUPS):
            predict[name](iq)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            predict[name](iq)
        torch.cuda.synchronize()
        rows[batch] = torch.cuda.max_memory_allocated()
        del iq
    return rows


@torch.inference_mode()
def measure(ma_checkpoint, dann_checkpoint):
    settings()
    models, predict, hashes = benchmark_models(ma_checkpoint, dann_checkpoint)
    full_params = {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()}
    if full_params != {"h0": 133690, "ma": 13935882, "dann": 13509263}:
        raise RuntimeError(f"Unexpected checkpoint parameter counts: {full_params}")

    inputs = synthetic_inputs()
    dann = models["dann"].cuda()
    for batch, value in inputs.items():
        gpu_input = value.cuda()
        torch.testing.assert_close(predict["dann"](gpu_input), dann(gpu_input), rtol=0, atol=0)
        del gpu_input
    # Remove training-only weights after checking exact parity with the full model.
    del dann.rx_input, dann.rx_classifier
    deployed_params = {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()}
    for model in models.values():
        model.cuda()

    start = gpu_snapshot()
    runs = {name: {} for name in models}
    orders = tuple(itertools.permutations(models))
    for batch, value in inputs.items():
        gpu_input = value.cuda()
        for name in models:
            for _ in range(WARMUPS):
                logits = predict[name](gpu_input)
            if logits.shape != (batch, 10) or not torch.isfinite(logits).all():
                raise RuntimeError(f"Invalid logits for {name}/{batch}")
            del logits
        torch.cuda.synchronize()
        ms = {name: [] for name in models}
        for _ in range(ORDER_BLOCKS):
            for order in orders:
                for name in order:
                    before = perf_counter_ns()
                    logits = predict[name](gpu_input)
                    torch.cuda.synchronize()
                    ms[name].append((perf_counter_ns() - before) / 1e6)
                    del logits
        del gpu_input
        for name, samples in ms.items():
            p50, p95 = np.percentile(samples, (50, 95)).tolist()
            runs[name][batch] = {
                "p50_ms": p50, "p95_ms": p95, "packets_per_second": 1000 * batch / p50,
                "samples_ms": samples,
            }
    end = gpu_snapshot()
    for model in models.values():
        model.cpu()
    torch.cuda.empty_cache()
    for name in models:
        memory = json.loads(subprocess.check_output(
            [sys.executable, __file__, "--memory-only", name,
             "--ma-checkpoint", str(ma_checkpoint), "--dann-checkpoint", str(dann_checkpoint)],
            text=True,
        ))
        for batch in BATCHES:
            runs[name][batch]["peak_pytorch_allocated_bytes"] = memory[str(batch)]

    return {
        "protocol": "Shen Native frozen fold 0 seed 201, GPU-resident IQ to TX logits",
        "measured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu_before": start, "gpu_after": end,
        "settings": {
            "network_precision": "FP32, native frontend precision, no AMP or TF32",
            "input": "identical synthetic float32 IQ, B x 2 x 8192",
            "input_seed": INPUT_SEED, "warmups_per_method_batch": WARMUPS,
            "orders_per_block": orders, "order_blocks": ORDER_BLOCKS,
            "repeats_per_method_batch": len(orders) * ORDER_BLOCKS,
            "timing": "synchronized wall time, all three models resident",
            "memory": "peak live PyTorch allocations with each model in an isolated process",
            "dann_tx_parity": "exact at both batch sizes before training-only receiver head removal",
        },
        "rows": {name: {
            "checkpoint_sha256": hashes[name], "params": full_params[name],
            "tx_params": deployed_params[name], "batches": runs[name],
        } for name in models},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ma-checkpoint", type=Path, required=True)
    parser.add_argument("--dann-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--memory-only", choices=("h0", "ma", "dann"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.memory_only:
        print(json.dumps(isolated_memory(args.memory_only, args.ma_checkpoint, args.dann_checkpoint)))
        sys.exit()
    if args.out is None:
        parser.error("--out is required")
    if args.out.exists():
        parser.error("output already exists")
    result = measure(args.ma_checkpoint, args.dann_checkpoint)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    for method, row in result["rows"].items():
        for size, batch in row["batches"].items():
            print(f'{method} B{size}: {batch["p50_ms"]:.3f}/{batch["p95_ms"]:.3f} ms, '
                  f'{batch["packets_per_second"]:.0f} pkt/s, '
                  f'{batch["peak_pytorch_allocated_bytes"] / 2**20:.1f} MiB')
