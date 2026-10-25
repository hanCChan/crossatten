"""Shared end-to-end cost measurements, preserving stage-01 timing conventions."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from .models import build_pipeline, n_params


def gpu_lat(model, x, warm=50, it=300):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    with torch.no_grad():
        for i in range(warm + it):
            start.record()
            model(x)
            end.record()
            torch.cuda.synchronize()
            if i >= warm:
                times.append(start.elapsed_time(end))
    return np.asarray(times)


def cpu_lat(model, x, warm=20, it=100):
    times = []
    with torch.no_grad():
        for i in range(warm + it):
            start = time.perf_counter()
            model(x)
            if i >= warm:
                times.append(1000 * (time.perf_counter() - start))
    return np.asarray(times)


def benchmark(arms, out: Path, length=8192):
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    rng = np.random.default_rng(0)
    x1 = torch.from_numpy(rng.standard_normal((1, 2, length)).astype(np.float32))
    xb = torch.from_numpy(rng.standard_normal((512, 2, length)).astype(np.float32))
    results = []
    for arm in arms:
        model = build_pipeline(arm).eval()
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            model(x1)
        cl = cpu_lat(model, x1)
        model.cuda()
        g1 = gpu_lat(model, x1.cuda())
        gb = gpu_lat(model, xb.cuda(), warm=5, it=20)
        row = dict(arm=arm["name"], params=n_params(model), mflops_per_packet=round(fc.get_total_flops() / 1e6, 1),
                   gpu_b1_ms_p50=round(float(np.percentile(g1, 50)), 3),
                   gpu_b1_ms_p95=round(float(np.percentile(g1, 95)), 3),
                   gpu_b1_ms_p99=round(float(np.percentile(g1, 99)), 3),
                   gpu_b512_pkts_per_s=round(512000 / float(np.median(gb))),
                   cpu4t_b1_ms_p50=round(float(np.percentile(cl, 50)), 2),
                   cpu4t_b1_ms_p95=round(float(np.percentile(cl, 95)), 2))
        results.append(row)
        print(json.dumps(row), flush=True)
        del model
        torch.cuda.empty_cache()
    env = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cpu_threads=4,
               note="random weights; raw IQ on device to logits; FFT/scalar ops included in timing, not fully in FLOPs")
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.json").write_text(json.dumps(dict(env=env, arms=results), indent=1) + "\n")
    cols = list(results[0])
    lines = [f"# End-to-end cost ({env['gpu']}, torch {env['torch']})\n\n{env['note']}\n\n",
             "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"]
    lines.extend("| " + " | ".join(str(row[c]) for c in cols) + " |\n" for row in results)
    (out / "bench.md").write_text("".join(lines))
    return results
