#!/usr/bin/env python3
"""Measure frozen S08 inference graphs after all training has stopped."""
import json
import subprocess
import sys
import time

from training import HERE, ROOT, build_model, n_params, sha

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from xa.bench import cpu_lat, gpu_lat


def main():
    snapshots = []
    for _ in range(2):
        active = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=process_name,used_gpu_memory",
            "--format=csv,noheader,nounits"], text=True).splitlines()
        utilization = subprocess.check_output([
            "nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True).splitlines()
        if any(float(row.rsplit(",", 1)[1]) > 64 for row in active) or any(float(v) > 0 for v in utilization):
            raise SystemExit("training/GPU work still active; do not benchmark")
        snapshots.append(dict(processes=active, utilization=utilization))
        time.sleep(2)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(8103)
    rows = []
    for dataset in ("ManySig", "ManyRx"):
        for feature in ("frequency", "phase", "phase_null"):
            model = build_model(dataset, feature).eval()
            x = torch.randn(1, 2, 256)
            with torch.no_grad(), FlopCounterMode(display=False) as counter:
                model(x)
            cpu = cpu_lat(model, x, warm=10, it=50)
            model.cuda()
            single = gpu_lat(model, x.cuda(), warm=20, it=100)
            batch = gpu_lat(model, torch.randn(256, 2, 256, device="cuda"), warm=5, it=20)
            row = dict(dataset=dataset, feature=feature, params=n_params(model),
                       mflops=counter.get_total_flops() / 1e6, cpu4t_b1_ms=float(np.median(cpu)),
                       gpu_b1_p50_ms=float(np.median(single)), gpu_b1_p95_ms=float(np.quantile(single, .95)),
                       gpu_b256_packets_per_second=256000 / float(np.median(batch)))
            rows.append(row)
            print(json.dumps(row), flush=True)
            model.cpu()
            torch.cuda.empty_cache()
    record = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cpu_threads=4,
                  quiescence_snapshots=snapshots, rows=rows,
                  registration_sha256=sha(HERE / "registration.json"),
                  decision_sha256=sha(HERE / "decision_final.json") if (HERE / "decision_final.json").exists() else None,
                  note="Random weights, synthetic resident IQ to logits; excludes transfer and detection. "
                       "FFT/scalar work included in timing but not completely counted in FLOPs. "
                       "EMA and source checkpoint rules do not change inference graphs.")
    (HERE / "costs.json").write_text(json.dumps(record, indent=1) + "\n")
    lines = ["# S08 Measured Costs\n\n", f"{record['gpu']}; PyTorch {record['torch']}; CPU4threads. ",
             record["note"], " Costs are this runtime's measurements, not universal deployment claims.\n\n",
             "| Dataset | Feature | Params | MFLOPs | CPU b1 ms | GPU b1 p50 ms | GPU b1 p95 ms | GPU b256 packet/s |\n",
             "|---|---|---:|---:|---:|---:|---:|---:|\n"]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['feature']} | {row['params']} | {row['mflops']:.3f} | "
                     f"{row['cpu4t_b1_ms']:.3f} | {row['gpu_b1_p50_ms']:.3f} | "
                     f"{row['gpu_b1_p95_ms']:.3f} | {row['gpu_b256_packets_per_second']:.0f} |\n")
    lines.append("\nParameter count alone does not imply speed. Full200, clean BN recalibration "
                 "and evaluating multiple source rules increase training/selection cost; "
                 "the final inference model remains a single checkpoint, not an ensemble.\n")
    (HERE / "S08_COSTS.md").write_text("".join(lines))


if __name__ == "__main__":
    main()
