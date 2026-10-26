"""Benchmark all five registered frontends only when training is quiescent."""
import json
import subprocess
import time
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
from training import HERE, paired_model, n_params, sha, save
from xa.bench import cpu_lat, gpu_lat


def main():
    snapshots = []
    for _ in range(2):
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=process_name,used_gpu_memory",
                                          "--format=csv,noheader,nounits"], text=True).splitlines()
        utilization = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu",
                                               "--format=csv,noheader,nounits"], text=True).splitlines()
        if any(float(row.rsplit(",", 1)[1]) > 64 for row in active) or any(float(v) > 0 for v in utilization):
            raise SystemExit("GPU work is active; benchmark after all training/evaluation")
        snapshots.append(dict(processes=active, utilization=utilization))
        time.sleep(2)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    rows = []
    for dataset in ("ManySig", "ManyRx"):
        for feature in ("frequency", "null", "first", "centered", "curvature"):
            model, _ = paired_model(dataset, feature, 17)
            model.eval()
            x = torch.randn(1, 2, 256)
            with torch.no_grad(), FlopCounterMode(display=False) as count:
                model(x)
            cpu = cpu_lat(model, x, warm=10, it=40)
            model.cuda()
            single = gpu_lat(model, x.cuda(), warm=20, it=80)
            batch = gpu_lat(model, torch.randn(256, 2, 256, device="cuda"), warm=5, it=20)
            row = dict(dataset=dataset, feature=feature, params=n_params(model),
                       mflops=count.get_total_flops() / 1e6, cpu4t_b1_ms=float(np.median(cpu)),
                       gpu_b1_p50_ms=float(np.median(single)), gpu_b1_p95_ms=float(np.quantile(single, .95)),
                       gpu_b256_packets_per_second=256000 / float(np.median(batch)))
            rows.append(row)
            print(json.dumps(row), flush=True)
            model.cpu()
            torch.cuda.empty_cache()
    result = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cpu_threads=4,
                  snapshots=snapshots, rows=rows, registration_sha256=sha(HERE / "registration.json"),
                  note="Random initialized weights, synthetic resident IQ to logits. Includes features/FFT "
                  "in time but not all scalar/FFT FLOPs. Excludes packet detection and transfers. "
                  "Runtime variance prevents speedup claims from parameter count alone.")
    save(HERE / "costs.json", result)
    lines = ["# S09 Measured Costs\n\n", result["gpu"], "; ", result["note"], "\n\n",
             "| Dataset | Feature | Params | MFLOPs | CPU b1 ms | GPU b1 p50 ms | GPU b1 p95 ms | GPU b256 packet/s |\n",
             "|---|---|---:|---:|---:|---:|---:|---:|\n"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['feature']} | {r['params']} | {r['mflops']:.3f} | "
                     f"{r['cpu4t_b1_ms']:.3f} | {r['gpu_b1_p50_ms']:.3f} | {r['gpu_b1_p95_ms']:.3f} | "
                     f"{r['gpu_b256_packets_per_second']:.0f} |\n")
    (HERE / "S09_COSTS.md").write_text("".join(lines))


if __name__ == "__main__":
    main()
