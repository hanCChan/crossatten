#!/usr/bin/env python3
"""Run end-to-end synthetic cost measurements only when all training is stopped."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from xa.bench import cpu_lat, gpu_lat
from xa.explore_models import build_explore_model
from xa.models import ShenCISPipeline, n_params
from xa.paper_models import ValidationRF


def candidates():
    models = []
    aliases = {}
    decision = json.loads((HERE / "decision_final.json").read_text())
    for job in decision["jobs"]:
        dataset = job["dataset"]
        frontend = job["frontend"]
        fusion = job.get("fusion", "pooled_concat")
        name = f"WiSig:{dataset}:{frontend}:{fusion}"
        aliases.setdefault(name, set()).add(job["id"])
    for dataset, classes in (("ManyRx", 10), ("ManySig", 6)):
        for frontend in ("legacy", "time", "diff", "split", "stf", "filter", "frequency"):
            aliases.setdefault(f"WiSig:{dataset}:{frontend}:pooled_concat", set())
        for name in sorted(n for n in aliases if n.startswith(f"WiSig:{dataset}:")):
            _, _, frontend, fusion = name.split(":")
            models.append((name, 256, build_explore_model(dict(
                frontend=frontend, num_classes=classes, fusion=fusion))))
    models.extend((f"OSU:{method}", 8192, ValidationRF(dict(
        length=8192, stem="patch" if method == "patch_hstu" else "hierarchical",
        mixer=method != "hstem_cnn", num_classes=25, input_fmax=62500)))
                  for method in ("patch_hstu", "hstem_hstu", "hstem_cnn"))
    models.append(("OSU:cis_rank16", 8192,
                   ShenCISPipeline(head="rank16", input_fmax=62500, num_classes=25)))
    return models, {k: sorted(v) for k, v in aliases.items()}


def main():
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader"], text=True
    ).splitlines()
    if any("python" in name.lower() for name in active):
        raise SystemExit("GPU Python processes still active; defer isolated cost measurements")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(7102)
    models, aliases = candidates()
    rows = []
    for name, length, model in models:
        model.eval()
        x = torch.randn(1, 2, length)
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            model(x)
        cpu = cpu_lat(model, x, warm=10, it=50)
        model.cuda()
        single = gpu_lat(model, x.cuda(), warm=20, it=100)
        batches = gpu_lat(model, torch.randn(256, 2, length, device="cuda"), warm=5, it=20)
        row = dict(model=name, params=n_params(model), mflops=counter.get_total_flops()/1e6,
                   cpu4t_b1_ms=float(np.median(cpu)), gpu_b1_ms=float(np.median(single)),
                   gpu_b1_p95_ms=float(np.quantile(single,.95)),
                   gpu_b256_packets_per_second=256000/float(np.median(batches)))
        rows.append(row)
        print(json.dumps(row), flush=True)
        model.cpu()
        torch.cuda.empty_cache()
    value = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cpu_threads=4, rows=rows,
                 weights="random; no target data", final_job_aliases=aliases,
                 final_decision_sha256=hashlib.sha256(
                     (HERE / "decision_final.json").read_bytes()).hexdigest(),
                 note="Resident IQ to logits, excludes transfer/detection; no concurrent GPU Python processes. "
                      "FFT and scalar operations counted in timing, partially omitted in FLOPs")
    (HERE / "costs.json").write_text(json.dumps(value, indent=1) + "\n")


if __name__ == "__main__":
    main()
