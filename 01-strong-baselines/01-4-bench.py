#!/usr/bin/env python3
"""01-4  Cost of every registered arm: parameters, model FLOPs, end-to-end latency/throughput.
End-to-end = raw I/Q tensor already on the device -> RMS norm / band limit / views or CIS -> logits.
FLOPs: torch FlopCounterMode on one packet (matmul/conv only; FFT/STFT not counted -> reported separately as timing).
GPU: CUDA-event timing, 50 warm-up + 300 timed iterations, batch 1 (p50/p95/p99) and batch 512 throughput.
CPU: 4 threads, batch 1, 20 warm-up + 100 timed iterations (p50/p95).
Output: summary/bench.json, summary/bench.md
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from xa.models import build_pipeline, n_params  # noqa: E402


def gpu_lat(m, x, warm=50, it=300):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    with torch.no_grad():
        for i in range(warm + it):
            s.record()
            m(x)
            e.record()
            torch.cuda.synchronize()
            if i >= warm:
                ts.append(s.elapsed_time(e))
    return np.array(ts)


def cpu_lat(m, x, warm=20, it=100):
    ts = []
    with torch.no_grad():
        for i in range(warm + it):
            t = time.perf_counter()
            m(x)
            if i >= warm:
                ts.append(1e3 * (time.perf_counter() - t))
    return np.array(ts)


def main():
    arms = json.loads((HERE / "01-arms.json").read_text())["arms"]
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    rng = np.random.default_rng(0)
    x1 = torch.from_numpy(rng.standard_normal((1, 2, 8192)).astype(np.float32))
    xb = torch.from_numpy(rng.standard_normal((512, 2, 8192)).astype(np.float32))
    res = []
    seen = {}
    for a in arms:
        sig = json.dumps({k: a.get(k) for k in ("family", "fusion", "head", "input_fmax", "oob_keep")}, sort_keys=True)
        if sig in seen:          # AWGN / recipe do not change inference cost
            res.append(dict(seen[sig], arm=a["name"]))
            continue
        m = build_pipeline(a).eval()
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            m(x1)
        flops = fc.get_total_flops()
        cl = cpu_lat(m, x1)
        mg = m.cuda()
        g1 = gpu_lat(mg, x1.cuda())
        gb = gpu_lat(mg, xb.cuda(), warm=5, it=20)
        r = dict(arm=a["name"], params=n_params(m), mflops_per_packet=round(flops / 1e6, 1),
                 gpu_b1_ms_p50=round(float(np.percentile(g1, 50)), 3), gpu_b1_ms_p95=round(float(np.percentile(g1, 95)), 3),
                 gpu_b1_ms_p99=round(float(np.percentile(g1, 99)), 3),
                 gpu_b512_pkts_per_s=round(512 / (float(np.median(gb)) / 1e3)),
                 cpu4t_b1_ms_p50=round(float(np.percentile(cl, 50)), 2), cpu4t_b1_ms_p95=round(float(np.percentile(cl, 95)), 2))
        seen[sig] = {k: v for k, v in r.items() if k != "arm"}
        res.append(r)
        print(json.dumps(r), flush=True)
    env = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cpu_threads=4,
               note="random weights; end-to-end from raw I/Q on device; FFT/STFT included in timing, not in FLOPs")
    (HERE / "summary").mkdir(exist_ok=True)
    (HERE / "summary/bench.json").write_text(json.dumps(dict(env=env, arms=res), indent=1))
    cols = list(res[0].keys())
    md = [f"# Cost benchmark ({env['gpu']}, torch {env['torch']})\n\n{env['note']}\n\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |\n" for r in res]
    (HERE / "summary/bench.md").write_text("".join(md))


if __name__ == "__main__":
    main()
