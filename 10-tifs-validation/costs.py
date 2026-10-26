"""Common-device end-to-end costs; partial FLOPs clearly disclosed."""
import json
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
from stage import HERE, save, ask_modules
from features import paired_model
from xa.bench import cpu_lat, gpu_lat


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    rows = []
    torch.manual_seed(10101)
    x = torch.randn(1, 2, 256)
    xb = torch.randn(128, 2, 256)
    for dataset in ("ManyRx", "ManySig"):
        for feature in ("hard", "soft", "unweighted", "no_hstu", "ask"):
            if feature == "ask":
                official = ask_modules()
                conf = __import__("argparse").Namespace(candidates=[8, 16, 24, 32, 48, 64],
                    patch_size=64, embed_dim=128, mlp_ratio=2.)
                model = official.build_model(conf, 10 if dataset == "ManyRx" else 6)
            else:
                model, _ = paired_model(dataset, feature, 0)
            model.eval()
            count = sum(p.numel() for p in model.parameters())
            real_scalars = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
            with FlopCounterMode(display=False) as fc, torch.no_grad():
                model(x)
            cpu = cpu_lat(model, x, warm=5, it=30)
            model.cuda()
            gpu1 = gpu_lat(model, x.cuda(), warm=10, it=80)
            gpub = gpu_lat(model, xb.cuda(), warm=5, it=30)
            rows.append(dict(dataset=dataset, model=feature, parameter_elements=count,
                real_parameter_scalars=real_scalars, counted_mflops=fc.get_total_flops() / 1e6,
                cpu4t_b1_p50_ms=float(np.median(cpu)), gpu_b1_p50_ms=float(np.median(gpu1)),
                gpu_b1_p95_ms=float(np.percentile(gpu1, 95)),
                gpu_b128_packets_per_second=float(128000 / np.median(gpub))))
            print(dataset, feature, rows[-1], flush=True)
    save(HERE / "costs.json", dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
        input="on-device raw IQ[batch,2,256] to logits; eval mode", batch=128, cpu_threads=4,
        scope="random weights, same hardware; no training concurrency",
        flop_limit="Torch counts supported convolution/linear/matmul only. FFT, QR, complex/scalar "
                   "and normalization work not fully counted; not a complete total-FLOP comparison",
        rows=rows))


if __name__ == "__main__":
    main()
