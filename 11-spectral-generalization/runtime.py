"""Same-device S11 frontend/cost measurements, not an efficacy claim."""
import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode
from core import HERE,ROOT,save,sha
from spectral import paired_model,center_weight
from xa.bench import cpu_lat,gpu_lat
from outcomes import table


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    torch.manual_seed(11101)
    x=torch.randn(1,2,256)
    xb=torch.randn(128,2,256)
    rows=[]
    for d in ("ManyRx","ManySig"):
        for kind in ("hard","F1","F2","ASKNet"):
            if kind=="ASKNet":
                from core import module
                official=module("s11_ask_model",ROOT/"external/ASKNet-RFF/WiSig/backbones/PatchNet.py")
                model=official.PatchNet(num_classes=10 if d=="ManyRx" else 6)
            else:
                weight=center_weight() if kind=="F1" else np.ones(256,np.float32)
                if kind=="F2":
                    p=next((HERE/"priors").glob(f"{d}__f0__last*"))
                    import json
                    weight=np.array(json.loads(p.read_text())["weight"],np.float32)
                model,_=paired_model(d,kind,0,weight)
            model.eval()
            with FlopCounterMode(display=False) as fc,torch.no_grad():
                model(x)
            cl=cpu_lat(model,x,warm=5,it=30)
            model.cuda()
            g1=gpu_lat(model,x.cuda(),warm=10,it=80)
            gb=gpu_lat(model,xb.cuda(),warm=5,it=30)
            rows.append(dict(dataset=d,model=kind,
                parameter_elements=sum(p.numel() for p in model.parameters()),
                real_parameter_scalars=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters()),
                counted_mflops=fc.get_total_flops()/1e6,
                cpu4t_b1_p50_ms=float(np.median(cl)),gpu_b1_p50_ms=float(np.median(g1)),
                gpu_b1_p95_ms=float(np.percentile(g1,95)),gpu_b128_packets_per_s=float(128000/np.median(gb))))
            print(d,kind,"measured",flush=True)
            del model
    save(HERE/"costs.json",dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
        cpu_threads=4,batch=128,rows=rows,random_weights=True,
        asknet_random_tie_selected_k=8,asknet_complex_operator_fixed_modes=127,
        prior_estimation_cost_excluded_from_inference=True,
        counted_flops_incomplete="supported conv/linear/matmul only; FFT/complex/QR/scalar not fully counted"))
    lines=["# Parameters And Runtime","",
        "Same A100, Torch2.5.0/CUDA12.4, CPU4threads, batch128, random frozen network weights. "
        "IQ already on device to logits; prior estimated offline only from training source packets.",
        table(pd.DataFrame(rows)),"",
        "No new network accuracy is inferred from these costs. ASKNet complex elements and real scalar "
        "freedom are separate. No speed superiority claim based on parameter count.",
        "Torch counter omits/not fully counts FFT, QR, complex and scalar operations; counted FLOPs "
        "are partial and not a complete total-operation comparison.",
        "ASKNet uses random weights with tied cutoff logits (eval k8), not trained checkpoints. "
        "Its complex operator stays127modes for every cutoff; this is an architectural cost probe, "
        "not a checkpoint-specific runtime superiority test.",
        "F2 buffer256floats is not a trainable parameter. Both F1/F2 preserve original token/HSTU dimensions."]
    (HERE/"PARAMETERS_AND_RUNTIME.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    main()
