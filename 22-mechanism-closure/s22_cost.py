import numpy as np
from s22_core import HERE,now,write,validate
from s22_report import csv_write

def cost():
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from s22_models import build
    validate();torch.set_num_threads(2);rows=[]
    for arm in ['H0','P0','H1','H2','S1','S2']:
        m=build(arm,123).eval(); params=sum(p.numel() for p in m.parameters());blocks=m.net.encoder.blocks
        dense=sum(2*32*(3*64*b.inner+b.inner*64)+4*32*32*b.inner for b in blocks)
        m.net.encoder.blocks=torch.nn.ModuleList()
        with torch.no_grad(),FlopCounterMode(display=False) as count:m(torch.randn(1,2,8192))
        flops=count.get_total_flops()+dense;m.net.encoder.blocks=blocks;m=m.cuda()
        for batch in [1,128]:
            x=torch.randn(batch,2,8192,device='cuda')
            for repeat in range(3):
                with torch.no_grad():
                    for _ in range(20):m(x)
                    times=[]
                    for _ in range(100):
                        a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)];a.record();m(x);b.record();b.synchronize();times.append(a.elapsed_time(b))
                rows.append(dict(arm=arm,repeat=repeat,batch=batch,params=params,counted_macs=flops/2,counted_mflops=flops/1e6,p50_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),packets_per_second=batch*1000/float(np.mean(times))))
            del x
        del m;torch.cuda.empty_cache()
    csv_write('costs.csv',rows);write(HERE/'costs.json',dict(time=now(),gpu=torch.cuda.get_device_name(0),torch=torch.__version__,scope='Resident FP32 IQ to logits;3 repeats;20warmup/100timings; FMA2; FFT/interpolation/elementwise incomplete; latency includes actual interpolation',rows=rows))

if __name__=='__main__':cost()
