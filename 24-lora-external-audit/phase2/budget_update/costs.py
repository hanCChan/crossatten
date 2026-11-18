import time
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
from common import HERE,PHASE,read,digest,atomic_json,verify_seal
from inference import h0_model
from models import build,forward
from data import preprocess
from xa.frontends import shen_cis,rms_normalize,band_limit
from legacy_rfhstu.features import torch_rf_views

@torch.no_grad()
def main():
    seal=verify_seal('retained54');torch.set_num_threads(4);torch.manual_seed(24016)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    rows=[]
    def timing(fn):
        for _ in range(3):fn()
        torch.cuda.synchronize();times=[]
        for _ in range(15):
            torch.cuda.synchronize();start=time.perf_counter();fn();torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
        return dict(p50_ms=float(np.quantile(times,.5)),p95_ms=float(np.quantile(times,.95)),samples=times)
    for method in ['h0','ce','dann','fan','ib','ask']:
        for view in ['native','strict_ib']:
            if method in ('h0','ask') and view=='strict_ib':continue
            jid=f'{method}__{view}__f0__s201'
            if method=='h0':model=h0_model(jid,'cuda')
            else:
                p=PHASE/'runs'/jid/'best.pt'
                if digest(p)!=seal['jobs'][jid]['checkpoint_sha256']:raise ValueError('unsealed cost model')
                model=build(method).cuda().eval();model.load_state_dict(torch.load(p,map_location='cuda',weights_only=True))
            def pipeline(x):return model(x) if method=='h0' else forward(model,method,preprocess(x,view,method))[0]
            def frontend(x):
                if method=='h0':return torch_rf_views(band_limit(rms_normalize(x),62500),fft_norm='log_zscore',oob_norm='ratio')
                x=preprocess(x,view,method)
                if method in ('ce','dann'):return shen_cis(x)
                if method=='ask':return model.cis(model.ask(x))
                return model.signal_denoising(x)
            item=dict(method=method,view=view,params=sum(p.numel() for p in model.parameters()),
                      real_degrees_of_freedom=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters()),batches={})
            for batch in (1,128):
                x=torch.randn(batch,2,8192,device='cuda');result=timing(lambda:pipeline(x))
                result['packets_per_second']=batch*1000/result['p50_ms'];result['native_frontend']=timing(lambda:frontend(x))
                if batch==1:
                    with FlopCounterMode(display=False) as counter:pipeline(x)
                    result['partial_dispatcher_MACs']=counter.get_total_flops()/2
                item['batches'][str(batch)]=result
            rows.append(item);atomic_json(HERE/'COSTS.json',dict(complete=False,rows=rows))
            del model;torch.cuda.empty_cache()
    attempts=[read(p) for p in (PHASE/'runs').glob('*/attempt*_timing.json') if not p.parent.name.startswith('ask__strict_ib')]
    atomic_json(HERE/'COSTS.json',dict(complete=True,gpu=torch.cuda.get_device_name(0),rows=rows,
                total_actual_fit_wall_seconds=sum(p['actual_wall_seconds'] for p in attempts),
                coverage='FP32 GPU resident raw IQ to logits; native learned preprocessing and original ASK QR retained. MACs are partial: FFT/QR/unsupported complex arithmetic excluded. No host transfer.',
                source_training_costs='Both IB stages included; see per-fit timings and original registered recipes.'))

if __name__=='__main__':main()
