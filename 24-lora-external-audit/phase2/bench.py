"""Same-device FP32 inference and native frontend timing, after all fits are sealed."""
import json,sys,time
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
from execution import HERE,ROOT,read,digest,write_json,committed,setup
from models import build,forward
from data import preprocess
from report import h0_model
from xa.frontends import shen_cis,rms_normalize,band_limit
from legacy_rfhstu.features import torch_rf_views

@torch.no_grad()
def main():
    committed(HERE/'global_seal.json');assert read(HERE/'global_seal.json')['status']=='ALL_REGISTERED_FITS_SEALED'
    setup(24016);records=[]
    def timing(fn,warmup=3,repeats=15):
        for _ in range(warmup):fn()
        torch.cuda.synchronize();times=[]
        for _ in range(repeats):
            torch.cuda.synchronize();t=time.perf_counter();fn();torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
        return dict(p50_ms=float(np.quantile(times,.5)),p95_ms=float(np.quantile(times,.95)),samples=times)
    for method in ['h0','ce','dann','fan','ib','ask']:
        for view in ['native','strict_ib']:
            if method=='h0' and view=='strict_ib':continue
            model=h0_model(0,201) if method=='h0' else build(method).cuda().eval()
            if method!='h0':
                job=f'{method}__{view}__f0__s201';p=HERE/'runs'/job/'best.pt'
                if digest(p)!=read(HERE/'global_seal.json')['jobs'][job]['checkpoint_sha256']:raise ValueError('benchmark unsealed model')
                model.load_state_dict(torch.load(p,map_location='cuda',weights_only=True))
            def pipeline(x):return model(x) if method=='h0' else forward(model,method,preprocess(x,view,method))[0]
            def frontend(x):
                if method=='h0':return torch_rf_views(band_limit(rms_normalize(x),62500),fft_norm='log_zscore',oob_norm='ratio')
                z=preprocess(x,view,method)
                if method in ('ce','dann'):return shen_cis(z)
                if method=='ask':return model.cis(model.ask(z))
                return model.signal_denoising(z)
            item=dict(method=method,view=view,params=sum(p.numel() for p in model.parameters()),
                      real_degrees_of_freedom=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters()),batches={})
            for batch in [1,128]:
                x=torch.randn(batch,2,8192,device='cuda')
                result=timing(lambda:pipeline(x));result['packets_per_second']=batch*1000/result['p50_ms']
                result['native_frontend']=timing(lambda:frontend(x))
                if batch==1:
                    with FlopCounterMode(display=False) as c:pipeline(x)
                    result['partial_dispatcher_flops']=c.get_total_flops()
                    result['partial_dispatcher_MACs']=c.get_total_flops()/2
                item['batches'][str(batch)]=result
            item['coverage']='FP32 GPU-resident raw IQ to logits, no host transfer; original QR recomputed each forward. Dispatcher MACs are partial and exclude FFT/QR/unsupported complex arithmetic; never a full cost claim.'
            item['frontend_definition']='H0: RMS/IB/five RF views; CIS: common input transform and original CIS; FAN/IB: common input transform and SAF; ASK: common input transform, original ASK and CIS. CNN/tokenizer excluded from frontend sub-timing, retained in complete pipeline.'
            records.append(item);write_json(HERE/'COSTS.json',dict(gpu=torch.cuda.get_device_name(0),records=records,complete=False))
            del model;torch.cuda.empty_cache()
    attempts=[read(p) for p in (HERE/'runs').glob('*/attempt*_timing.json')]
    write_json(HERE/'COSTS.json',dict(gpu=torch.cuda.get_device_name(0),records=records,complete=True,
               total_actual_fit_wall_seconds=sum(a['actual_wall_seconds'] for a in attempts),fits_attempt_timings=len(attempts)))

if __name__=='__main__':main()
