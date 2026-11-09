"""Synthetic P0 only; select fixed memory-safe microbatch before scientific fits."""
import time,math,gc
import torch
from s16_core import HERE,read,write,sha,now,validate
from s16_models import build,parameter_audit,trainable_state,load_trainable

@torch.no_grad()
def inference_bench(model,rf):
    model.eval();rf.eval();results=[]
    for batch in [1,16]:
        x=torch.randn(batch,2,256,device='cuda');z=rf(x)
        for mode in ['tail','end_to_end']:
            times=[]
            for step in range(13):
                torch.cuda.synchronize();start=time.perf_counter()
                tokens=rf(x) if mode=='end_to_end' else z
                with torch.autocast('cuda',dtype=torch.bfloat16):model(tokens)
                torch.cuda.synchronize()
                if step>=3:times.append((time.perf_counter()-start)*1000)
            results.append(dict(batch=batch,mode=mode,median_ms=sorted(times)[len(times)//2],repeats=10,synthetic=True,
                frontend_precision='FP32 FFT/CNN, same as cached tokens',tail_precision='BF16 autocast'))
    return results

def run():
    validate('design.json');torch.set_num_threads(2)
    if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 unavailable on registered device')
    for file in read(HERE/'MODEL_MANIFEST.json')['files']:
        from pathlib import Path
        if sha(Path(file['path']))!=file['sha256']:raise ValueError('Official asset changed')
    rows=[];checks=[]
    from s16_data import RFTokenizer,front_ref
    from wifi import build_wifi
    ref,_,cfg=front_ref(dict(dataset='ManyRx',fold=0,group='first',split_seed=16001))
    base=build_wifi('ManyRx','B0',cfg['data_seed']);base.load_state_dict(torch.load(ref/'model.pt',map_location='cpu',weights_only=True))
    rf=RFTokenizer(base).cuda();del base
    # Includes synthetic RF-token data-pool space, which is resident in actual fits.
    reserve=torch.empty(40000,32,64,device='cuda',dtype=torch.float32)
    for arm in ['R0_mha','R0_hstu','R1_qwen_small','R2_transformer','R3_pretrained_frozen','R4_random_frozen','R5_pretrained_lora','R6_random_lora']:
        m=build(arm,16001).cuda();audit=parameter_audit(m);params=[p for p in m.parameters() if p.requires_grad]
        sizes=[16,32,64,128] if arm=='R5_pretrained_lora' else [2,128] if arm.startswith(('R0','R1','R2')) else [2]
        initial=trainable_state(m)
        for batch in sizes:
            try:
                load_trainable(m,initial);opt=torch.optim.AdamW(params,lr=1e-3,weight_decay=5e-4)
                x=torch.randn(batch,32,64,device='cuda');y=torch.arange(batch,device='cuda')%10
                torch.cuda.reset_peak_memory_stats();times=[]
                for step in range(4):
                    opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
                    with torch.autocast('cuda',dtype=torch.bfloat16):logits=m(x).float();loss=torch.nn.functional.cross_entropy(logits,y)
                    if not torch.isfinite(loss):raise ValueError('Synthetic nonfinite loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                    if not any(p.grad is not None and bool((p.grad!=0).any()) for p in params):raise ValueError('Missing trainable gradient')
                    if hasattr(m,'project') and (m.project.weight.grad is None or m.cls.grad is None or not bool((m.project.weight.grad!=0).any()) or not bool((m.cls.grad!=0).any())):
                        raise ValueError('Frozen backbone blocks RF projection/CLS gradients')
                    if arm.startswith(('R3','R4')) and any(p.grad is not None for p in m.backbone.parameters()):raise ValueError('Frozen backbone got gradients')
                    opt.step();torch.cuda.synchronize()
                    if step:times.append(time.perf_counter()-start)
                allocated=torch.cuda.max_memory_allocated()/2**30;reserved=torch.cuda.max_memory_reserved()/2**30
                elapsed=sorted(times)[len(times)//2]
                safe=allocated<=5.75 and reserved<=6.5
                rows.append(dict(arm=arm,batch=batch,status='PASS' if safe else 'MEMORY_CAP',seconds=elapsed,samples_s=batch/elapsed,
                    peak_allocated_gib=allocated,peak_reserved_gib=reserved,synthetic=True,**{k:v for k,v in audit.items() if k!='details'}))
                print('P0',rows[-1],flush=True)
                del opt,x,y,logits,loss;gc.collect();torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                rows.append(dict(arm=arm,batch=batch,status='ENGINEERING_OOM',synthetic=True));m.zero_grad(set_to_none=True)
                opt=None;x=None;y=None;logits=None;loss=None;gc.collect();torch.cuda.empty_cache();break
        checks.append(dict(arm=arm,audit=audit,inference=inference_bench(m,rf)));del m,params,initial;gc.collect();torch.cuda.empty_cache()
    candidates=[r for r in rows if r['arm']=='R5_pretrained_lora' and r['status']=='PASS']
    if any(not any(r['arm']==arm and r['batch']==128 and r['status']=='PASS' for r in rows)
           for arm in ['R0_mha','R0_hstu','R1_qwen_small','R2_transformer']):
        write(HERE/'P0_ENGINEERING_FAILURE.json',dict(time=now(),rows=rows,checks=checks,status='SMALL_MODEL_BATCH128_FAILED'))
        raise RuntimeError('Fixed small-model batch128 failed engineering resource checks')
    best=max(candidates,key=lambda r:r['samples_s']) if candidates else None
    projected=None if best is None else 20*math.ceil(30000/128)*(128/best['batch'])*best['seconds']*1.4
    allowed=best is not None and projected<=7200
    write(HERE/'P0_ENGINEERING.json',dict(time=now(),status='GO_FULL_PILOT' if allowed else 'SMALL_MODELS_ONLY_RESOURCE_GATE',
        gpu=torch.cuda.get_device_name(0),large_models_allowed=allowed,large_microbatch=best['batch'] if allowed else None,
        projected_large_fit_seconds=projected,full_fit_limit_seconds=7200,rows=rows,checks=checks,
        fixed_RF_frontend=True,scientific_accuracy_evidence=False,random_labels=True,model_revision=read(HERE/'MODEL_MANIFEST.json')['revision']))
if __name__=='__main__':run()
