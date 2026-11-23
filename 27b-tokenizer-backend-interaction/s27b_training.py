"""Original Shen200 update order with atomic full-epoch recovery added."""
import csv
import hashlib
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from s27b_common import HERE,read,sha,write,validate,identity,now,replace_retry
from s27b_models import build,shared_hash,state_hash,surviving_hash
from s27b_data import source
from xa.frontends import awgn
from xa.train import stable_seed
import train as canonical


def setup(seed):
    canonical.setup(seed)
    # These match the archived training/runtime defaults, not S25's separate
    # TF32-off feature diagnostic execution.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True


def atomic_torch(value,path):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    torch.save(value,temporary);replace_retry(temporary,path)


def learning_rate(step,steps,budget=200):
    warm=5*steps;total=steps*budget
    return 1e-3*(step+1)/warm if step<warm else 1e-3*.5*(1+math.cos(math.pi*(step-warm)/max(1,total-warm)))


def save_resume(path,model,optimizer,order,noise,**state):
    atomic_torch(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
        numpy_rng=np.random.get_state(),python_rng=random.getstate(),
        order_rng=order.get_state(),noise_rng=noise.get_state(),**state),path)


def restore(path,model,optimizer,order,noise):
    state=torch.load(path,map_location='cuda',weights_only=False)
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['cpu_rng'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
    np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
    order.set_state(state['order_rng'].cpu());noise.set_state(state['noise_rng'].cpu())
    return state


def fit(job):
    registration=validate();assert job in registration['jobs']
    fold,seed_label=job['fold'],job['seed'];name=identity(fold,seed_label,job['arm']);out=HERE/'runs'/name
    if (out/'metrics.json').exists():
        result=read(out/'metrics.json');assert sha(out/'model.pt')==result['model_sha256'];return result
    out.mkdir(parents=True,exist_ok=True)
    seed=stable_seed('P-H5x160',fold,seed_label);setup(seed)
    tr,va=source(fold);model=build(job['arm'],seed)
    reference=next(r for r in registration['references'] if r['arm']==job['arm'].split('_')[0] and r['fold']==fold and r['seed']==seed_label)
    oldcfg=read(Path(reference['checkpoint']).parent/'config.json')
    assert oldcfg['data_seed']==seed and oldcfg['batch']==64 and oldcfg['epochs_budget']==200
    old=build(job['arm'].split('_')[0],seed);assert surviving_hash(old)==surviving_hash(model)
    config=dict(job=job,data_seed=seed,batch=64,epochs_budget=200,params=sum(p.numel() for p in model.parameters()),
        checkpoint='minimum_source_receiver_equal_ce',train_packets=8000,val_packets=2000,
        optimizer='AdamW',lr=1e-3,weight_decay=5e-4,warmup_epochs=5,augmentation='AWGN 0..50dB, source train only',
        initial_shared_sha256=shared_hash(model),initial_surviving_sha256=surviving_hash(model),registration_sha256=sha(HERE/'registration.json'),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(0),tf32_matmul=False,tf32_cudnn=True,
        evaluation_batch=128,original_training_loop_sha256=registration['canonical_training_sha256'])
    assert config['params']=={'H0_P':133562,'LP_P':138218}[job['arm']];del old
    if (out/'config.json').exists():assert read(out/'config.json')==config
    else:write(out/'config.json',config)
    write(out/'pairing.json',dict(reference=reference['checkpoint'],reference_config_sha256=sha(Path(reference['checkpoint']).parent/'config.json'),
        initial_shared_sha256=shared_hash(model),initial_full_sha256=state_hash(model),same_source_packets=True,same_data_seed=True,
        same_permutation_and_noise_generators=True,tokenizer_initialization='Exact corresponding tokenizer/frontend/input_proj/position/LNs/classifier initialization; fixed shared S27B-Readout CPU stream for newMLP head',
        source_selection='Original last40/TX/RX; minimum RX-equal clean CE, no target score'))
    model=model.cuda();tr={k:torch.from_numpy(np.ascontiguousarray(v)).cuda() for k,v in tr.items()}
    vg={k:torch.from_numpy(np.ascontiguousarray(v)).cuda() for k,v in va.items()};vr=va['rx']
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=5e-4)
    order=torch.Generator(device='cuda').manual_seed(seed);noise=torch.Generator(device='cuda').manual_seed(seed+1)
    steps=125;history=[];best=float('inf');best_epoch=0;step=0;elapsed=0.;start_epoch=1
    resume=out/'resume.pt'
    if resume.exists():
        state=restore(resume,model,optimizer,order,noise)
        assert state['registration_sha256']==sha(HERE/'registration.json')
        history=state['history'];best=state['best'];best_epoch=state['best_epoch'];step=state['step'];elapsed=state['elapsed']
        start_epoch=len(history)+1
        write(out/f'recovery_epoch{start_epoch}.json',dict(time=now(),from_completed_epoch=len(history),
            optimizer_rng_model_restored=True,no_seed_restart=True,resume_sha256=sha(resume)))
    elif (out/'attempt.json').exists():
        raise RuntimeError('Existing attempt without an epoch checkpoint; retain evidence and review engineering failure')
    else:write(out/'attempt.json',dict(status='RUNNING',started_at=now(),job=job))
    started=time.perf_counter()
    try:
        for ep in range(start_epoch,201):
            model.train();permutation=torch.randperm(len(tr['y']),generator=order,device='cuda')
            total_loss=0.;correct=0;seen=0
            for ids in list(permutation.split(64))[:steps]:
                for group in optimizer.param_groups:group['lr']=learning_rate(step,steps)
                x,y=tr['x'][ids],tr['y'][ids];x=awgn(x,0,50,noise)
                if ep==1 and seen==0:
                    write(out/'streams.json',dict(permutation_sha256=hashlib.sha256(permutation.cpu().numpy().tobytes()).hexdigest(),
                        first_augmented_batch_sha256=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),data_seed=seed))
                logits=model(x);loss=F.cross_entropy(logits,y)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();step+=1
                total_loss+=float(loss.detach())*len(ids);correct+=int((logits.argmax(1)==y).sum());seen+=len(ids)
            lv=canonical.predict(model,vg['x'])
            ce=float(torch.stack([F.cross_entropy(lv[vg['rx']==r],vg['y'][vg['rx']==r]) for r in torch.unique(vg['rx'])]).mean())
            if not math.isfinite(ce):raise ValueError('Nonfinite source CE')
            if ce<best:
                best=ce;best_epoch=ep;atomic_torch({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'model.pt')
            duration=elapsed+time.perf_counter()-started
            row=dict(epoch=ep,train_ce=total_loss/seen,train_acc=correct/seen,source_val_ce=ce,
                source_val_acc=float((lv.argmax(1)==vg['y']).float().mean()),lr=optimizer.param_groups[0]['lr'],seconds=duration)
            history.append(row)
            save_resume(resume,model,optimizer,order,noise,history=history,best=best,best_epoch=best_epoch,
                step=step,elapsed=duration,registration_sha256=sha(HERE/'registration.json'))
            with (out/'history.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row),lineterminator='\n');writer.writeheader();writer.writerows(history)
            write(out/'live.json',dict(status='RUNNING',updated_at=now(),job=job,epochs_budget=200,**row))
            completed=sum((HERE/'runs'/identity(j['fold'],j['seed'],j['arm'])/'metrics.json').exists() for j in registration['jobs'])
            recent=np.diff([r['seconds'] for r in history[-11:]])
            remaining=[]
            for other in registration['jobs']:
                folder=HERE/'runs'/identity(other['fold'],other['seed'],other['arm'])
                remaining.append(0 if (folder/'metrics.json').exists() else 200-(read(folder/'live.json')['epoch'] if (folder/'live.json').exists() else 0))
            workers=read(HERE/'EXECUTION.json')['workers']
            estimate=float(np.median(recent)*max(max(remaining),sum(remaining)/workers)) if len(recent)>=5 else None
            write(HERE/'LIVE_STATUS.json',dict(status='TRAINING',updated_at=now(),completed_fits=completed,total_fits=12,
                active_model=name,active_epoch=ep,active_budget=200,remaining_fit_seconds_estimate=estimate,
                estimate_scope='Fits only; excludes final evaluation, analysis and manuscript work.',new_dev_reads=0))
            if ep==1 or ep%10==0:print(name,f'epoch={ep}/200 sourceCE={ce:.6f} elapsed={duration:.1f}s',flush=True)
        model.load_state_dict(torch.load(out/'model.pt',map_location='cuda',weights_only=True))
        lv=canonical.predict(model,vg['x'])
        result=dict(status='COMPLETE',job=job,best_epoch=best_epoch,best_source_ce=best,epochs=200,params=config['params'],
            optimizer_steps=step,wall_seconds=elapsed+time.perf_counter()-started,model_sha256=sha(out/'model.pt'),
            source_validation=canonical.score(lv,vg['y'],vr),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,finished_at=now())
        write(out/'metrics.json',result);write(out/'attempt.json',dict(status='COMPLETE',job=job,finished_at=now()))
        return result
    except Exception:
        import traceback
        write(out/'failure.json',dict(time=now(),traceback=traceback.format_exc(),resume_from_completed_epoch_only=True))
        raise
