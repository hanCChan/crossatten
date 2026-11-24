"""Original Shen200 updates, fixed stage-specific source sets and full recovery."""
import csv
import hashlib
import math
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from ac_common import HERE,read,write,sha,replace,validate,identity,now
from ac_data import source
from s28_models import build,state_hash,shared_hash,surviving_equal
from s27b_training import setup,learning_rate,save_resume,restore
from xa.frontends import awgn
from xa.train import stable_seed
import train as canonical

def atomic_torch(value,path):
    temporary=path.with_name(path.name+'.tmp');torch.save(value,temporary);replace(temporary,path)

def update(model,opt,x,y,noise,step,steps):
    for g in opt.param_groups:g['lr']=learning_rate(step,steps)
    augmented=awgn(x,0,50,noise);logits=model(augmented);loss=F.cross_entropy(logits,y)
    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
    opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    return logits,loss,augmented

def fit(job):
    reg=validate();assert job in reg['jobs'];name=identity(job);out=HERE/'runs'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():
        m=read(out/'metrics.json');assert sha(out/'model.pt')==m['model_sha256'];return 0
    stage=job['stage'];seed=stable_seed('P-H5x160' if stage=='A' else 'S27C-H4x160',job['fold'],job['seed']);setup(seed)
    tr,va=source(stage,job['fold']);model=build(job['arm'],seed)
    initial=state_hash(model);shared=shared_hash(model)
    if stage=='A':
        reference=build('H2',seed);assert initial==state_hash(reference) and surviving_equal(build('H0',seed),model);del reference
    else:assert shared==shared_hash(build('LP' if job['arm']=='H0' else 'H0',seed))
    config=dict(job=job,data_seed=seed,batch=64,epochs_budget=200,train_packets=len(tr['y']),val_packets=len(va['y']),
        steps_per_epoch=len(tr['y'])//64,params=sum(p.numel() for p in model.parameters()),checkpoint='minimum_source_receiver_equal_ce',
        optimizer='AdamW',lr=.001,weight_decay=.0005,warmup_epochs=5,augmentation='Original AWGN0..50dB',
        initial_full_sha256=initial,initial_shared_sha256=shared,pretrained_checkpoint_loaded=False,
        registration_sha256=sha(HERE/'registration.json'),tf32_matmul=False,tf32_cudnn=True,torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
    if (out/'config.json').exists():assert read(out/'config.json')==config
    else:write(out/'config.json',config)
    write(out/'pairing.json',dict(initial_full_sha256=initial,initial_shared_sha256=shared,surviving_state_matches_H0=stage=='A',
        original_H2_full_initial_state_matches=stage=='A',fresh_C_initialization=stage=='C',no_pretrained_load=True))
    model=model.cuda();tr={k:torch.as_tensor(np.ascontiguousarray(v),device='cuda') for k,v in tr.items()};vr=va['rx']
    vg={k:torch.as_tensor(np.ascontiguousarray(v),device='cuda') for k,v in va.items()}
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0005)
    order=torch.Generator(device='cuda').manual_seed(seed);noise=torch.Generator(device='cuda').manual_seed(seed+1)
    history=[];best=float('inf');best_epoch=0;best_state=None;step=0;elapsed=0.;resume=out/'resume.pt';steps=config['steps_per_epoch']
    if resume.exists():
        snapshot_sha=sha(resume);state=restore(resume,model,opt,order,noise)
        assert state['registration_sha256']==config['registration_sha256']
        history=state['history'];best=state['best'];best_epoch=state['best_epoch'];step=state['step'];elapsed=state['elapsed']
        assert step==len(history)*steps
        best_state={k:v.detach().cpu() for k,v in state['best_model'].items()};atomic_torch(best_state,out/'model.pt')
        write(out/f'recovery_epoch{len(history)+1}.json',dict(time=now(),from_completed_epoch=len(history),snapshot_sha256=snapshot_sha,
            optimizer_rng_restored=True,best_checkpoint_restored_from_same_atomic_snapshot=True))
    elif (out/'attempt.json').exists():raise RuntimeError('Prior attempt has no complete epoch; preserve and review before retry')
    else:write(out/'attempt.json',dict(time=now(),status='RUNNING',job=job))
    start=time.perf_counter()
    try:
        for ep in range(len(history)+1,201):
            model.train();permutation=torch.randperm(len(tr['y']),generator=order,device='cuda');total=correct=seen=0
            for ids in list(permutation.split(64))[:steps]:
                logits,loss,augmented=update(model,opt,tr['x'][ids],tr['y'][ids],noise,step,steps)
                if ep==1 and seen==0:
                    streams=dict(data_seed=seed,permutation_sha256=hashlib.sha256(permutation.cpu().numpy().tobytes()).hexdigest(),
                        first_augmented_batch_sha256=hashlib.sha256(augmented.cpu().numpy().tobytes()).hexdigest())
                    if stage=='A':
                        reference=next(q for q in reg['reference_streams_A'] if q['fold']==job['fold'] and q['seed']==job['seed'])
                        assert all(streams[k]==reference[k] for k in ['permutation_sha256','first_augmented_batch_sha256'])
                    write(out/'streams.json',streams)
                step+=1;total+=float(loss.detach())*len(ids);correct+=int((logits.argmax(1)==tr['y'][ids]).sum());seen+=len(ids)
            lv=canonical.predict(model,vg['x']);ce=float(torch.stack([F.cross_entropy(lv[vg['rx']==r],vg['y'][vg['rx']==r]) for r in torch.unique(vg['rx'])]).mean())
            if not math.isfinite(ce):raise ValueError('Nonfinite source CE')
            if ce<best:best=ce;best_epoch=ep;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            duration=elapsed+time.perf_counter()-start
            row=dict(epoch=ep,train_ce=total/seen,train_acc=correct/seen,source_val_ce=ce,source_val_acc=float((lv.argmax(1)==vg['y']).float().mean()),lr=opt.param_groups[0]['lr'],seconds=duration)
            history.append(row)
            # Best weights and current optimizer/RNG/history form one atomic
            # epoch snapshot; interrupted best-file writes cannot mismatch it.
            save_resume(resume,model,opt,order,noise,history=history,best=best,best_epoch=best_epoch,best_model=best_state,
                step=step,elapsed=duration,registration_sha256=config['registration_sha256'])
            atomic_torch(best_state,out/'model.pt')
            with (out/'history.csv').open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(row),lineterminator='\n');writer.writeheader();writer.writerows(history)
            write(out/'live.json',dict(time=now(),job=job,status='RUNNING',**row))
            if ep==1 or ep%10==0:print(name,f'epoch={ep}/200 sourceCE={ce:.6f} elapsed={duration:.1f}s',flush=True)
            if (HERE/'PAUSE_REQUESTED').exists():
                write(out/'paused.json',dict(time=now(),complete_epoch=ep,optimizer_steps=step));return 75
        model.load_state_dict(best_state);lv=canonical.predict(model,vg['x'])
        write(out/'metrics.json',dict(status='COMPLETE',job=job,epochs=200,optimizer_steps=step,best_epoch=best_epoch,best_source_ce=best,
            params=config['params'],wall_seconds=elapsed+time.perf_counter()-start,model_sha256=sha(out/'model.pt'),
            source_validation=canonical.score(lv,vg['y'],vr),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,finished_at=now()))
        write(out/'attempt.json',dict(status='COMPLETE',job=job,time=now()));return 0
    except Exception:
        write(out/'failure.json',dict(time=now(),traceback=traceback.format_exc(),complete_epoch_checkpoint_retained=True));raise
