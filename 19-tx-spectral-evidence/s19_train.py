"""Strictly paired source fits; teacher evidence never touches held-source."""
import csv
import hashlib
import random
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from s19_core import HERE, ROOT, read, write, sha, now, validate, git, job_id, committed
from s19_data import load_fit, teacher
from s19_models import build_model
from s19_metrics import score


def setup(seed):
    torch.set_num_threads(2); torch.manual_seed(seed); np.random.seed(seed%2**32); random.seed(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    if not torch.cuda.is_available(): raise RuntimeError('Registered CUDA environment required')
    torch.cuda.reset_peak_memory_stats()


@torch.no_grad()
def predict(model,x,batch=128):
    model.eval()
    return np.concatenate([model(torch.as_tensor(np.array(x[i:i+batch],copy=True),device='cuda')).cpu().numpy() for i in range(0,len(x),batch)])


def save_predictions(path,logits,data):
    np.savez_compressed(path,logits=logits,**{k:v for k,v in data.items() if k!='x'})


def shared_hash(model):
    keys=build_model('B0','ManyRx',123).state_dict()
    # Hashing is independent of RNG: callers restore setup before fitting.
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for k,v in model.state_dict().items() if k in keys)).hexdigest()


def fit(fold,arm):
    validate()
    if arm not in ['B1','B2','B3','B4']: raise ValueError('Only registered new arms may fit')
    out=HERE/'runs'/job_id(fold,arm)
    if (out/'metrics.json').exists(): return
    if (out/'attempt.json').exists(): raise RuntimeError('Interrupted fit retained; no automatic scientific retry')
    write(out/'attempt.json',dict(time=now(),fold=fold,arm=arm,status='RUNNING'))
    started=time.perf_counter()
    try:
        data,meta,_,indices=load_fit(fold);tr,va=data['train'],data['val']
        _,base_cfg,_=teacher(fold);seed=base_cfg['data_seed'];setup(seed)
        model=build_model(arm,'ManyRx',seed).cuda()
        # Preserve RNG while constructing the key template for initial-state QA.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]): initial=shared_hash(model)
        if initial!=base_cfg['initial_state_sha256']: raise ValueError('Non-matching backbone initialization')
        qpath=HERE/'cache'/f'evidence_f{fold}.npz';qrecord=read(HERE/f'evidence_f{fold}.json')
        if sha(qpath)!=qrecord['cache_sha256']: raise ValueError('Changed teacher evidence cache')
        with np.load(qpath,allow_pickle=False) as cache: q=torch.as_tensor(cache['bar_q'],dtype=torch.float32,device='cuda')
        train={k:torch.as_tensor(v,device='cuda') for k,v in tr.items() if k in ['x','y','rx']}
        val={k:torch.as_tensor(v,device='cuda') for k,v in va.items() if k in ['x','y','rx']}
        cfg=dict(fold=fold,arm=arm,seed=18101,data_seed=seed,split_seed=16001,epochs_budget=40,batch=128,precision='FP32',
                 optimizer='Adam lr1e-3 wd0; ReduceLROnPlateau factor.1 patience5 min1e-6; earlywait10',
                 active_params=sum(p.numel() for p in model.parameters()),initial_backbone_sha256=initial,
                 initial_full_state_sha256=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest(),
                 evidence_lambda=.1 if arm=='B4' else 0.,evidence_sha256=qrecord['cache_sha256'],code_git_sha=git('rev-parse','HEAD'),
                 checkpoint_rule='fit-RX-equal source validation CE only',train_indices_sha256=hashlib.sha256(indices['train'].tobytes()).hexdigest(),
                 val_indices_sha256=hashlib.sha256(indices['val'].tobytes()).hexdigest())
        write(out/'config.json',cfg)
        opt=torch.optim.Adam(model.parameters(),lr=1e-3)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.1,patience=5,min_lr=1e-6)
        order=torch.Generator(device='cuda').manual_seed(seed)
        best=float('inf');best_epoch=0;wait=0;history=[]
        with (out/'stdout.log').open('w',encoding='utf-8',newline='\n') as log:
            for epoch in range(1,41):
                model.train();total=0.;total_ce=0.;total_evidence=0.;seen=0;correct=0
                permutation=torch.randperm(len(tr['y']),generator=order,device='cuda')
                order_digest=hashlib.sha256(permutation.cpu().numpy().tobytes()).hexdigest()
                for ids in permutation.split(128):
                    logits=model(train['x'][ids]);y=train['y'][ids]
                    ce=F.cross_entropy(logits,y)
                    evidence=-(q[y]*F.log_softmax(model.last_evidence_logits,dim=1)).sum(1).mean() if arm=='B4' else ce.new_zeros(())
                    loss=ce+(.1*evidence if arm=='B4' else 0.)
                    if not torch.isfinite(loss): raise ValueError('Nonfinite training objective')
                    opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                    total+=float(loss.detach())*len(ids);total_ce+=float(ce.detach())*len(ids);total_evidence+=float(evidence.detach())*len(ids)
                    seen+=len(ids);correct+=int((logits.argmax(1)==y).sum())
                model.eval()
                with torch.no_grad(): vl=torch.cat([model(val['x'][i:i+128]) for i in range(0,len(va['y']),128)])
                val_ce=float(torch.stack([F.cross_entropy(vl[val['rx']==r],val['y'][val['rx']==r]) for r in torch.unique(val['rx'])]).mean())
                if not np.isfinite(val_ce): raise ValueError('Nonfinite source CE')
                if val_ce<best:
                    best=val_ce;best_epoch=epoch;wait=0
                    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'model.pt')
                else: wait+=1
                scheduler.step(val_ce)
                row=dict(epoch=epoch,train_loss=total/seen,train_ce=total_ce/seen,evidence_ce=total_evidence/seen,train_acc=correct/seen,
                         source_val_ce=val_ce,source_val_acc=float((vl.argmax(1)==val['y']).float().mean()),lr=opt.param_groups[0]['lr'],seconds=time.perf_counter()-started,
                         order_sha256=order_digest)
                history.append(row)
                with (out/'history.csv').open('w',encoding='utf-8',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(row),lineterminator='\n');w.writeheader();w.writerows(history)
                line=f'{job_id(fold,arm)} epoch={epoch} sourceCE={val_ce:.6f}'
                log.write(line+'\n');log.flush()
                write(out/'live.json',dict(time=now(),status='RUNNING',**row))
                if wait>=10:break
        model.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True))
        logits=predict(model,va['x']);save_predictions(out/'source_val_predictions.npz',logits,va)
        write(out/'metrics.json',dict(fold=fold,arm=arm,status='COMPLETE',best_epoch=best_epoch,best_source_ce=best,epochs=len(history),
              source_validation=score(logits,va,meta['rx_names']),model_sha256=sha(out/'model.pt'),prediction_sha256=sha(out/'source_val_predictions.npz'),
              wall_seconds=time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,finished_at=now()))
        write(out/'attempt.json',dict(time=now(),fold=fold,arm=arm,status='COMPLETE'))
        print(job_id(fold,arm),'COMPLETE','epochs',len(history),'seconds',round(time.perf_counter()-started,2),flush=True)
    except Exception:
        write(out/'failure.json',dict(time=now(),traceback=traceback.format_exc(),automatic_retry=False))
        raise
