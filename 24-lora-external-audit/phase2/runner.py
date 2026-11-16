"""Single registered source-only fit, with exact RNG/optimizer recovery."""
import argparse,copy,csv,json,math,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from execution import (HERE,STAGE,ROOT,read,digest,atomic_json,atomic_torch,git,jobid,setup,allowed,validate_registry)
from models import build,forward,hsic
from data import source,preprocess
sys.path.insert(0,str(ROOT/'lib'))
from xa.frontends import awgn

def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

@torch.no_grad()
def validate(model,method,view,values,batch):
    model.eval();x,y,rx=values
    txs=[];rxs=[]
    for start in range(0,len(x),batch):
        z=preprocess(x[start:start+batch].cuda(),view,method)
        tx,rc,_=forward(model,method,z);txs.append(tx.cpu())
        if rc is not None:rxs.append(rc.cpu())
    tx=torch.cat(txs);ce=F.cross_entropy(tx,y,reduction='none')
    equal=float(torch.stack([ce[rx==r].mean() for r in range(5)]).mean())
    result=dict(tx_ce=equal,tx_acc=float((tx.argmax(1)==y).float().mean()))
    if rxs:result['rx_ce']=float(F.cross_entropy(torch.cat(rxs),rx))
    return result

def fit(job):
    fit_started=time.perf_counter()
    registry=allowed(job);jid=jobid(job);out=HERE/'runs'/jid;out.mkdir(parents=True,exist_ok=True)
    config=out/'config.json';metrics=out/'metrics.json';reg_sha=digest(HERE/'pilot_registry.json')
    if metrics.exists() and read(metrics)['status']=='COMPLETE':
        if digest(out/'best.pt')!=read(metrics)['checkpoint_sha256']:raise ValueError('completed checkpoint changed')
        return
    cfg=dict(job=job,registry_sha256=reg_sha,git=git('rev-parse','HEAD'),gpu=torch.cuda.get_device_name(0),torch=torch.__version__)
    if config.exists():
        if read(config)['job']!=job or read(config)['registry_sha256']!=reg_sha:raise ValueError('run identity changed')
    else:atomic_json(config,cfg)
    seed=job['seed']+10000*job['fold'];setup(seed)
    # Explicit per-job RNG streams; CE/DANN controls share encoder weights, order and noise.
    order=torch.Generator().manual_seed(seed+101000)
    noise=torch.Generator(device='cuda').manual_seed(seed+202000)
    model=build(job['method']).cuda()
    if job['method']=='ib':
        init=HERE/'runs'/f"fan__{job['view']}__f{job['fold']}__s{job['seed']}"
        if read(init/'metrics.json')['status']!='COMPLETE':raise ValueError('matching Stage1 incomplete')
        if digest(init/'best.pt')!=read(init/'metrics.json')['checkpoint_sha256']:raise ValueError('Stage1 hash changed')
        model.load_state_dict(torch.load(init/'best.pt',map_location='cuda',weights_only=True))
        cfg=read(config);cfg['initialization_checkpoint_sha256']=digest(init/'best.pt');atomic_json(config,cfg)
    optimizer=(torch.optim.SGD(model.parameters(),lr=job['lr'],momentum=.9) if job['optimizer']=='SGD'
               else torch.optim.Adam(model.parameters(),lr=job['lr'],weight_decay=0.))
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=job['scheduler_factor'],patience=job['scheduler_patience'],min_lr=job['min_lr'])
    cells=source(job['fold'],jid)
    tx,ty,tr=cells['train'];vx,vy,vr=cells['val']
    if job['awgn']:
        vg=torch.Generator().manual_seed(seed+303000)
        vx=awgn(vx,*job['awgn'],vg)
    best=float('inf');best_epoch=0;best_state=None;bad=0;start_epoch=1;steps=0;elapsed_before=0.
    def snapshot(epoch,elapsed):
        return dict(model=cpu_state(model),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,
                    best=best,best_epoch=best_epoch,best_state=best_state,bad=bad,steps=steps,elapsed=elapsed,
                    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                    order_rng=order.get_state(),noise_rng=noise.get_state(),numpy_rng=np.random.get_state(),
                    registry_sha256=reg_sha,job=job)
    last=out/'last.pt'
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False)
        if ck['registry_sha256']!=reg_sha or ck['job']!=job:raise ValueError('recovery registry mismatch')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        best,best_epoch,best_state,bad,steps=ck['best'],ck['best_epoch'],ck['best_state'],ck['bad'],ck['steps']
        start_epoch=ck['epoch']+1;elapsed_before=ck['elapsed']
        torch.set_rng_state(ck['cpu_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        order.set_state(ck['order_rng']);noise.set_state(ck['noise_rng']);np.random.set_state(ck['numpy_rng'])
        if best_state is not None:atomic_torch(out/'best.pt',best_state)
    else:atomic_torch(last,snapshot(0,0.))
    attempt=1+len(list(out.glob('history_attempt*.csv')))
    history=out/f'history_attempt{attempt}.csv'
    ledger=HERE/'INITIALIZATIONS.json'
    entries=read(ledger) if ledger.exists() else {}
    if jid not in entries:
        if len(entries)>=registry['maximum_scientific_fit_initializations']:raise ValueError('fit budget exhausted')
        entries[jid]=dict(config_sha256=digest(config),first_started_git=git('rev-parse','HEAD'))
        atomic_json(ledger,entries)
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats();current_epoch=start_epoch-1
    try:
        for epoch in range(start_epoch,job['epochs']+1):
            current_epoch=epoch;et=time.perf_counter();model.train()
            permutation=torch.randperm(len(tx),generator=order);sum_ce=sum_hsic=0.;n=0
            for bi,start in enumerate(range(0,len(tx),job['batch'])):
                ids=permutation[start:start+job['batch']]
                x=tx[ids].cuda();y=ty[ids].cuda();r=tr[ids].cuda()
                if job['awgn']:x=awgn(x,*job['awgn'],noise)
                x=preprocess(x,job['view'],job['method'])
                optimizer.zero_grad(set_to_none=True)
                logits,rx,h=forward(model,job['method'],x,epoch)
                ce=F.cross_entropy(logits,y);loss=ce
                if job['method']=='dann':loss=loss+F.cross_entropy(rx,r)
                reg=torch.tensor(0.)
                if job['method'] in ('fan','ib'):
                    reg=hsic(h,x);loss=loss+job['alpha']*reg.cuda()
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
                loss.backward();optimizer.step();steps+=1
                sum_ce+=float(ce)*len(ids);sum_hsic+=float(reg)*len(ids);n+=len(ids)
                if bi%20==0:print(jid,'epoch',epoch,'batch',bi,'/',math.ceil(len(tx)/job['batch']),flush=True)
            if not all(torch.isfinite(p).all().item() for p in model.parameters()):raise FloatingPointError('nonfinite weights')
            val=validate(model,job['method'],job['view'],(vx,vy,vr),job['batch'])
            if not math.isfinite(val['tx_ce']):raise FloatingPointError('nonfinite source validation')
            scheduler.step(val['tx_ce'])
            if val['tx_ce']<best:
                best=val['tx_ce'];best_epoch=epoch;bad=0;best_state=cpu_state(model);atomic_torch(out/'best.pt',best_state)
            else:bad+=1
            row=dict(epoch=epoch,train_tx_ce=sum_ce/n,train_hsic=sum_hsic/n,source_tx_ce=val['tx_ce'],source_tx_acc=val['tx_acc'],
                     source_rx_ce=val.get('rx_ce'),lr=optimizer.param_groups[0]['lr'],best_epoch=best_epoch,optimizer_steps=steps,
                     epoch_seconds=time.perf_counter()-et,total_seconds=elapsed_before+time.perf_counter()-started)
            if job['method']=='ask':
                row['ask_k']=model.get_current_k();row['ask_probabilities']=json.dumps(model.ask.cutoff.selection_logits.detach().softmax(0).cpu().tolist())
            with history.open('a',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row),lineterminator='\n')
                if f.tell()==0:w.writeheader()
                w.writerow(row)
            stop=bad>=job['early_stop'] or epoch==job['epochs']
            if epoch%job['last_checkpoint_every_epochs']==0 or stop:atomic_torch(last,snapshot(epoch,row['total_seconds']))
            atomic_json(out/'progress.json',dict(status='TRAINING_SOURCE_ONLY',epoch=epoch,**{k:v for k,v in row.items() if k!='epoch'},peak_bytes=torch.cuda.max_memory_allocated()))
            print(jid,json.dumps(row),flush=True)
            if stop:break
        model.load_state_dict(best_state)
        final=validate(model,job['method'],job['view'],(vx,vy,vr),job['batch'])
        result=dict(status='COMPLETE',job=job,best_epoch=best_epoch,best_source_tx_ce=best,source_metrics=final,
                    epochs=current_epoch,optimizer_steps=steps,wall_seconds=elapsed_before+time.perf_counter()-started,
                    attempt=attempt,params=sum(p.numel() for p in model.parameters()),
                    real_degrees_of_freedom=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters()),
                    peak_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=digest(out/'best.pt'),
                    registry_sha256=reg_sha,config_sha256=digest(config),dev_evaluations=0)
        atomic_json(metrics,result);atomic_json(out/'progress.json',result)
    except BaseException as exc:
        failure=dict(status='FAILED_SOURCE_ONLY',job=job,epoch=current_epoch,attempt=attempt,total_optimizer_steps_at_failure=steps,
                     exception=repr(exc),traceback=traceback.format_exc(),last_recoverable_checkpoint=str(last),
                     elapsed_seconds=elapsed_before+time.perf_counter()-started)
        atomic_json(out/f'failure_attempt{attempt}.json',failure);atomic_json(out/'progress.json',failure);raise
    finally:
        atomic_json(out/f'attempt{attempt}_timing.json',dict(actual_wall_seconds=time.perf_counter()-fit_started,
                    started_epoch=start_epoch,attempt=attempt,includes_model_build_data_hashing_and_checkpoint_io=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
    registry=validate_registry();job=next(j for j in registry['jobs'] if jobid(j)==a.job);fit(job)
