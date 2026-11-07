"""Single-GPU fits; only source validation chooses checkpoints."""
import argparse
import copy
import csv
import math
import time
import random
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from core import HERE, ROOT, read, write, once, now, sha, array_sha, validate_registration, require_committed, git
from data import shen_train, shen_cell, shen_roles, PROTOCOL, wisig_cells, source_groups, flatten
from models import build, nparams
from wifi import build_wifi, rotate
from xa.frontends import awgn
from xa.train import stable_seed
from xa.evaluate import summarize
from metrics import rx_metrics


def job_id(job):
    return f"{job['phase']}__{job['dataset']}__{job['arm']}__f{job['fold']}__{job.get('group','all')}__s{job['seed']}"


def setup(seed):
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed); np.random.seed(seed % 2**32); random.seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("S13 execution requires the registered CUDA environment")
    torch.cuda.reset_peak_memory_stats()


@torch.no_grad()
def predict(model, x, batch=128):
    model.eval()
    return torch.cat([model(torch.as_tensor(np.asarray(x[i:i+batch]), device="cuda"))
                      if not isinstance(x, torch.Tensor) else model(x[i:i+batch]) for i in range(0,len(x),batch)])


def score(logits, y, rx):
    logits = logits.detach().cpu().numpy() if isinstance(logits,torch.Tensor) else logits
    y = y.detach().cpu().numpy() if isinstance(y,torch.Tensor) else y
    per = {str(r):rx_metrics(logits[np.asarray(rx)==r],y[np.asarray(rx)==r]) for r in np.unique(rx)}
    return dict(summary=summarize(per),per_rx=per)


def smoke_subset(d, cap):
    # Round-robin TX/RX cells so a smoke fit exercises all labels and receivers.
    groups=[np.flatnonzero((d['y']==y)&(d['rx']==r)) for r in np.unique(d['rx']) for y in np.unique(d['y'])]
    ids=[]
    for i in range(max(map(len,groups))):
        ids.extend(g[i] for g in groups if len(g)>i)
        if len(ids)>=cap:break
    ids=np.asarray(ids[:cap])
    return {k:v[ids] for k,v in d.items()}


def fit(job):
    registration = validate_registration()
    matrix = HERE / ("jobs_smoke.json" if job["phase"] == "smoke" else f"jobs_{job['phase']}.json")
    require_committed(matrix)
    if job not in read(matrix):
        raise ValueError("Unregistered job")
    p2 = job["phase"].startswith("p2")
    if p2:
        require_committed(HERE / "decision_p0.json")
        require_committed(HERE / "registration_p2.json")
        if not read(HERE / "decision_p0.json")["cfo_pilot_allowed"]:
            raise ValueError("P2 has not passed the estimator gate")
    out = HERE / "runs" / job_id(job)
    if (out / "metrics.json").exists():
        return read(out / "metrics.json")
    if (out / "attempt.json").exists():
        raise RuntimeError(f"Prior interrupted/failed attempt retained; no automatic retry: {out.name}")
    out.mkdir(parents=True,exist_ok=True)
    write(out / "attempt.json",dict(started_at=now(),status="RUNNING",job=job))
    try:
        seed = stable_seed("P-H5x160",job["fold"],job["seed"]) if job["dataset"]=="Shen" else stable_seed("S13-WiSig",job["fold"],job["seed"],job.get("group","first"),job["dataset"])
        setup(seed)
        if job["dataset"]=="Shen":
            tr,va=shen_train(job["fold"])
            model=build(job["arm"],seed)
            batch=64; budget=200
        else:
            fit_rx,held_rx=source_groups(job["dataset"],job["fold"],job.get("group","first"))
            pool=flatten(wisig_cells(job["dataset"],job["fold"],fit_rx,role="fit"))
            train_ids,val_ids=train_test_split(np.arange(len(pool["y"])),test_size=.3,random_state=job["seed"])
            tr,va=[{k:pool[k][ids] for k in ("x","y","rx")} for ids in (train_ids,val_ids)]
            model=build_wifi(job["dataset"],job["arm"],seed)
            batch=128;budget=40
        if job["phase"]=="smoke":
            tr=smoke_subset(tr,256);va=smoke_subset(va,128);budget=2
        config=dict(job=job,data_seed=seed,batch=batch,epochs_budget=budget,params=nparams(model),
                    checkpoint="minimum_source_receiver_equal_ce",train_packets=len(tr["y"]),val_packets=len(va["y"]),
                    initial_shared_sha256=array_sha(np.concatenate([v.detach().cpu().numpy().reshape(-1) for k,v in model.state_dict().items() if "blocks." not in k])),
                    registration_sha256=sha(HERE/"registration.json"),torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
                    git_sha=git('rev-parse','HEAD'),amendments_sha256={p.name:sha(p) for p in sorted(HERE.glob('registration_amendment_*.json'))},
                    input_roles="source RX only; WiSig dates1-2 only",evaluation_batch=128)
        write(out/"config.json",config)
        tr={k:torch.from_numpy(np.ascontiguousarray(v)).cuda() for k,v in tr.items()}
        va_gpu={k:torch.from_numpy(np.ascontiguousarray(v)).cuda() for k,v in va.items()}
        vr=va["rx"]
        model=model.cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=5e-4) if job["dataset"]=="Shen" else torch.optim.Adam(model.parameters(),lr=1e-3)
        scheduler=None if job["dataset"]=="Shen" else torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.1,patience=5,min_lr=1e-6)
        order=torch.Generator(device="cuda").manual_seed(seed)
        noise=torch.Generator(device="cuda").manual_seed(seed+1)
        cfo_rng=torch.Generator(device="cuda").manual_seed(seed+2)
        cfo_bound=read(HERE/'registration_p2.json')['bounds_hz'][job['dataset']][str(job['fold'])] if job['arm']=='B1_cfo' else None
        steps=len(tr["y"])//batch if job["dataset"]=="Shen" else math.ceil(len(tr["y"])/batch)
        best=float("inf");best_epoch=0;wait=0;history=[];started=time.time();step=0
        for ep in range(1,budget+1):
            model.train();perm=torch.randperm(len(tr["y"]),generator=order,device="cuda");total_loss=0.;correct=0;seen=0
            for ids in list(perm.split(batch))[:steps]:
                if job["dataset"]=="Shen":
                    warm=5*steps;total=steps*budget
                    lr=1e-3*(step+1)/warm if step<warm else 1e-3*.5*(1+math.cos(math.pi*(step-warm)/max(1,total-warm)))
                    for group in optimizer.param_groups:group["lr"]=lr
                x,y=tr["x"][ids],tr["y"][ids]
                if job["dataset"]=="Shen":x=awgn(x,0,50,noise)
                if job["arm"]=="B1_cfo":
                    hz=(torch.rand(len(x),device="cuda",generator=cfo_rng)*2-1)*cfo_bound
                    x=rotate(x,hz)
                logits=model(x);loss=F.cross_entropy(logits,y)
                if not torch.isfinite(loss):raise ValueError("Nonfinite loss")
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();step+=1
                total_loss+=float(loss.detach())*len(ids);correct+=int((logits.argmax(1)==y).sum());seen+=len(ids)
            lv=predict(model,va_gpu["x"])
            ce=float(torch.stack([F.cross_entropy(lv[va_gpu['rx']==r],va_gpu['y'][va_gpu['rx']==r]) for r in torch.unique(va_gpu['rx'])]).mean())
            if not math.isfinite(ce):raise ValueError("Nonfinite source validation")
            improved=ce<best
            if improved:
                best=ce;best_epoch=ep;wait=0
                torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/"model.pt")
            else:wait+=1
            if scheduler is not None:scheduler.step(ce)
            history.append(dict(epoch=ep,train_ce=total_loss/seen,train_acc=correct/seen,source_val_ce=ce,
                                source_val_acc=float((lv.argmax(1)==va_gpu['y']).float().mean()),lr=optimizer.param_groups[0]['lr'],seconds=time.time()-started))
            with (out/"history.csv").open("w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=history[0]);writer.writeheader();writer.writerows(history)
            write(out/'live.json',dict(updated_at=now(),status='RUNNING',job=job,**history[-1],epochs_budget=budget))
            if ep==1 or ep%10==0:print(f"{out.name} ep={ep}/{budget} sourceCE={ce:.5f} elapsed={time.time()-started:.1f}s",flush=True)
            if scheduler is not None and wait>=10:break
        model.load_state_dict(torch.load(out/"model.pt",map_location="cuda",weights_only=True))
        lv=predict(model,va_gpu["x"])
        result=dict(job=job,status="COMPLETE",best_epoch=best_epoch,best_source_ce=best,epochs=len(history),
                    params=nparams(model),wall_seconds=time.time()-started,model_sha256=sha(out/"model.pt"),
                    source_validation=score(lv,va_gpu['y'],vr),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,finished_at=now())
        if job["dataset"]!="Shen":
            held=wisig_cells(job["dataset"],job["fold"],held_rx,role="held")
            per={};saved={}
            for rx in held_rx:
                x=np.concatenate([held[rx,d]['x'] for d in (1,2)]);y=np.concatenate([held[rx,d]['y'] for d in (1,2)])
                lg=predict(model,x).cpu().numpy();per[rx]=rx_metrics(lg,y);saved[rx]=lg;saved['y_'+rx]=y
            result['held_source']=dict(summary=summarize(per),per_rx=per)
            np.savez_compressed(out/'preds_held.npz',**saved)
        write(out/"metrics.json",result)
        write(out/"attempt.json",dict(started_at=read(out/"attempt.json")["started_at"],finished_at=now(),status="COMPLETE",job=job))
        return result
    except Exception:
        write(out/"failure.json",dict(time=now(),error=traceback.format_exc(),automatic_retry=False))
        raise


def evaluate_shen(job):
    validate_registration()
    require_committed(HERE/'jobs_p1.json')
    if job not in read(HERE/'jobs_p1.json'):raise ValueError('Unregistered evaluation')
    require_committed(HERE/"decision_p1_source.json")
    out=HERE/"runs"/job_id(job)
    if (out/'target_metrics.json').exists():return read(out/'target_metrics.json')
    m=read(out/'metrics.json');cfg=read(out/'config.json')
    frozen=read(HERE/'decision_p1_source.json')['checkpoints'][out.name]
    if sha(out/'model.pt')!=frozen or m['model_sha256']!=frozen:raise ValueError('Unsealed checkpoint')
    setup(cfg['data_seed']);model=build(job['arm'],cfg['data_seed']).cuda()
    model.load_state_dict(torch.load(out/'model.pt',map_location='cuda',weights_only=True))
    result={}
    for role,rxs in [('dev_val',shen_roles(job['fold'])['dev_val']),('confirm',PROTOCOL['confirm_receivers'])]:
        per={};saved={}
        for rx in rxs:
            d=shen_cell(rx,job['fold'],role);logits=predict(model,d['x']).cpu().numpy()
            per[rx]=rx_metrics(logits,d['y']);saved[rx]=logits;saved['y_'+rx]=d['y']
        result[role]=dict(summary=summarize(per),per_rx=per)
        np.savez_compressed(out/f'preds_{role}.npz',**saved)
    write(out/'target_metrics.json',result)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--matrix',required=True);ap.add_argument('--index',type=int,required=True);ap.add_argument('--evaluate',action='store_true')
    a=ap.parse_args();job=read(HERE/a.matrix)[a.index]
    evaluate_shen(job) if a.evaluate else fit(job)
