"""Paired end-to-end small MHA fits; only source CE selects checkpoints."""
import argparse,csv,math,random,time,traceback
import numpy as np
import torch
import torch.nn.functional as F
from s17_core import HERE,read,write,sha,now,validate,committed,git
from s17_models import build
from s17_data import load
from s17_augment import perturb
from xa.train import stable_seed
from xa.evaluate import summarize
from metrics import rx_metrics
from s17_jobs import run_id,profiles_path,split_seed

def baseline(j):return dict(j,arm='M0',phase=j['phase'])
def setup(seed):
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed%2**32);random.seed(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True,warn_only=True)
    if not torch.cuda.is_available():raise RuntimeError('Registered single CUDA GPU required')
    torch.cuda.reset_peak_memory_stats()
@torch.no_grad()
def predict(m,x,batch=128):
    m.eval();return torch.cat([m(torch.as_tensor(np.array(x[i:i+batch],copy=True),device='cuda')) for i in range(0,len(x),batch)]).cpu().numpy()
def score(logits,d,names):
    per={names[int(r)]:rx_metrics(logits[d['rx']==r],d['y'][d['rx']==r]) for r in np.unique(d['rx'])}
    return dict(summary=summarize(per),per_rx=per)
def check_job(j):
    validate();matrix=HERE/f"jobs_{j['phase']}.json";committed(matrix)
    r=HERE/f"registration_{j['phase']}.json";committed(r)
    if sha(matrix)!=read(r)['matrix_sha256'] or j not in read(matrix):raise ValueError('Job differs from frozen matrix')

def fit(j):
    check_job(j);out=HERE/'runs'/run_id(j)
    if (out/'metrics.json').exists():return
    if (out/'attempt.json').exists():raise RuntimeError('Failed/interrupted scientific attempt retained; no automatic retry')
    write(out/'attempt.json',dict(time=now(),job=j,status='RUNNING'))
    try:
        seed=stable_seed('S17-WiSig',j['dataset'],j['fold'],j['group'],j['seed']);setup(seed)
        data,meta=load(j);tr=data['train'];va=data['val'];m=build(j['dataset'],seed).cuda()
        train={k:torch.as_tensor(v,device='cuda') for k,v in tr.items() if k in ['x','y','rx']}
        val={k:torch.as_tensor(v,device='cuda') for k,v in va.items() if k in ['x','y','rx']}
        family=j.get('family');pool=read(profiles_path(j))['profiles'] if family else None
        cfg=dict(job=j,data_seed=seed,source_split_seed=split_seed(j),epochs_budget=40,batch=128,optimizer='Adam1e-3;ReduceLROnPlateau factor.1 patience5 min1e-6;earlywait10',
            precision='FP32',augmentation_probability=.5,active_params=sum(p.numel() for p in m.parameters()),
            profile_sha256=sha(profiles_path(j)) if family else None,git_sha=git('rev-parse','HEAD'),
            input_roles='fit RX dates1-2 only; source held evaluated only after phase checkpoint seal',
            initial_state_sha256=__import__('hashlib').sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in m.state_dict().values())).hexdigest())
        write(out/'config.json',cfg)
        opt=torch.optim.Adam(m.parameters(),lr=1e-3);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.1,patience=5,min_lr=1e-6)
        order=torch.Generator(device='cuda').manual_seed(seed);aug=torch.Generator(device='cuda').manual_seed(seed+1);noise=torch.Generator(device='cuda').manual_seed(seed+2)
        best=float('inf');best_ep=0;wait=0;history=[];start=time.perf_counter()
        for epoch in range(1,41):
            m.train();total=0.;correct=0;seen=0
            for ids in torch.randperm(len(tr['y']),generator=order,device='cuda').split(128):
                x=train['x'][ids];y=train['y'][ids]
                if family:x=perturb(x,family,j['tau'],pool,aug,noise,control=j.get('noise_control',False),probability=.5)
                logits=m(x);loss=F.cross_entropy(logits,y)
                if not torch.isfinite(loss):raise ValueError('Nonfinite source CE')
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                total+=float(loss.detach())*len(ids);correct+=int((logits.argmax(1)==y).sum());seen+=len(ids)
            m.eval()
            with torch.no_grad():lv=torch.cat([m(val['x'][i:i+128]) for i in range(0,len(va['y']),128)])
            ce=float(torch.stack([F.cross_entropy(lv[val['rx']==r],val['y'][val['rx']==r]) for r in torch.unique(val['rx'])]).mean())
            if not math.isfinite(ce):raise ValueError('Nonfinite source validation')
            if ce<best:best=ce;best_ep=epoch;wait=0;torch.save({k:v.detach().cpu() for k,v in m.state_dict().items()},out/'model.pt')
            else:wait+=1
            scheduler.step(ce);row=dict(epoch=epoch,train_ce=total/seen,train_acc=correct/seen,source_val_ce=ce,
                source_val_acc=float((lv.argmax(1)==val['y']).float().mean()),lr=opt.param_groups[0]['lr'],seconds=time.perf_counter()-start)
            history.append(row)
            with (out/'history.csv').open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row),lineterminator='\n');w.writeheader();w.writerows(history)
            write(out/'live.json',dict(time=now(),status='RUNNING',job=j,epochs_budget=40,**row))
            print(run_id(j),'epoch',epoch,'sourceCE',round(ce,5),flush=True)
            if wait>=10:break
        m.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True));logits=predict(m,va['x'])
        np.savez_compressed(out/'source_val_predictions.npz',logits=logits,**{k:v for k,v in va.items() if k!='x'})
        write(out/'metrics.json',dict(job=j,status='COMPLETE',best_epoch=best_ep,best_source_ce=best,epochs=len(history),
            source_validation=score(logits,va,meta['rx_names']),wall_seconds=time.perf_counter()-start,
            model_sha256=sha(out/'model.pt'),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,finished_at=now()))
        write(out/'attempt.json',dict(time=now(),job=j,status='COMPLETE'))
    except Exception:write(out/'failure.json',dict(time=now(),traceback=traceback.format_exc(),automatic_retry=False));raise

def evaluate(j):
    check_job(j);seal=HERE/f"seal_{j['phase']}.json";committed(seal);out=HERE/'runs'/run_id(j)
    if (out/'held_metrics.json').exists():return
    m=read(out/'metrics.json')
    if sha(out/'model.pt')!=read(seal)['checkpoints'][run_id(j)]:raise ValueError('Unsealed checkpoint')
    cfg=read(out/'config.json');setup(cfg['data_seed']);model=build(j['dataset'],cfg['data_seed']).cuda()
    model.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True));d,meta=load(j,held=True);held=d['held']
    logits=predict(model,held['x']);np.savez_compressed(out/'held_predictions.npz',logits=logits,**{k:v for k,v in held.items() if k!='x'})
    write(out/'held_metrics.json',dict(job=j,model_sha256=m['model_sha256'],held_source=score(logits,held,meta['rx_names'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--index',required=True,type=int);p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    j=read(HERE/f'jobs_{a.phase}.json')[a.index];evaluate(j) if a.evaluate else fit(j)
