"""Matched effective batches on fixed source-only RF tokens; no target CRD."""
import argparse,csv,math,time,random,traceback
import numpy as np
import torch
import torch.nn.functional as F
from s16_core import HERE,read,write,sha,now,validate,committed,git
from s16_models import build,parameter_audit,trainable_state,load_trainable,tensor_hash
from s16_data import load
from xa.train import stable_seed
from xa.evaluate import summarize
from metrics import rx_metrics

def run_id(j):return f"{j['phase']}__{j['dataset']}__{j['arm']}__f{j['fold']}__{j['group']}__s{j['seed']}"
def setup(seed):
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed%2**32);random.seed(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True,warn_only=True)
    if not torch.cuda.is_bf16_supported():raise RuntimeError('Registered BF16 device support missing')
    torch.cuda.reset_peak_memory_stats()
@torch.no_grad()
def predict(m,x,batch):
    m.eval();values=[]
    for i in range(0,len(x),batch):
        a=x[i:i+batch] if isinstance(x,torch.Tensor) else torch.tensor(np.array(x[i:i+batch],copy=True),device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):values.append(m(a).float())
    return torch.cat(values)
def score(logits,y,rx,names):
    if isinstance(logits,torch.Tensor):logits=logits.detach().float().cpu().numpy()
    if isinstance(y,torch.Tensor):y=y.cpu().numpy()
    per={names[int(r)]:rx_metrics(logits[rx==r],y[rx==r]) for r in np.unique(rx)}
    return dict(summary=summarize(per),per_rx=per)

def fit(job):
    reg=validate();matrix=HERE/f"jobs_{job['phase']}.json";committed(matrix)
    phase_registration=HERE/f"registration_{job['phase']}.json";committed(phase_registration)
    if sha(matrix)!=read(phase_registration)['matrix_sha256']:raise ValueError('Phase matrix differs from preregistration')
    if job not in read(matrix):raise ValueError('Unregistered head job')
    out=HERE/'runs'/run_id(job)
    if (out/'metrics.json').exists():return
    if (out/'attempt.json').exists():raise RuntimeError('Existing failed/interrupted attempt retained; no automatic retry')
    write(out/'attempt.json',dict(time=now(),status='RUNNING',job=job))
    try:
        seed=stable_seed('S16-head',job['dataset'],job['fold'],job['group'],job['seed']);setup(seed)
        data,cache=load(job);model=build(job['arm'],seed,10 if job['dataset']=='ManyRx' else 6)
        audit=parameter_audit(model);initial_hash=tensor_hash(model);model=model.cuda()
        large=job['arm'].startswith(('R3','R4','R5','R6'));micro=reg['large_microbatch'] if large else 128
        cfg=dict(job=job,model_seed=seed,source_split_seed=job['split_seed'],epochs_budget=20,effective_batch=128,microbatch=micro,
            optimizer='AdamW lr1e-3 wd5e-4;warmup2/cos20;clip1;no dropout',precision='autocast BF16;trainable parameters/optimizer FP32',
            initialization_sha256=initial_hash,rf_tokenizer_frozen=True,rf_front_sha256=cache['front_sha256'],rf_active_params=cache['rf_active_params'],
            total_active_params=audit['total_active_tail_params']+cache['rf_active_params'],trainable_params=audit['trainable_tail_params'],
            audit=audit,git_sha=git('rev-parse','HEAD'),registration_sha256=sha(HERE/'registration.json'),model_manifest_sha256=sha(HERE/'MODEL_MANIFEST.json'))
        write(out/'config.json',cfg)
        tr={k:torch.tensor(np.array(v,copy=True),device='cuda') for k,v in data['train'].items() if k in ['x','y','rx']}
        va={k:torch.tensor(np.array(v,copy=True),device='cuda') for k,v in data['val'].items() if k in ['x','y','rx']}
        params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-3,weight_decay=5e-4)
        generator=torch.Generator(device='cuda').manual_seed(seed);steps=math.ceil(len(tr['y'])/128);total=20*steps;warm=2*steps
        best=float('inf');best_ep=0;history=[];step=0;start=time.perf_counter()
        for epoch in range(1,21):
            model.train();permutation=torch.randperm(len(tr['y']),generator=generator,device='cuda');loss_sum=0.;correct=0;n=0
            for logical in permutation.split(128):
                lr=1e-3*(step+1)/warm if step<warm else .0005*(1+math.cos(math.pi*(step-warm)/max(1,total-warm)))
                for group in opt.param_groups:group['lr']=lr
                opt.zero_grad(set_to_none=True)
                for ids in logical.split(micro):
                    with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(tr['x'][ids]).float();loss=F.cross_entropy(logits,tr['y'][ids])
                    if not torch.isfinite(loss):raise ValueError('Nonfinite CE')
                    (loss*(len(ids)/len(logical))).backward()
                    loss_sum+=float(loss.detach())*len(ids);correct+=int((logits.argmax(1)==tr['y'][ids]).sum());n+=len(ids)
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();step+=1
            lv=predict(model,va['x'],micro)
            ce=float(torch.stack([F.cross_entropy(lv[va['rx']==r],va['y'][va['rx']==r]) for r in torch.unique(va['rx'])]).mean())
            if not math.isfinite(ce):raise ValueError('Nonfinite source validation')
            if ce<best:best=ce;best_ep=epoch;torch.save(trainable_state(model),out/'trainable.pt')
            row=dict(epoch=epoch,train_ce=loss_sum/n,train_acc=correct/n,source_val_ce=ce,source_val_acc=float((lv.argmax(1)==va['y']).float().mean()),
                lr=lr,seconds=time.perf_counter()-start,grad_norm=float(norm))
            history.append(row)
            with (out/'history.csv').open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row),lineterminator='\n');w.writeheader();w.writerows(history)
            write(out/'live.json',dict(time=now(),status='RUNNING',job=job,epochs_budget=20,**row))
            print(run_id(job),'epoch',epoch,'sourceCE',round(ce,5),'seconds',round(row['seconds'],1),flush=True)
        load_trainable(model,torch.load(out/'trainable.pt',map_location='cpu',weights_only=True))
        results={}
        for part in ['train','val','held']:
            d=data[part];logits=predict(model,d['x'],micro).cpu().numpy();results[part]=score(logits,np.asarray(d['y']),np.asarray(d['rx']),cache['rx_names'])
            if part=='held':np.savez_compressed(out/'held_predictions.npz',logits=logits,y=d['y'],rx=d['rx'],day=d['day'])
        write(out/'metrics.json',dict(job=job,status='COMPLETE',best_epoch=best_ep,best_source_ce=best,epochs=20,
            source_train=results['train'],source_validation=results['val'],held_source=results['held'],
            source_to_held_gap_pp=100*(results['val']['summary']['acc_mean']-results['held']['summary']['acc_mean']),
            active_params=cfg['total_active_params'],trainable_params=cfg['trainable_params'],peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,wall_seconds=time.perf_counter()-start,trainable_sha256=sha(out/'trainable.pt'),finished_at=now()))
        write(out/'attempt.json',dict(time=now(),status='COMPLETE',job=job))
    except Exception:
        write(out/'failure.json',dict(time=now(),traceback=traceback.format_exc(),automatic_retry=False));raise

def fit_front(job):
    # Only for independent-confirmation groups; retain the exact S13 B0 recipe.
    validate()
    phase={'front_r1':'confirm1','front_r2':'confirm2'}[job['phase']]
    phase_registration=HERE/f'registration_{phase}.json';committed(phase_registration)
    if sha(HERE/f"jobs_{job['phase']}.json")!=read(phase_registration)['front_matrix_sha256']:
        raise ValueError('RF front matrix differs from preregistration')
    import train as old
    from s16_data import cells,groups,flatten
    old.HERE=HERE;old.validate_registration=validate;old.wisig_cells=cells;old.source_groups=groups;old.flatten=flatten
    return old.fit(job)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--matrix',required=True);p.add_argument('--index',required=True,type=int);p.add_argument('--front',action='store_true');a=p.parse_args()
    job=read(HERE/a.matrix)[a.index];fit_front(job) if a.front else fit(job)
