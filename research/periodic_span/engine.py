"""Bounded source-CV execution; selection IQ is inaccessible before a committed seal."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import traceback

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


C=module('span_frozen_C',ROOT/'research/standard_method_competition_c/engine.py')
M=module('span_models',HERE/'models.py')
S=module('span_frozen_C_statistics',ROOT/'research/standard_method_competition_c/analyze.py')
A=C.A
read,write,sha,now,semantic_hash,cpu_tree=A.read,A.write,A.file_sha,A.now,A.semantic_hash,A.cpu_tree
ENDPOINTS=('selected','epoch10','final40')


def committed(path):
    A.require_committed(path)


def validate():
    reg=read(HERE/'REGISTRATION.json')
    committed(HERE/'REGISTRATION.json')
    if sha(ROOT/'research/periodic_hstem/AUDIT_RESULT.json')!=reg['prior_fixed_window_result_sha256']:
        raise ValueError('Original fixed-window STOP evidence changed')
    for name,digest in reg['code_sha256'].items():
        p=ROOT/name;committed(p)
        if sha(p)!=digest:raise ValueError('Frozen execution code changed: '+name)
    for name,digest in reg['artifacts_sha256'].items():
        p=HERE/name;committed(p)
        if sha(p)!=digest:raise ValueError('Frozen artifact changed: '+name)
    if str(torch.__version__)!=reg['environment']['torch'] or np.__version__!=reg['environment']['numpy']:
        raise ValueError('Frozen software environment changed')
    jobs=read(HERE/'JOBS.json')
    expected={(v,s,a) for v in ('V0','V1') for s in ('S0','S1','S2') for a in M.ARMS}
    if len(jobs)!=24 or {(j['view'],j['stream'],j['arm']) for j in jobs}!=expected:
        raise ValueError('Exactly24 predeclared architecture fits required')
    for j in jobs:
        k=int(j['stream'][-1])
        if (j['initialization_seed'],j['order_seed'],j['stochastic_seed'])!=(24101+k,24201+k,24301):raise ValueError('Seed drift')
        if j['epochs']!=40 or j['batch']!=128 or j['weight_decay']!=.001 or j['max_gradient_updates']!=5080:raise ValueError('Budget drift')
    return reg,jobs


def access(name,item,action,job):
    with (HERE/'ACCESS_LOG.jsonl').open('a',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(dict(time=now(),job=job,name=name,path=item['path'],action=action,sha256=item['sha256']))+'\n')


def input_path(reg,name,job):
    item=reg['source_input_whitelist'][name]
    access(name,item,'hash_registered_source_bytes',job)
    p=Path(item['path'])
    if sha(p)!=item['sha256']:raise ValueError('Source input changed: '+name)
    return p


def metadata(reg,job):
    pool={}
    for name in ('fit_y','fit_rx','fit_day','train_ids','val_ids'):
        path=input_path(reg,name,job)
        access(name,reg['source_input_whitelist'][name],'decode_source_metadata',job)
        pool[name]=np.load(path,allow_pickle=False)
    y,rx,day=(pool['fit_'+k] for k in ('y','rx','day'))
    if y.shape!=rx.shape or y.shape!=day.shape or y.shape!=(34668,) or set(day)!={1,2}:raise ValueError('Metadata scope changed')
    if {reg['rx_names'][int(r)] for r in np.unique(rx)}!=set(reg['fit_rx']):raise ValueError('Source RX scope changed')
    if not np.array_equal(np.sort(np.r_[pool['train_ids'],pool['val_ids']]),np.arange(len(y))):raise ValueError('Original packet split changed')
    pool['keys']=A.make_keys(y,rx,day,reg['rx_names'],reg['source_cell_sha256'])
    return pool


def materialize(reg,pool,ids,roles,job):
    if not set(roles)<={'fit_train','checkpoint_val','source_selection'}:raise ValueError('Unknown source role')
    if 'source_selection' in roles:
        if job!='sealed_selection_evaluation':raise ValueError('Selection reads only through sealed evaluation')
        committed(HERE/'TRAINING_SEAL.json')
    p=input_path(reg,'fit_x',job)
    access('fit_x',reg['source_input_whitelist']['fit_x'],'mmap_header_only',job)
    array=np.load(p,mmap_mode='r',allow_pickle=False)
    if array.shape!=(34668,2,256) or array.dtype!=np.float32:raise ValueError('IQ schema changed')
    result={}
    for role in roles:
        rows=ids[role]
        access('fit_x',reg['source_input_whitelist']['fit_x'],'copy_'+role+'_rows',job)
        result[role]=dict(x=np.array(array[rows],copy=True),y=pool['fit_y'][rows].astype(np.int64),
            rx=pool['fit_rx'][rows].astype(np.int64),day=pool['fit_day'][rows].astype(np.int64),
            keys=pool['keys'][rows],pool_ids=rows.copy())
    return result


def preflight(reg):
    if (HERE/'SOURCE_PREFLIGHT.json').exists():raise FileExistsError('Keep original preflight')
    pool=metadata(reg,'preflight');views={}
    for view in read(HERE/'SOURCE_PLAN.json')['views']:
        ids=C.role_ids(reg,view,pool)
        views[view['id']]={role:dict(rows=len(rows),key_sha256=semantic_hash(pool['keys'][rows])) for role,rows in ids.items()}
    value=dict(status='PASS_EXACT_C_SOURCE_ROLES',registration_sha256=sha(HERE/'REGISTRATION.json'),views=views,IQ_payload_loads=0,time=now())
    write(HERE/'SOURCE_PREFLIGHT.json',value)
    return value


def state(model,opt,scheduler,order,epoch,best,best_epoch,updates):
    return cpu_tree(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
        rng=A.rng_state(order),epoch=epoch,best=best,best_epoch=best_epoch,updates=updates))


def fit_loop(job,train,val,model,device,out,rx_names):
    """Same loop for synthetic acceptance and registered scientific fits."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name in ('epoch_states','orders'):(out/name).mkdir(exist_ok=True)
    torch.manual_seed(job['stochastic_seed']);np.random.seed(job['stochastic_seed']);random.seed(job['stochastic_seed'])
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=job['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.1,patience=5,min_lr=1e-6)
    order=torch.Generator(device=device).manual_seed(job['order_seed'])
    tensors={role:{k:torch.as_tensor(d[k],device=device,dtype=torch.float32 if k=='x' else torch.int64)
                   for k in ('x','y','rx')} for role,d in (('train',train),('val',val))}
    best,best_epoch,updates=math.inf,0,0
    initial=state(model,opt,scheduler,order,0,best,best_epoch,0)
    torch.save(initial,out/'initial.pt')
    write(out/'INITIAL.json',dict(model_sha256=semantic_hash(model.state_dict()),backbone_sha256=semantic_hash(M.common_state(model)),
        branch_common_sha256=semantic_hash({k:v for k,v in model.state_dict().items() if k.startswith(('span.reduce.','span.expand.','span.gate.'))}),
        rng_sha256=semantic_hash(initial['rng']),optimizer_sha256=semantic_hash(initial['optimizer']),scheduler_sha256=semantic_hash(initial['scheduler'])))
    np.savez_compressed(out/'SAMPLE_KEYS.npz',train_keys=train['keys'],val_keys=val['keys'],train_ids=train['pool_ids'],val_ids=val['pool_ids'])
    history=[];started=time.perf_counter();training_seconds=validation_seconds=0.
    for epoch in range(1,job['epochs']+1):
        t=time.perf_counter();model.train();lr=opt.param_groups[0]['lr']
        permutation=torch.randperm(len(train['y']),generator=order,device=device)
        positions=permutation.cpu().numpy()
        if not np.array_equal(np.sort(positions),np.arange(len(positions))):raise ValueError('Epoch coverage failed')
        np.save(out/'orders'/f'epoch_{epoch:02d}.npy',positions)
        loss_total,correct=0.,0
        for pos in permutation.split(job['batch']):
            logits=model(tensors['train']['x'][pos]);y=tensors['train']['y'][pos]
            loss=F.cross_entropy(logits,y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();updates+=1
            loss_total+=float(loss.detach())*len(pos);correct+=int((logits.argmax(1)==y).sum())
        if device.type=='cuda':torch.cuda.synchronize()
        training_seconds+=time.perf_counter()-t
        t=time.perf_counter();logits=C.predict_tensor(model,tensors['val']['x'],job['batch'])
        ce=C.rx_ce(torch.as_tensor(logits,device=device),tensors['val']['y'],tensors['val']['rx'])
        if not math.isfinite(ce):raise ValueError('Nonfinite validation loss')
        validation_seconds+=time.perf_counter()-t
        scheduler.step(ce)
        improved=ce<best
        if improved:best,best_epoch=ce,epoch
        endpoints=(['selected'] if improved else [])+(['epoch10'] if epoch==10 else [])+(['final40'] if epoch==job['epochs'] else [])
        for endpoint in endpoints:
            torch.save(dict(model=cpu_tree(model.state_dict()),epoch=epoch,checkpoint_val_rx_ce=ce),out/f'{endpoint}.pt')
            C.save_predictions(out/f'{endpoint}_val.npz',logits,val,rx_names)
        snapshot=state(model,opt,scheduler,order,epoch,best,best_epoch,updates)
        torch.save(snapshot,out/'epoch_states'/f'epoch_{epoch:02d}.pt')
        row=dict(epoch=epoch,updates=updates,train_ce=loss_total/len(train['y']),train_accuracy=correct/len(train['y']),
            val_rx_ce=ce,best_epoch=best_epoch,lr_used=lr,lr_next=opt.param_groups[0]['lr'],
            relative_order_sha256=semantic_hash(positions),canonical_order_sha256=semantic_hash(train['keys'][positions]),
            class_day_order_sha256=semantic_hash(np.stack((train['y'][positions],train['day'][positions]),1)),
            model_sha256=semantic_hash(snapshot['model']),state_sha256=semantic_hash(snapshot))
        history.append(row);write(out/'HISTORY.json',history)
        write(out/'LIVE.json',dict(time=now(),job=job['id'],elapsed_seconds=time.perf_counter()-started,**row))
        if epoch==1 or epoch%5==0:print(job['id'],f'epoch={epoch}',f'valCE={ce:.6f}',f'best={best_epoch}',flush=True)
    if updates!=job['max_gradient_updates']:raise ValueError('Gradient budget mismatch')
    replay={}
    # Only available auxiliary endpoints in short synthetic fixtures are checked.
    for endpoint in ('selected','final40')+ (('epoch10',) if job['epochs']>=10 else ()):
        ck=torch.load(out/f'{endpoint}.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model'])
        predicted=C.predict_tensor(model,tensors['val']['x'],job['batch'])
        with np.load(out/f'{endpoint}_val.npz',allow_pickle=False) as p:
            replay[endpoint]=np.array_equal(predicted,p['logits'])
        if not replay[endpoint]:raise ValueError('Checkpoint prediction replay failed')
    return dict(status='COMPLETE',job=job,epochs=len(history),updates=updates,best_epoch=best_epoch,best_val_rx_ce=best,
        active_parameters=sum(p.numel() for p in model.parameters()),val_inference_replay=replay,
        costs=dict(training_seconds=training_seconds,validation_seconds=validation_seconds,fit_loop_seconds=time.perf_counter()-started),
        checkpoints={endpoint:sha(out/f'{endpoint}.pt') for endpoint in ENDPOINTS if (out/f'{endpoint}.pt').exists()})


def train(reg,jobs,job):
    if (HERE/'TRAINING_SEAL.json').exists():raise ValueError('Training already sealed')
    committed(HERE/'SOURCE_PREFLIGHT.json')
    pf=read(HERE/'SOURCE_PREFLIGHT.json')
    if pf['registration_sha256']!=sha(HERE/'REGISTRATION.json') or pf['status']!='PASS_EXACT_C_SOURCE_ROLES':raise ValueError('Preflight mismatch')
    out=HERE/'artifacts'/job['id']
    if out.exists():raise FileExistsError('No overwrite/retry of scientific attempt')
    out.mkdir(parents=True)
    write(out/'ATTEMPT.json',dict(status='RUNNING',started_at=now(),job=job,registration_sha256=sha(HERE/'REGISTRATION.json')))
    t=time.perf_counter()
    try:
        A.setup(job['initialization_seed'])
        if torch.cuda.get_device_name()!=reg['environment']['gpu'] or torch.version.cuda!=reg['environment']['cuda'] or torch.backends.cudnn.version()!=reg['environment']['cudnn']:raise ValueError('CUDA environment mismatch')
        pool=metadata(reg,job['id']);view=next(v for v in read(HERE/'SOURCE_PLAN.json')['views'] if v['id']==job['view'])
        ids=C.role_ids(reg,view,pool)
        data=materialize(reg,pool,ids,('fit_train','checkpoint_val'),job['id'])
        model=M.build(job['arm'],job['initialization_seed']).cuda()
        if sum(p.numel() for p in model.parameters())!=reg['parameters'][job['arm']]:raise ValueError('Architecture parameter count mismatch')
        write(out/'CONFIG.json',dict(job=job,train_rows=len(data['fit_train']['y']),val_rows=len(data['checkpoint_val']['y']),
            train_keys_sha256=semantic_hash(data['fit_train']['keys']),val_keys_sha256=semantic_hash(data['checkpoint_val']['keys']),
            allowed_training_RX=view['fit_rx'],forbidden_training_RX=view['source_selection_rx'],source_selection_IQ_read=False))
        complete=fit_loop(job,data['fit_train'],data['checkpoint_val'],model,torch.device('cuda'),out,reg['rx_names'])
        complete.update(registration_sha256=sha(HERE/'REGISTRATION.json'),peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            total_wall_seconds=time.perf_counter()-t,finished_at=now())
        attempt=read(out/'ATTEMPT.json');attempt.update(status='COMPLETE',finished_at=now())
        write(out/'ATTEMPT.json',attempt)
        complete['files_sha256']={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
        write(out/'COMPLETE.json',complete)
        del model,data,pool
        torch.cuda.empty_cache()
        print('COMPLETE',job['id'],f'wall={complete["total_wall_seconds"]:.1f}s',flush=True)
        return complete
    except Exception:
        write(out/'FAILURE.json',dict(time=now(),status='FAILED',automatic_retry=False,traceback=traceback.format_exc()))
        raise


def verify_complete(job):
    out=HERE/'artifacts'/job['id'];value=read(out/'COMPLETE.json')
    if value['status']!='COMPLETE' or value['job']!=job or value['epochs']!=40 or value['updates']!=5080:raise ValueError('Incomplete scientific fit')
    if value['registration_sha256']!=sha(HERE/'REGISTRATION.json'):raise ValueError('Fit registration changed')
    for relative,digest in value['files_sha256'].items():
        p=(out/relative).resolve();p.relative_to(out.resolve())
        if sha(p)!=digest:raise ValueError('Completed fit artifact changed')
    return out,value


def seal(reg,jobs):
    if (HERE/'TRAINING_SEAL.json').exists():raise FileExistsError('Keep original training seal')
    evidence={};runs={};checks={}
    for job in jobs:
        out,c=verify_complete(job)
        evidence[job['id']]={n:read(out/(n+'.json')) for n in ('CONFIG','INITIAL','HISTORY')}
        evidence[job['id']]['COMPLETE']=c
        runs[job['id']]=dict(complete_sha256=sha(out/'COMPLETE.json'),checkpoints=c['checkpoints'])
    for view in ('V0','V1'):
        for stream in ('S0','S1','S2'):
            e=[evidence[j['id']] for j in jobs if j['view']==view and j['stream']==stream]
            prefix=view+'/'+stream+'/'
            for field in ('backbone_sha256','rng_sha256','scheduler_sha256'):
                checks[prefix+field]=len({a['INITIAL'][field] for a in e})==1
            span=[a for a in e if a['CONFIG']['job']['arm']!='P0']
            checks[prefix+'common_branch_layers']=len({a['INITIAL']['branch_common_sha256'] for a in span})==1
            p2,n19=[next(a for a in e if a['CONFIG']['job']['arm']==arm) for arm in ('P2','N19')]
            checks[prefix+'P2_N19_same_initial_weights']=p2['INITIAL']['model_sha256']==n19['INITIAL']['model_sha256']
            for field in ('train_keys_sha256','val_keys_sha256'):
                checks[prefix+field]=len({a['CONFIG'][field] for a in e})==1
            for field in ('relative_order_sha256','canonical_order_sha256','class_day_order_sha256'):
                checks[prefix+field]=all(len({a['HISTORY'][k][field] for a in e})==1 for k in range(40))
    checks['exact24_fixed_budgets']=len(evidence)==24 and all(a['COMPLETE']['updates']==5080 for a in evidence.values())
    if not all(checks.values()):raise ValueError('Paired design control failed')
    write(HERE/'RUN_EVIDENCE.json',evidence)
    result=dict(status='ALL24_FITS_SEALED',registration_sha256=sha(HERE/'REGISTRATION.json'),runs=runs,
        checks=checks,source_selection_scoring_started=False,time=now())
    write(HERE/'TRAINING_SEAL.json',result)
    return result


def evaluate(reg,jobs):
    committed(HERE/'TRAINING_SEAL.json');committed(HERE/'RUN_EVIDENCE.json')
    sealed=read(HERE/'TRAINING_SEAL.json')
    if sealed['status']!='ALL24_FITS_SEALED' or not all(sealed['checks'].values()):raise ValueError('Full committed training seal required')
    for j in jobs:
        out,_=verify_complete(j)
        if sha(out/'COMPLETE.json')!=sealed['runs'][j['id']]['complete_sha256']:raise ValueError('Seal mismatch')
    marker=HERE/'EVALUATION_ATTEMPT.json'
    if marker.exists():raise FileExistsError('Keep original evaluation attempt')
    write(marker,dict(status='STARTED',time=now(),seal_sha256=sha(HERE/'TRAINING_SEAL.json')))
    A.setup(24301);pool=metadata(reg,'sealed_selection_evaluation');files={}
    for view in read(HERE/'SOURCE_PLAN.json')['views']:
        ids=C.role_ids(reg,view,pool)
        data=materialize(reg,pool,ids,('source_selection',),'sealed_selection_evaluation')['source_selection']
        x=torch.as_tensor(data['x'],device='cuda')
        for job in [j for j in jobs if j['view']==view['id']]:
            out=HERE/'artifacts'/job['id'];model=M.build(job['arm'],job['initialization_seed']).cuda()
            for endpoint in ENDPOINTS:
                if sha(out/f'{endpoint}.pt')!=sealed['runs'][job['id']]['checkpoints'][endpoint]:raise ValueError('Sealed endpoint changed')
                ck=torch.load(out/f'{endpoint}.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
                before=semantic_hash(model.state_dict());logits=C.predict_tensor(model,x,128)
                if before!=semantic_hash(model.state_dict()):raise ValueError('Evaluation mutated model state')
                path=out/f'{endpoint}_selection.npz';C.save_predictions(path,logits,data,reg['rx_names'])
                files[path.relative_to(HERE/'artifacts').as_posix()]=sha(path)
            del model
        del data,x
    value=dict(status='ALL72_SEALED_SELECTION_PREDICTIONS',files=files,seal_sha256=sha(HERE/'TRAINING_SEAL.json'),time=now())
    write(HERE/'EVALUATION_MANIFEST.json',value)
    return value


def analyze(reg,jobs):
    manifest=read(HERE/'EVALUATION_MANIFEST.json')
    expected={f'{j["id"]}/{e}_selection.npz' for j in jobs for e in ENDPOINTS}
    if set(manifest['files'])!=expected:raise ValueError('Complete72 endpoint block required')
    for name,digest in manifest['files'].items():
        if sha(HERE/'artifacts'/name)!=digest:raise ValueError('Prediction changed')
    views={v['id']:v for v in read(HERE/'SOURCE_PLAN.json')['views']}
    draws=np.random.default_rng(24501).integers(0,12,size=(10000,12))
    endpoints={}
    for endpoint in ENDPOINTS:
        predictions={a:{} for a in M.ARMS}
        for job in jobs:
            with np.load(HERE/'artifacts'/job['id']/f'{endpoint}_selection.npz',allow_pickle=False) as f:
                data={k:f[k] for k in f.files}
            normalized=S.validate_source_predictions(data,views[job['view']],reg)
            predictions[job['arm']][(job['view'],job['stream'])]=normalized
        aggregates={a:S.aggregate_candidate({k:S._summary(v) for k,v in predictions[a].items()}) for a in M.ARMS}
        pairs={f'{right}-{left}':S.paired_comparison(predictions[left],predictions[right],draws)
            for left,right in (('P1','P2'),('P0','P2'),('N19','P2'),('P0','P1'),('P0','N19'))}
        endpoints[endpoint]=dict(aggregates=aggregates,comparisons=pairs)
    primary=endpoints['selected']['comparisons']
    gates={name:S.promotion_check(primary[name]) for name in ('P2-P1','P2-P0')}
    n=primary['P2-N19']
    specificity=dict(accuracy_positive=n['rx_equal']['accuracy_delta']>0,CE_improves=n['rx_equal']['ce_delta']<0,
        both_views_positive=all(r['accuracy_delta']>0 for r in n['per_view'].values()))
    promote=all(v['passes'] for v in gates.values()) and all(specificity.values())
    decision=dict(status='PERIODIC_SPAN_WORTH_SEPARATE_CONFIRMATION' if promote else 'STOP_NO_PERIODIC_SPAN_PROMOTION',
        primary_contrast='P2-P1',promotion_gates=gates,specificity_P2_minus_N19=specificity,
        no_additional_fits=True,fixed_window_STOP_unchanged=True,final_CRD_confirmed=False,
        interpretation='Observed source-CV development with reused source roles; not an independent RX or seed-population confirmation.')
    result=dict(stage='PERIODIC_SPAN',primary_endpoint='selected',auxiliary_endpoints=['epoch10','final40'],
        endpoints=endpoints,decision=decision,bootstrap=dict(replicates=10000,seed=24501,unit='12 observed RX clusters with 3 fixed streams averaged within RX'),
        evidence=dict(registration_sha256=sha(HERE/'REGISTRATION.json'),seal_sha256=sha(HERE/'TRAINING_SEAL.json'),evaluation_sha256=sha(HERE/'EVALUATION_MANIFEST.json')))
    with gzip.open(HERE/'ANALYSIS.json.gz','wt',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,allow_nan=False)
    write(HERE/'DECISION.json',decision)
    write(HERE/'SUMMARY.json',dict(stage=result['stage'],decision=decision,
        endpoints={e:dict(aggregates={a:{k:v for k,v in d.items() if k!='tx_rx_stream_mean'} for a,d in r['aggregates'].items()},
            comparisons={name:{k:v for k,v in d.items() if k not in ('cells','tx_rx_stream_mean')} for name,d in r['comparisons'].items()}) for e,r in endpoints.items()}))
    return decision


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['preflight','train-all','seal','evaluate','analyze'])
    args=parser.parse_args();reg,jobs=validate()
    if args.command=='preflight':r=preflight(reg)
    elif args.command=='train-all':
        for job in jobs:train(reg,jobs,job)
        r=seal(reg,jobs)
    elif args.command=='seal':r=seal(reg,jobs)
    elif args.command=='evaluate':r=evaluate(reg,jobs)
    else:r=analyze(reg,jobs)
    print(json.dumps(dict(command=args.command,status=r['status']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
