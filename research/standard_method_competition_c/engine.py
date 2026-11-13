"""Registered C execution. Only source fit18 rows; selection scoring after seal."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
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


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


A=load_module('C_frozen_A_helpers',HERE.parent/'controlled_training_a/runner.py')
M=load_module('C_frozen_method_kernels',HERE/'methods.py')
read,write,sha,now=A.read,A.write,A.file_sha,A.now
semantic_hash,cpu_tree=A.semantic_hash,A.cpu_tree
ENDPOINTS=('selected','epoch10','final40')


def require_committed(paths):
    relative=[Path(p).resolve().relative_to(ROOT).as_posix() for p in paths]
    subprocess.run(['git','-C',str(ROOT),'ls-files','--error-unmatch','--',*relative],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    subprocess.run(['git','-C',str(ROOT),'diff','--quiet','HEAD','--',*relative],check=True)


def validate_execution():
    reg=read(HERE/'PRE_REGISTRATION.json');ex=read(HERE/'EXECUTION_REGISTRATION.json')
    if ex['status']!='REGISTERED_FOR_FROZEN_24_JOB_EXECUTION' or ex['registration_sha256']!=sha(HERE/'PRE_REGISTRATION.json'):
        raise ValueError('Execution addendum differs from frozen C protocol')
    paths=[HERE/'PRE_REGISTRATION.json',HERE/'EXECUTION_REGISTRATION.json']
    for group in (reg['code_sha256'],ex['code_sha256'],reg['evidence_sha256']):
        for relative,digest in group.items():
            path=(ROOT/relative).resolve();path.relative_to(ROOT)
            if sha(path)!=digest:raise ValueError('Pinned code/evidence changed: '+relative)
            paths.append(path)
    for name,digest in reg['artifact_sha256'].items():
        path=HERE/name
        if sha(path)!=digest:raise ValueError('Frozen C artifact changed: '+name)
        paths.append(path)
    require_committed(paths)
    refs=read(HERE/'REFERENCES.json')['ema']
    if sha(Path(refs['actual_local_source']))!=refs['actual_local_source_sha256']:
        raise ValueError('Installed SWA/EMA source changed')
    if str(torch.__version__)!=ex['environment']['torch_version'] or np.__version__!=ex['environment']['numpy_version']:
        raise ValueError('Execution environment differs')
    jobs=read(HERE/'JOBS.json')
    if len(jobs)!=24 or sum(len(j['readouts']) for j in jobs)!=36:
        raise ValueError('Exactly24 gradient fits and36 readouts are allowed')
    return reg,ex,jobs


def access(item,action,job=None):
    with (HERE/'EXECUTION_ACCESS_LOG.jsonl').open('a',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(dict(time=now(),job=job,action=action,path=item['path'],sha256=item['sha256']))+'\n')


def verify_input(item,action,job):
    access(item,'hash_'+action,job)
    if sha(Path(item['path']))!=item['sha256']:raise ValueError('Registered source input changed')
    return Path(item['path'])


def metadata(reg,job):
    pool={}
    for name in ('fit_y','fit_rx','fit_day','train_ids','val_ids'):
        item=reg['source_input_whitelist'][name]
        path=verify_input(item,name,job);access(item,'load_'+name,job)
        pool[name]=np.load(path,allow_pickle=False)
    y,rx,day=[pool['fit_'+k] for k in ('y','rx','day')]
    if not (y.shape==rx.shape==day.shape==(34668,)) or set(day)!={1,2}:
        raise ValueError('Unexpected source metadata')
    if {reg['rx_names'][int(r)] for r in np.unique(rx)}!=set(reg['fit_rx']):
        raise ValueError('Source RX scope differs')
    if not np.array_equal(np.sort(np.r_[pool['train_ids'],pool['val_ids']]),np.arange(len(y))):
        raise ValueError('Original packet split differs')
    pool['keys']=A.make_keys(y,rx,day,reg['rx_names'],reg['source_cell_sha256'])
    return pool


def role_ids(reg,view,pool):
    rx=pool['fit_rx'];codes=[reg['rx_names'].index(r) for r in view['source_selection_rx']]
    ids=dict(fit_train=pool['train_ids'][~np.isin(rx[pool['train_ids']],codes)],
             checkpoint_val=pool['val_ids'][~np.isin(rx[pool['val_ids']],codes)],
             source_selection=np.flatnonzero(np.isin(rx,codes)).astype(np.int64))
    for role,rows in ids.items():
        expected=view[role]
        if rows.dtype.name!=expected['dtype'] or len(rows)!=expected['rows'] or hashlib.sha256(rows.tobytes()).hexdigest()!=expected['indices_array_sha256']:
            raise ValueError('Frozen role indices differ: '+role)
    if any(set(ids[a])&set(ids[b]) for a,b in (('fit_train','checkpoint_val'),('fit_train','source_selection'),('checkpoint_val','source_selection'))):
        raise ValueError('Role overlap')
    return ids


def materialize(reg,view,pool,ids,roles,job):
    if not set(roles)<=set(ids):raise ValueError('Unknown role')
    item=reg['source_input_whitelist']['fit_x']
    path=verify_input(item,'fit_x',job)
    access(item,'mmap_registered_fit_x_'+','.join(roles),job)
    x=np.load(path,mmap_mode='r',allow_pickle=False)
    if x.shape!=(34668,2,256) or x.dtype!=np.float32:raise ValueError('Registered IQ shape differs')
    result={}
    for role in roles:
        rows=ids[role]
        access(item,'copy_registered_role_rows_'+role,job)
        result[role]=dict(x=np.array(x[rows],copy=True),y=np.asarray(pool['fit_y'][rows],dtype=np.int64),
             rx=np.asarray(pool['fit_rx'][rows],dtype=np.int64),day=np.asarray(pool['fit_day'][rows],dtype=np.int64),
             keys=pool['keys'][rows],pool_ids=np.asarray(rows,dtype=np.int64))
    return result


def preflight(reg,ex):
    if (HERE/'SOURCE_PREFLIGHT.json').exists():raise FileExistsError('Keep original source preflight')
    pool=metadata(reg,'source_preflight');checks={}
    for view in read(HERE/'SOURCE_PLAN.json')['views']:
        ids=role_ids(reg,view,pool)
        checks[view['id']]={role:dict(rows=len(rows),keys_sha256=semantic_hash(pool['keys'][rows])) for role,rows in ids.items()}
    result=dict(status='PASS_EXACT_SOURCE_ROLES',registration_sha256=sha(HERE/'PRE_REGISTRATION.json'),
                execution_sha256=sha(HERE/'EXECUTION_REGISTRATION.json'),views=checks,IQ_payload_loads=0,scientific_fits=0,time=now())
    write(HERE/'SOURCE_PREFLIGHT.json',result)
    return result


def predict_tensor(model,x,batch):
    model.eval()
    with torch.no_grad():return torch.cat([model(v) for v in x.split(batch)]).cpu().numpy()


def rx_ce(logits,y,rx):
    return float(torch.stack([F.cross_entropy(logits[rx==r],y[rx==r]) for r in torch.unique(rx)]).mean())


def save_predictions(path,logits,data,names):
    np.savez_compressed(path,logits=logits,y=data['y'],rx=np.asarray([names[int(r)] for r in data['rx']]),
                        day=data['day'],keys=data['keys'],pool_ids=data['pool_ids'])


def full_state(model,opt,scheduler,order,shadow,dro,epoch,best,best_epochs,updates):
    return cpu_tree(dict(student=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
          rng=A.rng_state(order),ema=shadow.state_dict() if shadow is not None else None,
          group_dro=dro.state_dict() if dro is not None else None,epoch=epoch,best=best,best_epochs=best_epochs,updates=updates))


def fit_loop(job,train,val,model,device,out,rx_names):
    """The same scientific loop is exercised on synthetic fixtures before registration."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name in ('epoch_states','orders'): (out/name).mkdir(exist_ok=True)
    batch=job['batch'];arms=job['readouts'];family=job['family']
    if family not in ('ERM_EMA','RX_GDRO'):raise ValueError('Unregistered family')
    if arms!=(['C0','C1'] if family=='ERM_EMA' else ['C2']):raise ValueError('Readout mismatch')
    torch.manual_seed(job['stochastic_seed']);np.random.seed(job['stochastic_seed']);random.seed(job['stochastic_seed'])
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=job['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.1,patience=5,min_lr=1e-6)
    order=torch.Generator(device=device).manual_seed(job['order_seed'])
    shadow=M.make_ema(model) if family=='ERM_EMA' else None
    codes=sorted(map(int,np.unique(train['rx'])));group_names=[rx_names[r] for r in codes]
    group_lookup={r:i for i,r in enumerate(codes)}
    groups=torch.as_tensor([group_lookup[int(r)] for r in train['rx']],device=device,dtype=torch.int64)
    dro=M.RxGroupDRO(len(codes),device=device) if family=='RX_GDRO' else None
    tensors={role:{k:torch.as_tensor(d[k],device=device) for k in ('x','y','rx')} for role,d in (('train',train),('val',val))}
    best={arm:math.inf for arm in arms};best_epochs={arm:0 for arm in arms};history=[];updates=0
    initial=full_state(model,opt,scheduler,order,shadow,dro,0,best,best_epochs,0)
    torch.save(initial,out/'initial.pt')
    write(out/'INITIAL.json',dict(model_sha256=semantic_hash(initial['student']),optimizer_sha256=semantic_hash(initial['optimizer']),
          scheduler_sha256=semantic_hash(initial['scheduler']),rng_sha256=semantic_hash(initial['rng']),
          bn_buffers_sha256=semantic_hash({k:v for k,v in initial['student'].items() if k.endswith(('running_mean','running_var','num_batches_tracked'))})))
    np.savez_compressed(out/'SAMPLE_KEYS.npz',train_keys=train['keys'],val_keys=val['keys'],train_ids=train['pool_ids'],val_ids=val['pool_ids'])
    costs=dict(training_with_ema_seconds=0.,raw_validation_seconds=0.,ema_update_host_seconds=0.,ema_update_cuda_seconds=0.,ema_calibration_seconds=0.,ema_validation_seconds=0.,
               BN_calibration_samples=0,BN_calibration_passes=0,EMA_updates=0)
    group_trace={name:[] for name in ('q','loss','count')};started=time.perf_counter()
    for epoch in range(1,job['epochs']+1):
        t=time.perf_counter();model.train();used_lr=opt.param_groups[0]['lr']
        permutation=torch.randperm(len(train['y']),generator=order,device=device)
        positions=permutation.cpu().numpy()
        if not np.array_equal(np.sort(positions),np.arange(len(positions))):raise ValueError('Epoch coverage failed')
        np.save(out/'orders'/f'epoch_{epoch:02d}.npy',positions)
        loss_total,objective_total,correct,seen=0.,0.,0,0
        ema_seconds=0.;ema_events=[]
        for pos in permutation.split(batch):
            x=tensors['train']['x'][pos];y=tensors['train']['y'][pos]
            logits=model(x);per_ce=F.cross_entropy(logits,y,reduction='none')
            if not torch.isfinite(per_ce).all():raise ValueError('Nonfinite source loss')
            if dro is None:objective=per_ce.mean()
            else:
                objective,record=dro.loss(per_ce,groups[pos])
                for name,key in (('q','q'),('loss','group_loss'),('count','group_count')):group_trace[name].append(record[key].cpu().numpy())
            opt.zero_grad(set_to_none=True);objective.backward();opt.step();updates+=1
            if shadow is not None:
                if device.type=='cuda':
                    e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);e0.record()
                t_ema=time.perf_counter();shadow.update_parameters(model)
                if device.type=='cuda':e1.record();ema_events.append((e0,e1))
                ema_seconds+=time.perf_counter()-t_ema;costs['EMA_updates']+=1
            loss_total+=float(per_ce.detach().sum());objective_total+=float(objective.detach())*len(pos)
            correct+=int((logits.argmax(1)==y).sum());seen+=len(pos)
        if device.type=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-t;costs['training_with_ema_seconds']+=elapsed;costs['ema_update_host_seconds']+=ema_seconds
        costs['ema_update_cuda_seconds']+=sum(a.elapsed_time(b)/1000 for a,b in ema_events)
        t=time.perf_counter();model.eval()
        with torch.no_grad():lv=torch.cat([model(x) for x in tensors['val']['x'].split(batch)])
        ce=rx_ce(lv,tensors['val']['y'],tensors['val']['rx']);costs['raw_validation_seconds']+=time.perf_counter()-t
        values={arms[0]:(model,lv.cpu().numpy(),ce)}
        calibration_proof=True
        if shadow is not None:
            before=semantic_hash(A.rng_state(order));student_before=semantic_hash(model.state_dict());shadow_before=semantic_hash(shadow.state_dict())
            t=time.perf_counter()
            candidate,calibration=M.calibrated_ema_candidate(shadow,tensors['train']['x'].split(batch),role='fit_train')
            if device.type=='cuda':torch.cuda.synchronize()
            costs['ema_calibration_seconds']+=time.perf_counter()-t
            costs['BN_calibration_samples']+=calibration['samples'];costs['BN_calibration_passes']+=1
            calibration_proof=(before==semantic_hash(A.rng_state(order)) and student_before==semantic_hash(model.state_dict()) and shadow_before==semantic_hash(shadow.state_dict()))
            if not calibration_proof:raise ValueError('BN calibration changed training state')
            t=time.perf_counter();ema_logits=predict_tensor(candidate,tensors['val']['x'],batch)
            ema_ce=rx_ce(torch.as_tensor(ema_logits,device=device),tensors['val']['y'],tensors['val']['rx'])
            costs['ema_validation_seconds']+=time.perf_counter()-t
            values['C1']=(candidate,ema_logits,ema_ce)
        scheduler.step(ce)
        for arm,(readout,logits,value) in values.items():
            if not math.isfinite(value):raise ValueError('Nonfinite source validation')
            improved=value<best[arm]
            if improved:best[arm],best_epochs[arm]=value,epoch
            endpoint_names=(['selected'] if improved else [])+(['epoch10'] if epoch==10 else [])+(['final40'] if epoch==job['epochs'] else [])
            for endpoint in endpoint_names:
                torch.save(dict(model=cpu_tree(readout.state_dict()),arm=arm,epoch=epoch,checkpoint_val_rx_ce=value),out/f'{arm}_{endpoint}.pt')
                save_predictions(out/f'{arm}_{endpoint}_val.npz',logits,val,rx_names)
        state=full_state(model,opt,scheduler,order,shadow,dro,epoch,best,best_epochs,updates)
        torch.save(state,out/'epoch_states'/f'epoch_{epoch:02d}.pt')
        row=dict(epoch=epoch,updates=updates,train_ce=loss_total/seen,train_objective=objective_total/seen,train_accuracy=correct/seen,
             val_rx_ce={arm:values[arm][2] for arm in arms},best_epochs=dict(best_epochs),lr_used=used_lr,lr_next=opt.param_groups[0]['lr'],
             relative_order_sha256=semantic_hash(positions),canonical_order_sha256=semantic_hash(train['keys'][positions]),
             class_day_order_sha256=semantic_hash(np.stack((train['y'][positions],train['day'][positions]),1)),
             student_sha256=semantic_hash(state['student']),optimizer_sha256=semantic_hash(state['optimizer']),scheduler_sha256=semantic_hash(state['scheduler']),
             rng_sha256=semantic_hash(state['rng']),state_sha256=semantic_hash(state),calibration_preserved_training_state=calibration_proof)
        if dro is not None:row.update(q_min=float(dro.log_q.exp().min()),q_max=float(dro.log_q.exp().max()))
        history.append(row);write(out/'HISTORY.json',history)
        write(out/'LIVE.json',dict(time=now(),job=job['id'],seconds=time.perf_counter()-started,**row))
        print(job['id'],f'epoch {epoch:02d}',{arm:round(value,6) for arm,(_,_,value) in values.items()},'lr',used_lr,flush=True)
        if shadow is not None:del candidate
    if updates!=job['max_gradient_updates']:raise ValueError('Fixed gradient budget failed')
    if group_trace['q']:
        np.savez_compressed(out/'GROUP_TRACE.npz',**{name:np.asarray(values) for name,values in group_trace.items()},rx_names=np.asarray(group_names))
    # Deployment readouts must reproduce their own stored checkpoint-val logits.
    inference={}
    for arm in arms:
        for endpoint in ENDPOINTS:
            state=torch.load(out/f'{arm}_{endpoint}.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(state['model']);logits=predict_tensor(model,tensors['val']['x'],batch)
            with np.load(out/f'{arm}_{endpoint}_val.npz',allow_pickle=False) as z:check=A.compare_objects(logits,z['logits'])
            if not check['exact']:raise ValueError('Readout inference replay failed')
            inference[f'{arm}_{endpoint}']=check
    costs['total_fit_loop_seconds']=time.perf_counter()-started
    return dict(status='COMPLETE',job=job,epochs=len(history),updates=updates,best_epochs=best_epochs,best_val_rx_ce=best,
          active_parameters=sum(p.numel() for p in model.parameters()),costs=costs,val_inference_replay=inference,
          group_names=group_names,checkpoints={f'{arm}_{ep}.pt':sha(out/f'{arm}_{ep}.pt') for arm in arms for ep in ENDPOINTS})


def train(reg,ex,jobs,job_id):
    if (HERE/'STAGE_C_SEAL.json').exists():raise ValueError('The C training block is already sealed')
    require_committed([HERE/'SOURCE_PREFLIGHT.json'])
    pf=read(HERE/'SOURCE_PREFLIGHT.json')
    if pf['status']!='PASS_EXACT_SOURCE_ROLES' or pf['execution_sha256']!=sha(HERE/'EXECUTION_REGISTRATION.json'):
        raise ValueError('Source preflight is not admitted')
    job=next((j for j in jobs if j['id']==job_id),None)
    if job is None:raise ValueError('Unregistered job')
    out=HERE/'artifacts'/job_id
    if out.exists():raise FileExistsError('Scientific attempt exists; no overwrite or automatic retry')
    out.mkdir(parents=True)
    write(out/'ATTEMPT.json',dict(time=now(),status='RUNNING',job=job,registration_sha256=sha(HERE/'PRE_REGISTRATION.json'),execution_sha256=sha(HERE/'EXECUTION_REGISTRATION.json')))
    started=time.perf_counter()
    try:
        A.setup(job['initialization_seed'])
        if torch.cuda.get_device_name()!=ex['environment']['device_name'] or torch.version.cuda!=ex['environment']['cuda_version'] or torch.backends.cudnn.version()!=ex['environment']['cudnn_version']:
            raise ValueError('CUDA environment mismatch')
        view=next(v for v in read(HERE/'SOURCE_PLAN.json')['views'] if v['id']==job['view'])
        pool=metadata(reg,job_id);ids=role_ids(reg,view,pool)
        # Only the two permitted training-time roles are materialized.
        t=time.perf_counter();data=materialize(reg,view,pool,ids,('fit_train','checkpoint_val'),job_id)
        input_seconds=time.perf_counter()-t
        model=A.build_model(job['initialization_seed']);device=torch.device('cuda',torch.cuda.current_device())
        if sum(p.numel() for p in model.parameters())!=82426:raise ValueError('Frozen backbone parameter count differs')
        write(out/'CONFIG.json',dict(job=job,view=view['id'],registration_sha256=sha(HERE/'PRE_REGISTRATION.json'),
              execution_sha256=sha(HERE/'EXECUTION_REGISTRATION.json'),environment=ex['environment'],
              train_rows=len(data['fit_train']['y']),checkpoint_val_rows=len(data['checkpoint_val']['y']),
              train_keys_sha256=semantic_hash(data['fit_train']['keys']),val_keys_sha256=semantic_hash(data['checkpoint_val']['keys']),
              allowed_training_RX=view['fit_rx'],forbidden_training_RX=view['source_selection_rx'],source_selection_IQ_read=False))
        complete=fit_loop(job,data['fit_train'],data['checkpoint_val'],model,device,out,reg['rx_names'])
        complete.update(registration_sha256=sha(HERE/'PRE_REGISTRATION.json'),execution_sha256=sha(HERE/'EXECUTION_REGISTRATION.json'),
              source_input_load_seconds=input_seconds,peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        attempt=read(out/'ATTEMPT.json');attempt.update(status='COMPLETE',finished_at=now());write(out/'ATTEMPT.json',attempt)
        complete['files_sha256']={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
        complete.update(wall_seconds_since_attempt=time.perf_counter()-started,finished_at=now())
        write(out/'COMPLETE.json',complete)
        return complete
    except Exception:
        write(out/'FAILURE.json',dict(time=now(),status='FAILED',automatic_retry=False,traceback=traceback.format_exc()))
        raise


def verify_complete(job):
    out=HERE/'artifacts'/job['id'];complete=read(out/'COMPLETE.json')
    if complete['status']!='COMPLETE' or complete['job']!=job or complete['updates']!=5080 or complete['epochs']!=40:
        raise ValueError('Incomplete or mismatched frozen C fit')
    if complete['registration_sha256']!=sha(HERE/'PRE_REGISTRATION.json') or complete['execution_sha256']!=sha(HERE/'EXECUTION_REGISTRATION.json'):
        raise ValueError('Fit registration hash differs')
    for relative,digest in complete['files_sha256'].items():
        path=(out/relative).resolve();path.relative_to(out.resolve())
        if sha(path)!=digest:raise ValueError('Completed artifact changed: '+str(path))
    return out,complete


def seal(reg,ex,jobs):
    if (HERE/'STAGE_C_SEAL.json').exists():raise FileExistsError('Keep original C seal')
    evidence={};runs={};checks={}
    for job in jobs:
        out,c=verify_complete(job)
        evidence[job['id']]=dict(CONFIG=read(out/'CONFIG.json'),INITIAL=read(out/'INITIAL.json'),HISTORY=read(out/'HISTORY.json'),COMPLETE=c)
        runs[job['id']]=dict(complete_sha256=sha(out/'COMPLETE.json'),checkpoints=c['checkpoints'])
    buckets={}
    for job in jobs:buckets.setdefault((job['view'],job['stream'],job['weight_decay']),[]).append(job['id'])
    for key,ids in buckets.items():
        left,right=[evidence[i] for i in ids];prefix='|'.join(map(str,key))
        for field in ('model_sha256','bn_buffers_sha256','optimizer_sha256','scheduler_sha256','rng_sha256'):
            checks[prefix+'|initial|'+field]=left['INITIAL'][field]==right['INITIAL'][field]
        for field in ('train_keys_sha256','val_keys_sha256'):
            checks[prefix+'|'+field]=left['CONFIG'][field]==right['CONFIG'][field]
        for field in ('relative_order_sha256','canonical_order_sha256','class_day_order_sha256'):
            checks[prefix+'|'+field]=all(left['HISTORY'][i][field]==right['HISTORY'][i][field] for i in range(40))
    checks['all_EMA_calibrations_preserved_training_state']=all(row['calibration_preserved_training_state'] for e in evidence.values() for row in e['HISTORY'])
    checks['all24_exact_budget']=len(evidence)==24 and all(e['COMPLETE']['updates']==5080 for e in evidence.values())
    if not all(checks.values()):raise RuntimeError('C pairing or fit budget controls failed')
    write(HERE/'RUN_EVIDENCE.json',evidence)
    result=dict(status='SEALED_STAGE_C',registration_sha256=sha(HERE/'PRE_REGISTRATION.json'),execution_sha256=sha(HERE/'EXECUTION_REGISTRATION.json'),
                runs=runs,pairing_checks=checks,source_selection_scoring_started=False,time=now())
    write(HERE/'STAGE_C_SEAL.json',result)
    return result


def validate_seal(jobs):
    require_committed([HERE/'STAGE_C_SEAL.json',HERE/'RUN_EVIDENCE.json'])
    sealed=read(HERE/'STAGE_C_SEAL.json')
    if sealed['status']!='SEALED_STAGE_C' or set(sealed['runs'])!={j['id'] for j in jobs} or not all(sealed['pairing_checks'].values()):
        raise ValueError('Full committed C seal required before selection scoring')
    for job in jobs:
        out,c=verify_complete(job)
        if sha(out/'COMPLETE.json')!=sealed['runs'][job['id']]['complete_sha256']:raise ValueError('Sealed completion changed')
    return sealed


def evaluate(reg,ex,jobs):
    sealed=validate_seal(jobs)
    if (HERE/'SELECTION_EVALUATION_ATTEMPT.json').exists():raise FileExistsError('Selection evaluation was already attempted')
    write(HERE/'SELECTION_EVALUATION_ATTEMPT.json',dict(status='RUNNING',time=now(),seal_sha256=sha(HERE/'STAGE_C_SEAL.json')))
    A.setup(24301);pool=metadata(reg,'selection_evaluation');files={}
    for view in read(HERE/'SOURCE_PLAN.json')['views']:
        ids=role_ids(reg,view,pool)
        data=materialize(reg,view,pool,ids,('source_selection',),'selection_evaluation')[ 'source_selection']
        x=torch.as_tensor(data['x'],device='cuda')
        for job in [j for j in jobs if j['view']==view['id']]:
            out=HERE/'artifacts'/job['id'];model=A.build_model(job['initialization_seed'])
            for arm in job['readouts']:
                for endpoint in ENDPOINTS:
                    name=f'{arm}_{endpoint}.pt'
                    if sha(out/name)!=sealed['runs'][job['id']]['checkpoints'][name]:raise ValueError('Sealed readout changed')
                    ck=torch.load(out/name,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
                    before=semantic_hash(model.state_dict());logits=predict_tensor(model,x,128)
                    if before!=semantic_hash(model.state_dict()):raise ValueError('Inference altered frozen model')
                    path=out/f'{arm}_{endpoint}_selection.npz';save_predictions(path,logits,data,reg['rx_names'])
                    files[path.relative_to(HERE/'artifacts').as_posix()]=sha(path)
            del model
        del x
    if len(files)!=108:raise ValueError('Must score all108 frozen candidates/endpoints')
    manifest=dict(status='COMPLETE',registration_sha256=sha(HERE/'PRE_REGISTRATION.json'),execution_sha256=sha(HERE/'EXECUTION_REGISTRATION.json'),
                  seal_sha256=sha(HERE/'STAGE_C_SEAL.json'),files=files,source_only=True,time=now())
    write(HERE/'EVALUATION_MANIFEST.json',manifest)
    return manifest


def benchmark(reg,ex,jobs):
    validate_seal(jobs)
    if (HERE/'LATENCY_ATTEMPT.json').exists():raise FileExistsError('Latency attempt already exists')
    write(HERE/'LATENCY_ATTEMPT.json',dict(time=now(),status='RUNNING'))
    A.setup(24301);pool=metadata(reg,'latency_training_inputs_only');records={}
    for view in read(HERE/'SOURCE_PLAN.json')['views']:
        ids=role_ids(reg,view,pool)
        data=materialize(reg,view,pool,ids,('fit_train',),'latency_training_inputs_only')['fit_train']
        x=torch.as_tensor(data['x'][:128],device='cuda')
        for stream in ('S0','S1','S2'):
            for wd in (0.,.001):
                group=[]
                for job in jobs:
                    if job['view']==view['id'] and job['stream']==stream and job['weight_decay']==wd:
                        for arm in job['readouts']:
                            model=A.build_model(job['initialization_seed'])
                            ck=torch.load(HERE/'artifacts'/job['id']/f'{arm}_selected.pt',map_location='cpu',weights_only=False)
                            model.load_state_dict(ck['model']);model.eval();group.append((job,arm,model))
                group.sort(key=lambda item:item[1])
                for round_index in range(3):
                    rotated=group[round_index:]+group[:round_index]
                    for job,arm,model in rotated:
                        key=job['id']+'/'+arm;record=records.setdefault(key,dict(parameters=sum(p.numel() for p in model.parameters()),batches={}))
                        for batch,count in ((1,1000),(128,200)):
                            with torch.no_grad():
                                for _ in range(100):model(x[:batch])
                                torch.cuda.synchronize();events=[]
                                for _ in range(count):
                                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                                    a.record();model(x[:batch]);b.record();events.append((a,b))
                                torch.cuda.synchronize()
                            times=np.asarray([a.elapsed_time(b) for a,b in events])
                            record['batches'].setdefault(str(batch),[]).append(dict(round=round_index,p50_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),mean_ms=float(times.mean()),throughput_samples_per_second=float(batch*1000/times.mean()),measurements=count))
                for _,_,model in group:del model
                del group
        del x
    result=dict(status='COMPLETE',scope='selected deployment readouts; CUDA-resident input; whole model RF frontend included; no host copy/disk I/O',
                environment=ex['environment'],warmup_per_batch_round=100,rounds=3,readouts=records,time=now())
    write(HERE/'LATENCY.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    tr=sub.add_parser('train');tr.add_argument('--job',required=True)
    for name in ('preflight','seal','evaluate','benchmark'):sub.add_parser(name)
    args=p.parse_args();reg,ex,jobs=validate_execution()
    if args.command=='train':result=train(reg,ex,jobs,args.job)
    elif args.command=='preflight':result=preflight(reg,ex)
    elif args.command=='seal':result=seal(reg,ex,jobs)
    elif args.command=='evaluate':result=evaluate(reg,ex,jobs)
    else:result=benchmark(reg,ex,jobs)
    print(json.dumps(dict(status=result['status'],command=args.command)))


if __name__=='__main__':main()
