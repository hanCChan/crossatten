"""Authorization-gated12-fit execution; no source IQ is available in engineering mode."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time
import traceback

import numpy as np
import torch

import common as H
import kernel as K

HERE,ROOT,DESIGN=H.HERE,H.ROOT,H.DESIGN


def scientific_context():
    authorization,execution=H.scientific_gate()
    design=H.validate_design()
    jobs=H.validate_budget(H.read(DESIGN/'PROPOSED_JOBS.json'))
    source=H.read(ROOT/'research/periodic_span/REGISTRATION.json')
    if H.sha(HERE/'REFERENCE_RECOVERY.json')!=execution['artifact_sha256']['REFERENCE_RECOVERY.json']:raise ValueError('Reference recovery evidence changed')
    return authorization,execution,design,jobs,source


def read_metadata(source):
    H.scientific_gate()
    pool={}
    for name in ('fit_y','fit_rx','fit_day','train_ids','val_ids'):
        item=source['source_input_whitelist'][name];path=Path(item['path'])
        H.log_access(path,'hash_and_decode_registered_source_metadata',science=True)
        if H.sha(path)!=item['sha256']:raise ValueError('Source metadata changed')
        pool[name]=np.load(path,allow_pickle=False)
    y,rx,day=(pool['fit_'+k] for k in ('y','rx','day'))
    if not (y.shape==rx.shape==day.shape==(34668,)) or set(day)!={1,2}:raise ValueError('Wrong source metadata scope')
    if {source['rx_names'][int(r)] for r in np.unique(rx)}!=set(source['fit_rx']):raise ValueError('Source RX changed')
    if not np.array_equal(np.sort(np.r_[pool['train_ids'],pool['val_ids']]),np.arange(len(y))):raise ValueError('Original packet split changed')
    pool['keys']=H.A.make_keys(y,rx,day,source['rx_names'],source['source_cell_sha256'])
    return pool


def require_full_seal(jobs):
    H.scientific_gate()
    H.committed(HERE/'TRAINING_SEAL.json');H.committed(HERE/'RUN_EVIDENCE.json')
    seal=H.read(HERE/'TRAINING_SEAL.json')
    if seal['status']!='ALL12_FROZEN_BACKBONE_FITS_SEALED' or set(seal['runs'])!={j['id'] for j in jobs} or not all(seal['checks'].values()):raise ValueError('All12 committed fits are required')
    for job in jobs:
        out,_=verify_complete(job)
        if H.sha(out/'COMPLETE.json')!=seal['runs'][job['id']]['complete_sha256']:raise ValueError('Completed scientific fit changed')
    return seal


def materialize(source,pool,ids,roles,job):
    H.scientific_gate()
    if not set(roles)<={'fit_train','checkpoint_val','source_selection'}:raise ValueError('Unknown role')
    if 'source_selection' in roles:
        if job!='sealed_selection_evaluation':raise PermissionError('Selection IQ only through all12 sealed evaluation')
        require_full_seal(H.read(DESIGN/'PROPOSED_JOBS.json'))
    item=source['source_input_whitelist']['fit_x'];path=Path(item['path'])
    H.log_access(path,'hash_source_IQ_file_bytes',job,science=True)
    if H.sha(path)!=item['sha256']:raise ValueError('Registered source IQ changed')
    array=np.load(path,mmap_mode='r',allow_pickle=False)
    if array.shape!=(34668,2,256) or array.dtype!=np.float32:raise ValueError('Wrong source IQ shape/dtype')
    result={}
    for role in roles:
        rows=ids[role];H.log_access(path,'copy_'+role+'_rows',job,science=True)
        result[role]=dict(x=np.array(array[rows],copy=True),y=pool['fit_y'][rows].astype(np.int64),
            rx=pool['fit_rx'][rows].astype(np.int64),day=pool['fit_day'][rows].astype(np.int64),keys=pool['keys'][rows],pool_ids=rows.copy())
    return result


def load_reference(role,view,stream,device,arm='F0',side_seed=None):
    H.scientific_gate()
    record=H.read(HERE/'REFERENCE_INPUTS.json')['references'][view+'/'+stream][role]
    path=HERE/record['copy_relative'];H.log_access(path,'load_registered_frozen_source_model',view+'/'+stream+'/'+role,science=True)
    if H.sha(path)!=record['sha256']:raise ValueError('Recovered reference changed')
    ck=torch.load(path,map_location='cpu',weights_only=False)
    return H.M.build(arm,24101+int(stream[-1]),side_seed,backbone_state=ck['model']).to(device).eval()


def archived_reference_predictions(role,view,stream,which):
    H.scientific_gate()
    if which not in ('checkpoint_val','source_selection'):raise ValueError('Unknown reference role')
    if which=='source_selection':require_full_seal(H.read(DESIGN/'PROPOSED_JOBS.json'))
    record=H.read(HERE/'REFERENCE_INPUTS.json')['references'][view+'/'+stream][role][which+'_prediction']
    path=Path(record['path']);H.log_access(path,'decode_sealed_reference_'+which,view+'/'+stream+'/'+role,science=True)
    if H.sha(path)!=record['sha256']:raise ValueError('Archived reference prediction changed')
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def parity(logits,data,archived,names):
    expected_rx=np.asarray([names[int(r)] for r in data['rx']])
    for key,expected in (('keys',data['keys']),('pool_ids',data['pool_ids']),('y',data['y']),('day',data['day']),('rx',expected_rx)):
        if not np.array_equal(archived[key],expected):raise ValueError('Reference canonical role mismatch: '+key)
    if logits.shape!=archived['logits'].shape or not np.isfinite(logits).all() or not np.isfinite(archived['logits']).all():raise ValueError('Invalid replay logits')
    delta=float(np.max(np.abs(logits-archived['logits'])))
    same=np.array_equal(logits.argmax(1),archived['logits'].argmax(1))
    if delta>1e-6 or not same:raise ValueError(f'Reference parity failed:max_abs={delta},argmax_identical={same}')
    return dict(max_abs_logit_difference=delta,identical_argmax=same,rows=len(logits))


def preflight():
    authorization,ex,_,_,source=scientific_context()
    if (HERE/'SOURCE_PREFLIGHT.json').exists() or (HERE/'SOURCE_PREFLIGHT_ATTEMPT.json').exists():raise FileExistsError('Keep original source preflight attempt; no automatic retry')
    if torch.cuda.get_device_name()!=ex['environment']['gpu'] or torch.version.cuda!=ex['environment']['cuda'] or torch.backends.cudnn.version()!=ex['environment']['cudnn']:raise ValueError('Accepted CUDA environment changed')
    H.write(HERE/'SOURCE_PREFLIGHT_ATTEMPT.json',dict(status='STARTED',time=H.now(),authorization_sha256=H.sha(HERE/'EXECUTION_AUTHORIZATION.json')))
    H.A.setup(26301);pool=read_metadata(source);checks={}
    for view in H.read(DESIGN/'FUTURE_SOURCE_PLAN.json')['views']:
        ids=H.C.role_ids(source,view,pool)
        data=materialize(source,pool,ids,('checkpoint_val',),'authorized_source_preflight')['checkpoint_val']
        x=torch.as_tensor(data['x'],device='cuda')
        checks[view['id']]={}
        for stream in ('S0','S1','S2'):
            for role in ('F0','E0'):
                model=load_reference(role,view['id'],stream,torch.device('cuda'))
                before=H.semantic_hash(model.state_dict());logits=H.C.predict_tensor(model,x,128)
                proof=parity(logits,data,archived_reference_predictions(role,view['id'],stream,'checkpoint_val'),source['rx_names'])
                if before!=H.semantic_hash(model.state_dict()):raise ValueError('Parity inference changed frozen state')
                if role=='F0':
                    proof['alpha0_branch_bitwise']={}
                    for arm in ('F1','F2'):
                        branch=load_reference('F0',view['id'],stream,torch.device('cuda'),arm,26101+int(stream[-1]))
                        branch_before=H.semantic_hash(branch.state_dict())
                        branch_logits=H.C.predict_tensor(branch,x,128)
                        exact=np.array_equal(branch_logits,logits)
                        if not exact or H.semantic_hash(branch.state_dict())!=branch_before:raise ValueError('Real checkpoint_val alpha0 branch/F0 parity failed')
                        proof['alpha0_branch_bitwise'][arm]=exact
                        del branch
                checks[view['id']][stream+'/'+role]=proof
                del model
        del data,x
    value=dict(status='PASS_AUTHORIZED_SOURCE_CHECKPOINT_VAL_PARITY',checks=checks,design_sha256=H.sha(DESIGN/'REGISTRATION.json'),
        execution_sha256=H.sha(HERE/'EXECUTION_REGISTRATION.json'),authorization_sha256=H.sha(HERE/'EXECUTION_AUTHORIZATION.json'),
        source_selection_IQ_reads=0,scientific_gradient_fits=0,time=H.now())
    H.write(HERE/'SOURCE_PREFLIGHT.json',value);return value


def fit(job):
    _,ex,_,_,source=scientific_context()
    if (HERE/'TRAINING_SEAL.json').exists():raise ValueError('Scientific fits already sealed')
    H.committed(HERE/'SOURCE_PREFLIGHT.json')
    pf=H.read(HERE/'SOURCE_PREFLIGHT.json')
    if pf['status']!='PASS_AUTHORIZED_SOURCE_CHECKPOINT_VAL_PARITY' or pf['authorization_sha256']!=H.sha(HERE/'EXECUTION_AUTHORIZATION.json') or pf['execution_sha256']!=H.sha(HERE/'EXECUTION_REGISTRATION.json'):raise ValueError('Source preflight does not match authorization/execution')
    out=HERE/'artifacts/scientific'/job['id']
    if out.exists():raise FileExistsError('Scientific attempt exists; retries/resume are not authorized')
    out.mkdir(parents=True)
    H.write(out/'ATTEMPT.json',dict(status='RUNNING',time=H.now(),job=job,authorization_sha256=H.sha(HERE/'EXECUTION_AUTHORIZATION.json')))
    started=time.perf_counter()
    model=None
    try:
        H.A.setup(job['side_initialization_seed'])
        if torch.cuda.get_device_name()!=ex['environment']['gpu'] or torch.version.cuda!=ex['environment']['cuda'] or torch.backends.cudnn.version()!=ex['environment']['cudnn']:raise ValueError('Accepted CUDA environment changed')
        pool=read_metadata(source);view=next(v for v in H.read(DESIGN/'FUTURE_SOURCE_PLAN.json')['views'] if v['id']==job['view'])
        ids=H.C.role_ids(source,view,pool)
        data=materialize(source,pool,ids,('fit_train','checkpoint_val'),job['id'])
        model=load_reference('F0',job['view'],job['stream'],torch.device('cuda'),job['arm'],job['side_initialization_seed'])
        H.write(out/'CONFIG.json',dict(job=job,train_rows=len(data['fit_train']['y']),val_rows=len(data['checkpoint_val']['y']),
            train_keys_sha256=H.semantic_hash(data['fit_train']['keys']),val_keys_sha256=H.semantic_hash(data['checkpoint_val']['keys']),
            allowed_training_RX=view['fit_rx'],forbidden_training_RX=view['source_selection_rx'],source_selection_IQ_read=False,
            design_sha256=H.sha(DESIGN/'REGISTRATION.json'),execution_sha256=H.sha(HERE/'EXECUTION_REGISTRATION.json')))
        complete=K.fit_loop(job,data['fit_train'],data['checkpoint_val'],model,torch.device('cuda'),out,source['rx_names'])
        complete.update(total_wall_seconds=time.perf_counter()-started,peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            authorization_sha256=H.sha(HERE/'EXECUTION_AUTHORIZATION.json'),execution_sha256=H.sha(HERE/'EXECUTION_REGISTRATION.json'),finished_at=H.now())
        attempt=H.read(out/'ATTEMPT.json');attempt.update(status='COMPLETE',finished_at=H.now());H.write(out/'ATTEMPT.json',attempt)
        complete['files_sha256']={p.relative_to(out).as_posix():H.sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
        H.write(out/'COMPLETE.json',complete)
        return complete
    except Exception:
        if model is not None:torch.save(H.cpu_tree(model.state_dict()),out/'FAILURE_MODEL_STATE.pt')
        H.write(out/'FAILURE.json',dict(status='FAILED',time=H.now(),traceback=traceback.format_exc(),automatic_retry=False))
        raise


def verify_complete(job):
    out=HERE/'artifacts/scientific'/job['id'];value=H.read(out/'COMPLETE.json')
    if value['status']!='COMPLETE' or value['job']!=job or value['epochs']!=40 or value['updates']!=5080:raise ValueError('Incomplete or unregistered scientific fit')
    if value['execution_sha256']!=H.sha(HERE/'EXECUTION_REGISTRATION.json') or value['authorization_sha256']!=H.sha(HERE/'EXECUTION_AUTHORIZATION.json'):raise ValueError('Fit authorization/engineering registration changed')
    for name,digest in value['files_sha256'].items():
        path=(out/name).resolve();path.relative_to(out.resolve())
        if H.sha(path)!=digest:raise ValueError('Completed fit artifact changed')
    return out,value


def seal():
    _,_,_,jobs,_=scientific_context()
    if (HERE/'TRAINING_SEAL.json').exists():raise FileExistsError('Keep original seal')
    runs={};evidence={};checks={}
    for job in jobs:
        out,c=verify_complete(job)
        evidence[job['id']]={name:H.read(out/(name+'.json')) for name in ('CONFIG','INITIAL','HISTORY')};evidence[job['id']]['COMPLETE']=c
        runs[job['id']]=dict(complete_sha256=H.sha(out/'COMPLETE.json'),checkpoints=c['checkpoints'])
    for view in ('V0','V1'):
        for stream in ('S0','S1','S2'):
            a,b=[evidence[f'{view}_{stream}_{arm}'] for arm in ('F1','F2')];prefix=view+'/'+stream+'/'
            for field in ('frozen_backbone_sha256','common_residual_sha256','rng_sha256','scheduler_sha256'):
                checks[prefix+field]=a['INITIAL'][field]==b['INITIAL'][field]
            expected_backbone=H.read(HERE/'REFERENCE_RECOVERY.json')['references'][view+'/'+stream+'/F0']['semantic_state_sha256']
            checks[prefix+'registered_source_P0_identity']=a['INITIAL']['frozen_backbone_sha256']==b['INITIAL']['frozen_backbone_sha256']==expected_backbone
            for field in ('train_keys_sha256','val_keys_sha256'):checks[prefix+field]=a['CONFIG'][field]==b['CONFIG'][field]
            for field in ('relative_order_sha256','canonical_order_sha256','class_day_order_sha256'):
                checks[prefix+field]=all(a['HISTORY'][k][field]==b['HISTORY'][k][field] for k in range(40))
            for arm,item in (('F1',a),('F2',b)):
                checks[prefix+arm+'/frozen_and_rollback']=all(r['backbone_and_BN_unchanged'] and r['alpha0_rollback_bitwise'] for r in item['HISTORY'])
    checks['exact12_budget']=len(runs)==12 and sum(v['COMPLETE']['updates'] for v in evidence.values())==60960
    if not all(checks.values()):raise ValueError('Frozen-backbone/pairing/budget checks failed')
    H.write(HERE/'RUN_EVIDENCE.json',evidence)
    value=dict(status='ALL12_FROZEN_BACKBONE_FITS_SEALED',runs=runs,checks=checks,time=H.now(),
        authorization_sha256=H.sha(HERE/'EXECUTION_AUTHORIZATION.json'),source_selection_scoring_started=False)
    H.write(HERE/'TRAINING_SEAL.json',value);return value


def evaluate():
    _,_,_,jobs,source=scientific_context();sealed=require_full_seal(jobs)
    if (HERE/'EVALUATION_ATTEMPT.json').exists():raise FileExistsError('Evaluation was already attempted')
    H.write(HERE/'EVALUATION_ATTEMPT.json',dict(time=H.now(),seal_sha256=H.sha(HERE/'TRAINING_SEAL.json')))
    H.A.setup(26301);pool=read_metadata(source);files={};parities={}
    for view in H.read(DESIGN/'FUTURE_SOURCE_PLAN.json')['views']:
        ids=H.C.role_ids(source,view,pool)
        data=materialize(source,pool,ids,('source_selection',),'sealed_selection_evaluation')['source_selection']
        x=torch.as_tensor(data['x'],device='cuda')
        for stream in ('S0','S1','S2'):
            for role in ('F0','E0'):
                model=load_reference(role,view['id'],stream,torch.device('cuda'))
                before=H.semantic_hash(model.state_dict());logits=H.C.predict_tensor(model,x,128)
                parities[view['id']+'/'+stream+'/'+role]=parity(logits,data,archived_reference_predictions(role,view['id'],stream,'source_selection'),source['rx_names'])
                if H.semantic_hash(model.state_dict())!=before:raise ValueError('Reference evaluation changed frozen state')
                out=HERE/'artifacts/evaluation'/f'{view["id"]}_{stream}_{role}'/'reference.npz';out.parent.mkdir(parents=True,exist_ok=True)
                H.C.save_predictions(out,logits,data,source['rx_names']);files[out.relative_to(HERE/'artifacts').as_posix()]=H.sha(out)
                del model
        for job in [j for j in jobs if j['view']==view['id']]:
            out=HERE/'artifacts/scientific'/job['id']
            model=load_reference('F0',job['view'],job['stream'],torch.device('cuda'),job['arm'],job['side_initialization_seed'])
            for endpoint in H.ENDPOINTS:
                if H.sha(out/(endpoint+'.pt'))!=sealed['runs'][job['id']]['checkpoints'][endpoint]:raise ValueError('Sealed endpoint changed')
                model.load_state_dict(torch.load(out/(endpoint+'.pt'),map_location='cpu',weights_only=False)['model'])
                before=H.semantic_hash(model.state_dict());logits=H.C.predict_tensor(model,x,128)
                if H.semantic_hash(model.state_dict())!=before:raise ValueError('Evaluation changed state')
                path=HERE/'artifacts/evaluation'/job['id']/(endpoint+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
                H.C.save_predictions(path,logits,data,source['rx_names']);files[path.relative_to(HERE/'artifacts').as_posix()]=H.sha(path)
            del model
    if len(files)!=48:raise ValueError('All36 residual and12 frozen reference predictions required')
    value=dict(status='ALL48_SOURCE_PREDICTIONS_FROZEN',files=files,reference_parity=parities,seal_sha256=H.sha(HERE/'TRAINING_SEAL.json'),time=H.now())
    H.write(HERE/'EVALUATION_MANIFEST.json',value);return value


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['status','preflight','train-all','seal','evaluate']);args=p.parse_args()
    if args.command=='status':
        print(dict(status='SCIENTIFIC_EXECUTION_CLOSED' if not (HERE/'EXECUTION_AUTHORIZATION.json').exists() else 'AUTHORIZATION_PRESENT_REQUIRES_VALIDATION',scientific_attempts=len(list((HERE/'artifacts/scientific').glob('*/ATTEMPT.json')))))
        return
    if args.command=='preflight':value=preflight()
    elif args.command=='train-all':
        _,_,_,jobs,_=scientific_context()
        for job in jobs:fit(job)
        value=seal()
    elif args.command=='seal':value=seal()
    else:value=evaluate()
    print(dict(command=args.command,status=value['status']))


if __name__=='__main__':main()
