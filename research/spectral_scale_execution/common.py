"""Frozen design helpers and separate engineering/scientific access boundaries."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import torch
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
DESIGN=ROOT/'research/spectral_scale_review'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


BASE=module('frozen_span_helpers',ROOT/'research/periodic_span/engine.py')
A,C,S=BASE.A,BASE.C,BASE.S
M=module('frozen_spectral_scale_design',DESIGN/'models.py')
read,write,sha,now=A.read,A.write,A.file_sha,A.now
semantic_hash,cpu_tree=A.semantic_hash,A.cpu_tree
ENDPOINTS=('selected','epoch10','final40')


def committed(path):A.require_committed(path)


def validate_design():
    reg=read(DESIGN/'REGISTRATION.json')
    committed(DESIGN/'REGISTRATION.json')
    for name,digest in reg['code_sha256'].items():
        p=ROOT/name;committed(p)
        if sha(p)!=digest:raise ValueError('Scientific design code changed: '+name)
    for name,digest in reg['artifact_sha256'].items():
        p=DESIGN/name;committed(p)
        if sha(p)!=digest:raise ValueError('Scientific design artifact changed: '+name)
    return reg


def scientific_gate():
    """Must run before source-array IO, source reference predictions or a science fit."""
    path=HERE/'EXECUTION_AUTHORIZATION.json'
    if not path.exists():
        raise PermissionError('Scientific execution is closed: explicit separate user authorization has not been recorded.')
    committed(path)
    value=read(path)
    if value.get('status')!='EXPLICIT_USER_AUTHORIZATION_FOR_FROZEN_12_FITS' or not value.get('user_authorization_text'):
        raise PermissionError('No explicit scientific execution authorization')
    registration=HERE/'EXECUTION_REGISTRATION.json'
    if not registration.exists():raise PermissionError('Committed engineering registration is required')
    committed(registration)
    if value.get('design_sha256')!=sha(DESIGN/'REGISTRATION.json') or value.get('execution_sha256')!=sha(registration) or value.get('maximum_scientific_fits')!=12:
        raise PermissionError('Authorization does not match the frozen design/execution/budget')
    ex=read(registration)
    if ex['status']!='ENGINEERING_ACCEPTED_SCIENTIFIC_EXECUTION_CLOSED':raise PermissionError('Engineering acceptance is missing')
    for name,digest in ex['code_sha256'].items():
        p=ROOT/name;committed(p)
        if sha(p)!=digest:raise ValueError('Execution code changed: '+name)
    for name,digest in ex['artifact_sha256'].items():
        p=HERE/name;committed(p)
        if sha(p)!=digest:raise ValueError('Engineering evidence changed: '+name)
    if str(torch.__version__)!=ex['environment']['torch'] or np.__version__!=ex['environment']['numpy']:
        raise ValueError('Execution software differs from engineering acceptance')
    return value,ex


def log_access(path,action,job=None,science=False):
    name='SCIENTIFIC_ACCESS_LOG.jsonl' if science else 'ENGINEERING_ACCESS_LOG.jsonl'
    with (HERE/name).open('a',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(dict(time=now(),path=str(path),action=action,job=job))+'\n')


def recovered_weights(view,stream,role):
    registry=read(HERE/'REFERENCE_INPUTS.json')
    record=registry['references'][view+'/'+stream][role]
    path=HERE/record['copy_relative']
    log_access(path,'hash_and_load_registered_source_checkpoint')
    if sha(path)!=record['sha256']:raise ValueError('Recovered source checkpoint changed')
    ck=torch.load(path,map_location='cpu',weights_only=False)
    return ck['model'],record


def model_from_reference(arm,job,device):
    state,_=recovered_weights(job['view'],job['stream'],'F0')
    return M.build(arm,24101+int(job['stream'][-1]),job['side_initialization_seed'],backbone_state=state).to(device)


def named_reference(role,view,stream,device):
    state,_=recovered_weights(view,stream,role)
    return M.build('F0',24101+int(stream[-1]),backbone_state=state).to(device).eval()


def validate_budget(jobs):
    expected={(v,f'S{s}',a) for v in ('V0','V1') for s in range(3) for a in ('F1','F2')}
    if len(jobs)!=12 or {(j['view'],j['stream'],j['arm']) for j in jobs}!=expected:raise ValueError('Exactly the original12 jobs are allowed')
    for j in jobs:
        k=int(j['stream'][-1])
        if (j['side_initialization_seed'],j['order_seed'],j['stochastic_seed'])!=(26101+k,24201+k,26301):raise ValueError('Random streams changed')
        if (j['epochs'],j['batch'],j['lr'],j['weight_decay'],j['max_gradient_updates'])!=(40,128,.001,.001,5080):raise ValueError('Budget/recipe changed')
        if j['checkpoint_candidate_epochs']!=list(range(41)) or j['auxiliary_epochs']!=[10,40]:raise ValueError('Checkpoint selection endpoints changed')
    return jobs
