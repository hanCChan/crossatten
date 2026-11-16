import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import csv,hashlib,json,random,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;STAGE=HERE.parent;ROOT=STAGE.parent
sys.path.insert(0,str(STAGE))
from s24_common import write_json,sha,P

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8<<20):h.update(b)
    return h.hexdigest()
def atomic_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    write_json(tmp,obj);os.replace(tmp,p)
def atomic_torch(p,obj):
    tmp=p.with_suffix(p.suffix+'.tmp');torch.save(obj,tmp);os.replace(tmp,p)
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True,encoding='utf-8').strip()
def committed(p):
    rel=Path(p).resolve().relative_to(ROOT).as_posix()
    if subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:'+rel])!=Path(p).read_bytes():raise ValueError('not committed: '+rel)
def commit(paths,message):
    rels=[Path(p).resolve().relative_to(ROOT).as_posix() for p in paths]
    assert all(p.startswith('24-lora-external-audit/') for p in rels)
    subprocess.run(['git','-C',str(ROOT),'add','--',*rels],check=True)
    if subprocess.run(['git','-C',str(ROOT),'diff','--cached','--quiet','--',*rels]).returncode:
        subprocess.run(['git','-C',str(ROOT),'-c','user.name=Codex','-c','user.email=codex@local.invalid','commit','-m',message,'--',*rels],check=True)
    return git('rev-parse','HEAD')
def jobid(j):return f"{j['method']}__{j['view']}__f{j['fold']}__s{j['seed']}"
def setup(seed):
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def validate_registry():
    path=HERE/'pilot_registry.json';committed(path);r=read(path)
    for rel,expected in r['code_sha256'].items():
        if digest(ROOT/rel)!=expected:raise ValueError('registered code changed: '+rel)
    for rel,expected in r['artifact_sha256'].items():
        if digest(ROOT/rel)!=expected:raise ValueError('registered artifact changed: '+rel)
    return r
def allowed(job):
    r=validate_registry()
    if job not in r['jobs']:raise ValueError('unregistered job')
    jid=jobid(job)
    if jid in r['pilot_ids']:return r
    freeze=HERE/f"freeze_{'ib' if job['method'] in ('fan','ib') else job['method']}.json"
    committed(freeze);f=read(freeze)
    if f['registry_sha256']!=digest(HERE/'pilot_registry.json') or jid not in f['allowed_jobs']:raise ValueError('method not frozen')
    for rel,h in f['pilot_evidence'].items():
        if digest(HERE/rel)!=h:raise ValueError('pilot changed')
    return r
def freeze_method(method):
    r=validate_registry();group=['fan','ib'] if method=='ib' else [method]
    path=HERE/f'freeze_{method}.json'
    if path.exists():committed(path);return read(path)
    evidence={}
    for name in group:
        jid=f'{name}__native__f0__s201';out=HERE/'runs'/jid
        metrics=read(out/'metrics.json')
        if metrics['status']!='COMPLETE':raise ValueError('pilot not complete')
        for p in [out/'metrics.json',out/'config.json',out/'best.pt']:
            evidence[p.relative_to(HERE).as_posix()]=digest(p)
    obj=dict(status='FROZEN_AFTER_SOURCE_PILOT',registry_sha256=digest(HERE/'pilot_registry.json'),
             pilot_evidence=evidence,allowed_jobs=[jobid(j) for j in r['jobs'] if j['method'] in group],
             pilot_reuse='Promote native fold0 seed201 already preregistered weights; identical code/recipe, no target access, no score threshold',
             source_score_cannot_change_config=True,git=git('rev-parse','HEAD'))
    write_json(path,obj)
    commit([path]+[HERE/p for p in evidence if not p.endswith('.pt')],f'S24 freeze {method} after source-only pilot; retain registered configuration')
    return obj
