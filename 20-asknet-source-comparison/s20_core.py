"""S20 writes only its own stage; all historical evidence remains read-only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
S19=Path('D:/crossatten/worktrees/s19-tx-spectral-evidence/19-tx-spectral-evidence')
HISTORY=Path('D:/crossatten/worktrees/s17-rx-distortion')
UPSTREAM=Path('D:/crossatten/.reference-s20/ASKNet-RFF-f00846d')
for path in [ROOT/'lib',ROOT/'13-mixer-cfo-mechanisms',ROOT/'17-rx-distortion',ROOT/'19-tx-spectral-evidence']:
    sys.path.insert(0,str(path))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')

def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(16<<20):h.update(b)
    return h.hexdigest()
def write(p,value):
    p=Path(p)
    if not p.resolve().is_relative_to(HERE):raise ValueError('Only S20 output writes permitted')
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    os.replace(tmp,p)
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True,encoding='utf-8').strip()
def committed(p):
    rel=Path(p).resolve().relative_to(ROOT).as_posix()
    if subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:'+rel])!=Path(p).read_bytes():raise ValueError('Uncommitted evidence: '+rel)
def access(**fields):
    with (HERE/'access.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(dict(time=now(),**fields),ensure_ascii=False)+'\n')
def validate():
    committed(HERE/'registration.json');r=read(HERE/'registration.json')
    for rel,h in r['code_sha256'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Registered code changed: '+rel)
    for rel,h in r['design_sha256'].items():
        if sha(HERE/rel)!=h:raise ValueError('Registered design changed: '+rel)
    for rel,h in r['s19_evidence_sha256'].items():
        if sha(S19/rel)!=h:raise ValueError('Historical S19 evidence changed: '+rel)
    # The frozen S19 registration pins the actual fit-cache manifests and
    # original B0 metadata, beyond the copied S19 report artifacts.
    for rel,h in read(S19/'registration.json')['historical_sha256'].items():
        if sha(HISTORY/rel)!=h:raise ValueError('Original source evidence changed: '+rel)
    for rel,h in r['official_files_sha256'].items():
        if sha(UPSTREAM/rel)!=h:raise ValueError('Pinned official file changed: '+rel)
    return r
def official():
    manifest=read(UPSTREAM/'FETCH_MANIFEST.json')
    for rel,entry in manifest['files'].items():
        if sha(UPSTREAM/rel)!=entry['sha256']:raise ValueError('Official source SHA mismatch')
    directory=str(UPSTREAM/'WiSig')
    if directory not in sys.path:sys.path.insert(0,directory)
    spec=importlib.util.spec_from_file_location('s20_official_ask',UPSTREAM/'WiSig/main.py')
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    def forbidden_loader(*args,**kwargs):raise ValueError('Direct official dataset loader forbidden; guarded S19 samples only')
    obj.load_single_dataset=forbidden_loader
    return obj

def configure_data():
    import s19_data as data
    data.HERE=HERE;data.access=access;data.committed=committed
    return data
