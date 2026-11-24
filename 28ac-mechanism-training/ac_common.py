"""New execution activation; the S28-0 zero-fit record is immutable history."""
import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
BASE='de7d0100c5bcce4a389f756bc8fe46c73aa9787e'
CACHE=Path('D:/crossatten/stage_1/13-mixer-cfo-mechanisms/cache/shen')
for folder in ['lib','13-mixer-cfo-mechanisms','15-hstem-attribution','21-hstem-mechanisms','25-tifs-integration',
               '26-long-patch-control','27-mechanism-identification','27b-tokenizer-backend-interaction','28-affine-bn-gelu-audit']:
    sys.path.insert(0,str(ROOT/folder))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while block:=f.read(8<<20):h.update(block)
    return h.hexdigest()
_serial=itertools.count()
def replace(source,target):
    for i in range(201):
        try:os.replace(source,target);return
        except PermissionError as e:
            if getattr(e,'winerror',None) not in [5,32,33] or i==200:raise
            time.sleep(.01)
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(f'{p.name}.tmp.{os.getpid()}.{next(_serial)}')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n');replace(temp,p)
def committed(p):assert subprocess.check_output(['git','show','HEAD:'+Path(p).relative_to(ROOT).as_posix()],cwd=ROOT)==Path(p).read_bytes()
def validate():
    committed(HERE/'registration.json');r=read(HERE/'registration.json');assert r['maximum_new_fits']==18
    committed(HERE/'ACTIVATION.json');assert read(HERE/'ACTIVATION.json')['authorized_scientific_fits']==18
    for key in ['code_sha256','evidence_sha256']:
        for file,h in r[key].items():assert sha(ROOT/file)==h,file
    return r
def identity(j):return f"{j['stage']}_{j['arm']}_f{j['fold']}_s{j['seed']}"
def commit(message):
    subprocess.run(['git','add','--',HERE.name],cwd=ROOT,check=True)
    names=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
    assert all(p.startswith(HERE.name+'/') for p in names)
    if names:subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local.invalid','commit','-m',message],cwd=ROOT,check=True)
def event(**values):
    with (HERE/'access.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(dict(time=now(),**values))+'\n')
P=read(ROOT/'configs/protocol.json')
A=read(HERE/'PLAN_A.original.json');C=read(HERE/'PLAN_C.original.json')
