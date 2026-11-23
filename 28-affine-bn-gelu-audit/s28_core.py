"""Isolated no-fit audit; old scientific files and models remain read-only."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CACHE=Path('D:/crossatten/stage_1/13-mixer-cfo-mechanisms/cache/shen')
BASE='48faa0ec3a878ac096cb26b71c984f2c7ef46a56'
for folder in ['lib','13-mixer-cfo-mechanisms','15-hstem-attribution','21-hstem-mechanisms',
               '26-long-patch-control','27-mechanism-identification','27b-tokenizer-backend-interaction']:
    sys.path.insert(0,str(ROOT/folder))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while block:=f.read(8<<20):h.update(block)
    return h.hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temporary=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    os.replace(temporary,p)
def committed(p):
    assert subprocess.check_output(['git','show','HEAD:'+Path(p).relative_to(ROOT).as_posix()],cwd=ROOT)==Path(p).read_bytes()
def validate():
    committed(HERE/'registration.json');r=read(HERE/'registration.json')
    assert r['scientific_fit_budget']==0
    for file,h in r['code_sha256'].items():assert sha(ROOT/file)==h,file
    for file,h in r['evidence_sha256'].items():assert sha(ROOT/file)==h,file
    return r
def event(**fields):
    with (HERE/'access.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(dict(time=now(),**fields))+'\n')
P=read(ROOT/'configs/protocol.json')
