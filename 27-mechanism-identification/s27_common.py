"""S27 isolation, committed registration and source/report-only permissions."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = Path('D:/crossatten/stage_1/13-mixer-cfo-mechanisms/cache/shen')
for folder in ['lib','13-mixer-cfo-mechanisms','15-hstem-attribution','21-hstem-mechanisms','25-tifs-integration','26-long-patch-control']:
    sys.path.insert(0, str(ROOT/folder))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    value=hashlib.sha256()
    with Path(p).open('rb') as stream:
        while block:=stream.read(8<<20): value.update(block)
    return value.hexdigest()


def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True, exist_ok=True)
    temporary=p.with_name(p.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    os.replace(temporary,p)


def committed(p):
    relative=Path(p).resolve().relative_to(ROOT).as_posix()
    if subprocess.check_output(['git','show','HEAD:'+relative],cwd=ROOT)!=Path(p).read_bytes():
        raise PermissionError('Uncommitted scientific artifact: '+relative)


def validate():
    committed(HERE/'registration.json');r=read(HERE/'registration.json')
    assert r['maximum_new_fits']==6
    for relative, expected in r['code_sha256'].items():
        assert sha(ROOT/relative)==expected, 'Frozen code changed: '+relative
    for relative, expected in r['evidence_sha256'].items():
        assert sha(ROOT/relative)==expected, 'Historical evidence changed: '+relative
    return r


def commit(message):
    # One directory argument avoids Windows command-length failures; ignored raw
    # data/checkpoints/resume states never enter the index.
    subprocess.run(['git','add','--',HERE.name],cwd=ROOT,check=True)
    names=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
    if not all(n.startswith(HERE.name+'/') for n in names):
        raise PermissionError('Out-of-stage staged changes')
    if names:
        subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local.invalid','commit','-m',message],cwd=ROOT,check=True)
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def identity(fold,seed):
    return f'LA_f{fold}_s{seed}'


def event(**fields):
    with (HERE/'access.jsonl').open('a',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(dict(time=now(),**fields))+'\n')


P=read(ROOT/'configs/protocol.json')
