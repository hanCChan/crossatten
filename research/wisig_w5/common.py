"""W5 shared provenance utilities; writes are confined to the new W4 directory."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import hashlib,importlib.util,json,subprocess
from pathlib import Path
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
ART=HERE/'artifacts'
OLD=Path('D:/crossatten/worktrees/wisig-quality-preservation')
S20=Path('D:/crossatten/worktrees/s20-asknet-source-comparison/20-asknet-source-comparison')
S19=Path('D:/crossatten/worktrees/s19-tx-spectral-evidence/19-tx-spectral-evidence')
CART=Path('D:/crossatten/worktrees/training-instance-attribution/research/standard_method_competition_c/artifacts')

def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,obj):
    path=Path(path)
    if not path.is_absolute():path=HERE/path
    assert path.resolve().is_relative_to(HERE)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n');os.replace(temp,path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while data:=f.read(16<<20):h.update(data)
    return h.hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True,encoding='utf-8').strip()
def committed(paths):
    rels=[Path(p).resolve().relative_to(ROOT).as_posix() for p in paths]
    subprocess.run(['git','-C',str(ROOT),'ls-files','--error-unmatch','--',*rels],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(ROOT),'diff','--quiet','HEAD','--',*rels],check=True)
def record(path):return dict(path=str(Path(path).resolve()),sha256=sha(path),bytes=Path(path).stat().st_size)
def checked(item,action,role=None):
    p=Path(item['path'])
    if sha(p)!=item['sha256']:raise ValueError('Changed input: '+str(p))
    with (HERE/'ACCESS.jsonl').open('a',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(dict(time=now(),action=action,role=role,**item))+'\n')
    return p
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
def helpers():return module('W5_frozen_C',ROOT/'research/standard_method_competition_c/engine.py')
def commit(paths,message):
    subprocess.run(['git','-C',str(ROOT),'add','--',*[str(p) for p in paths]],check=True)
    subprocess.run(['git','-C',str(ROOT),'-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m',message],check=True)
def validate(name='EXECUTION_REGISTRATION.json'):
    p=HERE/name;reg=read(p);paths=[p]
    for rel,digest in reg['files_sha256'].items():
        q=ROOT/rel
        if sha(q)!=digest:raise ValueError('Frozen file changed: '+rel)
        paths.append(q)
    committed(paths);return reg
