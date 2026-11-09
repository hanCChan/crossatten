"""Isolated S16 bookkeeping; the live S13 tree is read-only to this stage."""
import hashlib
import json
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from contextlib import contextmanager

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
LIVE=Path('D:/crossatten/stage_1')
S13=LIVE/'13-mixer-cfo-mechanisms'
FROZEN=ROOT/'13-mixer-cfo-mechanisms'
sys.path.insert(0,str(ROOT/'lib'))
sys.path.insert(0,str(FROZEN))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')

def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(16<<20):h.update(b)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    os.replace(temp,path)
def frozen(path,value):
    if Path(path).exists():
        if read(path)!=value:raise ValueError(f'Frozen record exists: {path}')
    else:write(path,value)
def git(*args,cwd=ROOT):return subprocess.check_output(['git','-C',str(cwd),*args],text=True,encoding='utf-8').strip()
def committed(path,root=ROOT):
    rel=Path(path).resolve().relative_to(root).as_posix()
    if subprocess.check_output(['git','-C',str(root),'show',f'HEAD:{rel}'])!=Path(path).read_bytes():
        raise ValueError(f'Uncommitted evidence: {rel}')
def commit(paths,message):
    paths=[Path(p).resolve().relative_to(ROOT).as_posix() for p in paths]
    if any(not p.startswith(HERE.name+'/') for p in paths):raise ValueError('Only S16 may be committed')
    changed=[]
    for p in paths:
        tracked=subprocess.run(['git','-C',str(ROOT),'ls-files','--error-unmatch',p],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
        if not tracked or subprocess.run(['git','-C',str(ROOT),'diff','--quiet','HEAD','--',p]).returncode:changed.append(p)
    if changed:
        subprocess.run(['git','-C',str(ROOT),'add','--',*changed],check=True)
        subprocess.run(['git','-C',str(ROOT),'-c','user.name=Codex','-c','user.email=codex@local.invalid','commit','-m',message,'--',*changed],check=True)
    return git('rev-parse','HEAD')
def validate(name='registration.json'):
    p=HERE/name;committed(p);r=read(p)
    for rel,digest in r['code_sha256'].items():
        if sha(ROOT/rel)!=digest:raise ValueError(f'Registered code changed: {rel}')
    for rel,digest in r.get('evidence_sha256',{}).items():
        if sha(ROOT/rel)!=digest:raise ValueError(f'Registered evidence changed: {rel}')
    return r
def access(**fields):
    with (HERE/'access.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=now(),**fields))+'\n')
def job_id(job):return f"p1__Shen__{job['arm']}__f{job['fold']}__all__s{job['seed']}"
def s13_id(arm,fold,seed):return f'p1__Shen__{arm}__f{fold}__all__s{seed}'

@contextmanager
def lock():
    path=HERE/'locks/handoff.lock';path.parent.mkdir(exist_ok=True)
    with path.open('a+b') as f:
        if path.stat().st_size==0:f.write(b'0');f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f,fcntl.LOCK_UN)

def process_alive(pid):
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:return False
        code=wintypes.DWORD()
        try:return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
        finally:kernel.CloseHandle(handle)
    try:os.kill(int(pid),0);return True
    except ProcessLookupError:return False

S15=Path("D:/crossatten/worktrees/s15-hstem-attribution/15-hstem-attribution")
_DLL_HANDLES=[]
if os.name=="nt":
    for directory in ["D:/miniconda/envs/crossatten4060","D:/miniconda/envs/crossatten4060/Library/bin"]:
        if Path(directory).is_dir():_DLL_HANDLES.append(os.add_dll_directory(directory))
os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY","1")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
