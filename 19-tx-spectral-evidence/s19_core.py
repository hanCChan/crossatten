"""Isolated S19 bookkeeping: historical trees are read-only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HISTORY = Path('D:/crossatten/worktrees/s17-rx-distortion')
S18 = Path('D:/crossatten/worktrees/s18-selective-rx')
SOURCE_CELLS = Path('D:/crossatten/stage_1/13-mixer-cfo-mechanisms/cache/wisig')
PYTHON = Path('D:/miniconda/envs/crossatten4060/python.exe')
for path in [ROOT/'lib', ROOT/'13-mixer-cfo-mechanisms', ROOT/'17-rx-distortion']:
    sys.path.insert(0, str(path))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')


def now(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(16 << 20): h.update(chunk)
    return h.hexdigest()


def array_sha(array): return hashlib.sha256(array.tobytes()).hexdigest()


def write(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(HERE): raise ValueError('S19 write outside stage forbidden')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    os.replace(tmp, path)


def git(*args): return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True, encoding='utf-8').strip()


def committed(path):
    rel = Path(path).resolve().relative_to(ROOT).as_posix()
    if subprocess.check_output(['git', '-C', str(ROOT), 'show', 'HEAD:'+rel]) != Path(path).read_bytes():
        raise ValueError('Uncommitted evidence: '+rel)


def access(**fields):
    with (HERE/'access.jsonl').open('a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(dict(time=now(), **fields), ensure_ascii=False)+'\n')


def validate():
    committed(HERE/'registration.json')
    reg = read(HERE/'registration.json')
    for rel, digest in reg['code_sha256'].items():
        if sha(ROOT/rel) != digest: raise ValueError('Registered code changed: '+rel)
    for rel, digest in reg['design_sha256'].items():
        if sha(HERE/rel) != digest: raise ValueError('Registered design changed: '+rel)
    for rel, digest in reg['historical_sha256'].items():
        if sha(HISTORY/rel) != digest: raise ValueError('Historical source evidence changed: '+rel)
    return reg


def job_id(fold, arm): return f'ManyRx__{arm}__f{fold}__first__s18101'
