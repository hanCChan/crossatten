"""Bounded LoRa mechanism study. Historical stages are read-only."""
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
LIVE = Path('D:/crossatten/stage_1')
S13 = LIVE / '13-mixer-cfo-mechanisms'
S15 = Path('D:/crossatten/worktrees/s15-hstem-attribution/15-hstem-attribution')
for folder in [ROOT/'lib', ROOT/'13-mixer-cfo-mechanisms', ROOT/'15-hstem-attribution']:
    sys.path.insert(0, str(folder))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

def now(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        while data := f.read(16 << 20): h.update(data)
    return h.hexdigest()
def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    os.replace(temp, p)
def git(*args): return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True, encoding='utf-8').strip()
def committed(p):
    rel = Path(p).resolve().relative_to(ROOT).as_posix()
    if subprocess.check_output(['git', '-C', str(ROOT), 'show', 'HEAD:'+rel]) != Path(p).read_bytes():
        raise ValueError('Uncommitted evidence: '+rel)
def commit(paths, message):
    rels = [Path(p).resolve().relative_to(ROOT).as_posix() for p in paths]
    if any(not p.startswith(HERE.name+'/') for p in rels): raise ValueError('Out-of-stage mutation')
    subprocess.run(['git', '-C', str(ROOT), 'add', '--', *rels], check=True)
    if subprocess.run(['git', '-C', str(ROOT), 'diff', '--cached', '--quiet', '--', *rels]).returncode:
        subprocess.run(['git', '-C', str(ROOT), '-c', 'user.name=Codex', '-c', 'user.email=codex@local.invalid', 'commit', '-m', message, '--', *rels], check=True)
    return git('rev-parse', 'HEAD')
def job_id(j): return f"p1__Shen__{j['arm']}__f{j['fold']}__all__s{j['seed']}"
def baseline_path(arm, fold, seed):
    return (S13 if arm == 'H0' else S15)/'runs'/job_id(dict(arm='A1_mha' if arm == 'H0' else 'C2_lite_mha', fold=fold, seed=seed))
def validate():
    committed(HERE/'registration.json'); r = read(HERE/'registration.json')
    for rel, digest in r['code_sha256'].items():
        if sha(ROOT/rel) != digest: raise ValueError('Frozen code changed: '+rel)
    for filename, digest in r['artifacts_sha256'].items():
        if sha(HERE/filename) != digest: raise ValueError('Frozen artifact changed: '+filename)
    return r
def access(**fields):
    with (HERE/'access.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(dict(time=now(), **fields))+'\n')
