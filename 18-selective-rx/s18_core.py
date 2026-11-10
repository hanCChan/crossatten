"""S18 writes only to its own directory; all historical evidence is read-only."""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HISTORY = Path('D:/crossatten/worktrees/s17-rx-distortion')
PYTHON = Path('D:/miniconda/envs/crossatten4060/python.exe')
for p in [ROOT/'lib', ROOT/'13-mixer-cfo-mechanisms', ROOT/'17-rx-distortion']:
    sys.path.insert(0, str(p))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(16 << 20):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(HERE):
        raise ValueError('S18 may only write within its own stage directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    os.replace(tmp, path)


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True, encoding='utf-8').strip()


def access(**fields):
    with (HERE/'access.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(time=now(), **fields), ensure_ascii=False)+'\n')


def validate():
    p = HERE/'registration.json'
    if subprocess.check_output(['git', '-C', str(ROOT), 'show', 'HEAD:18-selective-rx/registration.json']) != p.read_bytes():
        raise ValueError('Registration must be committed before scientific data access')
    reg = read(p)
    for rel, digest in reg['code_sha256'].items():
        if sha(ROOT/rel) != digest:
            raise ValueError('Registered code changed: '+rel)
    for rel, digest in reg['historical_sha256'].items():
        if sha(HISTORY/rel) != digest:
            raise ValueError('Historical evidence changed: '+rel)
    for name, digest in reg['design_sha256'].items():
        if sha(HERE/name) != digest:
            raise ValueError('Registered design changed: '+name)
    return reg
