"""S13 isolated bookkeeping; no historical output mutations."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(16 << 20):
            h.update(b)
    return h.hexdigest()


def array_sha(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(temp, path)


def once(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f"Refusing to change frozen record: {path}")
    else:
        write(path, value)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, encoding="utf-8").strip()


def require_committed(path):
    relative = Path(path).resolve().relative_to(ROOT).as_posix()
    recorded = subprocess.check_output(["git", "-C", str(ROOT), "show", f"HEAD:{relative}"])
    if recorded != Path(path).read_bytes():
        raise ValueError(f"Uncommitted registration/evidence: {relative}")


def commit(paths, message):
    rel = [Path(p).resolve().relative_to(ROOT).as_posix() for p in paths]
    if any(not p.startswith(HERE.name + "/") for p in rel):
        raise ValueError("S13 may only commit S13 files")
    subprocess.run(["git", "-C", str(ROOT), "add", "--", *rel], check=True)
    subprocess.run(["git", "-C", str(ROOT), "-c", "user.name=Codex", "-c", "user.email=codex@local.invalid",
                    "commit", "-m", message, "--", *rel], check=True)
    return git("rev-parse", "HEAD")


def commit_changed(paths, message):
    changed=[]
    for p in paths:
        relative=Path(p).resolve().relative_to(ROOT).as_posix()
        tracked=subprocess.run(['git','-C',str(ROOT),'ls-files','--error-unmatch',relative],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
        if not tracked or subprocess.run(['git','-C',str(ROOT),'diff','--quiet','HEAD','--',relative]).returncode:
            changed.append(p)
    return commit(changed,message) if changed else git('rev-parse','HEAD')


def log_access(**entry):
    with (HERE / "access.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(time=now(), **entry), ensure_ascii=False) + "\n")


@contextmanager
def exclusive(name):
    path = HERE / "locks" / f"{name}.lock"
    path.parent.mkdir(exist_ok=True)
    with path.open("a+b") as f:
        if path.stat().st_size == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


def validate_registration():
    p = HERE / "registration.json"
    require_committed(p)
    r = read(p)
    effective=dict(r['code_sha256']);parent=sha(p)
    for amendment in sorted(HERE.glob('registration_amendment_*.json')):
        require_committed(amendment)
        value=read(amendment)
        if value['parent_registration_sha256']!=parent:raise ValueError('Broken registration amendment chain')
        if value['protocol_changes']:raise ValueError('Technical amendment may not change scientific configuration')
        effective.update(value['code_sha256']);parent=sha(amendment)
    for relative, digest in effective.items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f"Registered source changed: {relative}")
    return r
