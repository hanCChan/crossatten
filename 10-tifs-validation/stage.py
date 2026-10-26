"""S10 isolation, pinned imports and immutable protocol helpers."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(1, str(ROOT / "lib"))
sys.path.append(str(ROOT / "09-phase-invariance"))
import numpy as np
import torch
from xa.paper_train import require_committed, sha
from xa.wisig import DATA, RX, load_cell, receivers


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


s09 = module("s10_frozen_s09", ROOT / "09-phase-invariance/training.py")
ASK_DIR = ROOT / "external/ASKNet-RFF"
ASK_SHA = "f00846d68425315b68ae69b8c6b712e08559c2fe"


def ask_modules():
    sys.path.insert(0, str(ASK_DIR / "WiSig"))
    return module("s10_official_ask", ASK_DIR / "WiSig/main.py")


def save(path, value):
    path.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")


def once(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"immutable evidence changed: {path.name}")
    else:
        save(path, value)


def stream_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(job):
    return (f"{job['dataset']}__f{job['fold']}__{job.get('group', 'all')}"
            f"__s{job['seed']}__{job['feature']}__{job['policy']}")


def validate(job):
    reg_path = HERE / "registration.json"
    require_committed(reg_path)
    reg = json.loads(reg_path.read_text())
    for path, expected in reg["code_manifest"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"registered code changed: {path}")
    for path, expected in reg["upstream_files"].items():
        if sha(ASK_DIR / path) != expected:
            raise ValueError(f"ASKNet pinned file changed: {path}")
    require_committed(HERE / "preparation.json")
    if sha(HERE / "preparation.json") != reg["preparation_sha256"]:
        raise ValueError("preparation changed")
    phase = job["phase"]
    path = HERE / f"jobs_{phase}.json"
    require_committed(path)
    jobs = json.loads(path.read_text())
    if job not in jobs or job["id"] != identity(job) or len(jobs) > reg["caps"][phase]:
        raise ValueError("unregistered fit")
    if phase in ("final", "ablation"):
        require_committed(HERE / "decision_final.json")
        decision = json.loads((HERE / "decision_final.json").read_text())
        if not decision["admitted"] or job not in decision["jobs"]:
            raise ValueError("candidate not source-admitted")
        for path, expected in decision["evidence"].items():
            require_committed(ROOT / path)
            if sha(ROOT / path) != expected:
                raise ValueError("source evidence changed after gate")
    if phase == "ask":
        require_committed(HERE / "ask_acceptance.json")
        accepted = json.loads((HERE / "ask_acceptance.json").read_text())
        if not accepted["accepted"]:
            raise ValueError("ASKNet reproduction not accepted")
    return reg


def cells(job, names, days, role, task=None):
    source, target = receivers(job["dataset"], job["fold"])
    if role in ("fit", "held", "diagnostic"):
        if not set(names) <= set(source) or tuple(days) != (1, 2):
            raise ValueError("illegal source RX/day access")
    elif role == "target":
        validate(job)
        if job["phase"] not in ("ask", "final", "ablation") or not set(names) <= set(target):
            raise ValueError("unregistered target read")
        if (task, tuple(days)) not in (("CR", (1, 2)), ("CRD", (3, 4))):
            raise ValueError("invalid target dates")
        result_path = HERE / "runs" / job["phase"] / job["id"] / "metrics.json"
        require_committed(result_path)
        result = json.loads(result_path.read_text())
        out = result_path.parent
        if sha(out / "model.pt") != result["weights_sha256"]["model.pt"]:
            raise ValueError("unsealed target checkpoint")
    else:
        raise ValueError(role)
    prep = json.loads((HERE / "preparation.json").read_text())
    result = {}
    with (HERE / "access.jsonl").open("a") as log:
        for rx in names:
            xs, ys = [], []
            for day in days:
                path = DATA / "cache" / job["dataset"] / f"rx_{rx}_day{day}.npz"
                key = f"{job['dataset']}/{path.name}"
                if sha(path) != prep["cache_sha256"][key]:
                    raise ValueError("cache bytes changed")
                x, y = load_cell(job["dataset"], rx, day)
                xs.append(x)
                ys.append(y)
                log.write(json.dumps(dict(run=job["id"], phase=job["phase"],
                    dataset=job["dataset"], fold=job["fold"], rx=rx, day=day,
                    role=role, task=task)) + "\n")
            result[rx] = dict(x=np.concatenate(xs), y=np.concatenate(ys))
    return result


def git_commit(paths, message):
    subprocess.run(["git", "add", "--", *[str(p.relative_to(ROOT)) for p in paths]],
                   cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
