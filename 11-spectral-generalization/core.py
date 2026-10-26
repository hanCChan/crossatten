"""S11 scoped imports, immutable evidence, guarded RX/day/cache access."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(1, str(ROOT / "lib"))
sys.path.append(str(ROOT / "09-phase-invariance"))
import numpy as np
import torch
from xa.paper_train import RECIPES, require_committed, sha
from xa.train import stable_seed
from xa.wisig import DATA, RX, load_cell, receivers


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


s09 = module("s11_s09", ROOT / "09-phase-invariance/training.py")
prep = json.loads((ROOT / "10-tifs-validation/preparation.json").read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")


def once(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"immutable file differs: {path}")
    else:
        save(path, value)


def identity(job):
    return (f"{job['dataset']}__f{job['fold']}__{job.get('group', 'all')}"
            f"__s{job['seed']}__{job['feature']}__{job['phase']}")


def manifest(paths):
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def check_manifest(value):
    for path, expected in value.items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"frozen evidence changed: {path}")


def validate(job):
    require_committed(HERE / "registration.json")
    reg = json.loads((HERE / "registration.json").read_text())
    check_manifest(reg["code_manifest"])
    require_committed(HERE / f"jobs_{job['phase']}.json")
    jobs = json.loads((HERE / f"jobs_{job['phase']}.json").read_text())
    if job not in jobs or len(jobs) > reg["caps"][job["phase"]] or job["id"] != identity(job):
        raise ValueError("unregistered fit")
    for decision in (["decision_physics.json"] if job["phase"] == "source" else
                     ["decision_source.json"] if job["phase"] == "rx_day" else
                     ["decision_final.json"] if job["phase"] == "final" else
                     ["decision_ablation.json"] if job["phase"] == "ablation" else []):
        require_committed(HERE / decision)
        value = json.loads((HERE / decision).read_text())
        if not value["admitted"] or job not in value["jobs"]:
            raise ValueError("fit is not admitted")
        check_manifest(value["evidence"])
    return reg


def read(job, names, days, role, task=None):
    source, target = receivers(job["dataset"], job["fold"])
    if role in ("fit", "held", "diagnostic"):
        if not set(names) <= set(source) or not set(days) <= {1, 2}:
            raise ValueError("source requested target RX/date")
        if job["phase"] == "rx_day":
            expected = (1,) if role == "fit" else (2,)
            if tuple(days) != expected:
                raise ValueError("source CRD role/day violation")
    elif role == "target":
        validate(job)
        if job["phase"] not in ("final", "ablation") or not set(names) <= set(target):
            raise ValueError("unadmitted target read")
        if (task, tuple(days)) not in (("CR", (1, 2)), ("CRD", (3, 4))):
            raise ValueError("target dates do not match task")
        path = HERE / "runs" / job["phase"] / job["id"] / "metrics.json"
        require_committed(path)
        metric = json.loads(path.read_text())
        if sha(path.parent / "model.pt") != metric["weights_sha256"]["model.pt"]:
            raise ValueError("target checkpoint seal changed")
    else:
        raise ValueError(role)
    result = {}
    with (HERE / "access.jsonl").open("a") as log:
        for rx in names:
            xs, ys, dates = [], [], []
            for day in days:
                path = DATA / "cache" / job["dataset"] / f"rx_{rx}_day{day}.npz"
                if sha(path) != prep["cache_sha256"][f"{job['dataset']}/{path.name}"]:
                    raise ValueError("input cache differs from S10")
                x, y = load_cell(job["dataset"], rx, day)
                xs.append(x)
                ys.append(y)
                dates.append(np.full(len(y), day, np.int64))
                log.write(json.dumps(dict(run=job["id"], phase=job["phase"], role=role,
                    dataset=job["dataset"], fold=job["fold"], rx=rx, day=day, task=task)) + "\n")
            result[rx] = dict(x=np.concatenate(xs), y=np.concatenate(ys), day=np.concatenate(dates))
    return result


def pool(cells):
    return dict(x=np.concatenate([c["x"] for c in cells.values()]),
                y=np.concatenate([c["y"] for c in cells.values()]),
                day=np.concatenate([c["day"] for c in cells.values()]),
                rx=np.concatenate([np.full(len(c["y"]), rx) for rx, c in cells.items()]))


def partition(job, smoke=False):
    from sklearn.model_selection import train_test_split
    source, target = receivers(job["dataset"], job["fold"])
    grouped = job["phase"] in ("source", "restore", "rx_day", "smoke", "audit")
    fit_rx, held_rx = s09.s08.source_groups(job["dataset"], job["fold"], job["group"]) if grouped else (source, [])
    days = (1,) if job["phase"] == "rx_day" else (1, 2)
    data = pool(read(job, fit_rx, days, "fit"))
    tr, va = train_test_split(np.arange(len(data["y"])), test_size=.3, random_state=job["seed"])
    if smoke:
        tr, va = tr[:256], va[:128]
    splits = {key: {k: np.ascontiguousarray(v[ids]) for k, v in data.items()}
              for key, ids in (("train", tr), ("val", va))}
    held = read(job, held_rx, (2,) if job["phase"] == "rx_day" else (1, 2), "held") if held_rx else {}
    if smoke:
        held = {rx: {k: v[:64] for k, v in cell.items()} for rx, cell in held.items()}
    roles = dict(source=source, target=target, fit_rx=fit_rx, held_rx=held_rx, days=list(days))
    seals = dict(train=s09.array_sha(tr), val=s09.array_sha(va), pool_y=s09.array_sha(data["y"]))
    return splits, held, roles, seals
