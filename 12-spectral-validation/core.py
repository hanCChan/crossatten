"""S12 source-only access; historical code and evidence remain untouched."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


base = module("s12_frozen_core", ROOT / "11-spectral-generalization/core.py")
s09, sha, save, once = base.s09, base.sha, base.save, base.once
manifest, check_manifest = base.manifest, base.check_manifest
require_committed, receivers = base.require_committed, base.receivers


def identity(job):
    return base.identity(job)


def validate(job):
    require_committed(HERE / "registration.json")
    reg = json.loads((HERE / "registration.json").read_text())
    check_manifest(reg["code_manifest"])
    phase = job["phase"]
    if phase not in ("smoke", "pilot", "confirmation"):
        raise ValueError("target stages are not admitted")
    path = HERE / f"jobs_{phase}.json"
    require_committed(path)
    jobs = json.loads(path.read_text())
    cap = (reg["caps"]["confirmation_candidates"] + reg["caps"]["confirmation_fresh_hard"]
           if phase == "confirmation" else reg["caps"][phase])
    if job not in jobs or len(jobs) > cap or job["id"] != identity(job):
        raise ValueError("unregistered S12 fit")
    if phase == "confirmation":
        decision_path = HERE / "decision_pilot.json"
        require_committed(decision_path)
        decision = json.loads(decision_path.read_text())
        if not decision["admitted"] or job not in decision["jobs"]:
            raise ValueError("confirmation not admitted by source decision")
        check_manifest(decision["evidence"])
    return reg


def read(job, names, days, role, task=None):
    if job["phase"] not in ("smoke", "pilot", "diagnostic", "confirmation"):
        raise ValueError("unadmitted S12 phase")
    source, _ = receivers(job["dataset"], job["fold"])
    if role not in ("fit", "held", "diagnostic") or task is not None:
        raise ValueError("S12-A/B forbid target reads")
    if not set(names) <= set(source) or tuple(days) != (1, 2):
        raise ValueError("source requested target RX/date")
    result = {}
    with (HERE / "access.jsonl").open("a") as log:
        for rx in names:
            cells = []
            for day in days:
                path = base.DATA / "cache" / job["dataset"] / f"rx_{rx}_day{day}.npz"
                if sha(path) != base.prep["cache_sha256"][f"{job['dataset']}/{path.name}"]:
                    raise ValueError("source cache differs from the frozen preparation")
                x, y = base.load_cell(job["dataset"], rx, day)
                cells.append(dict(x=x, y=y, day=base.np.full(len(y), day, base.np.int64)))
                log.write(json.dumps(dict(run=job["id"], phase=job["phase"], role=role,
                                         dataset=job["dataset"], fold=job["fold"],
                                         rx=rx, day=day)) + "\n")
            result[rx] = {key: base.np.concatenate([c[key] for c in cells])
                          for key in ("x", "y", "day")}
    return result


def partition(job):
    from sklearn.model_selection import train_test_split
    source, target = receivers(job["dataset"], job["fold"])
    fit_rx, held_rx = s09.s08.source_groups(job["dataset"], job["fold"], job["group"])
    data = base.pool(read(job, fit_rx, (1, 2), "fit"))
    tr, va = train_test_split(base.np.arange(len(data["y"])), test_size=.3,
                             random_state=job["seed"])
    splits = {key: {k: base.np.ascontiguousarray(v[ids]) for k, v in data.items()}
              for key, ids in (("train", tr), ("val", va))}
    held = read(job, held_rx, (1, 2), "held")
    roles = dict(source=source, target=target, fit_rx=fit_rx, held_rx=held_rx, days=[1, 2])
    seals = dict(train=s09.array_sha(tr), val=s09.array_sha(va),
                 pool_y=s09.array_sha(data["y"]))
    return splits, held, roles, seals


def stratified_ids(cell, budget, seed):
    """Cover every observed TX/RX/day cell before round-robin allocation."""
    np = base.np
    keys = list(zip(cell["y"].tolist(), cell.get("rx", np.zeros(len(cell["y"]))).tolist(),
                    cell["day"].tolist()))
    groups = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    if budget < len(groups):
        raise ValueError("smoke budget cannot cover all observed TX/RX/day cells")
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(ids).tolist() for ids in groups.values()]
    selected = []
    for offset in range(max(map(len, groups), default=0)):
        for ids in groups:
            if offset < len(ids):
                selected.append(ids[offset])
                if len(selected) == budget:
                    return np.array(selected, dtype=np.int64)
    return np.array(selected, dtype=np.int64)


def smoke_subset(splits, held, seed):
    np = base.np
    def take(cell, budget):
        ids = stratified_ids(cell, budget, seed)
        return {k: np.ascontiguousarray(v[ids]) for k, v in cell.items()}
    return ({key: take(cell, 512) for key, cell in splits.items()},
            {rx: take(cell, 64) for rx, cell in held.items()})
