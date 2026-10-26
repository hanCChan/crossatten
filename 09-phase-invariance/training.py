"""S09 matched initialization, fixed-budget training and guarded evaluation."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from phase_features import phase_features_from_base
from xa.explore_models import ExploreRF, HierarchicalTokens, frequency_views
from xa.frontends import rms_normalize
from xa.models import n_params
from xa.paper_train import RECIPES, flatten, require_committed, sha
from xa.train import stable_seed
from xa.wisig import receivers, load_cell

spec = importlib.util.spec_from_file_location("frozen_s08", ROOT / "08-wisig-training/training.py")
s08 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s08)
FEATURES = ("frequency", "null", "first", "centered", "curvature")
RULES = ("raw_packet", "ema_packet", "raw_held", "ema_held")
BASELINE = "c0aceeb703de6d1a08b1b5b0fc14b1200062f0a8"


def save(path, value):
    path.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")


def write_once(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"frozen file changed: {path}")
    else:
        save(path, value)


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class PhaseRF(ExploreRF):
    def __init__(self, classes, feature):
        if feature not in FEATURES:
            raise ValueError(feature)
        super().__init__(dict(frontend="frequency", num_classes=classes))
        self.feature = feature
        if feature != "frequency":
            self.frequency = HierarchicalTokens(5)

    def tokens(self, x):
        return self.frequency(phase_features_from_base(
            frequency_views(rms_normalize(x)), self.feature))


def paired_model(dataset, feature, seed):
    classes = 10 if dataset == "ManyRx" else 6
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        canonical = PhaseRF(classes, "first")
        model = PhaseRF(classes, feature)
    state = canonical.state_dict()
    common = {name: value.clone() for name, value in state.items()}
    common["frequency.stem.0.weight"] = common["frequency.stem.0.weight"][:, :3].clone()
    model.load_state_dict(common if feature == "frequency" else state)
    return model, state_sha(common)


def identity(job):
    return (f"{job['dataset']}__f{job['fold']}__{job.get('group', 'all')}"
            f"__s{job['seed']}__{job['feature']}__{job['policy']}")


def validate(job):
    reg = json.loads((HERE / "registration.json").read_text())
    require_committed(HERE / "registration.json")
    for path, expected in reg["code_manifest"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"frozen code changed: {path}")
    phase = job["phase"]
    path = HERE / f"jobs_{phase}.json"
    require_committed(path)
    jobs = json.loads(path.read_text())
    if job not in jobs or job["id"] != identity(job) or len(jobs) > reg["caps"][phase]:
        raise ValueError("unregistered fit")
    if phase == "final":
        decision = json.loads((HERE / "decision_final.json").read_text())
        require_committed(HERE / "decision_final.json")
        if job not in decision["jobs"]:
            raise ValueError("final fit is not admitted")
    return reg


def read_cells(job, names, days, role, task=None):
    source, target = receivers(job["dataset"], job["fold"])
    if role in ("fit", "held", "diagnostic"):
        if not set(names) <= set(source) or tuple(days) != (1, 2):
            raise ValueError("source read requested target data")
    elif role == "target":
        decision_path = HERE / "decision_final.json"
        require_committed(decision_path)
        decision = json.loads(decision_path.read_text())
        if job not in decision["target_jobs"] or not set(names) <= set(target):
            raise ValueError("target read not in frozen whitelist")
        if (task, tuple(days)) not in (("CR", (1, 2)), ("CRD", (3, 4))):
            raise ValueError("invalid target dates")
        for path, expected in decision["evidence"].items():
            if sha(HERE / path) != expected:
                raise ValueError("source evidence changed after decision")
    else:
        raise ValueError(role)
    cells = {}
    with (HERE / "access.jsonl").open("a") as stream:
        for rx in names:
            xs, ys = [], []
            for day in days:
                x, y = load_cell(job["dataset"], rx, day)
                xs.append(x)
                ys.append(y)
                stream.write(json.dumps(dict(t=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                    run=job["id"], phase=job["phase"], dataset=job["dataset"], fold=job["fold"],
                    rx=rx, day=day, role=role, task=task)) + "\n")
            cells[rx] = dict(x=np.concatenate(xs), y=np.concatenate(ys))
    return cells


def load_data(job):
    source, target = receivers(job["dataset"], job["fold"])
    source_fit = job["phase"] in ("source", "smoke")
    fit_rx, held_rx = s08.source_groups(job["dataset"], job["fold"], job["group"]) \
        if source_fit else (source, [])
    pool = flatten(read_cells(job, fit_rx, (1, 2), "fit"))
    tr_ids, va_ids = train_test_split(np.arange(len(pool["y"])), test_size=.3,
                                     random_state=job["seed"])
    if job["phase"] == "smoke":
        tr_ids, va_ids = tr_ids[:256], va_ids[:128]
    tr, va = [{k: torch.from_numpy(np.ascontiguousarray(v[ids])).cuda()
               for k, v in pool.items() if k != "rx"} for ids in (tr_ids, va_ids)]
    held = s08.on_device(read_cells(job, held_rx, (1, 2), "held")) if held_rx else {}
    if job["phase"] == "smoke":
        held = {rx: {k: v[:64] for k, v in cell.items()} for rx, cell in held.items()}
    roles = dict(source=source, target=target, fit_rx=fit_rx, held_rx=held_rx, days=[1, 2])
    splits = dict(train=array_sha(tr_ids), val=array_sha(va_ids), pool_y=array_sha(pool["y"]))
    return tr, va, held, roles, splits


def fit(job):
    reg = validate(job)
    out = HERE / "runs" / job["phase"] / job["id"]
    if (out / "metrics.json").exists():
        raise ValueError("cannot overwrite complete fit")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    seed = stable_seed("S09-paired", job["dataset"], job["fold"],
                       job.get("group", "all"), job["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tr, va, held, roles, splits = load_data(job)
    model, common_sha = paired_model(job["dataset"], job["feature"], seed)
    model = model.cuda()
    params = n_params(model)
    if params != reg["expected_params"][f"{job['dataset']}:{job['feature']}"]:
        raise ValueError("model parameter count differs")
    recipe = dict(RECIPES["official"])
    batch = recipe["batch"]
    decay = 2 ** (-1 / math.ceil(len(tr["y"]) / batch))
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=False)
    ema.update_parameters(model)
    opt = torch.optim.Adam(model.parameters(), lr=recipe["lr"], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.1, patience=5, min_lr=1e-6)
    order = torch.Generator(device="cuda").manual_seed(seed)
    config = dict(job=job, params=params, recipe=recipe, data_seed=seed, roles=roles,
                  split_sha256=splits, shared_initialization_sha256=common_sha,
                  train_packets=len(tr["y"]), val_packets=len(va["y"]),
                  held_packets={rx: len(c["y"]) for rx, c in held.items()},
                  ema=dict(half_life_epochs=1, decay=decay, buffers="clean fit-only BN recalibration"),
                  registration_sha256=sha(HERE / "registration.json"), torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(0))
    save(out / "config.json", config)
    best = {r: dict(ce=float("inf"), epoch=0) for r in RULES}
    states, history, wait = {}, [], 0
    epochs = job.get("epochs", 200)
    started = time.time()
    classes = 10 if job["dataset"] == "ManyRx" else 6
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(tr["y"]), generator=order, device="cuda")
        ce_sum, correct = 0., 0
        for ids in perm.split(batch):
            logits = model(tr["x"][ids])
            loss = F.cross_entropy(logits, tr["y"][ids])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema.update_parameters(model)
            ce_sum += float(loss.detach()) * len(ids)
            correct += int((logits.argmax(1) == tr["y"][ids]).sum())
        s08.recalibrate_bn(ema.module, tr["x"], batch)
        row = dict(epoch=epoch, train_ce=ce_sum / len(tr["y"]), train_acc=correct / len(tr["y"]),
                   lr=opt.param_groups[0]["lr"])
        improved = False
        for name, network in (("raw", model), ("ema", ema.module)):
            lv = s08.predict(network, va["x"], batch)
            ce = float(F.cross_entropy(lv, va["y"]))
            row[f"{name}_packet_ce"] = ce
            row[f"{name}_packet_acc"] = float((lv.argmax(1) == va["y"]).float().mean())
            values = {"packet": ce}
            if held:
                result = s08.evaluate(network, held, classes)
                values["held"] = result["summary"]["ce_mean"]
                row[f"{name}_held_ce"] = values["held"]
                row[f"{name}_held_acc"] = result["summary"]["acc_mean"]
                row[f"{name}_held_worst"] = result["summary"]["worst_acc"]
            for criterion, value in values.items():
                if not math.isfinite(value):
                    raise ValueError("nonfinite source CE")
                rule = f"{name}_{criterion}"
                if value < best[rule]["ce"]:
                    best[rule] = dict(ce=value, epoch=epoch)
                    states[rule] = copy.deepcopy(network.state_dict())
                    improved |= rule == "raw_packet"
        wait = 0 if improved else wait + 1
        scheduler.step(row["raw_packet_ce"])
        history.append(row)
        if job["policy"] == "packet" and wait >= recipe["early_stop"]:
            break
    import csv
    with (out / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    selected, weights = {}, {}
    if held:
        for rule, state in states.items():
            model.load_state_dict(state)
            weight = out / f"model_{rule}.pt"
            torch.save(state, weight)
            weights[weight.name] = sha(weight)
            entry = s08.evaluate(model, held, classes, out, rule)
            entry["epoch"] = best[rule]["epoch"]
            selected[rule] = entry
    else:
        state = states["raw_packet"] if job["policy"] == "packet" else \
            (ema.module if job["rule"] == "ema" else model).state_dict()
        selected_epoch = best["raw_packet"]["epoch"] if job["policy"] == "packet" else epoch
        model.load_state_dict(state)
        torch.save(model.state_dict(), out / "model.pt")
        weights["model.pt"] = sha(out / "model.pt")
        lv = s08.predict(model, va["x"], batch)
        selected["source_packet"] = dict(epoch=selected_epoch, ce=float(F.cross_entropy(lv, va["y"])),
                                        acc=float((lv.argmax(1) == va["y"]).float().mean()))
    result = dict(job=job, params=params, stop_epoch=epoch, best={k: v for k, v in best.items()
                       if math.isfinite(v["ce"])}, selected=selected,
                  seconds=round(time.time() - started, 3), weights_sha256=weights,
                  config_sha256=sha(out / "config.json"), registration_sha256=sha(HERE / "registration.json"),
                  predictions_sha256={p.name: sha(p) for p in out.glob("preds_*.npz")})
    save(out / "metrics.json", result)
    print(json.dumps(dict(run=job["id"], stop_epoch=epoch, seconds=result["seconds"])), flush=True)


def evaluate_target(job):
    validate(job)
    out = HERE / "runs" / job["phase"] / job["id"]
    if (out / "target_metrics.json").exists():
        raise ValueError("target evaluation already completed")
    result = json.loads((out / "metrics.json").read_text())
    weight = out / "model.pt"
    if sha(weight) != result["weights_sha256"]["model.pt"]:
        raise ValueError("checkpoint changed")
    model, _ = paired_model(job["dataset"], job["feature"], 0)
    model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
    model = model.cuda().eval()
    torch.set_num_threads(2)
    source, target = receivers(job["dataset"], job["fold"])
    classes = 10 if job["dataset"] == "ManyRx" else 6
    entries = {}
    for task, days in (("CR", (1, 2)), ("CRD", (3, 4))):
        cells = s08.on_device(read_cells(job, target, days, "target", task))
        entries[task] = s08.evaluate(model, cells, classes, out, task)
    save(out / "target_metrics.json", dict(job=job, selected=entries,
        decision_sha256=sha(HERE / "decision_final.json"),
        predictions_sha256={p.name: sha(p) for p in out.glob("preds_*.npz")}))
