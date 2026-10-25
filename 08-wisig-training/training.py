"""Independent S08 training; never change the frozen shared S00-S07 library."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import subprocess
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
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn, update_bn

from xa.evaluate import predict, summarize
from xa.explore_models import ExploreRF, HierarchicalTokens, frequency_views
from xa.frontends import awgn, rms_normalize
from xa.models import n_params
from xa.paper_train import RECIPES, flatten, metrics, require_committed, sha
from xa.train import stable_seed
from xa.wisig import load_cell, receivers

RULES = ("raw_packet", "ema_packet", "raw_held", "ema_held")


def write_once(path, value):
    encoded = json.dumps(value, indent=1) + "\n"
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"frozen file differs: {path.name}")
    else:
        path.write_text(encoded)


def source_groups(dataset, fold, group):
    source, target = receivers(dataset, fold)
    count = 6 if dataset == "ManyRx" else 2
    held = source[-count:] if group == "last" else source[:count]
    if group not in ("first", "last") or set(held) & set(target):
        raise ValueError("invalid held-source group")
    return [rx for rx in source if rx not in held], held


def guarded_cells(dataset, fold, names, days, role, run, decision=None):
    source, target = receivers(dataset, fold)
    if role in ("fit", "held", "diagnostic"):
        if not set(names) <= set(source) or list(days) != [1, 2]:
            raise ValueError("source read attempted target RX/date")
    elif role == "target":
        if decision is None or not set(names) <= set(target) or list(days) not in ([1, 2], [3, 4]):
            raise ValueError("unfrozen or illegal target read")
        require_committed(HERE / "decision_final.json")
    else:
        raise ValueError("unknown data role")
    cells = {}
    with (HERE / "access.jsonl").open("a") as log:
        for rx in names:
            xs, ys = [], []
            for day in days:
                x, y = load_cell(dataset, rx, day)
                xs.append(x)
                ys.append(y)
                log.write(json.dumps(dict(
                    t=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                    run=run, dataset=dataset, fold=fold, rx=rx, day=day, role=role)) + "\n")
            cells[rx] = dict(x=np.concatenate(xs), y=np.concatenate(ys))
    return cells


def adjacent_spectrum_views(x, null=False, threshold=.05):
    base = frequency_views(x)
    z = torch.complex(base[:, 0], base[:, 1])
    amplitude = z.abs()
    product = z[:, 1:] * z[:, :-1].conj()
    reliable = (amplitude[:, 1:] >= threshold) & (amplitude[:, :-1] >= threshold)
    product = product / (amplitude[:, 1:] * amplitude[:, :-1]).clamp_min(1e-6)
    extra = F.pad(torch.stack((product.real, product.imag), 1) * reliable[:, None], (1, 0))
    if null:
        extra = torch.zeros_like(extra)
    return torch.cat((base, extra), 1)


class SpectrumRF(ExploreRF):
    def __init__(self, classes, feature="frequency"):
        if feature not in ("frequency", "phase", "phase_null"):
            raise ValueError(feature)
        super().__init__(dict(frontend="frequency", num_classes=classes))
        self.feature = feature
        if feature != "frequency":
            self.frequency = HierarchicalTokens(5)

    def tokens(self, x):
        x = rms_normalize(x)
        views = frequency_views(x) if self.feature == "frequency" else \
            adjacent_spectrum_views(x, null=self.feature == "phase_null")
        return self.frequency(views)


def build_model(dataset, feature="frequency"):
    return SpectrumRF(10 if dataset == "ManyRx" else 6, feature)


def recipe_for(condition):
    recipe = dict(RECIPES["official"])
    if condition not in ("official", "official_awgn", "official_awgn_full200"):
        raise ValueError("unregistered condition")
    if condition != "official":
        recipe["awgn"] = [10, 50]
    recipe["stop"] = condition != "official_awgn_full200"
    return recipe


def recalibrate_bn(model, x, batch=128):
    # All current RF-HSTU dropout probabilities are zero. Only fit-training
    # inputs enter BN calibration; neither packet validation nor held RX do.
    update_bn((x[i:i + batch] for i in range(0, len(x), batch)), model)
    model.eval()


def evaluate(model, cells, classes, out=None, label=None):
    per, saved = {}, {}
    for rx, cell in cells.items():
        logits = predict(model, cell["x"], 128).cpu().numpy().astype(np.float32)
        y = cell["y"].cpu().numpy()
        if not np.isfinite(logits).all():
            raise ValueError("nonfinite predictions")
        per[rx] = metrics(logits, y, classes)
        saved[rx], saved[f"y_{rx}"] = logits, y
    summary = summarize(per)
    summary["ce_mean"] = float(np.mean([r["ce"] for r in per.values()]))
    summary["packet_acc"] = sum(r["correct"] for r in per.values()) / sum(r["total"] for r in per.values())
    if out:
        np.savez_compressed(out / f"preds_{label}.npz", **saved)
    return dict(summary=summary, per_rx=per)


def on_device(cells):
    return {rx: {k: torch.from_numpy(np.ascontiguousarray(v)).cuda() for k, v in cell.items()}
            for rx, cell in cells.items()}


def job_id(job):
    return (f"{job['dataset']}_{job.get('variant', job['condition'])}_{job.get('feature', 'frequency')}"
            f"__f{job['fold']}__{job.get('group', 'all')}__s{job['seed']}")


def validate_job(job):
    registration = json.loads((HERE / "registration.json").read_text())
    require_committed(HERE / "registration.json")
    for name, value in registration["code_manifest"].items():
        if sha(ROOT / name) != value:
            raise ValueError(f"registered code changed: {name}")
    phase = job["phase"]
    if phase not in ("screen", "phase", "manyrx", "final", "smoke"):
        raise ValueError("unknown phase")
    jobs_path = HERE / f"jobs_{phase}.json"
    require_committed(jobs_path)
    jobs = json.loads(jobs_path.read_text())
    if job not in jobs or len(jobs) > registration["caps"][phase]:
        raise ValueError("unregistered job or exceeded phase cap")
    if job["id"] != job_id(job) or job["fold"] not in range(4):
        raise ValueError("invalid run identity")
    if phase != "smoke" and job["seed"] not in (
            registration["final_seeds"] if phase == "final" else registration["source_seeds"]):
        raise ValueError("invalid formal seed")
    decision = None
    if phase == "final":
        decision = json.loads((HERE / "decision_final.json").read_text())
        require_committed(HERE / "decision_final.json")
        if job not in decision["jobs"] or decision["registration_sha256"] != sha(HERE / "registration.json"):
            raise ValueError("unfrozen final job")
        for path, expected in decision["evidence"].items():
            require_committed(HERE / path)
            if sha(HERE / path) != expected:
                raise ValueError("changed source evidence")
    return registration, decision


def fit(job):
    registration, decision = validate_job(job)
    out = HERE / "runs" / job["phase"] / job["id"]
    if (out / "metrics.json").exists():
        raise ValueError("cannot overwrite completed run")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    seed = stable_seed("S08-paired", job["dataset"], job["fold"], job.get("group", "all"), job["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    source, target = receivers(job["dataset"], job["fold"])
    final, smoke = job["phase"] == "final", job["phase"] == "smoke"
    fit_rx, held_rx = (source, []) if final else source_groups(job["dataset"], job["fold"], job["group"])
    pool = flatten(guarded_cells(job["dataset"], job["fold"], fit_rx, (1, 2), "fit", job["id"]))
    train_ids, val_ids = train_test_split(np.arange(len(pool["y"])), test_size=.3, random_state=job["seed"])
    if smoke:
        train_ids, val_ids = train_ids[:256], val_ids[:128]
    tr, va = [{k: torch.from_numpy(np.ascontiguousarray(v[ids])).cuda() for k, v in pool.items()}
              for ids in (train_ids, val_ids)]
    held = on_device(guarded_cells(job["dataset"], job["fold"], held_rx, (1, 2), "held",
                                  job["id"])) if held_rx else {}
    if smoke:
        held = {rx: {k: v[:64] for k, v in cell.items()} for rx, cell in held.items()}
    model = build_model(job["dataset"], job.get("feature", "frequency")).cuda()
    params = n_params(model)
    if params != registration["expected_params"][f"{job['dataset']}:{job.get('feature', 'frequency')}"]:
        raise ValueError("unexpected model size")
    classes = 10 if job["dataset"] == "ManyRx" else 6
    recipe = recipe_for(job["condition"])
    batch = recipe["batch"]
    steps = math.ceil(len(tr["y"]) / batch)
    decay = 2 ** (-1 / steps)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=False)
    ema.update_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe["lr"], weight_decay=recipe["wd"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.1, patience=5, min_lr=1e-6)
    order = torch.Generator(device="cuda").manual_seed(seed)
    noise = torch.Generator(device="cuda").manual_seed(seed + 1)
    config = dict(job=job, recipe=recipe, data_seed=seed, params=params,
                  roles=dict(source=source, fit_rx=fit_rx, held_rx=held_rx, target=target, source_days=[1, 2]),
                  train_packets=len(tr["y"]), val_packets=len(va["y"]),
                  held_packets={rx: len(cell["y"]) for rx, cell in held.items()},
                  ema=dict(half_life_epochs=1, decay_per_step=decay, average_buffers=False,
                           bn_calibration="every epoch; clean fit-training subset only"),
                  registration_sha256=sha(HERE / "registration.json"),
                  decision_sha256=sha(HERE / "decision_final.json") if final else None,
                  dataset_manifest_sha256=sha(ROOT / f"05-wisig/out/05-1-{job['dataset']}-manifest.json"),
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(0), smoke=smoke)
    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    best = {name: dict(ce=float("inf"), epoch=0) for name in RULES}
    states, history, wait = {}, [], 0
    epochs = 2 if smoke else job.get("fixed_epochs", recipe["epochs"])
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(tr["y"]), device="cuda", generator=order)
        loss_sum, correct, seen = 0., 0, 0
        for ids in perm.split(batch):
            x, y = tr["x"][ids], tr["y"][ids]
            if recipe.get("awgn"):
                x = awgn(x, *recipe["awgn"], noise)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ema.update_parameters(model)
            loss_sum += float(loss.detach()) * len(y)
            correct += int((logits.argmax(1) == y).sum())
            seen += len(y)
        recalibrate_bn(ema.module, tr["x"], batch)
        row = dict(epoch=epoch, train_ce=loss_sum / seen, train_acc=correct / seen,
                   lr=optimizer.param_groups[0]["lr"])
        improved_raw = False
        for name, network in (("raw", model), ("ema", ema.module)):
            lv = predict(network, va["x"], batch)
            ce = float(F.cross_entropy(lv, va["y"]))
            row[f"{name}_packet_ce"] = ce
            row[f"{name}_packet_acc"] = float((lv.argmax(1) == va["y"]).float().mean())
            values = {"packet": ce}
            if held:
                entry = evaluate(network, held, classes)
                values["held"] = entry["summary"]["ce_mean"]
                row[f"{name}_held_ce"] = values["held"]
                row[f"{name}_held_acc"] = entry["summary"]["acc_mean"]
                row[f"{name}_held_worst"] = entry["summary"]["worst_acc"]
            for criterion, value in values.items():
                rule = f"{name}_{criterion}"
                if not math.isfinite(value):
                    raise ValueError("nonfinite source validation")
                if value < best[rule]["ce"]:
                    best[rule] = dict(ce=value, epoch=epoch)
                    states[rule] = copy.deepcopy(network.state_dict())
                    improved_raw |= rule == "raw_packet"
        wait = 0 if improved_raw else wait + 1
        scheduler.step(row["raw_packet_ce"])
        history.append(row)
        with (out / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator="\n")
            writer.writeheader()
            writer.writerows(history)
        if epoch == 1 or epoch % 25 == 0:
            print(json.dumps(dict(run=job["id"], epoch=epoch,
                                  raw_ce=round(row["raw_packet_ce"], 5),
                                  ema_ce=round(row["ema_packet_ce"], 5),
                                  seconds=round(time.time() - start))), flush=True)
        if recipe["stop"] and "fixed_epochs" not in job and not smoke and wait >= recipe["early_stop"]:
            break
    entries, checkpoints = {}, {}
    if not final:
        for rule, state in states.items():
            model.load_state_dict(state)
            weight = out / f"model_{rule}.pt"
            torch.save(state, weight)
            entry = evaluate(model, held, classes, out, rule)
            lv = predict(model, va["x"], batch)
            entry["packet_ce"] = float(F.cross_entropy(lv, va["y"]))
            entry["packet_acc"] = float((lv.argmax(1) == va["y"]).float().mean())
            entry["epoch"] = best[rule]["epoch"]
            entries[rule] = entry
            checkpoints[weight.name] = sha(weight)
    else:
        rule = job["rule"]
        if rule.endswith("held"):
            network = ema.module if rule.startswith("ema") else model
            state, selected_epoch = network.state_dict(), epoch
        else:
            state, selected_epoch = states[rule], best[rule]["epoch"]
        model.load_state_dict(state)
        torch.save(model.state_dict(), out / "model.pt")
        checkpoints["model.pt"] = sha(out / "model.pt")
        lv = predict(model, va["x"], batch)
        entries["source_packet"] = dict(ce=float(F.cross_entropy(lv, va["y"])),
                                      acc=float((lv.argmax(1) == va["y"]).float().mean()),
                                      epoch=selected_epoch)
        for task, days in (("CR", (1, 2)), ("CRD", (3, 4))):
            cells = on_device(guarded_cells(job["dataset"], job["fold"], target, days, "target",
                                           job["id"], decision))
            entries[task] = evaluate(model, cells, classes, out, task)
    result = dict(job=job, params=params, stop_epoch=epoch, best=best, selected=entries,
                  seconds=round(time.time() - start, 3), weights_sha256=checkpoints,
                  registration_sha256=config["registration_sha256"],
                  predictions_sha256={p.name: sha(p) for p in sorted(out.glob("preds_*.npz"))},
                  smoke=smoke)
    (out / "metrics.json").write_text(json.dumps(result, indent=1) + "\n")
    return result
