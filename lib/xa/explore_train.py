"""Bounded Stage-07 source selection and frozen final evaluation."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn import functional as F

from . import ROOT
from .evaluate import predict, summarize
from .explore_models import build_explore_model, physical_augmentation
from .frontends import awgn
from .models import n_params
from .paper_train import RECIPES, flatten, metrics, require_committed, sha
from .train import stable_seed
from .wisig import load_cell, receivers

HERE = ROOT / "07-wifi-osu"


def evaluate(model, cells, classes, out=None, label=None):
    per, saved = {}, {}
    for rx, data in cells.items():
        x = torch.from_numpy(np.ascontiguousarray(data["x"])).cuda()
        logits = predict(model, x, batch=128).float().cpu().numpy()
        if not np.isfinite(logits).all():
            raise ValueError("nonfinite evaluation logits")
        per[rx] = metrics(logits, data["y"], classes)
        saved[rx], saved[f"y_{rx}"] = logits, data["y"]
        del x
    summary = summarize(per)
    summary["packet_acc"] = sum(v["correct"] for v in per.values()) / sum(v["total"] for v in per.values())
    summary["ce_mean"] = float(np.mean([v["ce"] for v in per.values()]))
    if out is not None:
        np.savez_compressed(Path(out) / f"preds_{label}.npz", **saved)
    return dict(summary=summary, per_rx=per)


def source_cells(dataset, names, days, purpose, fold, run, decision=None):
    source, target = receivers(dataset, fold)
    if purpose in ("fit_source", "held_source"):
        if not set(names) <= set(source) or tuple(days) != (1, 2):
            raise ValueError("source read attempted target RX/day")
    elif purpose == "target_report":
        if decision is None or not set(names) <= set(target) or tuple(days) not in ((1, 2), (3, 4)):
            raise ValueError("target read without frozen decision")
    else:
        raise ValueError(purpose)
    result = {}
    with (HERE / "access.jsonl").open("a") as log:
        for rx in names:
            xs, ys = [], []
            for day in days:
                x, y = load_cell(dataset, rx, day)
                xs.append(x)
                ys.append(y)
                log.write(json.dumps(dict(t=time.strftime("%FT%T"), dataset=dataset, fold=fold,
                                          rx=rx, day=day, purpose=purpose, run=run)) + "\n")
            result[rx] = dict(x=np.concatenate(xs), y=np.concatenate(ys))
    return result


def source_data(dataset, fold, seed, final, run):
    source, target = receivers(dataset, fold)
    count = 6 if dataset == "ManyRx" else 2
    held = [] if final else source[-count:]
    fit_rx = source if final else source[:-count]
    data = flatten(source_cells(dataset, fit_rx, (1, 2), "fit_source", fold, run))
    train, val = train_test_split(np.arange(len(data["y"])), test_size=.3, random_state=seed)
    splits = [{k: v[ids] for k, v in data.items()} for ids in (train, val)]
    held_cells = source_cells(dataset, held, (1, 2), "held_source", fold, run) if held else {}
    return *splits, held_cells, dict(source=source, fit_rx=fit_rx, held_rx=held, target=target)


def validate_job(job):
    registered = json.loads((HERE / "registration.json").read_text())
    require_committed(HERE / "registration.json")
    for relative, expected in registered["wifi_code_manifest"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"WiSig implementation changed: {relative}")
    if sha(HERE / "source_signal_audit.json") != registered["source_audit_sha256"]:
        raise ValueError("source audit changed after registration")
    if job["dataset"] not in ("ManyRx", "ManySig") or job["fold"] not in range(4):
        raise ValueError("invalid dataset/fold")
    allowed = registered["frontends"]
    if job["frontend"] not in allowed:
        raise ValueError("unregistered frontend")
    if job.get("augmentation", "none") not in registered["augmentations"]:
        raise ValueError("unregistered augmentation")
    if job.get("recipe", "unified") not in ("official", "unified"):
        raise ValueError("unregistered recipe")
    if job.get("fusion", "pooled_concat") not in ("pooled_concat", "gate"):
        raise ValueError("unregistered fusion")
    if job.get("noise", "default") not in ("default", "mild"):
        raise ValueError("unregistered noise")
    if job.get("checkpoint", "packet") not in ("packet", "held_epoch", "last"):
        raise ValueError("unregistered checkpoint")
    if not 1 <= job.get("epochs", 200) <= 200:
        raise ValueError("unregistered epoch budget")
    if job["phase"] == "final":
        decision_path = HERE / "decision_final.json"
        decision = json.loads(decision_path.read_text())
        require_committed(decision_path)
        if decision["registration_sha256"] != sha(HERE / "registration.json"):
            raise ValueError("registration changed after freeze")
        if job not in decision["jobs"]:
            raise ValueError("final job not allowed")
        if job["seed"] not in registered["final_seeds"] or len(decision["jobs"]) > 72:
            raise ValueError("final seed/job budget violation")
        if job.get("checkpoint") == "held_epoch":
            raise ValueError("final fitting cannot select on held training receivers")
        for relative, value in decision["evidence"].items():
            path = HERE / relative
            if sha(path) != value:
                raise ValueError("source evidence changed")
            require_committed(path)
        return registered, decision
    if job["phase"] not in ("features", "replicate", "training", "training_replicate", "combine", "smoke"):
        raise ValueError("unregistered phase")
    if job["seed"] not in (3101, 3102, 0):
        raise ValueError("unregistered source seed")
    if job["phase"] != "smoke" and job["seed"] == 0:
        raise ValueError("smoke seed used for source")
    canonical = HERE / f"jobs_{job['phase']}.json"
    require_committed(canonical)
    jobs = json.loads(canonical.read_text())
    if len(jobs) > registered["phase_caps"][job["phase"]]:
        raise ValueError("phase job budget exceeded")
    if job not in jobs:
        raise ValueError("source job not in its committed queue")
    if job["frontend"] in ("stf", "diff_stf"):
        audit = json.loads((HERE / "source_signal_audit.json").read_text())
        row = next(r for r in audit if r["dataset"] == job["dataset"] and r["fold"] == job["fold"])
        if row["stf_coherence_median"] < .7:
            raise ValueError("STF source gate failed")
    return registered, None


def code_hash():
    h = hashlib.sha256()
    for path in sorted((ROOT / "lib").rglob("*.py")):
        h.update(path.relative_to(ROOT).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def fit(job, out):
    registered, decision = validate_job(job)
    out = Path(out)
    if out != HERE / "runs" / job["phase"] / job["id"]:
        raise ValueError("noncanonical run directory")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        raise ValueError("completed run must not be overwritten")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    seed = stable_seed("S07-data", job["dataset"], job["fold"], job["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    final = job["phase"] == "final"
    tr, va, held, roles = source_data(job["dataset"], job["fold"], job["seed"], final, job["id"])
    arm = dict(frontend=job["frontend"], num_classes=10 if job["dataset"] == "ManyRx" else 6,
               stf_stop=160, fusion=job.get("fusion", "pooled_concat"))
    model = build_explore_model(arm).cuda()
    expected_key = f"{job['dataset']}:{job['frontend']}:{arm['fusion']}"
    if n_params(model) != registered["expected_params"][expected_key]:
        raise ValueError("model parameter count differs from registration")
    recipe = dict(RECIPES[job.get("recipe", "unified")])
    smoke = job["phase"] == "smoke"
    epochs = 2 if smoke else job.get("epochs", recipe["epochs"])
    if not 1 <= epochs <= 200:
        raise ValueError("budget outside registration")
    if job.get("noise") == "mild":
        recipe["awgn"] = [10, 50]
    optimizer = (torch.optim.AdamW if recipe["opt"] == "adamw" else torch.optim.Adam)(
        model.parameters(), lr=recipe["lr"], weight_decay=recipe["wd"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.1, patience=5, min_lr=1e-6) if recipe["sched"] == "plateau" else None
    device = torch.device("cuda")
    tr = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in tr.items()}
    va = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in va.items()}
    order = torch.Generator(device=device).manual_seed(seed)
    noise = torch.Generator(device=device).manual_seed(seed + 1)
    physical = torch.Generator(device=device).manual_seed(seed + 2)
    batch = recipe["batch"]
    steps = len(tr["y"]) // batch if recipe["opt"] == "adamw" else math.ceil(len(tr["y"]) / batch)
    if job.get("balanced"):
        rx = tr["rx"]
        counts = torch.bincount(rx)
        weights = 1.0 / counts[rx].float()
    else:
        weights = None
    config = dict(job=job, roles=roles, recipe=recipe, arm=arm, data_seed=seed, smoke=smoke,
                  registration_sha256=sha(HERE / "registration.json"),
                  code_sha256=code_hash(), dataset_manifest_sha256=sha(
                      ROOT / f"05-wisig/out/05-1-{job['dataset']}-manifest.json"),
                  decision_sha256=sha(HERE / "decision_final.json") if final else None)
    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    history, best, wait = [], dict(ce=float("inf"), epoch=0), 0
    best_state, held_state, held_best, held_epoch = None, None, float("inf"), 0
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = (torch.multinomial(weights, steps * batch, replacement=True, generator=order)
                if weights is not None else torch.randperm(len(tr["y"]), device=device, generator=order))
        total_loss = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        seen = 0
        for i in range(steps):
            if recipe["sched"] == "warmcos":
                step = (epoch - 1) * steps + i
                warm, horizon = 5 * steps, 200 * steps
                lr = recipe["lr"] * ((step + 1) / warm if step < warm else
                                     .5 * (1 + math.cos(math.pi * (step - warm) / (horizon - warm))))
                for group in optimizer.param_groups:
                    group["lr"] = lr
            ids = perm[i * batch:(i + 1) * batch]
            x, y = tr["x"][ids], tr["y"][ids]
            x = physical_augmentation(x, job.get("augmentation", "none"), physical, registered["bounds"])
            if recipe.get("awgn"):
                x = awgn(x, *recipe["awgn"], noise)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * len(y)
            correct += (logits.argmax(1) == y).sum()
            seen += len(y)
            if smoke and i >= 3:
                break
        lv = predict(model, va["x"], 128)
        ce = float(F.cross_entropy(lv, va["y"]))
        row = dict(epoch=epoch, train_loss=float(total_loss) / seen, train_acc=float(correct) / seen,
                   val_ce=ce, val_acc=float((lv.argmax(1) == va["y"]).float().mean()),
                   lr=optimizer.param_groups[0]["lr"], held_ce="", held_acc="")
        if job.get("checkpoint") == "held_epoch" and held and (epoch == 1 or epoch % 5 == 0):
            entry = evaluate(model, held, arm["num_classes"])
            row["held_ce"] = entry["summary"]["ce_mean"]
            row["held_acc"] = entry["summary"]["acc_mean"]
            if row["held_ce"] < held_best:
                held_best, held_epoch = row["held_ce"], epoch
                held_state = copy.deepcopy(model.state_dict())
        history.append(row)
        with (out / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator="\n")
            writer.writeheader()
            writer.writerows(history)
        if ce < best["ce"]:
            best = dict(ce=ce, epoch=epoch)
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if scheduler:
            scheduler.step(ce)
        if epoch == 1 or epoch % 20 == 0:
            print(json.dumps(dict(run=job["id"], epoch=epoch, source_ce=round(ce, 5),
                                  seconds=round(time.time() - start))), flush=True)
        if scheduler and wait >= 10:
            break
    if job.get("checkpoint") == "last":
        selected_epoch = epoch
    elif job.get("checkpoint") == "held_epoch" and held:
        if held_state is None:
            raise ValueError("missing held-source epoch checkpoint")
        model.load_state_dict(held_state)
        selected_epoch = held_epoch
    else:
        model.load_state_dict(best_state)
        selected_epoch = best["epoch"]
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("nonfinite state; weights not saved")
    torch.save(model.state_dict(), out / "model.pt")
    selected = {}
    if held:
        selected["held_source"] = evaluate(model, held, arm["num_classes"], out, "held_source")
    if final:
        for task, days in (("CR", (1, 2)), ("CRD", (3, 4))):
            cells = source_cells(job["dataset"], roles["target"], days, "target_report",
                                 job["fold"], job["id"], decision)
            selected[task] = evaluate(model, cells, arm["num_classes"], out, task)
    result = dict(job=job, params=n_params(model), stop_epoch=epoch, selected_epoch=selected_epoch,
                  best_packet_epoch=best["epoch"], source_val_ce=best["ce"],
                  source_val_acc=history[best["epoch"] - 1]["val_acc"],
                  held_best_epoch=held_epoch, seconds=round(time.time() - start, 2),
                  selected_source_ce=history[selected_epoch - 1]["val_ce"],
                  selected_source_acc=history[selected_epoch - 1]["val_acc"],
                  selected=selected, code_sha256=config["code_sha256"],
                  registration_sha256=config["registration_sha256"],
                  weights_sha256=sha(out / "model.pt"),
                  predictions_sha256={p.name: sha(p) for p in sorted(out.glob("preds_*.npz"))})
    (out / "metrics.json").write_text(json.dumps(result, indent=1) + "\n")
    return result
