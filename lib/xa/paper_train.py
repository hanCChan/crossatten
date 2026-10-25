"""Source-only model selection and guarded final evaluation for paper validation."""
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
from .data import fold_roles, load_rx, source_split
from .evaluate import predict, summarize
from .frontends import awgn
from .guard import PROTOCOL_PATH
from .models import build_pipeline, n_params
from .paper_models import UPSTREAM, hsic_cca
from .train import code_sha, stable_seed
from .wisig import load_cell, receivers

RECIPES = {
    "official": dict(opt="adam", lr=1e-3, wd=0.0, batch=128, epochs=200,
                     sched="plateau", patience=5, early_stop=10, min_lr=1e-6),
    "unified": dict(opt="adamw", lr=1e-3, wd=5e-4, batch=64, epochs=200,
                    sched="warmcos", warmup=5, awgn=[0, 50]),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(logits, y, classes):
    pred = logits.argmax(1)
    cm = np.bincount(y * classes + pred, minlength=classes ** 2).reshape(classes, classes)
    tp = cm.diagonal().astype(float)
    precision = tp / np.maximum(cm.sum(0), 1)
    recall = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return dict(acc=float((pred == y).mean()), macro_f1=float(f1.mean()),
                ce=float(F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(y))),
                correct=int((pred == y).sum()), total=len(y), confusion=cm.tolist())


def read_wifi(dataset, names, days, purpose, run):
    cells = {}
    log = ROOT / "logs/paper_data_access.jsonl"
    with log.open("a") as f:
        for rx in names:
            xs, ys = [], []
            for day in days:
                x, y = load_cell(dataset, rx, day)
                xs.append(x)
                ys.append(y)
                f.write(json.dumps(dict(t=time.strftime("%FT%T"), dataset=dataset, rx=rx,
                                        day=day, purpose=purpose, run=run)) + "\n")
            cells[rx] = dict(x=np.concatenate(xs), y=np.concatenate(ys))
    return cells


def flatten(cells):
    return dict(x=np.concatenate([v["x"] for v in cells.values()]),
                y=np.concatenate([v["y"] for v in cells.values()]),
                rx=np.concatenate([np.full(len(v["y"]), i) for i, v in enumerate(cells.values())]))


def wifi_source(dataset, fold, seed, phase, run):
    train_rx, test_rx = receivers(dataset, fold)
    internal_n = 6 if dataset == "ManyRx" else 2
    fit_rx = train_rx[:-internal_n] if phase == "source" else train_rx
    held_rx = train_rx[-internal_n:] if phase == "source" else []
    data = flatten(read_wifi(dataset, fit_rx, (1, 2), "source", run))
    tr, va = train_test_split(np.arange(len(data["y"])), test_size=0.3, random_state=seed)
    split = {tag: {k: v[idx] for k, v in data.items()} for tag, idx in (("train", tr), ("val", va))}
    held = read_wifi(dataset, held_rx, (1, 2), "internal_rx_val", run) if held_rx else {}
    roles = dict(source=train_rx, fit_rx=fit_rx, internal_val_rx=held_rx, target=test_rx, source_days=[1, 2])
    return split["train"], split["val"], held, roles


def shen_source(fold, run):
    roles = fold_roles(fold)
    train, val = {}, {}
    for i, rx in enumerate(roles["source"]):
        d = load_rx(rx, "source", run)
        tr, va = source_split(d["y"], d["rows"])
        train[rx] = dict(x=d["iq"][tr], y=d["y"][tr])
        val[rx] = dict(x=d["iq"][va], y=d["y"][va])
    return flatten(train), flatten(val), {}, roles


def evaluate(model, cells, classes, out=None, label=None):
    per, saved = {}, {}
    for rx, data in cells.items():
        x = torch.from_numpy(np.ascontiguousarray(data["x"])).cuda()
        logits = predict(model, x, batch=128).cpu().numpy()
        per[rx] = metrics(logits, data["y"], classes)
        saved[rx] = logits.astype(np.float16)
        saved[f"y_{rx}"] = data["y"]
        del x
    summary = summarize(per)
    summary["packet_acc"] = sum(v["correct"] for v in per.values()) / sum(v["total"] for v in per.values())
    summary["ce_mean"] = float(np.mean([v["ce"] for v in per.values()]))
    if out is not None:
        np.savez_compressed(out / f"preds_{label}.npz", **saved)
    return dict(summary=summary, per_rx=per)


def load_init(model, arm, out, fold, seed, stage, decision_file):
    if arm.get("init_arm") is None:
        return None
    name = arm["init_arm"]
    init = out.parent / f"{name}__f{fold}__s{seed}" / "model.pt"
    if stage == "final":
        if decision_file is None:
            raise ValueError("refit IB requires frozen source selection")
        decision = json.loads(Path(decision_file).read_text())
        name = decision["fan_init"][f"{arm['dataset']}:{fold}"]
        init = out.parent / f"{name}__f{fold}__s{seed}" / "model.pt"
    if not init.is_file():
        raise ValueError(f"IB initialization not complete: {init}")
    state = torch.load(init, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return dict(path=str(init), sha256=sha(init))


def fit(arm, fold, seed, stage, out, decision_file=None, smoke=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.reset_peak_memory_stats()
    run = f"{arm['name']}__f{fold}__s{seed}"
    dataset = arm.get("dataset", "Shen")
    if stage in ("final", "confirm") and (decision_file is None or not Path(decision_file).is_file()):
        raise ValueError("final evaluation requires a frozen source decision")
    if stage in ("final", "confirm"):
        decision = json.loads(Path(decision_file).read_text())
        if [arm["name"], fold, seed] not in decision["allowed_final_jobs"]:
            raise ValueError("unregistered final arm/fold/seed")
        if decision["registration_sha256"] != sha(out.parents[2] / "arms.json"):
            raise ValueError("registration changed after source freeze")
    if dataset == "Shen":
        tr, va, held, roles = shen_source(fold, f"{stage}:{run}")
        data_seed = stable_seed("P-H5x160", fold, seed)
    else:
        tr, va, held, roles = wifi_source(dataset, fold, seed, "source" if stage == "source" else "final", run)
        data_seed = stable_seed("WiSig-paper-v1", dataset, fold, seed)
    torch.manual_seed(data_seed)
    np.random.seed(data_seed)
    random.seed(data_seed)
    model = build_pipeline(arm).cuda()
    init = load_init(model, arm, out, fold, seed, stage, decision_file)
    rec = dict(RECIPES[arm["recipe"]])
    if smoke:
        rec["epochs"] = 2
    opt_cls = torch.optim.Adam if rec["opt"] == "adam" else torch.optim.AdamW
    opt = opt_cls(model.parameters(), lr=rec["lr"], weight_decay=rec["wd"])
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.1, patience=rec["patience"], min_lr=rec["min_lr"])
        if rec["sched"] == "plateau" else None)
    device = torch.device("cuda")
    tr = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in tr.items()}
    va = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in va.items()}
    order = torch.Generator(device=device).manual_seed(data_seed)
    noise = torch.Generator(device=device).manual_seed(data_seed + 1)
    batch = rec["batch"]
    steps = len(tr["y"]) // batch if rec["opt"] == "adamw" else math.ceil(len(tr["y"]) / batch)
    budget = steps * rec["epochs"]
    config = dict(arm=arm, fold=fold, seed=seed, stage=stage, recipe=rec, data_seed=data_seed,
                  roles=roles, upstream=UPSTREAM, code_sha256=code_sha(), init=init,
                  decision_file=decision_file, decision_sha256=sha(decision_file) if decision_file else None,
                  protocol_sha256=sha(PROTOCOL_PATH), smoke=smoke,
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                  checkpoint="minimum source packet-validation CE; no target evaluation during training")
    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    hist, best, wait, step = [], dict(ce=float("inf"), epoch=0, state=None), 0, 0
    train_start = time.time()
    for epoch in range(1, rec["epochs"] + 1):
        model.train()
        perm = torch.randperm(len(tr["y"]), device=device, generator=order)
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        seen = 0
        for i in range(steps):
            idx = perm[i * batch:(i + 1) * batch]
            if len(idx) < 2 and arm.get("alpha", 0):
                raise ValueError("singleton HSIC batch")
            if rec["sched"] == "warmcos":
                warm = rec["warmup"] * steps
                lr = rec["lr"] * ((step + 1) / warm if step < warm else
                      0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, budget - warm))))
                for group in opt.param_groups:
                    group["lr"] = lr
            xb, yb = tr["x"][idx], tr["y"][idx]
            if rec.get("awgn"):
                xb = awgn(xb, *rec["awgn"], noise)
            if arm.get("alpha", 0):
                logits, hidden, inputs = model.forward_with_hidden(xb)
                loss = F.cross_entropy(logits, yb) + arm["alpha"] * hsic_cca(hidden, inputs)
            else:
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite loss at epoch {epoch}, step {i}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.detach() * len(idx)
            correct += (logits.argmax(1) == yb).sum()
            seen += len(idx)
            step += 1
            if smoke and i == 9:
                break
        lv = predict(model, va["x"], batch=128)
        ce = float(F.cross_entropy(lv, va["y"])) if dataset != "Shen" else float(
            torch.stack([F.cross_entropy(lv[va["rx"] == rx], va["y"][va["rx"] == rx])
                         for rx in va["rx"].unique()]).mean())
        if not math.isfinite(ce):
            raise ValueError("nonfinite source validation")
        row = dict(epoch=epoch, lr=opt.param_groups[0]["lr"],
                   train_loss=float(loss_sum) / seen, train_acc=float(correct) / seen,
                   val_ce=ce, val_acc=float((lv.argmax(1) == va["y"]).float().mean()))
        hist.append(row)
        with (out / "history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row), lineterminator="\n")
            writer.writeheader()
            writer.writerows(hist)
        if ce < best["ce"]:
            best = dict(ce=ce, epoch=epoch, state=copy.deepcopy(model.state_dict()))
            wait = 0
        else:
            wait += 1
        if sched:
            sched.step(ce)
        if epoch == 1 or epoch % 20 == 0:
            print(json.dumps(dict(run=run, epoch=epoch, source_ce=round(ce, 5),
                                  train_s=round(time.time() - train_start, 1))), flush=True)
        if sched and wait >= rec["early_stop"]:
            break
    train_seconds = time.time() - train_start
    model.load_state_dict(best["state"])
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("nonfinite checkpoint")
    torch.save(best["state"], out / "model.pt")
    lv = predict(model, va["x"], batch=128)
    result = dict(run_id=run, arm=arm["name"], dataset=dataset, fold=fold, seed=seed, stage=stage,
                  params=n_params(model), best_val_epoch=best["epoch"], stop_epoch=epoch,
                  epochs_budget=rec["epochs"], source_val_ce=best["ce"],
                  source_val_acc=float((lv.argmax(1) == va["y"]).float().mean()),
                  train_seconds=round(train_seconds, 1), selected={})
    if held:
        result["source_rx_val"] = evaluate(model, held, arm["num_classes"], out, "source_rx_val")
    if not smoke and dataset == "Shen":
        for role in (["dev_val", "confirm"] if stage == "confirm" else ["dev_val"]):
            cells = {}
            for rx in roles[role]:
                d = load_rx(rx, role, f"{stage}:{run}", decision_file=decision_file)
                cells[rx] = dict(x=d["iq"], y=d["y"])
            result["selected"][role] = evaluate(model, cells, 10, out, role)
    elif not smoke and stage == "final":
        for label, days in (("CR", (1, 2)), ("CRD", (3, 4))):
            cells = read_wifi(dataset, roles["target"], days, "target_report_only", run)
            result["selected"][label] = evaluate(model, cells, arm["num_classes"], out, label)
    result["wall_seconds"] = round(time.time() - t0, 1)
    result["gpu_peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
    (out / "metrics.json").write_text(json.dumps(result, indent=1) + "\n")
    return result
