"""Training recipes and the single-fit loop. Only source_val packets steer checkpoint / LR schedule / early stop."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from . import LIB, ROOT  # noqa: E402
from .data import load_fold  # noqa: E402
from .evaluate import predict, rx_metrics, summarize  # noqa: E402
from .frontends import awgn  # noqa: E402
from .guard import PROTOCOL_PATH  # noqa: E402
from .models import build_pipeline, n_params  # noqa: E402

RECIPES = {
    # paper-1 CrossAttn-OOB recipe transplanted literally (AdamW 1e-3 / wd 5e-4 / batch 64 / fixed epoch 4, last weights)
    "p1_fixed": dict(opt="adamw", lr=1e-3, wd=5e-4, sched="const", epochs=4, ckpt="last"),
    # + long horizon with source-only checkpoint (min receiver-equal source_val CE)
    "long_const": dict(opt="adamw", lr=1e-3, wd=5e-4, sched="const", epochs=100, ckpt="best_val_ce"),
    # + warmup (5 ep) and cosine decay to 0
    "long_cos": dict(opt="adamw", lr=1e-3, wd=5e-4, sched="warmcos", warmup=5, epochs=100, ckpt="best_val_ce"),
    # stage 02: same as long_cos with a 200-epoch horizon (stage-01 best epochs were 87-97 of 100)
    "long_cos_200": dict(opt="adamw", lr=1e-3, wd=5e-4, sched="warmcos", warmup=5, epochs=200, ckpt="best_val_ce"),
    # Shen et al. main.py: SGD 1e-3 m0.9, ReduceLROnPlateau(0.2, 10), EarlyStopping(20), <=500 ep, final weights
    "shen": dict(opt="sgd", lr=1e-3, momentum=0.9, sched="plateau", factor=0.2, patience=10, early_stop=20,
                 epochs=500, ckpt="last_at_stop"),
}
BATCH = 64


def stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def code_sha() -> str:
    h = hashlib.sha256()
    for p in sorted(list((LIB / "xa").glob("*.py")) + list((LIB / "legacy_rfhstu").glob("*.py"))):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _to(d, dev):
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev) for k, v in d.items()}


def _rx_equal_ce(logits, y, rx):
    return float(torch.stack([F.cross_entropy(logits[rx == r], y[rx == r]) for r in torch.unique(rx)]).mean())


def _eval_sets(model, sets: dict) -> dict:
    per = {}
    for rx, d in sets.items():
        lg = predict(model, d["x"]).cpu().numpy()
        per[rx] = dict(rx_metrics(lg, d["y"].cpu().numpy()), logits=lg)
    return per


def fit(arm: dict, fold: int, seed: int, stage: str, out_dir: Path, decision_file: str | None = None,
        smoke: bool = False, val_every: int | None = None) -> dict:
    t_start = time.time()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device("cuda")
    rec = dict(RECIPES[arm["recipe"]])
    if smoke:
        rec["epochs"] = min(rec["epochs"], 2)
    run_id = f"{arm['name']}__f{fold}__s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    D = load_fold(fold, who=f"{stage}:{run_id}", with_confirm=(stage == "confirm"), decision_file=decision_file)
    tr, va = _to(D["train"], dev), _to(D["source_val"], dev)
    dv = {rx: _to(d, dev) for rx, d in D["dev_val"].items()}
    cf = {rx: _to(d, dev) for rx, d in D["confirm"].items()}
    data_seed = stable_seed("P-H5x160", fold, seed)
    torch.manual_seed(data_seed)
    np.random.seed(data_seed % (2 ** 32))
    model = build_pipeline(arm).to(dev)
    g_order = torch.Generator(device=dev).manual_seed(data_seed)
    g_noise = torch.Generator(device=dev).manual_seed(data_seed + 1)
    snr = arm.get("awgn")
    if rec["opt"] == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=rec["lr"], weight_decay=rec["wd"])
    else:
        opt = torch.optim.SGD(model.parameters(), lr=rec["lr"], momentum=rec["momentum"])
    N = len(tr["y"])
    steps = N // BATCH
    total = steps * rec["epochs"]

    def lr_at(step):
        if rec["sched"] != "warmcos":
            return None
        w = rec["warmup"] * steps
        if step < w:
            return rec["lr"] * (step + 1) / w
        return rec["lr"] * 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, total - w)))

    monitor_x = va["x"]
    if rec["sched"] == "plateau" and snr is not None:   # Shen validates on AWGN-corrupted packets (fixed realisation)
        monitor_x = awgn(va["x"], snr[0], snr[1], torch.Generator(device=dev).manual_seed(data_seed + 2))
    val_every = val_every or (5 if rec["epochs"] <= 100 else 10)
    hist, step = [], 0
    best = dict(ce=float("inf"), epoch=0, state=None)
    plateau = dict(best=float("inf"), wait=0, stop_wait=0)
    stop_epoch = rec["epochs"]
    t_train = time.time()
    for ep in range(1, rec["epochs"] + 1):
        model.train()
        perm = torch.randperm(N, generator=g_order, device=dev)
        tl = torch.zeros((), device=dev)
        tc = torch.zeros((), device=dev)
        for i in range(steps):
            lr = lr_at(step)
            if lr is not None:
                for gp in opt.param_groups:
                    gp["lr"] = lr
            idx = perm[i * BATCH:(i + 1) * BATCH]
            xb, yb = tr["x"][idx], tr["y"][idx]
            if snr is not None:
                xb = awgn(xb, snr[0], snr[1], g_noise)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += loss.detach()
            tc += (logits.argmax(1) == yb).sum()
            step += 1
            if smoke and i >= 9:
                break
        lv = predict(model, va["x"])
        val_ce = _rx_equal_ce(lv, va["y"], va["rx"])
        val_acc = float((lv.argmax(1) == va["y"]).float().mean())
        mon = val_ce if monitor_x is va["x"] else float(F.cross_entropy(predict(model, monitor_x), va["y"]))
        row = dict(epoch=ep, lr=opt.param_groups[0]["lr"], train_loss=float(tl) / steps, train_acc=float(tc) / (steps * BATCH),
                   val_ce=val_ce, val_acc=val_acc, monitor=mon, dev_val_acc="")
        if ep % val_every == 0 or ep == rec["epochs"]:
            row["dev_val_acc"] = float(np.mean([float((predict(model, d["x"]).argmax(1) == d["y"]).float().mean()) for d in dv.values()]))
        hist.append(row)
        if mon < best["ce"]:
            best = dict(ce=mon, epoch=ep, state=copy.deepcopy(model.state_dict()))
        if rec["sched"] == "plateau":
            if mon < plateau["best"]:
                plateau.update(best=mon, wait=0, stop_wait=0)
            else:
                plateau["wait"] += 1
                plateau["stop_wait"] += 1
                if plateau["stop_wait"] >= rec["early_stop"]:
                    stop_epoch = ep
                    break
                if plateau["wait"] >= rec["patience"]:
                    for gp in opt.param_groups:
                        gp["lr"] *= rec["factor"]
                    plateau["wait"] = 0
    train_seconds = time.time() - t_train
    last_state = copy.deepcopy(model.state_dict())
    selected = best["state"] if rec["ckpt"] == "best_val_ce" else last_state
    alt_name = "last" if rec["ckpt"] == "best_val_ce" else "best_val"
    alt = last_state if rec["ckpt"] == "best_val_ce" else best["state"]
    res = dict(run_id=run_id, arm=arm["name"], fold=fold, seed=seed, stage=stage, ckpt_rule=rec["ckpt"],
               best_val_epoch=best["epoch"], stop_epoch=stop_epoch, epochs_budget=rec["epochs"],
               params=n_params(model), train_seconds=round(train_seconds, 1))
    for tag, state in (("selected", selected), (alt_name, alt)):
        model.load_state_dict(state)
        lv = predict(model, va["x"])
        ent = dict(source_val_acc=float((lv.argmax(1) == va["y"]).float().mean()), source_val_ce=_rx_equal_ce(lv, va["y"], va["rx"]))
        for role, sets in (("dev_val", dv), ("confirm", cf)):
            if not sets:
                continue
            per = _eval_sets(model, sets)
            if tag == "selected":
                np.savez_compressed(out_dir / f"preds_{role}.npz", **{rx: p.pop("logits").astype(np.float16) for rx, p in per.items()},
                                    **{f"y_{rx}": sets[rx]["y"].cpu().numpy() for rx in sets})
            else:
                for p in per.values():
                    p.pop("logits")
            ent[role] = dict(summary=summarize(per), per_rx=per)
        res[tag] = ent
    model.load_state_dict(selected)
    if not smoke:
        torch.save(selected, out_dir / "model.pt")
    res["wall_seconds"] = round(time.time() - t_start, 1)
    res["gpu_peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    cfg = dict(arm=arm, fold=fold, seed=seed, stage=stage, recipe=rec, data_seed=data_seed, batch=BATCH,
               roles=D["roles"], protocol_sha256=hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
               code_sha256=code_sha(), torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
               decision_file=decision_file, smoke=smoke)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader()
        w.writerows(hist)
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=1))
    return res
