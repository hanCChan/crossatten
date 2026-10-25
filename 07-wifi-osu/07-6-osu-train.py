#!/usr/bin/env python3
"""OSU independent 25-class fits; target loading occurs after source selection."""
import argparse
import copy
import csv
import fcntl
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
import numpy as np
import torch
from torch.nn import functional as F

from xa.evaluate import predict
from xa.frontends import awgn
from xa.models import ShenCISPipeline, n_params
from xa.paper_models import ValidationRF
from xa.paper_train import metrics, require_committed, sha
from xa.train import stable_seed


def build(method):
    if method == "cis_rank16":
        return ShenCISPipeline(head="rank16", input_fmax=62500, num_classes=25)
    return ValidationRF(dict(length=8192, stem="patch" if method == "patch_hstu" else "hierarchical",
                             mixer=method != "hstem_cnn", num_classes=25, input_fmax=62500))


def load(rx, split, reg, manifest, run):
    if rx not in ("RX1", "RX2") or split not in ("train", "val", "target"):
        raise ValueError("invalid OSU data role")
    xs, ys, starts = [], [], []
    audit = {r["tx"]: r for r in manifest["records"] if r["rx"] == rx}
    with (HERE / "osu_access.jsonl").open("a") as log:
        for tx in range(1, 26):
            path = HERE / "cache" / "osu" / f"{rx}_TX{tx}.npz"
            if sha(path) != audit[tx]["cache_sha256"]:
                raise ValueError("OSU cached input changed")
            with np.load(path) as archive:
                x, sample = archive[split], archive[f"{split}_start"]
            lo, hi = [int(t * reg["sample_rate"]) for t in
                      reg[{"train": "source_train_seconds", "val": "source_val_seconds",
                           "target": "target_seconds"}[split]]]
            if not np.all((sample >= lo) & (sample + 8192 <= hi)):
                raise ValueError("OSU crop escaped temporal split")
            xs.append(x)
            starts.append(sample)
            ys.append(np.full(len(x), tx - 1, dtype=np.int64))
            log.write(json.dumps(dict(run=run, rx=rx, tx=tx, split=split, count=len(x))) + "\n")
    return dict(x=np.concatenate(xs), y=np.concatenate(ys), start=np.concatenate(starts))


def fit(job, smoke=False):
    registration = HERE / "osu_registration.json"
    reg = json.loads(registration.read_text())
    require_committed(registration)
    if job not in json.loads((HERE / "jobs_osu.json").read_text()):
        raise ValueError("OSU job not registered")
    require_committed(HERE / "jobs_osu.json")
    manifest_path = HERE / "osu_data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require_committed(manifest_path)
    if not manifest["passed"] or manifest["registration_sha256"] != sha(registration):
        raise ValueError("OSU data gate failed")
    audit_path = HERE / "osu_cache_audit.json"
    require_committed(audit_path)
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["manifest_sha256"] != sha(manifest_path):
        raise ValueError("OSU exact-crop audit failed")
    if job["source"] == job["target"]:
        raise ValueError("OSU target must be an unseen receiver")
    out = HERE / "runs" / ("osu_smoke" if smoke else "osu") / job["id"]
    if (out / "metrics.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        archive = out / f"attempt_{time.time_ns()}"
        archive.mkdir()
        for p in out.iterdir():
            if p.is_file() and p.name not in ("stdout.log", "execution.lock"):
                p.rename(archive / p.name)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    seed = stable_seed("S07-OSU", job["source"], job["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tr = load(job["source"], "train", reg, manifest, job["id"])
    va = load(job["source"], "val", reg, manifest, job["id"])
    tr = {k: torch.from_numpy(v).cuda() for k, v in tr.items()}
    va = {k: torch.from_numpy(v).cuda() for k, v in va.items()}
    model = build(job["method"]).cuda()
    recipe = reg["recipe"]
    batch = recipe["batch"]
    epochs = 2 if smoke else recipe["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    order = torch.Generator(device="cuda").manual_seed(seed)
    noise = torch.Generator(device="cuda").manual_seed(seed + 1)
    steps = math.ceil(len(tr["y"]) / batch)
    config = dict(job=job, registration_sha256=sha(registration), manifest_sha256=sha(manifest_path),
                  trainer_sha256=sha(Path(__file__)), source_train=len(tr["y"]), source_val=len(va["y"]),
                  params=n_params(model), smoke=smoke, seed=seed, recipe=recipe)
    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    start = time.time()
    best, history = dict(ce=float("inf")), []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(tr["y"]), device="cuda", generator=order)
        losses, correct, seen = 0., 0, 0
        for i in range(steps):
            step, warm, horizon = (epoch-1)*steps + i, 5*steps, 200*steps
            lr = recipe["lr"] * ((step+1)/warm if step < warm else
                                 .5*(1 + math.cos(math.pi*(step-warm)/(horizon-warm))))
            for group in opt.param_groups:
                group["lr"] = lr
            ids = perm[i*batch:(i+1)*batch]
            x, y = tr["x"][ids], tr["y"][ids]
            logits = model(awgn(x, *recipe["awgn"], noise))
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite OSU loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses += float(loss.detach()) * len(y)
            correct += int((logits.argmax(1) == y).sum())
            seen += len(y)
            if smoke and i >= 1:
                break
        logits = predict(model, va["x"], batch=64)
        ce = float(F.cross_entropy(logits, va["y"]))
        row = dict(epoch=epoch, train_ce=losses/seen, train_acc=correct/seen,
                   val_ce=ce, val_acc=float((logits.argmax(1) == va["y"]).float().mean()), lr=lr)
        history.append(row)
        if ce < best["ce"]:
            best = dict(ce=ce, epoch=epoch, state=copy.deepcopy(model.state_dict()))
        with (out / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        if epoch == 1 or epoch % 20 == 0:
            print(json.dumps(dict(run=job["id"], epoch=epoch, source_ce=ce,
                                  seconds=round(time.time()-start))), flush=True)
    model.load_state_dict(best["state"])
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("nonfinite OSU weights")
    torch.save(model.state_dict(), out / "model.pt")
    result = dict(job=job, params=n_params(model), selected_epoch=best["epoch"],
                  source_val_ce=best["ce"], smoke=smoke, config=config)
    if not smoke:
        target = load(job["target"], "target", reg, manifest, job["id"])
        logits = predict(model, torch.from_numpy(target["x"]).cuda(), batch=64).cpu().numpy()
        if not np.isfinite(logits).all():
            raise ValueError("nonfinite target logits")
        result["target"] = metrics(logits, target["y"], 25)
        cm = np.asarray(result["target"]["confusion"])
        result["target"]["tx_equal_acc"] = float(np.mean(cm.diagonal() / cm.sum(1)))
        np.savez_compressed(out / "preds_target.npz", logits=logits, y=target["y"], start=target["start"])
    result.update(seconds=round(time.time()-start, 2), weights_sha256=sha(out / "model.pt"))
    (out / "metrics.json").write_text(json.dumps(result, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    if args.job:
        fit(json.loads(args.job), args.smoke)
        return
    jobs = json.loads((HERE / "jobs_osu.json").read_text())
    if args.smoke:
        jobs = [j for j in jobs if j["source"] == "RX1" and j["seed"] == 5101]
    gpus = args.gpus.split(",")
    import concurrent.futures
    def run(index_job):
        index, job = index_job
        phase = "osu_smoke" if args.smoke else "osu"
        out = HERE / "runs" / phase / job["id"]
        out.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[index % len(gpus)], OMP_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        command = [sys.executable, str(Path(__file__).resolve()), "--job", json.dumps(job)]
        if args.smoke:
            command.append("--smoke")
        with (out / "execution.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (out / "metrics.json").is_file():
                result = json.loads((out / "metrics.json").read_text())
                if result["job"] != job or sha(out / "model.pt") != result["weights_sha256"]:
                    raise ValueError("changed completed OSU run")
                return
            old = [p for p in out.iterdir() if p.is_file() and p.name != "execution.lock"]
            if old:
                archive = out / f"attempt_{time.time_ns()}"
                archive.mkdir()
                for p in old:
                    p.rename(archive / p.name)
            with (out / "stdout.log").open("w") as log:
                rc = subprocess.call(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            if rc or not (out / "metrics.json").is_file():
                raise RuntimeError(f"OSU job failed and retained: {job['id']}")
        print(job["id"], "done", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(run, enumerate(jobs)))


if __name__ == "__main__":
    main()
