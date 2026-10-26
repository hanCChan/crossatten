"""Reuse frozen S09 fitting without modifying any historical stage bytes."""
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from stage import HERE, ROOT, s09, validate, cells, save, sha, receivers, ask_modules
from features import paired_model
from xa.paper_train import flatten


def phase_adapter():
    # The independently loaded module has its own globals; S09 on disk is read-only.
    adapter = __import__("stage").module("s10_fit_adapter", ROOT / "09-phase-invariance/training.py")
    adapter.HERE = HERE
    adapter.validate = validate
    adapter.read_cells = cells
    adapter.paired_model = paired_model
    return adapter


def fit_phase(job):
    return phase_adapter().fit(job)


def evaluate_phase(job):
    return phase_adapter().evaluate_target(job)


def fit_ask(job):
    validate(job)
    out = HERE / "runs" / job["phase"] / job["id"]
    official = ask_modules()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    official.setup_seed(job["seed"])
    source, target = receivers(job["dataset"], job["fold"])
    pool = flatten(cells(job, source, (1, 2), "fit"))
    from sklearn.model_selection import train_test_split
    tr_ids, va_ids = train_test_split(np.arange(len(pool["y"])), test_size=.3,
                                     random_state=job["seed"])
    if job["phase"] == "smoke":
        tr_ids, va_ids = tr_ids[:256], va_ids[:128]
    datasets = [TensorDataset(torch.from_numpy(pool["x"][ids]),
                              torch.from_numpy(pool["y"][ids])) for ids in (tr_ids, va_ids)]
    train = DataLoader(datasets[0], batch_size=128, shuffle=True, drop_last=False)
    val = DataLoader(datasets[1], batch_size=128, shuffle=False, drop_last=False)
    classes = 10 if job["dataset"] == "ManyRx" else 6
    conf = __import__("argparse").Namespace(candidates=[8, 16, 24, 32, 48, 64],
                                         patch_size=64, embed_dim=128, mlp_ratio=2.)
    model = official.build_model(conf, classes)
    params = sum(p.numel() for p in model.parameters())
    config = dict(job=job, params=params, official_defaults=vars(conf),
        data_seed=job["seed"], split_sha256=dict(train=s09.array_sha(tr_ids),
        val=s09.array_sha(va_ids), pool_y=s09.array_sha(pool["y"])),
        train_packets=len(tr_ids), val_packets=len(va_ids), roles=dict(source=source, target=target),
        registration_sha256=sha(HERE / "registration.json"), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0), shuffle="official CPU DataLoader/global Torch RNG",
        precision="cached float32; parity audited against official float64-normalize-then-cast")
    save(out / "config.json", config)
    history = []
    train_fn, val_fn = official.train_epoch, official.evaluate_epoch

    def traced_train(*args, **kwargs):
        loss, acc = train_fn(*args, **kwargs)
        history.append(dict(epoch=args[4], train_ce=loss, train_acc=acc / 100,
                            lr=args[3].param_groups[0]["lr"]))
        return loss, acc

    def traced_val(*args, **kwargs):
        loss, acc = val_fn(*args, **kwargs)
        history[-1].update(val_ce=loss, val_acc=acc / 100)
        import csv
        with (out / "history.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[-1]))
            w.writeheader()
            w.writerows(history)
        return loss, acc

    official.train_epoch, official.evaluate_epoch = traced_train, traced_val
    start = time.time()
    best, trace = official.train_and_evaluate(model, train, val, job["epochs"],
        str(out / "model.pt"), log_path=str(out / "k_trace.txt"))
    selected_epoch = min(history, key=lambda r: r["val_ce"])["epoch"]
    save(out / "metrics.json", dict(job=job, params=params, stop_epoch=len(history),
        selected=dict(source_packet=dict(ce=best, epoch=selected_epoch)),
        seconds=time.time() - start, weights_sha256={"model.pt": sha(out / "model.pt")},
        config_sha256=sha(out / "config.json"), registration_sha256=config["registration_sha256"],
        k_history=trace))


def evaluate_ask(job):
    validate(job)
    torch.set_num_threads(2)
    out = HERE / "runs" / job["phase"] / job["id"]
    if (out / "target_metrics.json").exists():
        raise ValueError("target evaluation already complete")
    result = json.loads((out / "metrics.json").read_text())
    if sha(out / "model.pt") != result["weights_sha256"]["model.pt"]:
        raise ValueError("checkpoint changed")
    official = ask_modules()
    conf = __import__("argparse").Namespace(candidates=[8, 16, 24, 32, 48, 64],
                                         patch_size=64, embed_dim=128, mlp_ratio=2.)
    classes = 10 if job["dataset"] == "ManyRx" else 6
    model = official.build_model(conf, classes)
    model.load_state_dict(torch.load(out / "model.pt", map_location="cpu", weights_only=True))
    model.cuda().eval()
    _, target = receivers(job["dataset"], job["fold"])
    selected = {}
    for task, days in (("CR", (1, 2)), ("CRD", (3, 4))):
        selected[task] = s09.s08.evaluate(model,
            s09.s08.on_device(cells(job, target, days, "target", task)), classes, out, task)
    save(out / "target_metrics.json", dict(job=job, selected=selected,
        registration_sha256=sha(HERE / "registration.json"),
        predictions_sha256={p.name: sha(p) for p in out.glob("preds_*.npz")}))
