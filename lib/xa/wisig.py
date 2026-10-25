"""WiSig preparation and source-only audit helpers. No model selection on target RX."""
from __future__ import annotations

import math
import pickle
import re
from pathlib import Path

import numpy as np

from . import ROOT

DATA = ROOT / "data/wisig"
RX = {
    "ManyRx": ["1-1", "1-19", "1-20", "2-1", "2-19", "3-19", "7-7", "7-14",
               "8-7", "8-8", "8-14", "13-7", "13-14", "14-7", "18-2", "18-19",
               "19-1", "19-2", "19-19", "19-20", "20-1", "20-19", "20-20",
               "23-1", "23-3", "23-5", "23-6", "23-7", "24-5", "24-6", "24-13", "24-16"],
    "ManySig": ["1-1", "1-19", "2-1", "2-19", "3-19", "7-7", "7-14", "8-8",
                "14-7", "18-2", "19-2", "20-1"],
}


def receivers(dataset, fold):
    names = RX[dataset]
    if fold not in range(4):
        raise ValueError(fold)
    k = len(names) // 4
    test = names[fold * k:(fold + 1) * k]
    return [r for r in names if r not in test], test


def power_normalize(x):
    power = (x[:, 0] ** 2 + x[:, 1] ** 2).mean(-1)
    if np.any(power <= 0) or not np.isfinite(power).all():
        raise ValueError("nonfinite or zero-power WiSig packet")
    return x / np.sqrt(power)[:, None, None]


def log_counts(dataset, fold):
    path = ROOT / f"external/IB-RFF/logs/stage1/{dataset}_pre0_a0.0_round{fold}.log"
    text = path.read_text()
    tr = re.search(r"Train Epoch: 1\s+.*?Accuracy: \d+/(\d+)", text)
    va = re.search(r"Validation set:.*?Accuracy: \d+/(\d+)", text)
    if tr is None or va is None:
        raise ValueError(f"cannot parse count evidence: {path}")
    return int(tr[1]), int(va[1])


def build_dataset(dataset):
    source = DATA / f"{dataset}.pkl"
    # Inputs are the user-provided Kaggle archive or UCLA's official compact archive.
    with source.open("rb") as f:
        d = pickle.load(f)
    max_sig = {"ManyRx": 200, "ManySig": 1000}[dataset]
    if d["equalized_list"] != [0, 1] or d["max_sig"] != max_sig:
        raise ValueError("unexpected equalization/max_sig metadata")
    nc = {"ManyRx": 10, "ManySig": 6}[dataset]
    if len(d["tx_list"]) != nc or len(d["capture_date_list"]) != 4:
        raise ValueError("unexpected TX/day count")
    days = sorted(range(4), key=lambda i: d["capture_date_list"][i])
    indices = {name: d["rx_list"].index(name) for name in RX[dataset]}
    counts = {}
    npz_dir = DATA / "cache" / dataset
    npz_dir.mkdir(parents=True, exist_ok=True)
    for rx, ri in indices.items():
        counts[rx] = {}
        for day, di in enumerate(days, 1):
            arrays, xs, ys = [], [], []
            for tx in range(nc):
                a = np.asarray(d["data"][tx][ri][di][0])
                if a.ndim != 3 or a.shape[1:] != (256, 2) or len(a) > max_sig:
                    raise ValueError((dataset, rx, day, tx, a.shape))
                a = a[:100]
                arrays.append(a)
                x = power_normalize(a.transpose(0, 2, 1))
                xs.append(x.astype(np.float32))
                ys.append(np.full(len(a), tx, np.int64))
            counts[rx][str(day)] = [len(a) for a in arrays]
            np.savez_compressed(npz_dir / f"rx_{rx}_day{day}.npz",
                                x=np.concatenate(xs), y=np.concatenate(ys))
            layout = DATA / "official_layout" / dataset / "non_equalized" / f"date{day}"
            layout.mkdir(parents=True, exist_ok=True)
            # The official loader transposes (N,256,2), then normalizes once.
            # Keep source precision/raw scale for exact compatibility with that loader.
            with (layout / f"rx_{rx}_data.pkl").open("wb") as f:
                pickle.dump({"data": arrays}, f, protocol=4)
    checks = []
    for fold in range(4):
        train, test = receivers(dataset, fold)
        pool = sum(sum(counts[rx][str(day)]) for rx in train for day in (1, 2))
        actual_val = math.ceil(0.3 * pool)
        actual_train = pool - actual_val
        expected_train, expected_val = log_counts(dataset, fold)
        checks.append(dict(fold=fold, pool=pool, train=actual_train, val=actual_val,
                           expected_train=expected_train, expected_val=expected_val,
                           matches=(actual_train, actual_val) == (expected_train, expected_val),
                           train_rx=train, test_rx=test))
    return dict(dataset=dataset, tx_list=d["tx_list"], max_sig=max_sig,
                capture_dates=[d["capture_date_list"][i] for i in days],
                official_rx_order=RX[dataset], non_equalized_index=0, first_packets_per_cell=100,
                counts=counts, checks=checks, counts_match=all(c["matches"] for c in checks),
                official_layout="raw float64 (N,256,2); official loader transposes and power-normalizes",
                own_layout="normalized float32 (N,2,256)")


def link_official_layout(dataset):
    for repo in ("IB-RFF", "WaveMLP-RFF"):
        base = ROOT / "external" / repo / "dataset"
        base.mkdir(exist_ok=True)
        link = base / dataset
        target = DATA / "official_layout" / dataset
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError(f"existing unrelated dataset symlink: {link}")
        elif link.exists():
            raise ValueError(f"refusing to overwrite real upstream dataset directory: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)


def load_cell(dataset, rx, day):
    if rx not in RX[dataset] or day not in (1, 2, 3, 4):
        raise ValueError((dataset, rx, day))
    with np.load(DATA / "cache" / dataset / f"rx_{rx}_day{day}.npz") as d:
        return d["x"], d["y"]
