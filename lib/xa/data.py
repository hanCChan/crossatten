"""Shen packet cache (2000 selected packets per dev RX) and role-checked loaders."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import ROOT
from .guard import check_raw_path, check_rx, load_protocol

CACHE = ROOT / "data" / "cache"
RAW = ROOT / "data" / "raw"


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def select_rows(raw_labels: np.ndarray, seed: int, per_tx: int) -> np.ndarray:
    """Exactly gai src/shen_p2a8/g1_data.py::build_shen (train role)."""
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(raw_labels == c), per_tx, replace=False) for c in range(31, 41)]))


def build_rx_cache(rx: str, who: str = "00-1-build-cache") -> dict:
    import h5py

    P = load_protocol()
    check_rx(rx, "cache_build", who)
    path = check_raw_path(RAW / P["dataset"]["file_pattern"].format(rx=rx))
    raw_sha = sha256_file(path)
    sel = P["packet_selection"]
    with h5py.File(path, "r") as f:
        raw = np.asarray(f["label"][:]).reshape(-1).astype(np.int64)
        assert set(raw.tolist()) == set(range(31, 41)), rx
        assert f["data"].shape == (len(raw), 2 * P["dataset"]["packet_len"]), (rx, f["data"].shape)
        rows = select_rows(raw, sel["seed"], sel["per_tx"])
        x = np.asarray(f["data"][rows], dtype=np.float64)
        snr = np.asarray(f["SNR"][:]).reshape(-1)[rows].astype(np.float32)
        cfo = np.asarray(f["CFO"][:]).reshape(-1)[rows].astype(np.float32)
    L = P["dataset"]["packet_len"]
    iq = np.stack([x[:, :L], x[:, L:]], axis=1).astype(np.float32)  # [N, 2, L], raw (unnormalised)
    labels = (raw[rows] - 31).astype(np.int64)
    out = CACHE / rx
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "iq.npy", iq)
    np.save(out / "labels.npy", labels)
    np.save(out / "rows.npy", rows.astype(np.int64))
    np.save(out / "snr_field.npy", snr)
    np.save(out / "cfo_field.npy", cfo)
    man = dict(rx=rx, source=str(path.name), source_sha256=raw_sha, source_bytes=path.stat().st_size,
               packets=int(len(rows)), per_tx=sel["per_tx"], selection_seed=sel["seed"],
               iq_sha256=sha256_array(iq), rows_sha256=sha256_array(rows.astype(np.int64)),
               labels_sha256=sha256_array(labels), iq_layout="float32 [N,2,8192] raw I/Q from HDF5 float64 (no normalisation)",
               fields_note="snr_field/cfo_field are HDF5 metadata, diagnostics only")
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


def load_rx(rx: str, purpose: str, who: str, decision_file: str | None = None, mmap: bool = False) -> dict:
    check_rx(rx, purpose, who, decision_file)
    d = CACHE / rx
    man = json.loads((d / "manifest.json").read_text())
    iq = np.load(d / "iq.npy", mmap_mode="r" if mmap else None)
    lab = np.load(d / "labels.npy")
    rows = np.load(d / "rows.npy")
    assert np.array_equal(np.bincount(lab, minlength=10), np.full(10, man["per_tx"])), rx
    return dict(rx=rx, iq=iq, y=lab, rows=rows, manifest=man)


def source_split(y: np.ndarray, rows: np.ndarray, n_train: int = 160) -> tuple[np.ndarray, np.ndarray]:
    """first n_train selected rows per TX (ascending row) -> train; remainder -> source_val."""
    tr, va = [], []
    for t in range(10):
        idx = np.flatnonzero(y == t)
        idx = idx[np.argsort(rows[idx], kind="stable")]
        tr.append(idx[:n_train])
        va.append(idx[n_train:])
    return np.concatenate(tr), np.concatenate(va)


def fold_roles(fold: int) -> dict:
    P = load_protocol()
    f = [x for x in P["folds"] if x["fold"] == fold][0]
    return dict(source=list(f["source"]), dev_val=list(f["dev_val"]), confirm=list(P["confirm_receivers"]))


def load_fold(fold: int, who: str, with_confirm: bool = False, decision_file: str | None = None) -> dict:
    """Returns numpy arrays: train / source_val (with rx ids) and per-RX eval sets."""
    roles = fold_roles(fold)
    xs, ys, rs, vx, vy, vr = [], [], [], [], [], []
    for i, rx in enumerate(roles["source"]):
        d = load_rx(rx, "source", who)
        tr, va = source_split(d["y"], d["rows"])
        xs.append(d["iq"][tr]); ys.append(d["y"][tr]); rs.append(np.full(len(tr), i))
        vx.append(d["iq"][va]); vy.append(d["y"][va]); vr.append(np.full(len(va), i))
    out = dict(roles=roles,
               train=dict(x=np.concatenate(xs), y=np.concatenate(ys), rx=np.concatenate(rs)),
               source_val=dict(x=np.concatenate(vx), y=np.concatenate(vy), rx=np.concatenate(vr)),
               dev_val={}, confirm={})
    for rx in roles["dev_val"]:
        d = load_rx(rx, "dev_val", who)
        out["dev_val"][rx] = dict(x=d["iq"], y=d["y"])
    if with_confirm:
        for rx in roles["confirm"]:
            d = load_rx(rx, "confirm", who, decision_file=decision_file)
            out["confirm"][rx] = dict(x=d["iq"], y=d["y"])
    return out
