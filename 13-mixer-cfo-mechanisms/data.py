"""S13 role-gated data access; never writes historical caches or access logs."""
from functools import lru_cache
import numpy as np
from core import HERE, ROOT, read, write, sha, log_access, array_sha
from pickle_index import index, Array
from xa.data import select_rows, source_split
from xa.wisig import RX, receivers, power_normalize

PROTOCOL = read(ROOT / "configs/protocol.json")


def shen_roles(fold):
    return next(f for f in PROTOCOL["folds"] if f["fold"] == fold)


def shen_cell(rx, fold, role):
    legal = shen_roles(fold)["source"] if role == "source" else shen_roles(fold)["dev_val"] if role == "dev_val" else PROTOCOL["confirm_receivers"] if role == "confirm" else []
    if rx not in legal or rx in PROTOCOL["blind_receivers_never_open"]:
        raise ValueError(f"Illegal Shen access: {rx}/{role}/fold{fold}")
    if role != "source":
        from core import require_committed
        require_committed(HERE / "decision_p1_source.json")
    cache = HERE / "cache" / "shen" / rx
    if not (cache / "manifest.json").exists():
        import h5py
        raw = ROOT / "data/raw" / f"{rx}_train.h5"
        historical = read(ROOT / "00-signal-audit/out/00-1-cache-manifest.json")
        expected = next(r["source_sha256"] for r in historical["receivers"] if r["rx"] == rx)
        if sha(raw) != expected:
            raise ValueError(f"Shen raw hash mismatch: {rx}")
        log_access(dataset="Shen", rx=rx, fold=fold, role=role, action="raw_train_hdf5")
        with h5py.File(raw, "r") as f:
            lab = np.asarray(f["label"][:]).reshape(-1).astype(np.int64)
            rows = select_rows(lab, PROTOCOL["packet_selection"]["seed"], 200)
            selected = np.asarray(f["data"][rows], np.float64)
        x = np.stack([selected[:, :8192], selected[:, 8192:]], 1).astype(np.float32)
        y = lab[rows] - 31
        cache.mkdir(parents=True, exist_ok=True)
        for key, a in dict(x=x, y=y, rows=rows).items():
            np.save(cache / f"{key}.npy", a)
        write(cache / "manifest.json", dict(source_sha256=expected, rows_sha256=array_sha(rows), x_sha256=array_sha(x)))
    log_access(dataset="Shen", rx=rx, fold=fold, role=role, action="s13_cache")
    return {k: np.load(cache / f"{k}.npy", mmap_mode="r") for k in ("x", "y", "rows")}


def shen_train(fold):
    tr, va = [], []
    for ri, rx in enumerate(shen_roles(fold)["source"]):
        d = shen_cell(rx, fold, "source")
        train, val = source_split(d["y"], d["rows"])
        for output, ids in ((tr, train), (va, val)):
            output.append(dict(x=np.asarray(d["x"][ids]), y=np.asarray(d["y"][ids]), rx=np.full(len(ids), ri)))
    return [dict(x=np.concatenate([a["x"] for a in rows]), y=np.concatenate([a["y"] for a in rows]),
                 rx=np.concatenate([a["rx"] for a in rows])) for rows in (tr, va)]


@lru_cache(maxsize=2)
def metadata(dataset):
    if dataset not in RX:
        raise ValueError(dataset)
    path = ROOT / "data/wisig" / f"{dataset}.pkl"
    d = index(path)
    expected = read(ROOT / f"05-wisig/out/05-1-{dataset}-manifest.json")
    if list(d["tx_list"]) != expected['tx_list'] or set(d['rx_list']) != set(expected['official_rx_order']) or d["equalized_list"] != [0, 1]:
        raise ValueError("WiSig metadata mismatch")
    if sorted(d["capture_date_list"]) != expected["capture_dates"]:
        raise ValueError("WiSig dates differ")
    return d


def wisig_cells(dataset, fold, names, days=(1, 2), role="fit"):
    source, target = receivers(dataset, fold)
    if role not in ("fit", "held", "diagnostic") or not set(names) <= set(source) or tuple(days) != (1, 2):
        raise ValueError("P0/P2 cannot read target receivers or dates3-4")
    d = metadata(dataset)
    di = sorted(range(4), key=lambda i: d["capture_date_list"][i])
    cells = {}
    for rx in names:
        ri = d["rx_list"].index(rx)
        for day in days:
            path = HERE / "cache" / "wisig" / dataset / f"rx_{rx}_day{day}.npz"
            log_access(dataset=dataset, fold=fold, rx=rx, day=day, role=role, action="authorized_cell_only")
            if not path.exists():
                arrays = []
                for tx in range(len(d["tx_list"])):
                    array = d["data"][tx][ri][di[day - 1]][0]
                    a = array.get() if isinstance(array, Array) else np.asarray(array)
                    a = a[:100]
                    if a.ndim != 3 or a.shape[1:] != (256, 2):
                        raise ValueError("WiSig cell shape mismatch")
                    arrays.append(a)
                expected = read(ROOT / f"05-wisig/out/05-1-{dataset}-manifest.json")['counts'][rx][str(day)]
                if [len(a) for a in arrays] != expected:
                    raise ValueError(f"Historical WiSig packet count mismatch: {dataset}/{rx}/{day}")
                x = power_normalize(np.concatenate(arrays).transpose(0, 2, 1)).astype(np.float32)
                y = np.concatenate([np.full(len(a), tx, np.int64) for tx, a in enumerate(arrays)])
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, x=x, y=y)
            with np.load(path) as z:
                cells[(rx, day)] = dict(x=z["x"], y=z["y"])
    return cells


def source_groups(dataset, fold, group="first"):
    source, _ = receivers(dataset, fold)
    count = 6 if dataset == "ManyRx" else 2
    if group not in ("first", "last"):
        raise ValueError(group)
    held = source[:count] if group == "first" else source[-count:]
    return [rx for rx in source if rx not in held], held


def flatten(cells):
    xs, ys, rs, ds = [], [], [], []
    names = sorted({rx for rx, day in cells})
    for (rx, day), d in cells.items():
        xs.append(d["x"]); ys.append(d["y"])
        rs.append(np.full(len(d["y"]), names.index(rx)))
        ds.append(np.full(len(d["y"]), day))
    return dict(x=np.concatenate(xs), y=np.concatenate(ys), rx=np.concatenate(rs), day=np.concatenate(ds), rx_names=names)
