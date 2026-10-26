"""Restore derived caches only after matching historical raw-payload hashes."""
import pickle
import numpy as np
from stage import HERE, ROOT, DATA, RX, stream_sha, once, sha, ask_modules, ASK_DIR, ASK_SHA
from xa.wisig import power_normalize


def main():
    import json
    import subprocess
    if subprocess.check_output(["git", "-C", str(ASK_DIR), "rev-parse", "HEAD"]).decode().strip() != ASK_SHA:
        raise ValueError("unexpected ASKNet commit")
    result = dict(role="inert deterministic preparation, not outcome-based selection",
                  cache_sha256={}, official_parity={}, datasets={})
    official = ask_modules()
    for dataset in ("ManyRx", "ManySig"):
        manifest_path = ROOT / f"05-wisig/out/05-1-{dataset}-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        raw = DATA / f"{dataset}.pkl"
        digest = stream_sha(raw)
        if digest != manifest["source_sha256"]:
            raise ValueError("raw payload hash differs from historical audited data")
        with raw.open("rb") as f:
            d = pickle.load(f)
        nc = 10 if dataset == "ManyRx" else 6
        if d["equalized_list"] != [0, 1] or len(d["tx_list"]) != nc:
            raise ValueError("metadata mismatch")
        days = sorted(range(4), key=lambda i: d["capture_date_list"][i])
        if [d["capture_date_list"][i] for i in days] != manifest["capture_dates"]:
            raise ValueError("capture-date order mismatch")
        cache = DATA / "cache" / dataset
        cache.mkdir(parents=True, exist_ok=True)
        max_error = 0.
        for rx in RX[dataset]:
            ri = d["rx_list"].index(rx)
            for day, di in enumerate(days, 1):
                arrays = [np.asarray(d["data"][tx][ri][di][0])[:100] for tx in range(nc)]
                if [len(a) for a in arrays] != manifest["counts"][rx][str(day)]:
                    raise ValueError("cell count mismatch")
                if any(a.ndim != 3 or a.shape[1:] != (256, 2) for a in arrays):
                    raise ValueError("shape mismatch")
                raw_x = np.concatenate(arrays).transpose(0, 2, 1)
                x = power_normalize(raw_x).astype(np.float32)
                y = np.concatenate([np.full(len(a), tx, np.int64) for tx, a in enumerate(arrays)])
                own_path = cache / f"rx_{rx}_day{day}.npz"
                if not own_path.exists():
                    np.savez_compressed(own_path, x=x, y=y)
                else:
                    with np.load(own_path) as old:
                        if not np.array_equal(old["x"], x) or not np.array_equal(old["y"], y):
                            raise ValueError("existing cache mismatch")
                result["cache_sha256"][f"{dataset}/{own_path.name}"] = sha(own_path)
                layout = DATA / "official_layout" / dataset / "non_equalized" / f"date{day}"
                layout.mkdir(parents=True, exist_ok=True)
                path = layout / f"rx_{rx}_data.pkl"
                if not path.exists():
                    with path.open("wb") as f:
                        pickle.dump({"data": arrays}, f, protocol=4)
                # Execute only inspected loader normalization, never train or score target data.
                ref = official.load_single_dataset.__globals__["preprocessing"](raw_x.copy()).astype(np.float32)
                max_error = max(max_error, float(np.max(np.abs(ref - x))))
        linkbase = ASK_DIR / "WiSig/dataset"
        linkbase.mkdir(exist_ok=True)
        link = linkbase / dataset
        if not link.exists():
            link.symlink_to(DATA / "official_layout" / dataset, target_is_directory=True)
        elif link.resolve() != (DATA / "official_layout" / dataset).resolve():
            raise ValueError("unrelated official data link")
        result["official_parity"][dataset] = dict(max_abs_error=max_error, all_cells=True)
        if max_error > 1e-6 or official.load_single_dataset.__globals__["RX_INDEXES"][dataset] != RX[dataset]:
            raise ValueError("official loader parity failed")
        result["datasets"][dataset] = dict(raw_sha256=digest, manifest_sha256=sha(manifest_path),
                                          cells=len(RX[dataset]) * 4, counts_exact=True)
        print(dataset, "cache restored; parity", max_error, flush=True)
        del d
    once(HERE / "preparation.json", result)


if __name__ == "__main__":
    main()
