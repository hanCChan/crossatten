#!/usr/bin/env python3
"""Independently reconstruct every cached crop and verify purged temporal bounds."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = Path("/workspace/gai/data/raw/osu_lora/Diff_Receivers_Setup/Diff_Receivers_Setup_Indoor_SameTx")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    manifest_path = HERE / "osu_data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    reg = json.loads((HERE / "osu_registration.json").read_text())
    if not manifest["passed"]:
        raise ValueError("detector minimum counts failed; do not override using model scores")
    rows = []
    for entry in manifest["records"]:
        rx, tx = entry["rx"], entry["tx"]
        raw = RAW / rx / f"Device{tx}_IQ.dat"
        cache = HERE / "cache" / "osu" / f"{rx}_TX{tx}.npz"
        if sha(raw) != entry["raw_sha256"] or sha(cache) != entry["cache_sha256"]:
            raise ValueError("raw or cache changed")
        mm = np.memmap(raw, dtype="<c8", mode="r")
        seen = []
        with np.load(cache) as archive:
            for split, field in (("train", "source_train_seconds"), ("val", "source_val_seconds"),
                                 ("target", "target_seconds")):
                x, starts = archive[split], archive[f"{split}_start"]
                lo, hi = [int(s*reg["sample_rate"]) for s in reg[field]]
                if len(x) != len(starts) or len(x) != entry["counts"][split]:
                    raise ValueError("cache count mismatch")
                if x.dtype != np.float32 or x.shape[1:] != (2, 8192) or not np.isfinite(x).all():
                    raise ValueError("invalid cached IQ")
                if not np.all((starts >= lo) & (starts + 8192 <= hi)):
                    raise ValueError("cache temporal leakage")
                if np.any(np.diff(starts) < 16384):
                    raise ValueError("overlapping packet crops")
                for iq, start in zip(x, starts):
                    expected = np.stack((mm[start:start+8192].real, mm[start:start+8192].imag))
                    if not np.array_equal(iq, expected):
                        raise ValueError("cache does not equal original capture crop")
                seen.extend(starts.tolist())
        if len(set(seen)) != len(seen):
            raise ValueError("train/val/target crop identity leakage")
        rows.append(dict(rx=rx, tx=tx, counts=entry["counts"], exact_crop_parity=True,
                         raw_sha256=entry["raw_sha256"], cache_sha256=entry["cache_sha256"]))
        print(rx, tx, "PASS", flush=True)
    if {(r["rx"], r["tx"]) for r in rows} != {(rx, tx) for rx in ("RX1", "RX2") for tx in range(1, 26)}:
        raise ValueError("missing transmitter/receiver")
    value = dict(passed=True, manifest_sha256=sha(manifest_path), records=rows,
                 note="exact raw-crop parity and disjoint purged time blocks; classifier targets not accessed")
    (HERE / "osu_cache_audit.json").write_text(json.dumps(value, indent=1) + "\n")
    print("PASS all 50 raw hashes, cached crops, counts and temporal bounds")


if __name__ == "__main__":
    main()
