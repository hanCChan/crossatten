#!/usr/bin/env python3
"""00-1  Build the protocol packet cache for the 14 dev receivers and verify it.

- Selects 200 packets/TX per RX with the frozen rl1 rule (configs/protocol.json).
- Verifies byte-equality against the rl1 g1_cache_v1 I/Q for the 10 RX that exist there,
  and the source HDF5 SHA256 against the gai g1 registration where available.
- Never opens blind receiver files (guard).
Output: data/cache/<rx>/*, 00-signal-audit/out/00-1-cache-manifest.json
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from xa import ROOT  # noqa: E402
from xa.data import CACHE, build_rx_cache  # noqa: E402
from xa.guard import load_protocol  # noqa: E402

RL1_CACHE = Path("/workspace/rl1/gai_retrain/outputs/shen_p2a8/g1_cache_v1")
G1_REG = Path("/workspace/gai/branches/p2-rx-null/configs/shen_p2a8/g1_registration.json")


def main():
    P = load_protocol()
    rxs = P["dev_receivers"]
    with ProcessPoolExecutor(len(rxs)) as ex:
        mans = list(ex.map(build_rx_cache, rxs))
    expected = {}
    if G1_REG.is_file():
        for it in json.loads(G1_REG.read_text())["files"]:
            if it["relative"].endswith("_train.h5"):
                expected[it["receiver"]] = it["expected_sha256"]
    report = []
    for m in mans:
        rx = m["rx"]
        r = dict(rx=rx, source_sha256=m["source_sha256"], packets=m["packets"],
                 sha_vs_g1_registration=("match" if expected.get(rx) == m["source_sha256"] else
                                         "absent" if rx not in expected else "MISMATCH"))
        theirs = RL1_CACHE / f"shen_{rx}_train"
        if theirs.is_dir():
            ours_iq = np.load(CACHE / rx / "iq.npy", mmap_mode="r")
            r["iq_equal_rl1_cache"] = bool(np.array_equal(np.load(theirs / "iq.npy", mmap_mode="r"), ours_iq))
            r["rows_equal_rl1_cache"] = bool(np.array_equal(np.load(theirs / "rows.npy"), np.load(CACHE / rx / "rows.npy")))
            r["labels_equal_rl1_cache"] = bool(np.array_equal(np.load(theirs / "labels.npy"), np.load(CACHE / rx / "labels.npy")))
        else:
            r["iq_equal_rl1_cache"] = "absent (rx not in rl1 cache)"
        report.append(r)
        print(json.dumps(r))
    bad = [r for r in report if r["sha_vs_g1_registration"] == "MISMATCH" or r.get("iq_equal_rl1_cache") is False
           or r.get("rows_equal_rl1_cache") is False]
    out = ROOT / "00-signal-audit" / "out" / "00-1-cache-manifest.json"
    out.write_text(json.dumps(dict(receivers=report, failures=len(bad)), indent=1))
    if bad:
        raise SystemExit(f"cache verification failed: {bad}")
    print("OK: cache built and verified ->", out)


if __name__ == "__main__":
    main()
