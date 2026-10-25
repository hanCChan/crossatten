"""Synthetic eight-upchirp preamble, CFO, payload-only and noise controls."""
import importlib.util
from pathlib import Path

import numpy as np

path = Path(__file__).resolve().parents[1] / "07-wifi-osu/07-4-osu-detect.py"
spec = importlib.util.spec_from_file_location("s07_osu", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rng = np.random.default_rng(4)
n = np.arange(1024)
up = np.exp(1j*np.pi*.125*(n*n/1024-n))
packet = np.concatenate([np.tile(up, 8), np.roll(up, 64), np.roll(up, 128),
                         np.tile(up.conj(), 2), up.conj()[:256], np.roll(up, 80)])
x = .005*(rng.normal(size=60000) + 1j*rng.normal(size=60000))
start = 6000
x[start:start + len(packet)] += packet
x *= np.exp(2j*np.pi*2000*np.arange(len(x))/1e6)
records, _ = module.detect(x.astype("complex64"))
assert len(records) == 1, records
assert abs(records[0]["start"] - start) <= 32, records
noise = rng.normal(size=60000) + 1j*rng.normal(size=60000)
assert not module.detect(noise.astype("complex64"))[0]
assert not module.detect(np.tile(up, 60).astype("complex64"))[0]
print("PASS synthetic preamble boundary, CFO, noise and repeating-payload rejection")
