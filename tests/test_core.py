"""Core correctness tests (CPU). Run: python tests/test_core.py   (or pytest)"""
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from xa.frontends import (awgn, band_limit, cis_row_freqs, oob_keep_mask, rms_normalize, shen_cis,  # noqa: E402
                          shen_cis_numpy_reference)
from xa.guard import AccessDenied, check_raw_path, check_rx  # noqa: E402
from xa.models import build_pipeline, n_params  # noqa: E402
from xa.train import stable_seed  # noqa: E402

FS, L = 1e6, 8192


def chirp_packet(n=4, snr_db=30, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(1024) / FS
    c = np.exp(1j * (np.pi * 125e3 / 1.024e-3 * t ** 2 - np.pi * 125e3 * t))
    x = np.tile(c, 8)[None].repeat(n, 0) * (1 + 0.05 * rng.standard_normal((n, 1)))
    x = x + 10 ** (-snr_db / 20) * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)) / np.sqrt(2)
    return x


def to_t(x):
    return torch.from_numpy(np.stack([x.real, x.imag], 1).astype(np.float32))


def test_param_counts():
    exp = {("crossattn", "cross_attn_oob", None): 625_896, ("crossattn", "no_oob", None): 580_008,
           ("shen_cis", None, "flatten"): 13_442_954, ("shen_cis", None, "gap"): 270_218}
    for (fam, fusion, head), n in exp.items():
        arm = dict(family=fam, fusion=fusion, head=head)
        got = n_params(build_pipeline({k: v for k, v in arm.items() if v is not None}))
        assert got == n, (fam, fusion, head, got, n)
    tf = n_params(build_pipeline(dict(family="tfsplit")))
    assert abs(tf - 625_896) / 625_896 < 0.02, tf


def test_forward_shapes():
    x = to_t(chirp_packet(2))
    arms = [dict(family="crossattn"), dict(family="crossattn", oob_keep=[62.5e3, 200e3]),
            dict(family="crossattn", input_fmax=200e3), dict(family="crossattn", fusion="no_oob", input_fmax=62.5e3),
            dict(family="tfsplit"), dict(family="shen_cis", head="flatten"), dict(family="shen_cis", head="gap"),
            dict(family="shen_cis", head="flatten", input_fmax=62.5e3)]
    for a in arms:
        m = build_pipeline(a).eval()
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, 10) and torch.isfinite(y).all(), a


def test_cis_port_matches_numpy():
    x = chirp_packet(3, snr_db=25)
    ref = shen_cis_numpy_reference(x)
    got = shen_cis(to_t(x)).numpy()
    assert ref.shape == got.shape == (3, 1, 52, 126)
    # reference on the same float32-quantised I/Q, computed in complex128 (numpy keeps complex64 for float32*1j)
    t = to_t(x).double().numpy()
    ref32 = shen_cis_numpy_reference(t[:, 0] + 1j * t[:, 1])
    assert np.max(np.abs(ref32 - got)) < 1e-5, np.max(np.abs(ref32 - got))
    f = cis_row_freqs()
    assert len(f) == 52 and abs(f[0] + 203125) < 1e-6 and abs(f[-1] - 195312.5) < 1e-6


def test_band_limit_and_masks():
    x = rms_normalize(to_t(chirp_packet(2, snr_db=10)))
    y = band_limit(x, 62.5e3)
    Z = torch.fft.fft(torch.complex(y[:, 0], y[:, 1]).to(torch.complex128))
    f = torch.fft.fftfreq(L, 1 / FS, dtype=torch.float64)
    out = Z[:, f.abs() > 62.5e3].abs().max() / Z.abs().max()
    assert out < 1e-5, out
    X = torch.fft.fft(torch.complex(x[:, 0], x[:, 1]).to(torch.complex128))
    keep = f.abs() <= 62.5e3
    assert torch.allclose(Z[:, keep], X[:, keep], atol=1e-3)
    m = oob_keep_mask(L, 62.5e3, 200e3)
    assert int(m.sum()) == 2 * (1638 - 512), int(m.sum())


def test_awgn_snr():
    x = to_t(chirp_packet(64, snr_db=80))
    g = torch.Generator().manual_seed(0)
    y = awgn(x, 20.0, 20.0 + 1e-9, g)
    n = y - x
    snr = 10 * torch.log10((x ** 2).sum((1, 2)) / (n ** 2).sum((1, 2)))
    assert abs(float(snr.mean()) - 20.0) < 0.3, float(snr.mean())


def test_guard():
    import xa.guard as G
    real_log = G.ACCESS_LOG
    G.ACCESS_LOG = Path(tempfile.mkdtemp()) / "access.jsonl"   # keep the real access log free of test entries
    try:
        _guard_checks()
    finally:
        G.ACCESS_LOG = real_log


def _guard_checks():
    for bad in ["b200_2_test.h5", "b200_mini_2_test.h5", "Location_A_n210_2_test.h5", "pluto_2_test.h5", "n210_3_test.h5"]:
        try:
            check_raw_path(bad)
            raise AssertionError(bad)
        except AccessDenied:
            pass
    for ok in ["b200_mini_1_train.h5", "n210_1_train.h5", "rtl_2_train.h5"]:
        check_raw_path(ok)
    for purpose in ("source", "dev_val", "audit"):
        try:
            check_rx("rtl_4", purpose, "test")
            raise AssertionError(purpose)
        except AccessDenied:
            pass
    try:
        check_rx("n210_1", "confirm", "test", decision_file=None)
        raise AssertionError("confirm without decision")
    except AccessDenied:
        pass
    with tempfile.NamedTemporaryFile() as tf:
        check_rx("n210_1", "confirm", "test", decision_file=tf.name)


def test_stable_seed_cross_process():
    code = "import sys;sys.path.insert(0,'lib');from xa.train import stable_seed;print(stable_seed('P-H5x160',0,101))"
    outs = {subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, env={"PYTHONHASHSEED": s}).strip() for s in ("1", "2")}
    assert len(outs) == 1 and int(outs.pop()) == stable_seed("P-H5x160", 0, 101)


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__, flush=True)
    print(json.dumps(dict(passed=len(fns))))
