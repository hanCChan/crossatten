"""CPU invariants for the pre-registered stage-03 single-factor models."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from xa.models import build_pipeline, n_params
from xa.models2 import IBVariantPipeline, upchirp_tiled
from xa.models3 import ChirpPipeline, ifreq_deviation, legacy5_views, nofft_views, symbol_residual
from xa.frontends import band_limit, rms_normalize


def test_exact_legacy_control():
    for preln in (False, True):
        torch.manual_seed(19)
        a = IBVariantPipeline(preln=preln).eval()
        torch.manual_seed(19)
        b = ChirpPipeline(preln=preln).eval()
        sa, sb = a.state_dict(), b.state_dict()
        assert sa.keys() == sb.keys()
        assert all(torch.equal(sa[k], sb[k]) for k in sa)
        with torch.no_grad():
            x = torch.randn(2, 2, 8192)
            assert torch.equal(a(x), b(x))


def test_nofft_views_equal_legacy():
    x = torch.randn(2, 2, 8192)
    assert torch.equal(nofft_views(x), legacy5_views(x)[:, [0, 1, 3, 4]])


def test_residual():
    x = torch.randn(2, 2, 1024).repeat(1, 1, 8)
    r = symbol_residual(x)
    assert torch.count_nonzero(r) == 0, r.abs().max()
    with torch.no_grad():
        assert torch.isfinite(ChirpPipeline(channels="resid").eval()(x)).all()
    x = torch.randn(3, 2, 8192)
    raw = symbol_residual(x, normalized=False)
    mean = x.reshape(3, 2, 8, 1024).mean(2, keepdim=True).expand(-1, -1, 8, -1).reshape_as(x)
    assert torch.allclose(mean + raw, x, atol=1e-6)
    assert torch.allclose(symbol_residual(x).square().sum(1).mean(-1), torch.ones(3), atol=1e-6)


def test_symstack_locality_and_shared_init():
    torch.manual_seed(23)
    a = ChirpPipeline(channels="nofft").eval()
    torch.manual_seed(23)
    b = ChirpPipeline(channels="nofft", stem="symstack").eval()
    assert n_params(a) == n_params(b)
    sa, sb = a.state_dict(), b.state_dict()
    assert all(torch.equal(sa[k].reshape(-1), sb[k].reshape(-1)) for k in sa)
    x = torch.randn(1, 4, 8192)
    stem = b.net.embedder.cnn_stem
    with torch.no_grad():
        h = stem(x)
        for p in (0, 31, 32, 510, 1023):
            y = x.clone()
            y[:, 0, 3 * 1024 + p] += 10
            changed = torch.nonzero((stem(y) - h).abs().amax(-1)[0] > 1e-7).flatten().tolist()
            allowed = set(range(max(0, (p - 5) // 32), min(31, (p + 5) // 32) + 1))
            assert changed and set(changed) <= allowed, (p, changed, allowed)


def test_ifreq():
    z = upchirp_tiled()
    x = torch.stack([z.real, z.imag], 0)[None]
    p = torch.arange(8192) % 1024
    keep = (p > 2) & (p < 1021)
    assert ifreq_deviation(x)[0, 0, keep].abs().median() < 0.005


def test_params_shapes_gradients():
    arms = json.loads((ROOT / "03-chirp-structure/arms.json").read_text())["arms"]
    counts = {}
    x = band_limit(rms_normalize(torch.randn(2, 2, 8192)), 62500)
    for arm in arms:
        m = build_pipeline(arm).eval()
        n = n_params(m)
        assert n == arm["expected_params"], (arm["name"], n)
        h, _ = m.net.embedder(x)
        assert h.shape == (2, 32, 64)
        out = m(x)
        assert out.shape == (2, 10) and torch.isfinite(out).all()
        out.square().sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        counts[arm["name"]] = n
    print(json.dumps(counts))


if __name__ == "__main__":
    torch.set_num_threads(1)
    for test in (test_exact_legacy_control, test_nofft_views_equal_legacy, test_residual,
                 test_symstack_locality_and_shared_init, test_ifreq, test_params_shapes_gradients):
        test()
        print("PASS", test.__name__, flush=True)
