"""CPU tests for the stage-02 pipelines (lib/xa/models2.py) and the rank-16 CIS control."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "tests"))
import xa.models2 as M  # noqa: E402
from test_core import chirp_packet, to_t  # noqa: E402
from xa.frontends import band_limit, rms_normalize  # noqa: E402
from xa.models import build_pipeline, n_params  # noqa: E402


def test_cis_rank16_params():
    assert n_params(build_pipeline(dict(family="shen_cis", head="rank16"))) == 658_314


def test_dispatch_through_build_pipeline():
    assert n_params(build_pipeline(dict(family="ib_variant"))) == 580_008


def test_params_and_shapes():
    x = to_t(chirp_packet(2))
    base = M.build_pipeline2(dict(family="ib_variant"))
    assert n_params(base) == n_params(build_pipeline(dict(family="crossattn", fusion="no_oob", input_fmax=62500))) == 580_008
    assert n_params(M.build_pipeline2(dict(family="ib_rawoob"))) == 625_896
    arms = [dict(family="ib_variant", meanchirp=True), dict(family="ib_variant", combfft=True), dict(family="ib_variant", dechirp=True),
            dict(family="ib_variant", depth=4), dict(family="ib_rawoob", oob_band=[62500, 200000]),
            dict(family="ib_rawoob", oob_band=[62500, 100000], oob_side="L"), dict(family="ib_cisoob"), dict(family="ib_tfsplit")]
    out = {}
    for a in arms:
        m = M.build_pipeline2(a).eval()
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, 10) and torch.isfinite(y).all(), a
        out[str(a)] = n_params(m)
    return out


def test_base_identical_to_stage01_trueib():
    torch.manual_seed(7)
    a = M.build_pipeline2(dict(family="ib_variant")).eval()
    torch.manual_seed(7)
    b = build_pipeline(dict(family="crossattn", fusion="no_oob", input_fmax=62500)).eval()
    x = to_t(chirp_packet(3))
    with torch.no_grad():
        assert torch.equal(a(x), b(x))


def test_meanchirp_is_comb_projection():
    x = rms_normalize(to_t(chirp_packet(2, snr_db=5)))
    B = x.shape[0]
    xm = x.view(B, 2, 8, 1024).mean(2, keepdim=True).expand(B, 2, 8, 1024).reshape(B, 2, 8192)
    Z = torch.fft.fft(torch.complex(xm[:, 0], xm[:, 1]).to(torch.complex128))
    k = torch.arange(8192)
    off = Z[:, k % 8 != 0].abs().max() / Z.abs().max()
    X = torch.fft.fft(torch.complex(x[:, 0], x[:, 1]).to(torch.complex128))
    assert off < 1e-5 and torch.allclose(Z[:, k % 8 == 0], X[:, k % 8 == 0], atol=1e-3)


def test_dechirp_ideal_chirp_is_tone():
    x = torch.from_numpy(np.stack([chirp_packet(1, snr_db=200).real, chirp_packet(1, snr_db=200).imag], 1).astype(np.float32))
    x = band_limit(rms_normalize(x), 62500)
    z = torch.complex(x[:, 0], x[:, 1]) * M.upchirp_tiled().conj()
    S = torch.fft.fft(z).abs() ** 2
    assert S[0, 0] / S.sum() > 0.85, float(S[0, 0] / S.sum())


def test_cis_oob_rows():
    m = M.IBCISOOBPipeline()
    assert len(m.oob_rows) == 35


def test_preln_control():
    torch.manual_seed(5)
    base = M.build_pipeline2(dict(family="ib_variant")).eval()
    torch.manual_seed(5)
    pre = M.build_pipeline2(dict(family="ib_variant", preln=True)).eval()
    assert n_params(pre) == n_params(base) + 128
    sb, sp = base.state_dict(), pre.state_dict()
    assert all(torch.equal(sb[k], sp[k]) for k in sb)          # identical shared initial weights
    x = to_t(chirp_packet(2))
    with torch.no_grad():
        assert not torch.equal(base(x), pre(x)) and torch.isfinite(pre(x)).all()


def test_null_oob_controls_carry_no_packet_information():
    x = to_t(chirp_packet(2))
    for fam, n in (("ib_rawoob", 625_896), ("ib_cisoob", 619_816)):
        torch.manual_seed(3)
        info = M.build_pipeline2(dict(family=fam)).eval()
        torch.manual_seed(3)
        null = M.build_pipeline2(dict(family=fam, oob_null=True)).eval()
        assert n_params(null) == n_params(info) == n
        si, sn = info.state_dict(), null.state_dict()
        assert all(torch.equal(si[k], sn[k]) for k in si)
        proj = null.net.encoder.oob_proj if fam == "ib_rawoob" else null.oob_proj
        seen = []
        h = proj.register_forward_hook(lambda m, inp, out: seen.append(inp[0].abs().max().item()))
        with torch.no_grad():
            null(x)
        h.remove()
        assert seen == [0.0], seen


def test_cis_row_options():
    x = to_t(chirp_packet(2))
    for rows, n in (("oob", 35), ("ib", 17), ("all", 52)):
        m = M.build_pipeline2(dict(family="ib_cisoob", cis_rows=rows)).eval()
        assert len(m.oob_rows) == n and m.oob_pos.shape[1] == n
        with torch.no_grad():
            assert m(x).shape == (2, 10)
    m = M.build_pipeline2(dict(family="ib_cisoob", combfft=True)).eval()
    assert m.embedder.combfft


if __name__ == "__main__":
    print(test_params_and_shapes())
    for fn in (test_cis_rank16_params, test_dispatch_through_build_pipeline, test_base_identical_to_stage01_trueib,
               test_meanchirp_is_comb_projection, test_dechirp_ideal_chirp_is_tone, test_cis_oob_rows,
               test_preln_control, test_null_oob_controls_carry_no_packet_information, test_cis_row_options):
        fn()
        print("PASS", fn.__name__)
