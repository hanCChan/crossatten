import torch
import pytest
from phase_features import phase_features_from_base


def make_base(n=256, seed=17):
    torch.manual_seed(seed)
    k = torch.arange(n, dtype=torch.float64)
    amp = 0.6 + torch.rand(n, dtype=torch.float64)
    phase = 0.10 * k + 0.27 * torch.sin(k * (2 * torch.pi / n))
    X = torch.polar(amp, phase)
    x = torch.fft.ifft(torch.fft.ifftshift(X))
    return x.unsqueeze(0)


def base_from_iq(x):
    X = torch.fft.fftshift(torch.fft.fft(x), dim=-1)
    norm = X.abs().square().mean(-1, keepdim=True).clamp_min(1e-6).sqrt()
    mag = torch.log1p(X.abs())
    mag = (mag - mag.mean(-1, keepdim=True)) / mag.std(-1, keepdim=True).clamp_min(1e-6)
    return torch.stack((X.real / norm, X.imag / norm, mag), 1)


def test_exact_s08_first_order_parity():
    base = base_from_iq(make_base())
    z = torch.complex(base[:, 0], base[:, 1])
    amp = z.abs()
    prod = z[:, 1:] * z[:, :-1].conj()
    valid = (amp[:, 1:] >= .05) & (amp[:, :-1] >= .05)
    prod = prod / (amp[:, 1:] * amp[:, :-1]).clamp_min(1e-6)
    old = torch.nn.functional.pad(torch.stack((prod.real, prod.imag), 1) * valid[:, None], (1, 0))
    out = phase_features_from_base(base, "first")
    torch.testing.assert_close(out[:, :3], base)
    torch.testing.assert_close(out[:, 3:], old)


def test_rotation_and_circular_shift():
    x = make_base()
    shifted = torch.roll(x * torch.exp(1j * torch.tensor(1.23)), shifts=7, dims=-1)
    b0, b1 = base_from_iq(x), base_from_iq(shifted)
    for kind in ("first", "centered", "curvature"):
        u = phase_features_from_base(b0, kind)[:, 3:]
        v = phase_features_from_base(b1, kind)[:, 3:]
        if kind == "first":
            assert (u - v).abs().max() > 0.05
        else:
            torch.testing.assert_close(u, v, atol=1e-9, rtol=1e-9)


def test_finite_sparse_spectrum_and_zero_null():
    b = torch.zeros((2, 3, 256), dtype=torch.float64)
    b[0, 0, 10] = 1.0
    for kind in ("first", "centered", "curvature", "null", "frequency"):
        o = phase_features_from_base(b, kind)
        assert torch.isfinite(o).all()
        assert o.shape == (2, 3 if kind == "frequency" else 5, 256)
        if kind == "null":
            assert torch.count_nonzero(o[:, 3:]) == 0


def test_gradients_finite():
    b = base_from_iq(make_base()).float()
    for kind in ("first", "centered", "curvature"):
        c = b.detach().clone().requires_grad_()
        o = phase_features_from_base(c, kind)
        o.square().mean().backward()
        assert c.grad is not None and torch.isfinite(c.grad).all()


@pytest.mark.parametrize("mode", ["curvature", "centered"])
def test_phase_features_not_supposed_to_be_invariant_to_arbitrary_channel(mode):
    x = make_base()
    X = torch.fft.fft(x)
    k = torch.arange(x.shape[-1], dtype=torch.float64)
    channel = torch.exp(1j * 0.3 * torch.sin(k * (2*torch.pi/x.shape[-1]) * 3))
    y = torch.fft.ifft(X * channel)
    a = phase_features_from_base(base_from_iq(x), mode)[:, 3:]
    b = phase_features_from_base(base_from_iq(y), mode)[:, 3:]
    assert (a - b).abs().max() > 1e-4
