"""Synthetic physics and shape tests; never load benchmark target receivers."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from xa.explore_models import (SmoothComplexFilter, build_explore_model, phase_product,
                               periodic_views, physical_augmentation)
from xa.paper_models import ValidationRF
from xa.models import n_params


def main():
    torch.set_num_threads(1)
    x = torch.randn(2, 2, 256)
    z = torch.complex(x[:, 0], x[:, 1]) * torch.exp(torch.tensor(1.23j))
    rotated = torch.stack((z.real, z.imag), 1)
    for lag in (1, 20):
        assert torch.allclose(phase_product(x, lag), phase_product(rotated, lag), atol=3e-6)
    assert periodic_views(x)[:, :, 160:].count_nonzero() == 0
    filt = SmoothComplexFilter()
    assert torch.allclose(filt(x), x, atol=1e-6)
    torch.manual_seed(7)
    a = ValidationRF(dict(length=256, stem="hierarchical", num_classes=6))
    torch.manual_seed(7)
    b = build_explore_model(dict(frontend="legacy", num_classes=6))
    assert all(torch.equal(a.state_dict()[k], v) for k, v in b.state_dict().items())
    for name in ("legacy", "time", "frequency", "diff", "split", "stf", "filter", "diff_stf", "diff_split"):
        m = build_explore_model(dict(frontend=name, num_classes=6)).eval()
        out = m(x)
        assert out.shape == (2, 6) and torch.isfinite(out).all()
        out.square().sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        print(name, n_params(m))
    gen = torch.Generator().manual_seed(4)
    bounds = dict(shift_samples=2, cfo_hz=2000, channel_log_gain=.1, channel_phase=.1)
    for kind in ("none", "phase", "shift", "cfo", "channel"):
        augmented = physical_augmentation(x, kind, gen, bounds)
        assert augmented.shape == x.shape and torch.isfinite(augmented).all()
    print("PASS phase invariance, STF bounds, identity filter, legacy parity and all model gradients")


if __name__ == "__main__":
    main()
