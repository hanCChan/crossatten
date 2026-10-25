"""Paper-validation invariants; synthetic data only."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from xa.frontends import band_limit
from xa.models import build_pipeline, n_params
from xa.models3 import ChirpPipeline
from xa.paper_models import hsic_cca, upstream_module
from xa.paper_train import metrics
from xa.wisig import receivers


def main():
    torch.set_num_threads(1)
    registered = json.loads((ROOT / "06-paper-validation/arms.json").read_text())
    for arm in registered["arms"]:
        model = build_pipeline(arm).eval()
        assert n_params(model) == arm["expected_params"]
        x = torch.randn(2, 2, arm["length"])
        y = model(x)
        assert y.shape == (2, arm["num_classes"]) and torch.isfinite(y).all(), arm
        y.sum().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()), arm
        if arm["family"] == "paper_rf":
            assert model.net.embedder(x)[0].shape == (2, 32, 64)
    torch.manual_seed(1)
    model = build_pipeline(dict(family="paper_rf", length=8192, stem="hierarchical",
                                num_classes=10, input_fmax=62500))
    torch.manual_seed(1)
    old = ChirpPipeline(stem="hierarchical")
    assert model.state_dict().keys() == old.state_dict().keys()
    assert all(torch.equal(v, old.state_dict()[k]) for k, v in model.state_dict().items())
    model.eval(); old.eval()
    x = torch.randn(2, 2, 8192)
    assert torch.equal(model(x), old(x))
    spec = importlib.util.spec_from_file_location("reference_hsic", ROOT / "external/IB-RFF/utils/hsic.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    x, y = torch.randn(16, 32, requires_grad=True), torch.randn(16, 512)
    assert torch.allclose(hsic_cca(x, y), reference.hsic_normalized_cca(x, y), rtol=1e-3, atol=1e-5)
    hsic_cca(x, y).backward()
    assert torch.isfinite(x.grad).all()
    for dataset in ("ManyRx", "ManySig"):
        for fold in range(4):
            train, target = receivers(dataset, fold)
            assert not set(train) & set(target)
    score = metrics(np.eye(6, dtype=np.float32), np.arange(6, dtype=np.int64), 6)
    assert np.asarray(score["confusion"]).shape == (6, 6)
    assert score["acc"] == score["macro_f1"] == 1.0
    x = torch.randn(2, 2, 256)
    filtered = band_limit(x, 7675781.25, 25e6)
    spectrum = torch.fft.fft(torch.complex(filtered[:, 0], filtered[:, 1]))
    outside = torch.fft.fftfreq(256, 1 / 25e6).abs() > 7675781.25
    assert float(spectrum[:, outside].abs().max()) < 1e-4
    print("PASS all registered parameters/shapes/gradients, S03 exact parity, HSIC, splits, 6-class F1, WiFi cutoff")


if __name__ == "__main__":
    main()
