"""CPU checks of post-training interventions without reading any experiment packets."""
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from xa.frontends import band_limit, rms_normalize
from xa.models3 import ChirpPipeline, symbol_residual

spec = importlib.util.spec_from_file_location("mechanisms", ROOT / "03-chirp-structure/03-5-mechanisms.py")
mechanisms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mechanisms)


def test_symbol_interventions():
    x = band_limit(rms_normalize(torch.randn(3, 2, 8192)), 62500)
    raw = symbol_residual(x, normalized=False)
    mean = mechanisms.intervene_symbols(None, (x,), "mean_chirp")[0]
    assert torch.allclose(mean + raw, x, atol=1e-6)
    assert symbol_residual(mean, normalized=False).abs().max() < 1e-6
    shuffled = mechanisms.intervene_symbols(None, (x,), "shuffle_symbol_residual")[0]
    assert torch.equal(shuffled, x - raw + raw.roll(1, 0))


def test_restore_hooks_and_state():
    x = torch.randn(3, 2, 8192)
    for channels, stem in (("resid", "patch"), ("nofft", "symstack")):
        torch.manual_seed(7)
        model = ChirpPipeline(channels=channels, stem=stem).eval()
        state = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            original = model(x)
            if channels == "resid":
                target = model.net.embedder
                modes = ("zero_residual", "shuffle_residual")
            else:
                target = model.net.embedder.cnn_stem
                modes = ("reverse_symbols", "misalign_symbols")
            for mode in modes:
                target.intervention = mode
                output = model(x)
                assert torch.isfinite(output).all() and not torch.equal(output, original)
            target.intervention = None
            if stem == "symstack":
                for mode in ("mean_chirp", "shuffle_symbol_residual"):
                    hook = model.net.embedder.register_forward_pre_hook(
                        lambda module, inputs: mechanisms.intervene_symbols(module, inputs, mode))
                    output = model(x)
                    assert torch.isfinite(output).all() and not torch.equal(output, original)
                    hook.remove()
            assert torch.equal(model(x), original)
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())


if __name__ == "__main__":
    torch.set_num_threads(1)
    for test in (test_symbol_interventions, test_restore_hooks_and_state):
        test()
        print("PASS", test.__name__, flush=True)
