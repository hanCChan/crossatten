import unittest
import torch
from stage import ask_modules
from robustness import perturb
from source_gate import gate


class EvidenceTests(unittest.TestCase):
    def test_non_circular_shift(self):
        x = torch.arange(16.).reshape(1, 2, 8)
        y = perturb(x, "shift", 2, 1)
        self.assertTrue((y[:, :, :2] == 0).all())
        torch.testing.assert_close(y[:, :, 2:], x[:, :, :-2])
        y = perturb(x, "shift", -2, 1)
        self.assertTrue((y[:, :, -2:] == 0).all())
        torch.testing.assert_close(y[:, :, :-2], x[:, :, 2:])

    def test_ask_unitary_and_gradient(self):
        official = ask_modules()
        model = official.build_model(__import__("argparse").Namespace(
            candidates=[8, 16, 24, 32, 48, 64], patch_size=64, embed_dim=128, mlp_ratio=2.), 6)
        x = torch.randn(4, 2, 256)
        model(x).square().sum().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        layer = model.koopman
        z = torch.randn(4, layer.n_atoms, dtype=torch.complex64)
        torch.testing.assert_close(layer(z).abs().square().sum(1), z.abs().square().sum(1),
                                   rtol=1e-5, atol=1e-5)

    def test_gate_failure_not_rescued(self):
        rows = [dict(dataset=d, fold=f, group=g, seed=s, rx=f"r{f}",
                     delta_pp=.5 if d == "ManyRx" else 1.)
                for d in ("ManyRx", "ManySig") for f in range(4)
                for g in ("first", "last") for s in (8101, 8102)]
        passed, _, _ = gate(rows, {})
        self.assertFalse(passed)
        for r in rows:
            r["delta_pp"] = 1.1
        self.assertTrue(gate(rows, {})[0])
        rows[-1]["delta_pp"] = -100
        self.assertFalse(gate(rows, {})[0])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
