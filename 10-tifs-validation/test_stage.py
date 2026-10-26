import unittest
import numpy as np
import torch
from features import phase_view, paired_model
from stage import cells, s09


class ReliabilityTests(unittest.TestCase):
    def test_hard_parity_and_capacity(self):
        torch.manual_seed(3)
        base = torch.randn(8, 3, 256)
        torch.testing.assert_close(phase_view(base, "hard"),
                                   s09.phase_features_from_base(base, "first"), rtol=0, atol=0)
        for dataset in ("ManyRx", "ManySig"):
            hard, a = paired_model(dataset, "hard", 42)
            soft, b = paired_model(dataset, "soft", 42)
            self.assertEqual(a, b)
            self.assertEqual(sum(p.numel() for p in hard.parameters()),
                             sum(p.numel() for p in soft.parameters()))
            for k, v in hard.state_dict().items():
                torch.testing.assert_close(v, soft.state_dict()[k], rtol=0, atol=0)

    def test_exact_weight_and_preserved_base(self):
        base = torch.randn(4, 3, 256, dtype=torch.float64)
        out = phase_view(base)
        torch.testing.assert_close(out[:, :3], base, rtol=0, atol=0)
        z = torch.complex(base[:, 0], base[:, 1])
        a = z.abs()
        p = z[:, 1:] * z[:, :-1].conj() / (a[:, 1:] * a[:, :-1]).clamp_min(1e-6)
        r = a.square() / (a.square() + .05 ** 2)
        torch.testing.assert_close(out[:, 3, 1:], (p * r[:, 1:] * r[:, :-1]).real)
        self.assertTrue((out[:, 3:, 0] == 0).all())

    def test_zero_sparse_and_gradients(self):
        for scale in (0., 1e-12, .001, .05, 1.):
            x = (torch.randn(2, 3, 16, dtype=torch.float64) * scale).requires_grad_()
            y = phase_view(x)
            y.square().sum().backward()
            self.assertTrue(torch.isfinite(y).all())
            self.assertTrue(torch.isfinite(x.grad).all())
        x = (torch.randn(1, 3, 8, dtype=torch.float64) * .05).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(phase_view, (x,)))

    def test_continuity_and_bounds(self):
        base = torch.ones(2, 3, 16)
        base[:, 1] = 0
        base[0, 0] = .05 - 1e-7
        base[1, 0] = .05 + 1e-7
        out = phase_view(base)[:, 3:]
        self.assertLess(float((out[0] - out[1]).abs().max()), 1e-4)
        self.assertLessEqual(float(out.abs().max()), 1.)

    def test_source_guard(self):
        job = dict(dataset="ManyRx", fold=0)
        with self.assertRaises(ValueError):
            cells(job, ["1-1"], (1, 2), "fit")
        with self.assertRaises(ValueError):
            cells(job, ["8-7"], (3, 4), "held")


if __name__ == "__main__":
    unittest.main()
