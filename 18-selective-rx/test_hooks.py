"""Synthetic CPU checks: hooks never read scientific data/checkpoints."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '17-rx-distortion'))
from s17_models import build
from s18_hooks import LAYERS, capture, intervene


class HookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = build('ManyRx', 18001).eval()
        cls.x = torch.randn(3, 2, 256)

    def hook_count(self):
        return sum(len(module._forward_hooks) for module in self.model.modules())

    def test_shapes_and_capture_identity(self):
        with torch.no_grad():
            clean = self.model(self.x)
            with capture(self.model) as c:
                actual = self.model(self.x)
                self.model(self.x[:1])
        self.assertTrue(torch.equal(actual, clean))
        self.assertEqual({k: v.shape for k, v in c.arrays().items()},
                         {'stem1': (4, 32), 'stem_mid': (4, 48),
                          'tokens': (4, 64), 'mixer': (4, 64)})
        self.assertEqual(self.hook_count(), 0)
        c.clear()
        with self.assertRaises(ValueError):
            c.arrays()

    def test_zero_alpha_identity_and_state_unchanged(self):
        original = {k: v.clone() for k, v in self.model.state_dict().items()}
        with torch.no_grad():
            clean = self.model(self.x)
            for layer, channels in [('stem1', 32), ('stem_mid', 48),
                                    ('tokens', 64), ('mixer', 64)]:
                u = np.eye(channels)[:, :3]
                with intervene(self.model, layer, u, np.zeros(channels), alpha=0):
                    self.assertTrue(torch.equal(clean, self.model(self.x)))
                with intervene(self.model, layer, u, np.zeros(channels)):
                    self.assertTrue(torch.isfinite(self.model(self.x)).all())
        for k, v in self.model.state_dict().items():
            self.assertTrue(torch.equal(v, original[k]), k)
        self.assertEqual(self.hook_count(), 0)

    def test_channel_projection_matches_manual(self):
        # Attach observer after intervention so its pooled values are the
        # altered outputs; check every channel at every position implicitly.
        for layer, channels in [('stem1', 32), ('stem_mid', 48),
                                ('tokens', 64), ('mixer', 64)]:
            u = np.eye(channels)[:, :3]
            center = np.arange(channels) / channels
            with torch.no_grad():
                with capture(self.model) as clean:
                    self.model(self.x)
                with intervene(self.model, layer, u, center, alpha=.5):
                    with capture(self.model) as changed:
                        self.model(self.x)
            z = clean.arrays()[layer]
            expected = z - .5 * ((z - center) @ u) @ u.T
            np.testing.assert_allclose(changed.arrays()[layer], expected,
                                       atol=3e-7, rtol=3e-6)

    def test_exception_cleanup_and_invalid_basis(self):
        with self.assertRaises(RuntimeError):
            with capture(self.model):
                raise RuntimeError('synthetic')
        with self.assertRaises(RuntimeError):
            with intervene(self.model, 'tokens', np.eye(64)[:, :3], np.zeros(64)):
                raise RuntimeError('synthetic')
        self.assertEqual(self.hook_count(), 0)
        for alpha in [-.1, 1.1, float('nan')]:
            with self.assertRaises(ValueError):
                with intervene(self.model, 'tokens', np.eye(64)[:, :3],
                               np.zeros(64), alpha):
                    pass
        with self.assertRaises(ValueError):
            with intervene(self.model, 'tokens', np.ones((64, 3)), np.zeros(64)):
                pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
