import json
import unittest
from unittest.mock import patch
import numpy as np

from geometry_stats import geometry
from s25_common import HERE, P, authorize_source, write


class SourceGeometryTests(unittest.TestCase):
    def test_forbidden_roles_before_io(self):
        with patch('numpy.load', side_effect=AssertionError('IO must not run')):
            for fold in (0, 1):
                for rx in P['folds'][fold]['dev_val'] + P['confirm_receivers'] + P['blind_receivers_never_open']:
                    with self.assertRaises(PermissionError):
                        authorize_source(fold, rx)
                with self.assertRaises(PermissionError):
                    authorize_source(fold, P['folds'][fold]['source'][0], role='dev')

    def test_scatter_decomposition_and_permutation(self):
        rng = np.random.default_rng(25025)
        tx = np.repeat([0, 1], 12)
        rx = np.tile(np.repeat([0, 1, 2], 4), 2)
        features = rng.normal(size=(24, 8))
        result = geometry(features, tx, rx)
        self.assertLess(result['decomposition_abs_error'], 1e-12)
        order = rng.permutation(24)
        changed = geometry(features[order], tx[order], rx[order])
        for key in result:
            self.assertAlmostEqual(result[key], changed[key], places=11)

    def test_tx_only_centroids_have_zero_rx_scatter(self):
        tx = np.repeat([0, 1], 8)
        rx = np.tile(np.repeat([0, 1], 4), 2)
        features = np.eye(2)[tx]
        result = geometry(features, tx, rx)
        self.assertAlmostEqual(result['between_tx'], .5)
        self.assertEqual(result['conditional_rx'], 0)
        self.assertEqual(result['within_tx_rx'], 0)

    def test_invalid_or_unbalanced_rejected(self):
        with self.assertRaises(ValueError):
            geometry(np.zeros((4, 3)), [0, 0, 1, 1], [0, 1, 0, 1])
        with self.assertRaises(ValueError):
            geometry(np.ones((5, 3)), [0, 0, 0, 1, 1], [0, 0, 1, 0, 1])


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SourceGeometryTests))
    write(HERE/'ENGINEERING_TESTS.json', dict(status='PASS' if result.wasSuccessful() else 'FAIL',
          tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
          new_waveform_reads=0, new_training_steps=0))
    raise SystemExit(0 if result.wasSuccessful() else 1)
