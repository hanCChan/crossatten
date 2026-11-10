"""Scientific unit checks on synthetic data only; no source or target reads."""
import unittest

import numpy as np

from s18_stats import (compress, direction_metrics, fit_directions, fit_rx_probe,
                       overlap, random_basis, scatters, score_rx_probe)


def synthetic(noise=0, seed=19):
    rng = np.random.default_rng(seed)
    z, y, rx, day = [], [], [], []
    for t in range(4):
        for r in range(5):
            for d in range(2):
                for _ in range(8):
                    # TX spans coordinate 0, RX spans 1, date spans 2.
                    z.append([3 * t, 2 * r, .3 * d, 0, 0, 0] + rng.normal(0, noise, 6))
                    y.append(t)
                    rx.append(r)
                    day.append(d)
    return tuple(np.asarray(v) for v in (z, y, rx, day))


class ScatterTests(unittest.TestCase):
    def test_population_scatter_and_separation(self):
        z, y, rx, day = synthetic()
        result = fit_directions(z, y, rx, day, rank=1)
        np.testing.assert_allclose(np.diag(result["S_R"]), [0, 8, 0, 0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(np.diag(result["S_T"]), [11.25, 0, 0, 0, 0, 0], atol=1e-12)
        self.assertGreater(abs(result["U"][1, 0]), .999999)
        self.assertGreater(abs(result["U_tx"][0, 0]), .999999)
        self.assertLess(result["selective_metrics"]["tx_fraction"], 1e-12)
        self.assertAlmostEqual(result["selective_metrics"]["rx_fraction"], 1)
        self.assertAlmostEqual(result["epsilon"], 1e-3 * (8 + 11.25) / 6)

    def test_packet_imbalance_does_not_reweight_cells(self):
        z, y, rx, day = synthetic()
        reference = scatters(z, y, rx, day)
        # Duplicate all packets from one cell while leaving its mean unchanged.
        selected = np.flatnonzero((y == 0) & (rx == 0) & (day == 0))
        ids = np.concatenate((np.arange(len(z)), np.tile(selected, 31)))
        result = scatters(z[ids], y[ids], rx[ids], day[ids])
        for name in ("S_R", "S_T", "center", "cell_means"):
            np.testing.assert_allclose(result[name], reference[name], atol=1e-12)
        self.assertEqual(result["cell_counts"][0, 0, 0], 32 * 8)

    def test_missing_cells_fail_closed(self):
        z, y, rx, day = synthetic()
        keep = ~((y == 1) & (rx == 3) & (day == 0))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            scatters(z[keep], y[keep], rx[keep], day[keep])

    def test_generalized_subspace_orthonormal_and_stable(self):
        z, y, rx, day = synthetic(noise=.005)
        result = fit_directions(z, y, rx, day, rank=3)
        np.testing.assert_allclose(result["U"].T @ result["U"], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(result["U_tx"].T @ result["U_tx"], np.eye(3), atol=1e-12)
        # A reparameterized basis must have exactly the same principal overlap.
        rotation = random_basis(3, 3, 123)
        self.assertAlmostEqual(overlap(result["U"], result["U"] @ rotation), 1)
        u = random_basis(6, 3, 119)
        np.testing.assert_array_equal(u, random_basis(6, 3, 119))
        np.testing.assert_allclose(u.T @ u, np.eye(3), atol=1e-12)

    def test_compression_is_bounded_and_preserves_complement(self):
        z, y, rx, day = synthetic(noise=.01)
        result = fit_directions(z, y, rx, day, rank=3)
        u, center = result["U"], result["center"]
        after = compress(z, center, u, alpha=.5)
        np.testing.assert_allclose((after - center) @ u, .5 * (z - center) @ u, atol=1e-12)
        complement = np.eye(6) - u @ u.T
        np.testing.assert_allclose((after - center) @ complement, (z - center) @ complement, atol=1e-12)
        self.assertTrue(np.all(np.linalg.norm(after - center, axis=-1) <= np.linalg.norm(z - center, axis=-1) + 1e-12))
        np.testing.assert_array_equal(compress(z, center, u, alpha=0), z)
        # Shared token-channel transform accepts unpooled arrays as well.
        token = np.stack((z, z + .1), axis=1)
        self.assertEqual(compress(token, center, u).shape, token.shape)
        with self.assertRaises(ValueError):
            compress(z, center, u, alpha=1.1)

    def test_fixed_conditional_rx_probe(self):
        z, y, rx, day = synthetic(noise=.001)
        probe = fit_rx_probe(z[::2], y[::2], rx[::2], day[::2])
        result = score_rx_probe(probe, z[1::2], y[1::2], rx[1::2], day[1::2])
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["chance"], .2)
        basis = np.eye(z.shape[1])[:, 1:2]
        compressed_probe = fit_rx_probe(compress(z[::2], probe["offsets"].mean(axis=(0, 1)), basis, 1), y[::2], rx[::2], day[::2])
        # A fully removed receiver coordinate has zero receiver centroid spread.
        self.assertLess(np.ptp(compressed_probe["centroids"][:, 1]), 1e-12)

    def test_invalid_inputs(self):
        z, y, rx, day = synthetic()
        bad = z.copy()
        bad[0, 0] = np.nan
        for rank in (0, 7, 1.5):
            with self.assertRaises(ValueError):
                fit_directions(z, y, rx, day, rank=rank)
        with self.assertRaises(ValueError):
            scatters(bad, y, rx, day)
        with self.assertRaises(ValueError):
            overlap(np.ones((6, 3)), np.ones((6, 3)))
        empty = np.zeros((6, 6))
        metrics = direction_metrics(empty, empty, np.eye(6)[:, :3])
        self.assertEqual(metrics["rx_fraction"], 0)
        self.assertTrue(np.isfinite(metrics["rho"]).all())


if __name__ == "__main__":
    unittest.main()
