"""Synthetic checks for the prospective sparse-support scatter definitions."""
import unittest

import numpy as np

import s18_stats as dense
import s18_sparse_stats as sparse


def synthetic(sparse_support=False, noise=0, seed=5):
    rng = np.random.default_rng(seed)
    z, y, rx, day = [], [], [], []
    for t in range(4):
        for r in range(6):
            for d in range(2):
                # Different TX/date support means could otherwise leak a large
                # receiver coordinate into the between-TX scatter.
                if sparse_support and r >= 3 and r == 3+(t+d) % 3:
                    continue
                for _ in range(6):
                    z.append(np.array([3*t, 20*r, .4*d, 0, 0, 0])+rng.normal(0, noise, 6))
                    y.append(t); rx.append(r); day.append(d)
    return tuple(np.asarray(v) for v in (z, y, rx, day))


class SparseScatterTests(unittest.TestCase):
    def test_complete_rectangle_matches_original_statistics(self):
        z, y, rx, day = synthetic(noise=.03)
        original = dense.scatters(z, y, rx, day)
        result = sparse.scatters(z, y, rx, day)
        for key in original:
            np.testing.assert_allclose(result[key], original[key], rtol=1e-12, atol=1e-12)
        self.assertTrue(result["support_mask"].all())
        old = dense.fit_directions(z, y, rx, day)
        new = sparse.fit_directions(z, y, rx, day)
        self.assertGreater(dense.overlap(old["U"], new["U"]), 1-1e-10)
        np.testing.assert_allclose(new["epsilon"], old["epsilon"], rtol=1e-12)
        for key in ("offsets", "centroids"):
            np.testing.assert_allclose(sparse.fit_rx_probe(z, y, rx, day)[key], dense.fit_rx_probe(z, y, rx, day)[key], atol=1e-12)

    def test_sparse_common_support_removes_additive_rx_confounding(self):
        z, y, rx, day = synthetic(sparse_support=True)
        result = sparse.scatters(z, y, rx, day)
        np.testing.assert_allclose(np.diag(result["S_T"]), [11.25, 0, 0, 0, 0, 0], atol=1e-12)
        self.assertEqual(result["S_R"][0, 0], 0)
        self.assertGreater(result["S_R"][1, 1], 0)
        self.assertFalse(result["support_mask"].all())
        self.assertTrue(np.isnan(result["cell_means"][~result["support_mask"]]).all())
        for matrix in ("S_R", "S_T"):
            self.assertGreater(np.linalg.eigvalsh(result[matrix]).min(), -1e-10)
        fitted = sparse.fit_directions(z, y, rx, day, rank=1)
        self.assertGreater(abs(fitted["U"][1, 0]), .999999)
        self.assertGreater(abs(fitted["U_tx"][0, 0]), .999999)

    def test_unequal_packet_counts_preserve_observed_cell_weight(self):
        z, y, rx, day = synthetic(sparse_support=True, noise=.001)
        original = sparse.scatters(z, y, rx, day)
        selected = np.flatnonzero((y == 0) & (rx == 0) & (day == 0))
        ids = np.r_[np.arange(len(z)), np.tile(selected, 41)]
        result = sparse.scatters(z[ids], y[ids], rx[ids], day[ids])
        for key in ("S_R", "S_T", "center"):
            np.testing.assert_allclose(result[key], original[key], atol=1e-10)

    def test_no_common_support_fails(self):
        z, y, rx, day = synthetic()
        keep = ((y == 0) & (rx < 2)) | ((y > 0) & (rx >= 2))
        with self.assertRaisesRegex(ValueError, "common receivers"):
            sparse.scatters(z[keep], y[keep], rx[keep], day[keep])

    def test_missing_tx_day_or_single_receiver_group_fails(self):
        z, y, rx, day = synthetic()
        for keep in (~((y == 0) & (day == 0)), ~((y == 0) & (day == 0) & (rx != 0))):
            with self.assertRaisesRegex(ValueError, "TX/day group"):
                sparse.scatters(z[keep], y[keep], rx[keep], day[keep])

    def test_sparse_probe_fixed_support_and_equal_cell_scoring(self):
        z, y, rx, day = synthetic(sparse_support=True, noise=.001)
        a, b = np.arange(0, len(z), 2), np.arange(1, len(z), 2)
        fitted = sparse.fit_rx_probe(z[a], y[a], rx[a], day[a])
        result = sparse.score_rx_probe(fitted, z[b], y[b], rx[b], day[b])
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["chance"], 1/6)
        self.assertTrue(np.isnan(result["cell_accuracy"][~fitted["support_mask"]]).all())
        selected = b[(y[b] == 0) & (rx[b] == 0) & (day[b] == 0)]
        unequal = np.r_[b, np.tile(selected, 31)]
        more = sparse.score_rx_probe(fitted, z[unequal], y[unequal], rx[unequal], day[unequal])
        np.testing.assert_allclose(more["per_rx"], result["per_rx"])
        keep = ~((y[b] == 0) & (rx[b] == 0) & (day[b] == 0))
        with self.assertRaisesRegex(ValueError, "support mask"):
            sparse.score_rx_probe(fitted, z[b][keep], y[b][keep], rx[b][keep], day[b][keep])


if __name__ == "__main__":
    unittest.main()
