import unittest
import numpy as np
from stats import packet_stats, group_stats, paired_stats, equal_tail, macro_f1, wilson


class Tests(unittest.TestCase):
    def test_score_scale_and_offset(self):
        rng = np.random.default_rng(5)
        z = rng.normal(size=(20, 10)); y = np.arange(20, dtype=np.int64)%10
        a = packet_stats(z, y); b = packet_stats(7*z+100, y)
        np.testing.assert_allclose(a["normalized_margin"], b["normalized_margin"], atol=1e-13)
        np.testing.assert_array_equal(a["pred"], b["pred"])
        np.testing.assert_allclose(b["margin"], 7*a["margin"])

    def test_ce_and_tie_semantics(self):
        y = np.array([0, 1], dtype=np.int64)
        p = packet_stats(np.zeros((2, 10)), y)
        np.testing.assert_allclose(p["ce"], np.log(10))
        self.assertEqual(p["error"].tolist(), [False, True])
        self.assertTrue(np.all(p["margin"]==0))
        self.assertTrue(np.all(p["normalized_margin"]==0))

    def test_extreme_logits(self):
        z = np.zeros((2, 10)); z[0, 0]=1e4; z[1, 0]=-1e4
        p = packet_stats(z, np.array([0, 0], dtype=np.int64))
        self.assertTrue(np.isfinite(p["ce"]).all())
        self.assertGreaterEqual(p["ce"][1], 10000)

    def test_zero_support_is_missing(self):
        y = np.array([0], dtype=np.int64); p=packet_stats(np.zeros((1, 10)), y)
        g=group_stats(p, np.array([False]), y)
        self.assertIsNone(g["accuracy"]); self.assertEqual(g["n"], 0)

    def test_pair_disagreement_identity(self):
        a={"error":np.array([0, 1, 0, 1], dtype=bool), "pred":np.array([0, 1, 0, 1])}
        b={"error":np.array([0, 0, 1, 1], dtype=bool), "pred":np.array([0, 0, 1, 2])}
        r=paired_stats(a, b, np.ones(4, dtype=bool))
        self.assertEqual([r[k] for k in ["both_correct", "both_wrong", "b0_wrong_ask_correct", "b0_correct_ask_wrong"]], [1, 1, 1, 1])
        self.assertEqual(r["delta_accuracy_ask_minus_b0"], 0)
        self.assertEqual(r["diagnostic_oracle_accuracy"], .75)

    def test_fractional_tail(self):
        self.assertAlmostEqual(equal_tail([1, 2, 3], .5), 8/3)
        self.assertAlmostEqual(equal_tail([1, 2, 3], 1), 2)

    def test_missing_class_macro_and_interval(self):
        score, cm=macro_f1(np.array([0, 0]), np.array([0, 0]))
        self.assertAlmostEqual(score, .1)
        self.assertIsNone(wilson(0, 0))
        self.assertAlmostEqual(wilson(5, 10)[0], 1-wilson(5, 10)[1])


if __name__ == "__main__":
    unittest.main()
