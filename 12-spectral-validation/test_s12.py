import unittest
from unittest.mock import patch
import numpy as np
import torch
import core
import training
from diagnostics import geometry, scatter_diagnostics
from register import job, protocol
from spectral import estimate, paired_model
from report import gate, verify_pair


def fixture():
    rng = np.random.default_rng(12)
    y = np.repeat(np.arange(3), 3 * 2 * 20)
    rx = np.tile(np.repeat(np.arange(3), 2 * 20), 3)
    day = np.tile(np.repeat([1, 2], 20), 9)
    x = rng.normal(size=(len(y), 2, 256)).astype(np.float32)
    return dict(x=x, y=y, rx=rx, day=day)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_single_sample_rejected_without_nan(self):
        data = fixture()
        with self.assertRaisesRegex(ValueError, "min_cell >= 2"):
            estimate(np.zeros((len(data["y"]), 5, 256)), data["y"], data["rx"],
                     data["day"], min_cell=1)

    def test_estimator_unchanged_for_legal_cells(self):
        from spectral import old
        data = fixture()
        v = np.random.default_rng(3).normal(size=(len(data["y"]), 5, 256))
        v += data["y"][:, None, None]
        a = estimate(v, data["y"], data["rx"], data["day"])
        b = old.estimate(v, data["y"], data["rx"], data["day"])
        np.testing.assert_array_equal(a["weight"], b["weight"])
        self.assertTrue(np.isfinite(a["weight"]).all())

    def test_smoke_covers_every_observed_cell(self):
        data = fixture()
        ids = core.stratified_ids(data, 100, 42)
        expected = set(zip(data["y"], data["rx"], data["day"]))
        observed = set(zip(data["y"][ids], data["rx"][ids], data["day"][ids]))
        self.assertEqual(expected, observed)
        self.assertEqual(len(np.unique(ids)), 100)
        np.testing.assert_array_equal(ids, core.stratified_ids(data, 100, 42))
        with self.assertRaisesRegex(ValueError, "cannot cover"):
            core.stratified_ids(data, 5, 42)

    def test_manyrx_smoke_budgets_cover_full_domain_cells(self):
        y, rx, day = np.meshgrid(np.arange(10), np.arange(18), [1, 2], indexing="ij")
        data = dict(y=y.ravel(), rx=rx.ravel(), day=day.ravel(),
                    x=np.zeros((y.size, 2, 256), np.float32))
        splits, _ = core.smoke_subset(dict(train=data, val=data), {}, 8101)
        self.assertEqual(len(splits["train"]["y"]), 360)
        self.assertEqual(len(splits["val"]["y"]), 360)

    def test_smoke_and_pilot_share_full_fit_prior(self):
        data = {k: np.concatenate([v, v]) for k, v in fixture().items()}
        splits = dict(train=data, val={k: v[:120] for k, v in data.items()})
        seals = dict(train="full-train", val="full-val", pool_y="full-pool")
        with patch.object(training, "partition", return_value=(splits, {}, {}, seals)):
            pilot = training.load_arrays(job("pilot", "ManyRx", 0, "F2"))
            smoke = training.load_arrays(job("smoke", "ManyRx", 0, "F2"))
        np.testing.assert_array_equal(pilot[4]["weight"], smoke[4]["weight"])
        self.assertEqual(pilot[4]["split_sha256"], smoke[4]["split_sha256"])
        self.assertEqual(smoke[4]["fit_packets"], len(data["y"]))
        self.assertEqual(len(smoke[0]["train"]["y"]), 512)
        self.assertGreater(smoke[4]["fit_packets"], len(smoke[0]["train"]["y"]))

    def test_prior_does_not_use_validation_or_held(self):
        data = fixture()
        invalid = dict(x=np.full((1, 2, 256), np.nan, np.float32),
                       y=np.array([999]), rx=np.array([999]), day=np.array([4]))
        with patch.object(training, "partition",
                          return_value=(dict(train=data, val=invalid),
                                        {"forbidden": invalid}, {}, {})):
            prior = training.load_arrays(job("pilot", "ManyRx", 0, "F2"))[4]
        self.assertTrue(np.isfinite(prior["weight"]).all())

    def test_target_and_date_guards_before_io(self):
        j = job("pilot", "ManyRx", 0, "F1")
        for names, days, role in (([], (1, 2), "target"),
                                  ([], (3, 4), "fit"),
                                  (["1-1"], (1, 2), "fit")):
            with self.assertRaises(ValueError):
                core.read(j, names, days, role)

    def test_scale_free_ratio_and_zero_nuisance(self):
        p = dict(b=np.ones(256), r=np.ones(256) * 2, d=np.ones(256))
        a = scatter_diagnostics(p, np.ones(256))
        b = scatter_diagnostics(p, np.full(256, .5))
        self.assertEqual(a["J"], b["J"])
        self.assertEqual(b["tx_scatter_retention"], .25)
        self.assertFalse(b["admission_veto"])
        p["r"][:] = 0
        p["d"][:] = 0
        self.assertIsNone(scatter_diagnostics(p, np.ones(256))["J"])

    def test_geometry_constant_scaling_and_zero_inputs(self):
        data = fixture()
        v = np.random.default_rng(7).normal(size=(len(data["y"]), 5, 256)).astype(np.float32)
        a = geometry(v, v, data["y"], data["y"], np.ones(256, np.float32), "F2")
        b = geometry(v, v, data["y"], data["y"], np.full(256, .75, np.float32), "F2")
        self.assertAlmostEqual(a["centroid_acc_pct"], b["centroid_acc_pct"])
        self.assertAlmostEqual(a["unit_between"], b["unit_between"], places=6)
        z = geometry(v * 0, v * 0, data["y"], data["y"], np.ones(256, np.float32), "F2")
        self.assertIsNone(z["minimum_center_angle_rad"])

    def test_shared_initialization_and_forward(self):
        a, ha = paired_model("ManyRx", "hard", 42)
        b, hb = paired_model("ManyRx", "F2", 42, np.ones(256, np.float32))
        self.assertEqual(ha, hb)
        a.eval()
        b.eval()
        x = torch.randn(2, 2, 256)
        torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)
        x.requires_grad_()
        b(x).square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_matrix_and_independent_control_budget(self):
        reg, smoke, pilot = protocol()
        self.assertEqual(len(smoke), 4)
        self.assertEqual(len(pilot), 16)
        self.assertEqual(len({j["id"] for j in pilot}), 16)
        self.assertEqual({j["seed"] for j in pilot}, {8101})
        self.assertEqual(reg["confirmation"]["fresh_paired_hard_fits"], 32)
        self.assertTrue(set(reg["confirmation"]["seeds"]).isdisjoint({8101, 8102}))

    def test_negative_controls_are_deterministic_and_bounded(self):
        from spectral import negative_weights
        data = fixture()
        v = np.random.default_rng(5).normal(size=(len(data["y"]), 5, 256))
        v += data["y"][:, None, None]
        weight = estimate(v, data["y"], data["rx"], data["day"])["weight"]
        a = negative_weights(v, data["y"], data["rx"], data["day"], weight, 12026)
        b = negative_weights(v, data["y"], data["rx"], data["day"], weight, 12026)
        np.testing.assert_array_equal(np.sort(a["shuffled"]), np.sort(weight))
        for key, value in a.items():
            np.testing.assert_array_equal(value, b[key])
            self.assertTrue(np.isfinite(value).all())
            self.assertTrue(((value >= .5) & (value <= 1)).all())

    def test_frozen_gate_rejects_local_receiver_regression(self):
        reg, _, _ = protocol()
        rows = [dict(dataset=d, fold=f, group="first", seed=8101,
                     rx=f"rx{f}", delta_pp=2 if d == "ManyRx" else 0)
                for d in ("ManyRx", "ManySig") for f in range(4)]
        self.assertTrue(gate(rows, reg["source_gate"])["admitted"])
        rows[0]["delta_pp"] = -3
        result = gate(rows, reg["source_gate"])
        self.assertFalse(result["admitted"])
        self.assertFalse(result["checks"]["worst_rx"])
        with self.assertRaisesRegex(ValueError, "pairing"):
            verify_pair(dict(params=1), dict(params=2))

    def test_real_prior_network_has_finite_gradients(self):
        from spectral import center_weight
        for kind, weight in (("F1", center_weight()),
                             ("F2", np.linspace(.5, 1, 256, dtype=np.float32))):
            model, _ = paired_model("ManyRx", kind, 42, weight)
            x = torch.randn(2, 2, 256, requires_grad=True)
            model(x).square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                if p.grad is not None))


if __name__ == "__main__":
    unittest.main()
