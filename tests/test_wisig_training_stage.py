"""S08 CPU tests: paired roles, physical features, EMA and source gates."""
import copy
import importlib.util
import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08-wisig-training"))
import training

spec = importlib.util.spec_from_file_location("s08_select_test", ROOT / "08-wisig-training/08-3-select.py")
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


class StageTrainingTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_groups_are_source_only_disjoint_and_complete(self):
        from xa.wisig import receivers
        for dataset in ("ManySig", "ManyRx"):
            for fold in range(4):
                source, target = receivers(dataset, fold)
                groups = [training.source_groups(dataset, fold, g) for g in ("first", "last")]
                self.assertFalse(set(groups[0][1]) & set(groups[1][1]))
                for fit, held in groups:
                    self.assertFalse(set(fit) & set(held))
                    self.assertFalse(set(held) & set(target))
                    self.assertEqual(set(fit) | set(held), set(source))
        with self.assertRaises(ValueError):
            training.source_groups("ManySig", 0, "invalid")

    def test_frequency_baseline_matches_frozen_model(self):
        from xa.explore_models import build_explore_model
        torch.manual_seed(12)
        reference = build_explore_model(dict(frontend="frequency", num_classes=6)).eval()
        model = training.build_model("ManySig").eval()
        model.load_state_dict(reference.state_dict())
        x = torch.randn(3, 2, 256)
        self.assertTrue(torch.equal(reference(x), model(x)))
        self.assertEqual(training.n_params(model), 90420)
        self.assertEqual(training.n_params(training.build_model("ManyRx")), 90680)

    def test_relative_phase_rotation_mask_and_null_control(self):
        x = torch.randn(4, 2, 256)
        z = torch.complex(x[:, 0], x[:, 1]) * torch.exp(torch.tensor(.7j))
        rotated = torch.stack((z.real, z.imag), 1)
        a, b = training.adjacent_spectrum_views(x), training.adjacent_spectrum_views(rotated)
        self.assertTrue(torch.allclose(a[:, 3:], b[:, 3:], atol=2e-5))
        self.assertTrue(torch.isfinite(training.adjacent_spectrum_views(torch.zeros_like(x))).all())
        self.assertTrue(torch.equal(a[:, 3:, 0], torch.zeros(4, 2)))
        null = training.adjacent_spectrum_views(x, null=True)
        self.assertTrue(torch.equal(a[:, :3], null[:, :3]))
        self.assertEqual(int(torch.count_nonzero(null[:, 3:])), 0)
        self.assertEqual(training.n_params(training.build_model("ManySig", "phase")),
                         training.n_params(training.build_model("ManySig", "phase_null")))
        model = training.build_model("ManySig", "phase")
        model(x).sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_ema_parameters_not_integer_buffers_and_bn_fit_only(self):
        raw = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3), nn.Linear(3, 2))
        ema = AveragedModel(raw, multi_avg_fn=get_ema_multi_avg_fn(.5), use_buffers=False)
        ema.update_parameters(raw)
        previous = [p.clone() for p in ema.module.parameters()]
        with torch.no_grad():
            for p in raw.parameters():
                p.add_(2)
            raw[1].num_batches_tracked.fill_(17)
        ema.update_parameters(raw)
        for old, current in zip(previous, ema.module.parameters()):
            self.assertTrue(torch.allclose(current, old + 1))
        self.assertEqual(int(ema.module[1].num_batches_tracked), 17)
        weights = copy.deepcopy(ema.module.state_dict())
        fit = torch.randn(40, 2) + 4
        training.recalibrate_bn(ema.module, fit, 20)
        self.assertEqual(int(ema.module[1].num_batches_tracked), 2)
        for name, p in ema.module.named_parameters():
            self.assertTrue(torch.equal(p, weights[name]))
        self.assertFalse(ema.module.training)

    def test_paired_gate_rejects_one_group_gain_and_missing_pair(self):
        rows, controls = [], []
        for group in ("first", "last"):
            for seed in (6101, 6102):
                job = dict(group=group, seed=seed)
                ref = dict(summary=dict(acc_mean=.8, ce_mean=.2),
                           per_rx={group: dict(acc=.8)}, epoch=5)
                good = dict(summary=dict(acc_mean=.82, ce_mean=.15),
                            per_rx={group: dict(acc=.82)}, epoch=6)
                controls.append(dict(job=job, entry=ref))
                rows.append(dict(job=job, entry=good))
        self.assertTrue(selection.paired_gate(rows, controls)["supported"])
        rows[-1]["entry"]["summary"]["acc_mean"] = .75
        rows[-1]["entry"]["per_rx"]["last"]["acc"] = .75
        self.assertFalse(selection.paired_gate(rows, controls)["supported"])
        with self.assertRaises(ValueError):
            selection.paired_gate(rows[:-1], controls)

    def test_full_budget_and_minimal_recipe_change(self):
        base = training.recipe_for("official")
        mild = training.recipe_for("official_awgn")
        full = training.recipe_for("official_awgn_full200")
        self.assertNotIn("awgn", base)
        self.assertEqual(mild["awgn"], [10, 50])
        self.assertTrue(mild["stop"])
        self.assertFalse(full["stop"])
        self.assertEqual(full["epochs"], 200)
        self.assertEqual({k: v for k, v in base.items() if k != "stop"},
                         {k: v for k, v in mild.items() if k not in ("stop", "awgn")})


if __name__ == "__main__":
    unittest.main()
