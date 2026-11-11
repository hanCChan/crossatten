"""Synthetic Stage B controls; never read scientific arrays/checkpoints or use GPU."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


R = module('B_runtime_test', 'research/controlled_training_b/runner.py')
M = module('B_analysis_test', 'research/controlled_training_b/analyze.py')
OLD = module('old_panel_design_test', 'research/training_instance_attribution/audit.py')


def fixture():
    names = [f'rx{i:02d}' for i in range(18)]
    y, rx, day, keys, train, val, hashes = [], [], [], [], [], [], {}
    for r, name in enumerate(names):
        for date in (1, 2):
            digest = f'synthetic-cell-{name}-{date}'
            hashes[f'{name}|{date}'] = digest
            ordinal = 0
            for tx in range(10):
                for k in range(3):
                    i = len(y)
                    y.append(tx); rx.append(r); day.append(date)
                    keys.append((name, date, digest, ordinal)); ordinal += 1
                    (train if k < 2 else val).append(i)
    arrays = [np.asarray(a, dtype=np.int32) for a in (y, rx, day, train, val)]
    y, rx, day, train, val = arrays
    pool = dict(fit_rx=names, keys=keys, fit_y=y, fit_day=day, train_ids=train, val_ids=val,
                held_rx=['held'], forbidden_target_rx=['target'])
    plan = OLD.panels(pool)
    return y, rx, day, train, val, names, hashes, plan


def predictions(pred, y=(0, 1, 0, 1)):
    logits = np.full((4, 2), -1., dtype=np.float32)
    logits[np.arange(4), pred] = 1.
    return dict(logits=logits, y=np.asarray(y), rx=np.asarray(['r1', 'r1', 'r2', 'r2']),
                day=np.ones(4, dtype=int), keys=np.asarray(['r1|1|s|0', 'r1|1|s|1', 'r2|1|s|0', 'r2|1|s|1']))


class StageBTests(unittest.TestCase):
    def test_panel_exactly_matches_independent_legacy_rule(self):
        args = fixture()
        panels, val = R.build_panels(*args)
        y, rx, day, train, original_val, names, hashes, plan = args
        self.assertEqual(panels[0].dtype, np.dtype('int64'))
        self.assertEqual(val.dtype, np.dtype('int32'))
        np.testing.assert_array_equal(y[panels[0]], y[panels[1]])
        np.testing.assert_array_equal(day[panels[0]], day[panels[1]])
        np.testing.assert_array_equal(panels[0][:240], panels[1][:240])
        self.assertEqual(set(rx[val]), set(range(6)))
        self.assertFalse(set(val) & set(panels[0]))
        self.assertFalse(set(rx[panels[0][240:]] ) & set(rx[panels[1][240:]]))

    def test_tuple_key_lineage_and_cell_ordinals_are_not_pool_positions(self):
        keys = R.tuple_keys(np.array([0, 1, 0]), np.array([0, 1, 0]), np.array([1, 1, 1]), ['a', 'b'], {'a|1':'ha', 'b|1':'hb'})
        self.assertEqual(keys, [('a', 1, 'ha', 0), ('b', 1, 'hb', 0), ('a', 1, 'ha', 1)])

    def test_panel_rejects_changed_dtype_digest_and_class_caps(self):
        args = list(fixture())
        plan = copy.deepcopy(args[-1]); plan['arms'][0]['validation_indices_sha256'] = 'bad-int64-substitution'
        args[-1] = plan
        with self.assertRaisesRegex(ValueError, 'validation index SHA'):
            R.build_panels(*args)
        args = list(fixture()); args[-1]['extension_caps'][0]['retained_per_extension'] += 1
        with self.assertRaisesRegex(ValueError, 'caps differ'):
            R.build_panels(*args)

    def test_sequence_blocks_future_fit_before_any_source_access(self):
        reg = dict(runs=[dict(id='B1_S00')], run_sequence=list(R.EXPECTED))
        with patch.object(R, 'verify_complete', side_effect=ValueError('preceding incomplete')), patch.object(R, 'load_fit') as load, patch.object(R, 'setup') as setup:
            with self.assertRaisesRegex(ValueError, 'preceding incomplete'):
                R.train(reg, Path('synthetic-registration'), 'B1_S00')
            load.assert_not_called(); setup.assert_not_called()

    def test_held_access_requires_committed_seal_before_setup_or_array_read(self):
        with patch.object(R, 'require_committed', side_effect=ValueError('uncommitted seal')), patch.object(R, 'load_held') as load, patch.object(R, 'setup') as setup:
            with self.assertRaisesRegex(ValueError, 'uncommitted seal'):
                R.evaluate({}, Path('synthetic-registration'))
            load.assert_not_called(); setup.assert_not_called()

    def test_held_loader_maps_only_the_twelve_registered_cells(self):
        names = [f'h{i}' for i in range(6)]
        inputs = [dict(path='synthetic', sha256=f'hash-{r}-{d}', rx=r, day=d) for r in names for d in (1, 2)]
        reg = dict(held_inputs=inputs, held_rx=names, rx_names=names,
                   source_cell_sha256={f"{i['rx']}|{i['day']}":i['sha256'] for i in inputs})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'synthetic.npz'
            np.savez(path, x=np.zeros((2,2,256), dtype=np.float32), y=np.array([0,1]))
            with patch.object(R, 'verify_input', return_value=path), patch.object(R, 'access'):
                result = R.load_held(reg)
            self.assertEqual(result['x'].shape, (24,2,256))
            self.assertEqual(len(set(result['keys'])), 24)
            self.assertEqual(set(result['day']), {1,2})

    def test_mean_effect_and_shared_cluster_bootstrap_for_identical_arms(self):
        p = predictions([0,1,0,1])
        result = M.analyze_endpoint({name:p for name in M.RUNS}, replicates=100)
        for effect in result['composition_effects'].values():
            self.assertEqual(effect['average']['estimate'], 0)
            self.assertEqual(effect['average']['paired_rx_bootstrap_95_interval'], [0,0])
            self.assertFalse(effect['consistent_finite_effect_support'])

    def test_opposite_stream_effects_are_not_consistent_even_when_average_changes(self):
        good = predictions([0,1,0,1]); bad = predictions([1,0,1,0]); mixed = predictions([0,1,1,0])
        result = M.analyze_endpoint(dict(B0_S00=bad, B1_S00=good, B0_S11=good, B1_S11=mixed), replicates=100)
        effect = result['composition_effects']['accuracy']
        self.assertAlmostEqual(effect['streams']['S00']['estimate'], 1.)
        self.assertAlmostEqual(effect['streams']['S11']['estimate'], -.5)
        self.assertAlmostEqual(effect['average']['estimate'], .25)
        self.assertFalse(effect['same_direction']); self.assertFalse(effect['consistent_finite_effect_support'])

    def test_local_risk_screen_preserves_opposite_and_low_support_groups(self):
        pairs = {flow:dict(tx_rx=[dict(rx='r', tx_class_index=0, packets=200, accuracy_delta=delta)])
                 for flow,delta in (('S00',.10),('S11',-.10))}
        self.assertEqual(M.local_risk_consistency(pairs)['repeated_local_groups'], 0)
        pairs['S11']['tx_rx'][0]['accuracy_delta'] = .05
        self.assertEqual(M.local_risk_consistency(pairs)['repeated_local_groups'], 1)
        pairs['S11']['tx_rx'][0]['packets'] = 49
        result = M.local_risk_consistency(pairs)
        self.assertEqual(result['total_groups'], 1); self.assertEqual(result['repeated_local_groups'], 0)


if __name__ == '__main__':
    unittest.main()
