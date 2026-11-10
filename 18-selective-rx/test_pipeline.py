"""Integration and gate checks on synthetic examples, never scientific caches."""
import unittest
import numpy as np
from s18_decide import decide, paired_rx_interval
from s18_data import halves, locations
from s18_run import scores, delta, jsonable
from s18_stats import scatters, random_basis, compress


class PipelineTests(unittest.TestCase):
    def test_split_guard_and_halves(self):
        with self.assertRaises(ValueError): locations(4)
        grid = np.array([(t, r, d) for t in range(2) for r in range(3) for d in [1, 2] for _ in range(6)])
        a, b = halves(*grid.T)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(len(a)+len(b), len(grid))
        self.assertEqual(len(a), len(b))

    def test_scores_margin_and_paired_delta(self):
        logits = np.array([[2., 0.], [0., 3.], [0., 2.], [4., 0.]])
        y = np.array([0, 1, 0, 0]); rx = np.array([0, 0, 1, 1])
        s = scores(logits, y, rx, ['a', 'b'])
        self.assertEqual(s['summary']['acc'], .75)
        self.assertEqual(s['summary']['margin'], 1.75)
        self.assertTrue(all(v == 0 for v in delta(s, s).values()))

    def test_scatter_projection_agrees_with_linear_operator(self):
        rng = np.random.default_rng(7)
        grid = np.array([(t, r, d) for t in range(2) for r in range(3) for d in [1, 2] for _ in range(4)])
        z = rng.normal(size=(len(grid), 8)); U = random_basis(8, 3, 6)
        s = scatters(z, *grid.T); v = scatters(compress(z, s['center'], U), *grid.T)
        A = np.eye(8)-.5*U@U.T
        for key in ['S_R', 'S_T']:
            np.testing.assert_allclose(v[key], A@s[key]@A.T, atol=1e-12)

    def test_duplicate_fold_rx_is_one_bootstrap_unit(self):
        row = dict(interventions=[dict(arm='selective', score=dict(per_rx={'a':dict(acc=.9), 'b':dict(acc=.8)})),
                                  dict(arm='random_0', score=dict(per_rx={'a':dict(acc=.8), 'b':dict(acc=.8)}))])
        single = paired_rx_interval([row], draws=100)
        repeat = paired_rx_interval([row]*4, draws=100)
        self.assertEqual(single, repeat)
        self.assertEqual(single['independent_rx'], 2)

    def test_gate_needs_stability_specificity_and_all_fold_preservation(self):
        rows = []
        for fold in range(4):
            score = lambda a: dict(per_rx={f'r{fold}':dict(acc=a)})
            s = dict(arm='selective', delta_acc_pp=0., delta_f1_pp=0., delta_ce=0., normalized_delta_margin=0.,
                     worst_rx_paired_pp=0., worst_tx_paired_pp=0., rx_scatter_reduction=.4, tx_scatter_reduction=.01, score=score(.9))
            rnd = dict(s, arm='random_0', delta_ce=.02, rx_scatter_reduction=.05, score=score(.85))
            tx = dict(s, arm='tx_dominant', delta_ce=.10, tx_scatter_reduction=.5)
            rows.append(dict(fold=fold, interventions=[s, rnd, tx], layers=dict(tokens=dict(stability=dict(packet_halves=.8, days=.8, tx_halves=.8, source_validation=.8), val_metrics=dict(rx_fraction=.5)))))
        self.assertTrue(decide(rows)['admitted'])
        rows[0]['interventions'][0]['worst_rx_paired_pp'] = -3.
        self.assertFalse(decide(rows)['admitted'])
        rows[0]['interventions'][0]['worst_rx_paired_pp'] = 0.
        for row in rows[:2]: row['layers']['tokens']['stability']['days'] = .1
        self.assertFalse(decide(rows)['admitted'])
        import json
        json.dumps(decide(rows), allow_nan=False)


if __name__ == '__main__': unittest.main()
