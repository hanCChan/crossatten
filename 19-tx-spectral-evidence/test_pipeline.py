"""Synthetic gate/metric/role integration checks, no scientific files."""
import copy
import unittest
import numpy as np
import s19_core
from s19_metrics import score, decide, cluster_comparison
from s19_data import roles, locations, evidence_ids


class PipelineTests(unittest.TestCase):
    def test_source_roles_and_invalid_fold(self):
        for f in range(4):
            fit,held,target=roles(f)
            self.assertEqual((len(fit),len(held),len(target)),(18,6,8))
            self.assertFalse(set(fit)&set(held+target))
            self.assertFalse(set(held)&set(target))
        with self.assertRaises(ValueError):locations(4)

    def test_training_sampling_skips_missing_without_inserting(self):
        labels=np.array([(t,r,d) for t in range(10) for r in range(3) for d in [1,2] if (t,r,d)!=(0,0,1) for _ in range(20)])
        data=dict(zip(['y','rx','day'],labels.T))
        ids=evidence_ids(data,0)
        self.assertEqual(len(ids),59*16)
        np.testing.assert_array_equal(ids,evidence_ids(data,0))
        self.assertEqual(len(np.unique(ids)),len(ids))

    def test_receiver_equal_metrics_and_true_margin(self):
        lg=np.array([[3.,0.],[0.,2.],[0.,1.]])
        d=dict(y=np.array([0,1,0]),rx=np.array([0,0,1]))
        s=score(lg,d,['a','b'])
        self.assertEqual(s['summary']['acc'],.5)
        self.assertEqual(s['summary']['margin'],.75)
        self.assertEqual(s['summary']['worst_rx'],0.)

    def test_gate_separates_evidence_from_module_gain(self):
        rows=[]
        for f in range(4):
            for arm in ['B0','B1','B2','B3','B4']:
                acc={'B0':.8,'B1':.81,'B2':.81,'B3':.81,'B4':.84}[arm]
                per={rx:dict(acc=acc) for rx in ['a','b']}
                rows.append(dict(fold=f,arm=arm,held_source=dict(summary=dict(acc=acc,macro_f1=acc,ce=1-acc),per_rx=per)))
        self.assertTrue(decide(rows)['admitted'])
        for row in rows:
            if row['arm']=='B3':row['held_source']=copy.deepcopy(next(r['held_source'] for r in rows if r['fold']==row['fold'] and r['arm']=='B4'))
        d=decide(rows)
        self.assertFalse(d['admitted']);self.assertFalse(d['evidence_supervision_supported'])
        self.assertTrue(all(d['checks']['B0'].values()))

    def test_rx_identity_not_repeated_fold_is_bootstrap_unit(self):
        rows=[]
        for f in range(4):
            for arm,acc in [('B0',.8),('B4',.82)]:
                rows.append(dict(fold=f,arm=arm,held_source=dict(summary=dict(acc=acc,macro_f1=acc,ce=1-acc),per_rx={'same':dict(acc=acc)})))
        c=cluster_comparison(rows,'B4','B0',draws=50)
        self.assertEqual(c['unique_rx'],1)
        self.assertAlmostEqual(c['ci95_pp'][0],2.)


if __name__=='__main__':unittest.main()
