"""NumPy-only D0 validation; does not load the blocked PyTorch DLL."""
import unittest
import numpy as np
from s17_features import features,cell_profiles,analyze
from s17_stats import delta_stats,gate
from s17_data import cells,groups
from s17_jobs import split_seed,cache_key
from xa.wisig import receivers

def synthetic(receiver_effect):
    data={k:[] for k in ['y','rx','day']};v=[]
    for tx in range(4):
        for rx in range(4):
            for day in [1,2]:
                for _ in range(6):
                    data['y'].append(tx);data['rx'].append(rx);data['day'].append(day)
                    v.append([100*tx+20*day+(rx if receiver_effect else 0)])
    return np.asarray(v,dtype=float),{k:np.asarray(a) for k,a in data.items()}

class CPUStats(unittest.TestCase):
    def test_tx_day_offsets_are_removed(self):
        value,data=synthetic(True);c=cell_profiles(value,data,['a','b','c','d'])
        np.testing.assert_allclose(c['residual'][:,0],c['rx']-1.5)
        r=analyze(value,data,['a','b','c','d'],{0:.1,1:.2,2:.3,3:.4},permutations=19)
        self.assertAlmostEqual(r['cross_day_correlation'],1);self.assertAlmostEqual(r['leave_tx_out_rx_accuracy'],1)
    def test_tx_day_only_negative_control(self):
        value,data=synthetic(False);r=analyze(value,data,['a','b','c','d'],{0:.1,1:.2,2:.3,3:.4},permutations=19)
        self.assertFalse(r['stable']);self.assertAlmostEqual(r['leave_tx_out_rx_accuracy'],.25);self.assertEqual(r['rx_permutation_p'],1)
    def test_descriptor_invariances_and_finite_values(self):
        x=np.random.default_rng(2).normal(size=(8,2,256)).astype(np.float32);z=x[:,0]+1j*x[:,1]
        rot=z*np.exp(.8j);r=np.stack([rot.real,rot.imag],1);a=features(x);b=features(r);c=features(np.roll(x,7,axis=2))
        self.assertEqual(a['amp'].shape,(8,16));self.assertEqual(a['phase'].shape,(8,32));self.assertEqual(a['iq'].shape,(8,2))
        for key in ['amp','phase']:
            np.testing.assert_allclose(a[key],b[key],atol=2e-5,rtol=2e-5)
            np.testing.assert_allclose(a[key],c[key],atol=2e-5,rtol=2e-5)
        self.assertTrue(all(np.isfinite(v).all() for v in a.values()))
    def test_roles_rejected_before_io(self):
        for dataset in ['ManyRx','ManySig']:
            source,target=receivers(dataset,0)
            for kwargs in [dict(names=target),dict(names=source[:1],days=(3,4)),dict(names=source[:1],role='target')]:
                with self.assertRaises(ValueError):cells(dataset,0,**kwargs)
            a,b=groups(dataset,0,'first');self.assertFalse(set(a)&set(b));self.assertEqual(set(a+b),set(source))
    def test_split_is_separate_from_initialization_seed(self):
        j=dict(dataset='ManyRx',fold=0,group='first',seed=18001,split_seed=16001)
        self.assertEqual(split_seed(j),16001);self.assertEqual(cache_key(j),'ManyRx_f0_first_s16001')
    def test_receiver_clusters_and_complete_folds(self):
        rows=[dict(fold=f,rx=f'r{r}',delta_pp=2.) for f in range(4) for r in range(6)]
        self.assertEqual(delta_stats(rows),delta_stats(rows+rows));self.assertTrue(gate(dict(ManyRx=delta_stats(rows)))['admitted'])
        with self.assertRaises(ValueError):delta_stats(rows[:6])

if __name__=='__main__':unittest.main(verbosity=2)
