"""Synthetic CPU checks before real source data access."""
import unittest
import numpy as np
import torch
from s17_core import HERE
from s17_models import build
from s17_data import cells,groups
from s17_features import features,cell_profiles,analyze
from s17_augment import fir,bounded_kernel,perturb,as_complex
from s17_stats import delta_stats,gate
from xa.wisig import receivers

class Engineering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_linear_fir_identity_and_frequency_bound(self):
        x=torch.randn(3,2,256);torch.testing.assert_close(fir(x),x,atol=1e-6,rtol=1e-6)
        for phase in [False,True]:
            a=torch.randn(3,16)*20
            h=bounded_kernel(3,'cpu',phase=a) if phase else bounded_kernel(3,'cpu',log_db=a)
            response=torch.fft.fft(h,n=2048).abs()
            self.assertGreaterEqual(float(response.min()),.75-1e-6);self.assertLessEqual(float(response.max()),1.25+1e-6)
        # An impulse at the final sample cannot wrap around to the first sample.
        edge=torch.zeros(1,2,256);edge[0,0,-1]=1
        y=fir(edge,log_db=torch.randn(1,16));self.assertLess(float(y[:,:,:230].abs().max()),1e-6)
    def test_evm_matched_noise_and_zero_intensity(self):
        x=torch.randn(8,2,256);profiles=dict(amp=np.random.default_rng(0).normal(size=(5,16)).tolist(),iq=[[.1,.1],[-.1,-.1],[0,0]])
        for family in ['amp','iq']:
            def sample(control,tau=1):return perturb(x,family,tau,profiles,torch.Generator().manual_seed(1),torch.Generator().manual_seed(2),control=control)
            a=as_complex(sample(False))-as_complex(x);b=as_complex(sample(True))-as_complex(x)
            torch.testing.assert_close(a.abs().square().mean(1),b.abs().square().mean(1),rtol=1e-5,atol=1e-8)
            torch.testing.assert_close(sample(False,0),x,rtol=1e-6,atol=1e-6)
    def test_descriptors_and_model_gain_phase_invariance(self):
        x=torch.randn(3,2,256);z=as_complex(x);w=z*np.exp(.7j);rot=torch.stack([w.real,w.imag],1)
        a=features(x.numpy());b=features(rot.numpy())
        for key in ['amp','phase']:np.testing.assert_allclose(a[key],b[key],atol=2e-5,rtol=2e-5)
        m=build('ManyRx',18001).eval()
        with torch.no_grad():
            torch.testing.assert_close(m(x),m(2*x),atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(m(x),m(rot),atol=2e-5,rtol=2e-5)
        self.assertEqual(sum(p.numel() for p in m.parameters()),82426)
    def test_role_guards_before_io(self):
        for dataset in ['ManyRx','ManySig']:
            source,target=receivers(dataset,0)
            for kwargs in [dict(names=target),dict(names=source[:1],days=(3,4)),dict(names=source[:1],role='target')]:
                with self.assertRaises(ValueError):cells(dataset,0,**kwargs)
            a,b=groups(dataset,0,'first');self.assertEqual(set(a+b),set(source));self.assertFalse(set(a)&set(b))
    def test_conditioning_removes_tx_day_offsets(self):
        data={k:[] for k in ['x','y','rx','day']};values=[]
        for tx in range(4):
            for rx in range(4):
                for day in [1,2]:
                    for _ in range(6):
                        data['y'].append(tx);data['rx'].append(rx);data['day'].append(day);values.append([100*tx+20*day+rx])
        data={k:np.asarray(v) for k,v in data.items()};c=cell_profiles(np.asarray(values),data,['a','b','c','d'])
        np.testing.assert_allclose(c['residual'][:,0],c['rx']-1.5)
        result=analyze(np.asarray(values,dtype=float),data,['a','b','c','d'],{0:.1,1:.2,2:.3,3:.4},permutations=19)
        self.assertAlmostEqual(result['cross_day_correlation'],1);self.assertAlmostEqual(result['leave_tx_out_rx_accuracy'],1)
    def test_paired_cluster_statistics(self):
        rows=[dict(fold=f,rx=f'r{r}',delta_pp=2.) for f in range(4) for r in range(6)]
        self.assertEqual(delta_stats(rows),delta_stats(rows+rows));self.assertTrue(gate(dict(ManyRx=delta_stats(rows)))['admitted'])
        with self.assertRaises(ValueError):delta_stats(rows[:6])

if __name__=='__main__':unittest.main(verbosity=2)
