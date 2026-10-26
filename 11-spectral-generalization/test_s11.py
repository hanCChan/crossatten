import unittest
import numpy as np
import torch
from core import s09,read
from spectral import views,estimate,apply_prior,center_weight,paired_model,split_stability


class Tests(unittest.TestCase):
    def test_identity_and_pairing(self):
        x=torch.randn(4,2,256)
        b=views(x)
        torch.testing.assert_close(apply_prior(b,torch.ones(256),"F2"),b,rtol=0,atol=0)
        a,h=paired_model("ManyRx","hard",42)
        c,j=paired_model("ManyRx","F2",42,np.ones(256))
        self.assertEqual(h,j)
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in c.parameters()))
        for key,value in a.state_dict().items():
            torch.testing.assert_close(value,c.state_dict()[key],rtol=0,atol=0)
        a.eval();c.eval()
        torch.testing.assert_close(a(x),c(x),rtol=0,atol=0)

    def test_band_boundary(self):
        w=torch.from_numpy(center_weight())
        out=apply_prior(torch.ones(1,5,256),w,"F1")
        self.assertEqual(int(w.sum()),63)
        self.assertEqual(float(out[0,3,97]),0)
        self.assertEqual(float(out[0,3,98]),1)
        self.assertEqual(float(out[0,0,160]),0)

    def test_gradient_and_zero(self):
        for scale in (0,1e-8,1):
            x=(torch.randn(2,2,256)*scale).requires_grad_()
            # views is diagnostic no-grad; model's training path must retain gradients.
            b=s09.phase_features_from_base(__import__("xa.explore_models",fromlist=["frequency_views"]).frequency_views(x),"first")
            out=apply_prior(b,torch.linspace(.5,1,256),"F2")
            out.square().sum().backward()
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_stats_invariance_and_missing_cells(self):
        rng=np.random.default_rng(42)
        y=np.repeat(np.arange(3),3*2*24)
        r=np.tile(np.repeat(np.arange(3),2*24),3)
        d=np.tile(np.repeat([1,2],24),9)
        v=rng.normal(size=(len(y),5,256))*.05 + y[:,None,None]*rng.normal(size=(1,5,256))
        a=estimate(v,y,r,d)
        perm=rng.permutation(len(y))
        b=estimate(v[perm],y[perm],r[perm],d[perm])
        np.testing.assert_allclose(a["weight"],b["weight"],rtol=1e-5,atol=1e-6)
        self.assertTrue((a["weight"]>=.5).all() and (a["weight"]<=1).all())
        self.assertGreater(split_stability(v,y,r,d)["correlation"],.5)
        with self.assertRaises(ValueError):
            estimate(v[d==1],y[d==1],r[d==1],d[d==1])

    def test_source_guard(self):
        j=dict(dataset="ManyRx",fold=0,phase="audit")
        with self.assertRaises(ValueError):
            read(j,["1-1"],(1,2),"fit")
        with self.assertRaises(ValueError):
            read(j,["8-7"],(3,),"diagnostic")


if __name__=="__main__":
    torch.set_num_threads(2)
    unittest.main()
