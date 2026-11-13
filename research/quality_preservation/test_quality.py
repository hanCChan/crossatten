"""Synthetic checks for physical invariances, corruption, and confounding control."""
import unittest
import numpy as np
from quality import proxies, within_cell_low, conditional_effect, bootstrap_rx


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(29)
        z = self.rng.normal(size=(64,20)) + 1j*self.rng.normal(size=(64,20))
        self.z = np.tile(z, (1,13))[:, :256]

    @staticmethod
    def iq(z):
        return np.stack((z.real,z.imag),1)

    def test_repetition_and_cfo(self):
        q, _, _ = proxies(self.iq(self.z))
        f, _, _ = proxies(self.iq(self.z*2.4*np.exp(1j*(.3+.047*np.arange(256)))))
        np.testing.assert_allclose(q[:,:3],1,atol=1e-12)
        np.testing.assert_allclose(f[:,:3],1,atol=1e-12)

    def test_noise_and_local_corruption(self):
        noise = self.rng.normal(size=self.z.shape)+1j*self.rng.normal(size=self.z.shape)
        clean,_,_=proxies(self.iq(self.z))
        dirty,_,_=proxies(self.iq(self.z+noise))
        self.assertGreater(clean[:,0].mean(),dirty[:,0].mean()+.2)
        local=self.z.copy();local[:,60:80]=noise[:,60:80]
        _,pairs,_=proxies(self.iq(local))
        self.assertLess(pairs[:,1:3].mean(),.5)

    def test_invalid_packets_are_retained(self):
        x=self.iq(self.z);x[0]=0;x[1,0,30]=np.nan
        f,_,valid=proxies(x)
        self.assertEqual(f.shape,(64,6));self.assertFalse(valid[0]);self.assertFalse(valid[1])
        self.assertTrue(np.isfinite(f).all());np.testing.assert_equal(f[:2],0)

    def test_conditioning_removes_fixed_identity_confounding(self):
        y=np.repeat([0,1],80);rx=np.array(['r']*160);day=np.ones(160)
        q=np.r_[np.linspace(0,.2,80),np.linspace(.8,1,80)]
        error=(y==0).astype(float)
        low=within_cell_low(q,y,rx,day,np.arange(160))
        cells,receivers=conditional_effect(error,low,y,rx,day)
        self.assertEqual(low.sum(),40);self.assertEqual(receivers[0]['difference_pp'],0)
        self.assertEqual(len(cells),2)

    def test_ties_do_not_drop_rows(self):
        low=within_cell_low(np.ones(80),np.zeros(80),np.array(['r']*80),np.ones(80),np.arange(80))
        self.assertEqual(low.sum(),20);np.testing.assert_equal(np.flatnonzero(low),np.arange(20))

    def test_positive_control_and_cluster_interval(self):
        y=np.zeros(160);rx=np.repeat(['a','b'],80);day=np.ones(160);q=np.tile(np.arange(80),2)
        low=within_cell_low(q,y,rx,day,np.arange(160))
        _,r=conditional_effect(low.astype(float),low,y,rx,day)
        np.testing.assert_equal([v['difference_pp'] for v in r],[100,100])
        self.assertEqual(bootstrap_rx([100,100]),[100,100])


if __name__=='__main__':unittest.main()
