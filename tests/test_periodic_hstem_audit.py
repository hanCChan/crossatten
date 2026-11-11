"""Synthetic adversarial checks; no scientific arrays are accessed."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

PATH=Path(__file__).resolve().parents[1]/"research/periodic_hstem/audit.py"
SPEC=importlib.util.spec_from_file_location("periodic_audit",PATH)
A=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
REG=dict(period=20,primary_window=[20,140],window_offsets=list(range(20)),min_adjacent_coherence=.7,min_neighbor_lag_margin=.05)


def iq(z):
    return np.stack((z.real,z.imag),axis=1)


class PeriodicAuditTests(unittest.TestCase):
    def repeated(self, batch=4):
        rng=np.random.default_rng(179)
        pattern=rng.normal(size=(batch,20))+1j*rng.normal(size=(batch,20))
        return np.tile(pattern,(1,13))[:,:256]

    def test_periodic_shifted_with_cfo_passes(self):
        z=self.repeated()
        z=np.stack([np.roll(row,i) for i,row in enumerate(z)])
        z*=np.exp(.037j*np.arange(256))
        self.assertTrue(A.packet_metrics(iq(z),REG)["passed"].all())

    def test_noise_fails(self):
        rng=np.random.default_rng(32)
        z=rng.normal(size=(10,256))+1j*rng.normal(size=(10,256))
        self.assertFalse(A.packet_metrics(iq(z),REG)["passed"].any())

    def test_single_tone_high_autocorrelation_is_insufficient(self):
        z=np.exp(.137j*np.arange(256))[None]
        m=A.packet_metrics(iq(z),REG)
        self.assertGreater(m["rho20_by_offset"][0,0],.99)
        self.assertFalse(m["passed"][0])

    def test_zero_and_nonfinite_fail_and_remain_in_denominator(self):
        x=iq(self.repeated())
        x[0]=0
        x[1,0,30]=np.nan
        m=A.packet_metrics(x,REG)
        self.assertEqual(len(m["passed"]),4)
        self.assertEqual(m["passed"].tolist(),[False,False,True,True])
        self.assertTrue(all(np.isfinite(v).all() for v in m.values()))

    def test_partial_repetition_and_late_boundary_fail(self):
        z=self.repeated()
        rng=np.random.default_rng(23)
        z[:,120:]=rng.normal(size=(4,136))+1j*rng.normal(size=(4,136))
        self.assertFalse(A.packet_metrics(iq(z),REG)["passed"].any())

    def test_region_description_does_not_change_eligibility(self):
        z=self.repeated()
        rng=np.random.default_rng(12)
        z[:,180:]=rng.normal(size=(4,76))+1j*rng.normal(size=(4,76))
        self.assertTrue(A.packet_metrics(iq(z),REG)["passed"].all())

    def test_wrong_shape_and_changed_window_rejected(self):
        with self.assertRaises(ValueError):A.packet_metrics(np.zeros((3,2,200)),REG)
        with self.assertRaises(ValueError):A.packet_metrics(iq(self.repeated()),dict(REG,primary_window=[0,120]))

    def test_scale_invariance(self):
        x=iq(self.repeated())
        a,b=[A.packet_metrics(v,REG) for v in (x,x*13)]
        np.testing.assert_array_equal(a["passed"],b["passed"])
        np.testing.assert_allclose(a["rho20_by_offset"],b["rho20_by_offset"],atol=1e-12)

    def test_row_selection_is_bounded_deterministic_and_train_only(self):
        n=34668
        y=np.arange(n)%10
        rx=np.zeros(n,dtype=int)
        day=np.ones(n,dtype=int)
        day[1::2]=2
        train=np.arange(0,n,3,dtype=np.int32)
        val=np.setdiff1d(np.arange(n,dtype=np.int32),train)
        reg=dict(rx_names=["a"],audit_rx=["a"],allowed_days=[1,2],maximum_packets_per_cell=16)
        a,groups,missing=A.select_rows(y,rx,day,train,val,reg)
        b,_,_=A.select_rows(y,rx,day,train[::-1],val,reg)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(len(a),160)
        self.assertEqual(len(missing),10)
        self.assertTrue(all(g["rows"]==16 for g in groups))
        self.assertTrue(np.isin(a,train).all())
        self.assertFalse(np.isin(a,val).any())


if __name__=="__main__":unittest.main()
