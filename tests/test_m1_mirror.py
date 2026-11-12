"""M1 physical counterexamples, numerical rejection and held-segment isolation."""
from pathlib import Path
import sys
import unittest
from unittest import mock
import numpy as np
HERE=Path(__file__).resolve().parents[1]/'research/m1_mirror_identifiability'
sys.path.insert(0,str(HERE))
import core as C
import runner as R


class PhysicsTests(unittest.TestCase):
    def test_25MHz_reference_and_conjugate_time_ambiguity(self):
        n=np.arange(160);s=C.template(n)
        np.testing.assert_allclose(s,C.template(n+20),atol=1e-12)
        np.testing.assert_allclose(s.conj(),-1j*C.template(n+10),atol=1e-12)
        occupied=np.flatnonzero(abs(np.fft.fft(s[:20]))>1e-10)
        self.assertEqual(set(occupied),set(np.r_[1:7,14:20]))

    def test_SVD_recovery_and_rank_condition_rejection(self):
        s=C.template(np.arange(80));U=np.c_[s,s.conj()];theta=np.array([.8+.2j,.05-.03j])
        f=C.svd_fit(U,U@theta);self.assertTrue(f['valid']);self.assertLess(f['condition'],1.0000001)
        np.testing.assert_allclose(f['theta'],theta,atol=1e-12)
        for V in (np.c_[s,s],np.zeros((80,2)),np.c_[s,s+1e-10*np.arange(80)]):
            f=C.svd_fit(V,s);self.assertFalse(f['valid']);self.assertIsNone(f['theta'])

    def test_full_cascade_and_wrong_common_rotation(self):
        n=C.N;t=.08*np.exp(.4j);r=.06*np.exp(-.7j);f=100000;h=np.exp(.2j);phase=.3
        actual=C.forward(n,t,r,f,channel=((0,h),))
        coeff=np.array([np.exp(1j*phase)*h,np.exp(1j*phase)*h*t,r*np.exp(-1j*phase)*h.conjugate(),r*np.exp(-1j*phase)*h.conjugate()*t.conjugate()])
        np.testing.assert_allclose(actual,C.full_design(n,0,f)@coeff,atol=1e-12)
        wrong=C.forward(n,t,r,0,channel=((0,h),))*np.exp(2j*np.pi*f*n/C.FS)
        self.assertGreater(np.linalg.norm(actual-wrong)/np.linalg.norm(actual),.01)
        self.assertFalse(C.svd_fit(C.full_design(n,0,0),actual)['valid'])
        self.assertTrue(C.svd_fit(C.full_design(n,0,f),actual,max_condition=100)['valid'])

    def test_RX_only_is_not_a_TX_parameter(self):
        n=C.N;s=C.template(n);r=.06*np.exp(-.7j)
        f=C.svd_fit(np.c_[s,s.conj()],C.forward(n,rx=r))
        self.assertAlmostEqual(abs(f['theta'][1]),.06,places=12)
        out=C.Estimator().packet(C.forward(np.arange(256),rx=r))
        self.assertIsNone(out['tx_iq_estimate']);self.assertTrue(out['structural_half_period_ambiguity'])

    def test_admissible_cascade_gauge(self):
        def matrix(a,b):return np.array([[a,b],[b.conjugate(),a.conjugate()]],complex)
        mt,mr,Q=matrix(1+.0j,.08+.03j),matrix(1+.0j,.02-.01j),matrix(1+.0j,.03+.01j)
        left=mr@Q;right=np.linalg.solve(Q,mt)
        np.testing.assert_allclose(left@right,mr@mt,atol=1e-14)
        self.assertGreater(np.linalg.norm(right-mt),.01)
        np.testing.assert_allclose(left[1],left[0,::-1].conj(),atol=1e-14)

    def test_control_same_support_and_distinct_subspace(self):
        n=np.arange(80);s=C.template(n);c=C.template(n,control=True)
        np.testing.assert_allclose(abs(np.fft.fft(s[:20])),abs(np.fft.fft(c[:20])),atol=1e-12)
        U=np.c_[s,s.conj()]
        residual=np.linalg.norm(c-U@np.linalg.lstsq(U,c,rcond=1e-12)[0])/np.linalg.norm(c)
        self.assertGreater(residual,.1)
        est=C.Estimator();self.assertLess(max(est.conditions['NC']),10)

    def test_held_segment_cannot_change_fitted_quantities(self):
        est=C.Estimator();a=C.forward(np.arange(256),.05,.03,10000,.5);b=a.copy();b[100:180]+=20j
        x,y=est.packet(a),est.packet(b)
        self.assertEqual(x['cfo_hz'],y['cfo_hz'])
        for arm in ('L','WL','NC'):
            for field in ('phase','coefficients','fit_nmse'):self.assertEqual(x['models'][arm][field],y['models'][arm][field])
        self.assertNotEqual(x['models']['WL']['held_nmse'],y['models']['WL']['held_nmse'])

    def test_invalid_boundaries_and_IO_guard(self):
        est=C.Estimator()
        for x in (np.zeros(256),np.ones(255),np.full(256,np.nan),np.exp(2j*np.pi*7*np.arange(256)/20)):
            self.assertFalse(est.packet(x)['valid'])
        with mock.patch.object(R,'validate',side_effect=PermissionError('closed')):
            with mock.patch.object(np,'load',side_effect=AssertionError('forbidden array decode')):
                with self.assertRaises(PermissionError):R.load_source_metadata({})


if __name__=='__main__':unittest.main()
