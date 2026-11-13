import unittest
import numpy as np
import torch
from p_models import relation,balanced_batches


class RelationChecks(unittest.TestCase):
    def setUp(self):torch.manual_seed(42001)
    def test_same_geometry_across_RX_and_noncollapse(self):
        c=torch.randn(8,1,1,12,dtype=torch.float64)
        x=c.repeat(1,6,4,1);loss,s=relation(x)
        self.assertLess(float(loss),1e-20);self.assertGreater(float(s['minimum_relative_spread']),.1)
    def test_rotation_translation_and_scale_invariance(self):
        c=torch.randn(8,12,dtype=torch.float64);q=torch.linalg.qr(torch.randn(12,12,dtype=torch.float64)).Q
        x=torch.stack([c,c@q+10,(c@q)*2-4],1)[:,:,None,:].repeat(1,1,4,1)
        loss,_=relation(x);self.assertLess(float(loss),1e-14)
    def test_nonrigid_receiver_distortion_has_nonzero_gradient(self):
        x=torch.randn(8,6,4,12,dtype=torch.float64,requires_grad=True)
        loss,_=relation(x);loss.backward();self.assertGreater(float(loss),.01);self.assertGreater(float(x.grad.norm()),1e-5)
    def test_TX_permutation_breaking_correspondence_is_detected(self):
        c=torch.randn(8,12,dtype=torch.float64);x=torch.stack([c,c,c[torch.randperm(8)]],1)[:,:,None,:].repeat(1,1,4,1)
        loss,_=relation(x);self.assertGreater(float(loss),.01)
    def test_collapse_is_detected(self):
        _,s=relation(torch.ones(8,6,4,12,dtype=torch.float64));self.assertEqual(float(s['minimum_relative_spread']),0)
    def test_insufficient_support_rejected(self):
        with self.assertRaises(ValueError):relation(torch.ones(8,6,3,12))
    def test_sampler_exact_coverage_support_and_pairing(self):
        batches=balanced_batches(72,41301,1);self.assertEqual(len(batches),108)
        for ids in batches:
            tx=ids//288;rx=(ids%288)//48
            self.assertEqual(len(set(tx)),8)
            for t in set(tx):
                for r in range(6):self.assertEqual(int(np.sum((tx==t)&(rx==r))),4)
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)),np.arange(20736))
        np.testing.assert_array_equal(batches,balanced_batches(72,41301,1))
        self.assertFalse(np.array_equal(batches,balanced_batches(72,41302,1)))


if __name__=='__main__':unittest.main()
