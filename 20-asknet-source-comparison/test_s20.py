"""Synthetic science/engineering checks; never load real IQ or model weights."""
import copy
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader
from s20_core import official,configure_data
from s20_run import CONF,RecordedDataset
from s20_stats import decision

class S20Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_official_default_model_and_deterministic_eval(self):
        api=official();api.setup_seed(20201);m=api.build_model(CONF,10)
        self.assertEqual(sum(p.numel() for p in m.parameters()),100240)
        self.assertEqual(sum(p.numel()*(2 if p.is_complex() else 1) for p in m.parameters()),116369)
        self.assertEqual(m.koopman.n_atoms,127)
        m.eval();x=torch.randn(4,2,256)
        with torch.no_grad():a=m(x);b=m(x)
        self.assertEqual(tuple(a.shape),(4,10));self.assertTrue(torch.equal(a,b))
        with self.assertRaises(ValueError):api.load_single_dataset('ManyRx',0,1,10,'non_equalized')
    def test_complex_unitarity_and_backward(self):
        api=official();api.setup_seed(20202);m=api.build_model(CONF,10)
        m.train();x=torch.randn(4,2,256);loss=torch.nn.functional.cross_entropy(m(x),torch.tensor([0,1,2,3]));loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
        U=torch.linalg.qr(m.koopman.U.detach())[0];K=U@torch.diag(torch.exp(1j*m.koopman.theta.detach()))@U.conj().T
        self.assertLess(float((K.conj().T@K-torch.eye(127)).abs().max()),1e-5)
    def test_observed_dataloader_order_preserves_global_rng(self):
        x=torch.randn(17,2,256);y=torch.arange(17)
        torch.manual_seed(9);a=RecordedDataset(x,y);loader=DataLoader(a,batch_size=5,shuffle=True)
        order=[]
        for _,labels in loader:order.extend(labels.tolist())
        self.assertEqual(order,a.order);self.assertEqual(sorted(order),list(range(17)))
        torch.manual_seed(9);plain=torch.utils.data.TensorDataset(x,y);reference=[]
        for _,labels in DataLoader(plain,batch_size=5,shuffle=True):reference.extend(labels.tolist())
        self.assertEqual(order,reference)
    def test_source_roles_and_forbidden_fold(self):
        data=configure_data()
        for f in range(4):
            fit,held,target=data.roles(f)
            self.assertEqual([len(fit),len(held),len(target)],[18,6,8]);self.assertFalse(set(fit)&set(held+target))
        with self.assertRaises(ValueError):data.locations(4)
    def rows(self,effects):
        rows=[]
        for fold in range(4):
            for arm in ['B0','ASK']:
                acc=.8+(effects[fold]/100 if arm=='ASK' else 0)
                rows.append(dict(fold=fold,arm=arm,held_source=dict(summary=dict(acc=acc,macro_f1=acc,ce=1-acc),per_rx={f'r{fold}':dict(acc=acc)})))
        return rows
    def test_clear_advantage_and_symmetric_definition(self):
        self.assertEqual(decision(self.rows([2,2,2,2]))['status'],'ASKNET_CLEAR_SOURCE_ADVANTAGE')
        self.assertEqual(decision(self.rows([-2,-2,-2,-2]))['status'],'HSTEM_CLEAR_SOURCE_ADVANTAGE')
    def test_non_significance_does_not_prove_equivalence(self):
        self.assertEqual(decision(self.rows([2,-2,2,-2]))['status'],'INCONCLUSIVE_SOURCE_COMPARISON')
        self.assertEqual(decision(self.rows([.1,.2,-.1,.1]))['status'],'PRACTICALLY_CLOSE_ON_THIS_SOURCE_PROTOCOL')

if __name__=='__main__':unittest.main()
