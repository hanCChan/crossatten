import sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from s15_models import build,search,shared_hash,receptive_field
from s15_stats import paired
import s15_data

class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_search_is_data_free_and_matches_both_actual_models(self):
        a=search();b=search();self.assertEqual(a,b)
        best=a['selected'];self.assertEqual(best['groups'],8);self.assertEqual(best['mid'],80)
        for new,old in [('C2_lite_mha','A1_mha'),('C2_lite_hstu','A3_rfhstu')]:
            n=sum(p.numel() for p in build(new,1).parameters());ref=sum(p.numel() for p in build(old,1).parameters())
            self.assertLess(abs(n/ref-1),.01)
    def test_all_non_tokenizer_initialization_is_paired(self):
        for new,old in [('C2_lite_mha','A1_mha'),('C2_lite_hstu','A3_rfhstu')]:self.assertEqual(shared_hash(build(new,201)),shared_hash(build(old,201)))
        a,b=[build(name,201).net.embedder.cnn_stem.tokenize for name in ['C2_lite_mha','C2_lite_hstu']]
        for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]))
    def test_token_grid_and_receptive_fields_are_audited(self):
        for arm,field in [('E1_patch_mha',266),('A1_mha',350),('C2_lite_mha',266)]:
            r=receptive_field(build(arm,1));self.assertEqual(r['cnn_receptive_field'],field)
            self.assertEqual(r['token_stride'],256);self.assertEqual(r['tokens'],32);self.assertEqual(r['first_token_center'],127.5)
    def test_forward_backward_and_no_hidden_nonlinearity(self):
        for arm in ['C2_lite_mha','C2_lite_hstu']:
            model=build(arm,3);x=torch.randn(2,2,8192);y=model(x)
            self.assertEqual(y.shape,(2,10));y.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.assertEqual(len(model.net.embedder.cnn_stem.tokenize),2)
    def test_blind_and_unsealed_target_refused_before_cache_io(self):
        with patch.object(s15_data,'read',side_effect=AssertionError('not allowed')):
            with self.assertRaises(ValueError):s15_data.shen_cell('b200_2',0,'source')
        with patch.object(s15_data,'committed',side_effect=ValueError('no seal')):
            with self.assertRaises(ValueError):s15_data.shen_cell('b200_1',0,'confirm')
    def test_receiver_bootstrap_ignores_duplicated_rows(self):
        rows=[dict(rx=r,fold=f,seed=s,delta_pp=v) for r,v in [('a',1),('b',3)] for f in [0,1] for s in [201,202,203]]
        self.assertEqual(paired(rows)['ci95_pp'],paired(rows*4)['ci95_pp'])
if __name__=='__main__':unittest.main()
