import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from models import ARMS, build, shared_state, nparams, PlainMHA
from pickle_index import index, Array, Span
from wifi import cfo, rotate, corrected, build_wifi, FS
import data
from reports import source_gate, paired_stats
from train import smoke_subset
from metrics import rx_metrics


class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_shared_and_single_factor_weights(self):
        ref=build('A3_rfhstu',99)
        for arm in ARMS:
            model=build(arm,99)
            for k,v in shared_state(model).items():self.assertTrue(torch.equal(v,shared_state(ref)[k]),(arm,k))
            if arm in ['A4_softmax_gate','A5_silu_gate']:
                for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,ref.state_dict()[k]),(arm,k))
        for a in ['A1_mha_matched','A2_transformer_matched']:
            self.assertLess(abs(nparams(build(a,1))/nparams(ref)-1),.05)

    def test_plain_mha_matches_torch(self):
        model=PlainMHA();native=torch.nn.MultiheadAttention(64,4,batch_first=True,dropout=0)
        native.in_proj_weight.data.copy_(model.qkv.weight);native.in_proj_bias.data.copy_(model.qkv.bias)
        native.out_proj.load_state_dict(model.out.state_dict())
        x=torch.randn(3,32,64);y,_=native(x,x,x,need_weights=True)
        torch.testing.assert_close(model(x),model.norm(x+y),atol=2e-6,rtol=1e-5)

    def test_models_forward_backward(self):
        for arm in ARMS:
            model=build(arm,17);y=model(torch.randn(2,2,8192));self.assertEqual(tuple(y.shape),(2,10));y.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_cfo_estimator_and_inverse(self):
        # Exact repeated nonconstant prefix; known injected CFO is recoverable.
        torch.manual_seed(3)
        z=torch.randn(4,2,20).repeat(1,1,13)[:,:,:256]
        hz=torch.tensor([-10000.,-2000.,3000.,10000.]);shifted=rotate(z,hz)
        est,quality=cfo(shifted);torch.testing.assert_close(est,hz,atol=.05,rtol=1e-4)
        self.assertTrue((quality>.999).all());torch.testing.assert_close(corrected(shifted),z,atol=1e-5,rtol=1e-4)

    def test_dual_controls_parameter_and_shared_parity(self):
        a,b=[build_wifi('ManySig',kind,5) for kind in ['B2_dual','B2_dual_raw']]
        self.assertEqual(nparams(a),nparams(b))
        for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]))

    def test_pickle_index_does_not_read_iq(self):
        a=np.arange(200*256*2,dtype=np.float64).reshape(200,256,2)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'test.pkl';p.write_bytes(pickle.dumps({'data':a,'label':[1,2]},protocol=4))
            with patch.object(Span,'read',side_effect=AssertionError('payload read during indexing')):
                d=index(p);self.assertIsInstance(d['data'],Array)
            np.testing.assert_array_equal(d['data'].get(),a)

    def test_targets_refused_before_metadata_or_payload(self):
        source,target=data.receivers('ManyRx',0)
        with patch.object(data,'metadata',side_effect=AssertionError('must validate first')):
            for names,days in [(target,(1,2)),(source,(3,4))]:
                with self.assertRaises(ValueError):data.wisig_cells('ManyRx',0,names,days)
        with self.assertRaises(ValueError):data.shen_cell('b200_2',0,'source')

    def test_source_gate_requires_joint_evidence(self):
        rows=[dict(dataset=d,fold=f,rx=f'{d}-{f}',delta_pp=1.1 if d=='ManyRx' else 0.)
              for d in ['ManyRx','ManySig'] for f in range(4)]
        self.assertTrue(source_gate(rows)['admitted'])
        rows[0]['delta_pp']=-3
        self.assertFalse(source_gate(rows)['admitted'])
        with self.assertRaises(ValueError):source_gate(rows[:4])

    def test_paired_stats_clusters_receivers(self):
        rows=[dict(rx=r,fold=f,seed=s,delta_pp=v) for r,v in [('a',2),('b',-1)] for f in [0,1] for s in [1,2,3]]
        a=paired_stats(rows);self.assertEqual(a['mean_delta_pp'],.5);self.assertEqual(a['positive_pairs'],6)
        self.assertEqual(a['worst_receiver_delta_pp'],-1)
        self.assertEqual(a['rx_cluster_ci95_pp'],paired_stats(rows*4)['rx_cluster_ci95_pp'])

    def test_smoke_subset_preserves_classes_and_receivers(self):
        y=np.tile(np.repeat(np.arange(10),20),5);rx=np.repeat(np.arange(5),200)
        d=smoke_subset(dict(x=np.arange(1000)[:,None],y=y,rx=rx),128)
        self.assertEqual(len(d['y']),128);self.assertEqual(len(set(zip(d['y'],d['rx']))),50)

    def test_macro_f1_uses_declared_classes(self):
        for k in [6,10]:
            value=rx_metrics(np.eye(k),np.arange(k))
            self.assertEqual(value['macro_f1'],1.);self.assertEqual(np.asarray(value['confusion']).shape,(k,k))
        from sklearn.metrics import f1_score
        y=np.array([0,0,1,1,2,3,4,5]);p=np.array([0,1,1,1,2,3,4,4])
        self.assertAlmostEqual(rx_metrics(np.eye(6)[p],y)['macro_f1'],f1_score(y,p,average='macro',labels=range(6),zero_division=0))


if __name__=='__main__':unittest.main()
