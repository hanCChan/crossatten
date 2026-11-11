"""Synthetic acceptance of geometry, boundaries, gradients and pairing."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m
M=module("periodic_span_model",ROOT/"research/periodic_span/models.py")
torch.set_num_threads(2)


class SpanTests(unittest.TestCase):
    def test_all_positions_mask_matches_explicit_gather(self):
        for length in (1,2,19,20,21,39,40,41,256):
            for kernel,dilation in ((3,20),(3,19),(41,1)):
                x=torch.arange(length,dtype=torch.float64)[None,None]
                w=torch.tensor([1.3]*(kernel),dtype=torch.float64)[None,None]
                actual=M.masked_spatial(x,w,dilation)
                expected=[]
                for n in range(length):
                    taps=[n+(k-kernel//2)*dilation for k in range(kernel)]
                    taps=[t for t in taps if 0<=t<length]
                    expected.append(sum(1.3*t for t in taps)/len(taps))
                torch.testing.assert_close(actual[0,0],torch.tensor(expected,dtype=torch.float64))

    def test_edges_do_not_wrap(self):
        x=torch.zeros(1,1,256);x[0,0,-1]=1
        y=M.masked_spatial(x,torch.ones(1,1,3),20)
        self.assertEqual(y[0,0,0].item(),0.)
        self.assertEqual(torch.nonzero(y[0,0]).flatten().tolist(),[235,255])

    def test_effective_receptive_span_and_parameter_tolerance(self):
        models={a:M.build(a,24101) for a in M.ARMS}
        counts={a:sum(p.numel() for p in m.parameters()) for a,m in models.items()}
        self.assertEqual(counts['P0'],82426)
        self.assertEqual(counts['P1']-counts['P2'],608)
        self.assertLess((counts['P1']-counts['P2'])/counts['P2'],.01)
        self.assertEqual(counts['N19'],counts['P2'])
        for arm in ('P1','P2'):
            s=models[arm].span
            self.assertEqual(1+(s.kernel-1)*s.dilation,41)

    def test_shared_backbone_and_negative_control_initialization(self):
        models=[M.build(a,24101) for a in M.ARMS]
        for model in models[1:]:
            for key,value in M.common_state(models[0]).items():torch.testing.assert_close(value,M.common_state(model)[key],rtol=0,atol=0)
        for key,value in models[2].state_dict().items():torch.testing.assert_close(value,models[3].state_dict()[key],rtol=0,atol=0)
        x=torch.randn(3,2,256)
        for model in models:model.eval()
        with torch.no_grad():
            for model in models[2:]:torch.testing.assert_close(models[1](x),model(x),rtol=0,atol=0)

    def test_baseline_is_original_and_every_arm_has_32_tokens(self):
        x=torch.randn(3,2,256)
        baseline=M.build('P0',24101).eval()
        original=M.original_build('ManyRx',24101).eval()
        with torch.no_grad():torch.testing.assert_close(baseline(x),original(x),rtol=0,atol=0)
        for arm in M.ARMS:
            model=M.build(arm,24101).eval()
            with torch.no_grad():
                self.assertEqual(tuple(model.tokens(x).shape),(3,32,64))
                self.assertEqual(tuple(model(x).shape),(3,10))

    def test_temporal_features_equal_existing_s07(self):
        from xa.explore_models import time_views
        from xa.frontends import rms_normalize
        x=torch.randn(3,2,256)
        torch.testing.assert_close(M.time_features(x),time_views(rms_normalize(x),differential=True),rtol=0,atol=0)

    def test_all_branch_parameters_receive_gradients(self):
        model=M.build('P2',24101).train()
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        x=torch.randn(10,2,256);y=torch.arange(10)
        seen=set()
        for _ in range(4):
            opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(x),y);loss.backward()
            seen|={n for n,p in model.span.named_parameters() if p.grad is not None and p.grad.abs().sum()>0}
            opt.step()
        self.assertEqual(seen,{n for n,p in model.span.named_parameters()})

    def test_zero_initialized_branch_preserves_details(self):
        h=torch.randn(2,32,256)
        for arm in M.ARMS[1:]:torch.testing.assert_close(M.ResidualSpan(arm,24101)(h),h,rtol=0,atol=0)

    def test_invalid_arms_and_shapes_rejected(self):
        with self.assertRaises(ValueError):M.build('N21',24101)
        with self.assertRaises(ValueError):M.build('P2',24101).tokens(torch.zeros(1,2,200))
        with self.assertRaises(ValueError):M.valid_counts(0,3,20)

    def test_real_fit_loop_has_paired_orders_fixed_budget_and_replay(self):
        E=module('span_engine_cpu_test',ROOT/'research/periodic_span/engine.py')
        rng=np.random.default_rng(7)
        def fixture(n):
            return dict(x=rng.normal(size=(n,2,256)).astype(np.float32),y=np.arange(n)%10,
                rx=np.arange(n)%2,day=np.ones(n,dtype=np.int64),keys=np.asarray([f'k{i}' for i in range(n)]),pool_ids=np.arange(n))
        train,val=fixture(20),fixture(10)
        job=dict(id='synthetic',epochs=2,batch=8,stochastic_seed=24301,order_seed=24201,weight_decay=.001,max_gradient_updates=6)
        with tempfile.TemporaryDirectory() as tmp:
            histories=[]
            for arm in ('P1','P2'):
                out=Path(tmp)/arm
                result=E.fit_loop(dict(job,arm=arm),train,val,M.build(arm,24101),torch.device('cpu'),out,['rx0','rx1'])
                self.assertEqual(result['updates'],6)
                self.assertTrue(all(result['val_inference_replay'].values()))
                histories.append(E.read(out/'HISTORY.json'))
            self.assertEqual([r['canonical_order_sha256'] for r in histories[0]],[r['canonical_order_sha256'] for r in histories[1]])

    def test_selection_IQ_rejected_outside_sealed_entry(self):
        E=module('span_engine_guard_test',ROOT/'research/periodic_span/engine.py')
        with self.assertRaises(ValueError):E.materialize({}, {}, {'source_selection':np.arange(2)}, ('source_selection',), 'unregistered_job')


if __name__=='__main__':unittest.main()
