"""Synthetic engineering only: no optimizer steps, IQ files or checkpoints."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
M=module('spectral_scale_models',ROOT/'research/spectral_scale_review/models.py')
A=module('spectral_scale_audit',ROOT/'research/spectral_scale_review/audit.py')
torch.set_num_threads(2)


class SpectralDesignTests(unittest.TestCase):
    def test_exact_parameter_matching_and_frozen_backbone(self):
        for arm in ('F1','F2'):
            m=M.build(arm,24101,26101)
            self.assertEqual(sum(p.numel() for p in m.parameters()),85627)
            self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad),3201)
            self.assertFalse(any(p.requires_grad for p in m.backbone.parameters()))

    def test_identity_fallback_matches_original_logits_and_tokens(self):
        x=torch.randn(4,2,256)
        old=M.build_original('ManyRx',24101).eval()
        with torch.no_grad():
            unfrozen_reference=old(x);tokens=old.tokens(x)
            old.requires_grad_(False)
            reference=old(x)
            # CPU may select a slightly different floating-point path after
            # requires_grad changes. The fallback reference is the frozen F0.
            torch.testing.assert_close(reference,unfrozen_reference,rtol=0,atol=1e-6)
            for arm in M.ARMS:
                m=M.build(arm,24101,26101).train()
                torch.testing.assert_close(m(x),reference,rtol=0,atol=0)
                torch.testing.assert_close(m.tokens(x),tokens,rtol=0,atol=0)
                self.assertEqual(tuple(m.tokens(x).shape),(4,32,64))

    def test_BN_stays_frozen_in_training_mode(self):
        m=M.build('F2',24101,26101).train()
        state={k:v.clone() for k,v in m.backbone.state_dict().items()}
        self.assertFalse(m.backbone.training)
        self.assertTrue(all(not b.training for b in m.backbone.modules() if isinstance(b,nn.BatchNorm1d)))
        m(torch.randn(3,2,256))
        for k,v in state.items():torch.testing.assert_close(v,m.backbone.state_dict()[k],rtol=0,atol=0)

    def test_alpha_initial_gradient_is_live_and_backbone_has_none(self):
        for arm in ('F1','F2'):
            m=M.build(arm,24101,26101).train()
            m(torch.randn(5,2,256))[:,0].sum().backward()
            self.assertTrue(torch.isfinite(m.residual.alpha.grad))
            self.assertGreater(float(m.residual.alpha.grad.abs()),0)
            self.assertTrue(all(p.grad is None for p in m.backbone.parameters()))

    def test_all_residual_parameters_can_receive_gradients_without_updates(self):
        for arm in ('F1','F2'):
            m=M.build(arm,24101,26101)
            with torch.no_grad():m.residual.alpha.fill_(.05)
            m(torch.randn(5,2,256))[:,0].sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.residual.parameters()))

    def test_turning_alpha_off_restores_exact_P0_after_side_changes(self):
        m=M.build('F2',24101,26101)
        x=torch.randn(3,2,256)
        with torch.no_grad():
            base=m(x)
            for name,p in m.residual.named_parameters():
                if name!='alpha':p.add_(.2)
            m.residual.alpha.fill_(.05)
            self.assertFalse(torch.equal(m(x),base))
            m.residual.alpha.zero_()
            torch.testing.assert_close(m(x),base,rtol=0,atol=0)

    def test_saved_P0_state_prefix_is_accepted_strictly(self):
        source=M.build('F0',24101)
        state={k:v.clone() for k,v in source.state_dict().items()}
        target=M.build('F2',999,26101,backbone_state=state)
        x=torch.randn(2,2,256)
        with torch.no_grad():torch.testing.assert_close(source(x),target(x),rtol=0,atol=0)
        with self.assertRaises(RuntimeError):M.build('F2',24101,26101,backbone_state={'bad':torch.zeros(1)})

    def test_spatial_geometry_has_equal_wide_span_without_wrap(self):
        x=torch.zeros(1,8,256);x[:,:,0]=1
        for arm in ('F1','F2'):
            s=M.SpectralResidual(arm,26101)
            self.assertEqual(s.wide.kernel_size,(33,))
            self.assertEqual(s.wide.dilation,(1,))
            self.assertEqual(s.wide.padding,(16,))
            self.assertEqual(tuple(s(x.new_zeros(1,32,256)).shape),(1,32,64))
            with torch.no_grad():
                s.wide.weight.fill_(1)
                y=s.wide(x)
                self.assertEqual(float(y[:,:,-1].abs().sum()),0)

    def test_residual_token_centers_match_the_original_tokenizer(self):
        m=M.build('F2',24101,26101)
        center,jump,receptive=0.,1,11
        for layer in m.backbone.frequency.tokenize:
            if isinstance(layer,nn.Conv1d):
                center+=((layer.kernel_size[0]-1)/2-layer.padding[0])*jump
                receptive+=(layer.kernel_size[0]-1)*jump
                jump*=layer.stride[0]
        self.assertEqual((center,jump,receptive),(3.5,8,32))
        self.assertEqual((8-1)/2,center)
        self.assertEqual(11+(3-1)+(8-1),20)
        self.assertEqual(11+(33-1)+(8-1),50)

    def test_common_layers_and_p0_initial_state_match(self):
        a,b=[M.build(arm,24101,26101) for arm in ('F1','F2')]
        for k,v in a.backbone.state_dict().items():torch.testing.assert_close(v,b.backbone.state_dict()[k],rtol=0,atol=0)
        for name in ('reduce','wide','output'):
            torch.testing.assert_close(getattr(a.residual,name).weight,getattr(b.residual,name).weight,rtol=0,atol=0)

    def test_oracle_pairing_rejects_conflicting_input_identity(self):
        y=np.array([0,1,2,3]);logits=np.eye(10)[y]
        a=dict(logits=logits,y=y,rx=np.array(['r0']*3+['r1']),day=np.ones(4),keys=np.array(['a','b','c','d']),pool_ids=np.arange(4))
        b={k:v[::-1].copy() for k,v in a.items()}
        rows=A.pair_metrics(a,b)
        self.assertEqual(rows['r0']['oracle'],1.)
        b['pool_ids'][0]=100
        with self.assertRaises(ValueError):A.pair_metrics(a,b)

    def test_RX_equal_metrics_do_not_turn_into_packet_weighting(self):
        y=np.array([0,1,2,3]);a_pred=np.array([0,1,2,0]);b_pred=np.array([1,2,0,3])
        meta=dict(y=y,rx=np.array(['r0']*3+['r1']),day=np.ones(4),keys=np.array(['a','b','c','d']),pool_ids=np.arange(4))
        rows=A.pair_metrics(dict(meta,logits=np.eye(10)[a_pred]),dict(meta,logits=np.eye(10)[b_pred]))
        self.assertEqual(np.mean([r['baseline_accuracy'] for r in rows.values()]),.5)
        self.assertEqual(np.mean([r['corrected'] for r in rows.values()]),.5)
        self.assertEqual(np.mean([r['oracle'] for r in rows.values()]),1.)


if __name__=='__main__':unittest.main()
