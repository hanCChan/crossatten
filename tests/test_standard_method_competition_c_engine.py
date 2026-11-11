"""Synthetic engine integration. No scientific IQ, weights, or target data."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('C_engine_under_test',ROOT/'research/standard_method_competition_c/engine.py')
E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
torch.set_num_threads(2)


def toy_model():
    return nn.Sequential(nn.Flatten(),nn.Linear(16,6),nn.BatchNorm1d(6),nn.GELU(),nn.Linear(6,3))


def toy_data(n,seed):
    return dict(x=np.random.default_rng(seed).normal(size=(n,2,8)).astype(np.float32),
        y=np.arange(n,dtype=np.int64)%3,rx=np.arange(n,dtype=np.int64)%2,day=np.ones(n,dtype=np.int64),
        keys=np.asarray([f'r{i%2}|1|synthetic-{seed}|{i}' for i in range(n)]),pool_ids=np.arange(n,dtype=np.int64))


class EngineSyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory();cls.root=Path(cls.directory.name)
        torch.manual_seed(24101);cls.initial=copy.deepcopy(toy_model().state_dict())
        cls.results={}
        for name,family in (('ERM','ERM_EMA'),('REPLAY','ERM_EMA'),('DRO','RX_GDRO')):
            model=toy_model();model.load_state_dict(cls.initial)
            job=dict(id=name,family=family,readouts=['C0','C1'] if family=='ERM_EMA' else ['C2'],
                batch=4,epochs=40,weight_decay=.001,stochastic_seed=24301,order_seed=24201,max_gradient_updates=120)
            cls.results[name]=E.fit_loop(job,toy_data(12,10),toy_data(8,11),model,torch.device('cpu'),cls.root/name,['r0','r1'])

    @classmethod
    def tearDownClass(cls):cls.directory.cleanup()

    def test_fixed_budget_and_all_endpoints_replay(self):
        for result in self.results.values():
            self.assertEqual(result['updates'],120);self.assertEqual(result['epochs'],40)
            self.assertTrue(all(x['exact'] for x in result['val_inference_replay'].values()))
        self.assertEqual(set(self.results['ERM']['checkpoints']),{f'{a}_{e}.pt' for a in ('C0','C1') for e in E.ENDPOINTS})

    def test_full_engine_replays_without_extra_scientific_fit(self):
        a=E.read(self.root/'ERM/HISTORY.json');b=E.read(self.root/'REPLAY/HISTORY.json')
        self.assertEqual(a,b)
        for arm in ('C0','C1'):
            for endpoint in E.ENDPOINTS:
                first=torch.load(self.root/'ERM'/f'{arm}_{endpoint}.pt',weights_only=False)
                second=torch.load(self.root/'REPLAY'/f'{arm}_{endpoint}.pt',weights_only=False)
                self.assertTrue(E.A.compare_objects(first,second)['exact'])

    def test_same_initialization_and_order_across_loss_families(self):
        self.assertEqual(E.read(self.root/'ERM/INITIAL.json'),E.read(self.root/'DRO/INITIAL.json'))
        a=E.read(self.root/'ERM/HISTORY.json');b=E.read(self.root/'DRO/HISTORY.json')
        for left,right in zip(a,b):
            for field in ('relative_order_sha256','canonical_order_sha256','class_day_order_sha256'):
                self.assertEqual(left[field],right[field])

    def test_ema_costs_and_bn_calibration_are_explicit(self):
        costs=self.results['ERM']['costs']
        self.assertEqual(costs['EMA_updates'],120)
        self.assertEqual(costs['BN_calibration_passes'],40)
        self.assertEqual(costs['BN_calibration_samples'],480)
        self.assertTrue(all(x['calibration_preserved_training_state'] for x in E.read(self.root/'ERM/HISTORY.json')))
        self.assertEqual(self.results['DRO']['costs']['BN_calibration_passes'],0)

    def test_group_trace_has_every_update_and_probability_range(self):
        with np.load(self.root/'DRO/GROUP_TRACE.npz',allow_pickle=False) as z:
            self.assertEqual(z['q'].shape,(120,2))
            self.assertTrue(np.isfinite(z['q']).all());self.assertTrue((z['q']>=0).all());self.assertTrue((z['q']<=1).all())
            np.testing.assert_allclose(z['q'].sum(1),np.ones(120),rtol=0,atol=1e-12)
            self.assertTrue((z['count'].sum(1)==4).all())

    def test_checkpoint_epoch_and_primary_ce_match_strict_history_minimum(self):
        for name,result in self.results.items():
            history=E.read(self.root/name/'HISTORY.json')
            for arm in result['best_epochs']:
                best=min(history,key=lambda x:x['val_rx_ce'][arm])
                self.assertEqual(best['epoch'],result['best_epochs'][arm])
                state=torch.load(self.root/name/f'{arm}_selected.pt',weights_only=False)
                self.assertEqual(state['epoch'],best['epoch'])
                self.assertEqual(state['checkpoint_val_rx_ce'],best['val_rx_ce'][arm])

    def test_selection_scoring_rejects_missing_seal_before_metadata_or_IQ(self):
        with patch.object(E,'validate_seal',side_effect=ValueError('not sealed')),patch.object(E,'metadata') as meta,patch.object(E,'materialize') as iq:
            with self.assertRaisesRegex(ValueError,'not sealed'):E.evaluate({}, {}, [])
            meta.assert_not_called();iq.assert_not_called()

    def test_unknown_role_rejected_before_file_open(self):
        with patch.object(E,'verify_input') as verify:
            with self.assertRaisesRegex(ValueError,'Unknown role'):E.materialize({}, {}, {}, {'fit_train':[]},('target',),'synthetic')
            verify.assert_not_called()

    def test_existing_attempt_cannot_be_overwritten(self):
        tmp=self.root/'attempt_test';(tmp/'artifacts'/'j').mkdir(parents=True)
        with patch.object(E,'HERE',tmp),patch.object(E,'require_committed'),patch.object(E,'read',return_value=dict(status='PASS_EXACT_SOURCE_ROLES',execution_sha256='hash')),patch.object(E,'sha',return_value='hash'),patch.object(E.A,'setup') as setup:
            with self.assertRaisesRegex(FileExistsError,'attempt exists'):E.train({}, {}, [dict(id='j')], 'j')
            setup.assert_not_called()


if __name__=='__main__':unittest.main()
