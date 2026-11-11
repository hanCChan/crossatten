"""Engineering tests only; no scientific checkpoint/array reads or source fits."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'research/spectral_scale_execution'
sys.path.insert(0,str(HERE))
import common as H
import kernel as K
import executor as E
import analyze as S
torch.set_num_threads(2)


def fixture(n,seed):
    rng=np.random.default_rng(seed)
    return dict(x=rng.normal(size=(n,2,256)).astype(np.float32),y=(np.arange(n)%10).astype(np.int64),
        rx=(np.arange(n)%2).astype(np.int64),day=np.ones(n,dtype=np.int64),
        keys=np.asarray([f'synthetic|{i}' for i in range(n)]),pool_ids=np.arange(n,dtype=np.int64))


class ExecutorTests(unittest.TestCase):
    def test_scientific_authorization_is_absent_and_required(self):
        self.assertFalse((HERE/'EXECUTION_AUTHORIZATION.json').exists())
        with self.assertRaises(PermissionError):E.scientific_context()

    def test_source_IO_guard_runs_before_any_numpy_load(self):
        with mock.patch.object(np,'load',side_effect=AssertionError('must not load scientific arrays')):
            with self.assertRaises(PermissionError):E.read_metadata({})
            with self.assertRaises(PermissionError):E.materialize({}, {}, {}, ('checkpoint_val',),'test')
            with self.assertRaises(PermissionError):E.archived_reference_predictions('F0','V0','S0','source_selection')

    def test_all12_job_budget_and_streams_are_fixed(self):
        jobs=H.read(H.DESIGN/'PROPOSED_JOBS.json');H.validate_budget(jobs)
        broken=[dict(j) for j in jobs];broken[0]['side_initialization_seed']+=1
        with self.assertRaises(ValueError):H.validate_budget(broken)

    def test_epoch0_and_earliest_tie_remain_eligible(self):
        self.assertEqual(K.choose_checkpoint(.5,0,.5,1),(.5,0,False))
        self.assertEqual(K.choose_checkpoint(.5,0,.6,2),(.5,0,False))
        self.assertEqual(K.choose_checkpoint(.5,0,.4,1),(.4,1,True))
        self.assertEqual(K.choose_checkpoint(.4,1,.4,2),(.4,1,False))
        with self.assertRaises(ValueError):K.choose_checkpoint(.5,0,float('nan'),1)

    def test_parity_rejects_identity_drift_and_numeric_drift(self):
        d=fixture(10,7);names=['r0','r1'];logits=np.eye(10,dtype=np.float32)
        archive=dict(d,logits=logits.copy());archive.pop('x');archive['rx']=np.asarray(names)[d['rx']]
        self.assertTrue(E.parity(logits,d,archive,names)['identical_argmax'])
        bad={k:v.copy() for k,v in archive.items()};bad['pool_ids'][0]=999
        with self.assertRaises(ValueError):E.parity(logits,d,bad,names)
        with self.assertRaises(ValueError):E.parity(logits+1e-4,d,archive,names)
        bad={k:v.copy() for k,v in archive.items()};bad['logits'][0,0]=float('nan')
        with self.assertRaises(ValueError):E.parity(logits,d,bad,names)

    def test_same_real_fit_kernel_preserves_backbone_orders_and_gradients(self):
        train,val=fixture(20,8),fixture(10,9)
        job=dict(id='CPU_SYNTHETIC',epochs=2,batch=8,lr=.001,weight_decay=.001,stochastic_seed=26301,order_seed=24201,max_gradient_updates=6)
        with tempfile.TemporaryDirectory() as directory:
            histories=[];initials=[]
            for arm in ('F1','F2'):
                out=Path(directory)/arm;m=H.M.build(arm,24101,26101)
                original=H.semantic_hash(m.backbone.state_dict())
                result=K.fit_loop(dict(job,arm=arm),train,val,m,torch.device('cpu'),out,['r0','r1'])
                self.assertEqual(result['updates'],6)
                self.assertEqual(result['initial_task_side_gradient_l2'],0)
                self.assertGreater(abs(result['initial_task_alpha_gradient']),0)
                self.assertGreater(result['first_side_parameter_delta_l2'],0)
                self.assertTrue(result['all_task_gradients_finite'])
                self.assertTrue(result['all_backbone_gradients_absent'])
                self.assertEqual(H.semantic_hash(m.backbone.state_dict()),original)
                self.assertTrue(all(result['validation_endpoint_replay'].values()))
                history=H.read(out/'HISTORY.json');histories.append(history);initials.append(H.read(out/'INITIAL.json'))
                self.assertTrue(all(r['alpha0_rollback_bitwise'] for r in history))
                with np.load(out/'GRADIENT_TRACE.npz',allow_pickle=False) as f:
                    values=f['values'];cols=list(f['columns'])
                    self.assertEqual(values.shape,(6,len(K.TRACE_COLUMNS)))
                    self.assertTrue(np.isfinite(values).all())
                    self.assertGreater(values[1:,cols.index('task_side_gradient_l2')].max(),0)
                all_scores=[initials[-1]['epoch0_val_rx_ce']]+[r['val_rx_ce'] for r in history]
                self.assertEqual(result['best_epoch'],min(range(3),key=lambda k:all_scores[k]))
            self.assertEqual(initials[0]['common_residual_sha256'],initials[1]['common_residual_sha256'])
            self.assertEqual(initials[0]['rng_sha256'],initials[1]['rng_sha256'])
            self.assertEqual([r['canonical_order_sha256'] for r in histories[0]],[r['canonical_order_sha256'] for r in histories[1]])


def synthetic_predictions(improved_selected=True,ema_already_best=False):
    result={}
    for endpoint in H.ENDPOINTS:
        arms={a:{} for a in S.ARMS}
        for view in ('V0','V1'):
            rx=np.repeat([f'{view}r{i}' for i in range(6)],50)
            y=np.zeros(300,dtype=np.int64)
            for stream in ('S0','S1','S2'):
                for arm in S.ARMS:
                    correct=np.ones(300,dtype=bool)
                    for i in range(6):correct[i*50]=False
                    improved=(arm=='F2' and (improved_selected or endpoint!='selected')) or (arm=='E0' and ema_already_best)
                    if improved:correct[:]=True
                    logits=np.full((300,10),-2.,dtype=np.float64);logits[:,0]=1;logits[~correct,1]=2
                    arms[arm][(view,stream)]=dict(logits=logits,y=y,rx=rx,day=np.ones(300,dtype=np.int64),
                        keys=np.asarray([f'{view}|{i}' for i in range(300)]),pool_ids=np.arange(300,dtype=np.int64))
        result[endpoint]=arms
    return result


class DecisionTests(unittest.TestCase):
    def test_all_three_references_must_be_beaten(self):
        result=S.analyze_arrays(synthetic_predictions(),128)
        self.assertEqual(result['decision']['status'],'F2_WORTH_SEPARATE_CONFIRMATION')
        result=S.analyze_arrays(synthetic_predictions(ema_already_best=True),128)
        self.assertEqual(result['decision']['status'],'STOP_F2_FROZEN_BACKBONE_RESIDUAL')
        self.assertFalse(result['decision']['gates']['F2-E0']['passes'])

    def test_auxiliary_results_cannot_replace_selected(self):
        result=S.analyze_arrays(synthetic_predictions(improved_selected=False),128)
        self.assertEqual(result['decision']['status'],'STOP_F2_FROZEN_BACKBONE_RESIDUAL')
        self.assertGreater(result['endpoints']['final40']['comparisons']['F2-F1']['rx_equal']['accuracy_delta'],.01)


if __name__=='__main__':unittest.main()
