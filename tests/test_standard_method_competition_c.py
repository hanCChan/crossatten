"""Synthetic CPU-only method/permission checks; no scientific data or models."""
import copy
import importlib.util
import random
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'research/standard_method_competition_c'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M, P = load('C_kernels_test', 'methods.py'), load('C_plan_test', 'prepare.py')


def tensor_state_equal(first, second):
    return first.keys()==second.keys() and all(torch.equal(first[k],second[k]) for k in first)


def toy_source():
    names=[f'rx{i:02d}' for i in range(18)]
    y,rx,day,train,val=[],[],[],[],[]
    for receiver in range(18):
        for date in (1,2):
            for tx in range(10):
                for packet in range(3):
                    i=len(y); y.append(tx);rx.append(receiver);day.append(date)
                    (train if packet<2 else val).append(i)
    return (*[np.asarray(a,dtype=np.int32) for a in (y,rx,day,train,val)],names,names)


class StandardMethodCompetitionTests(unittest.TestCase):
    def test_ema_first_step_copies_then_uses_registered_decay(self):
        student=nn.Linear(1,1,bias=False)
        student.weight.data.fill_(0)
        shadow=M.make_ema(student)
        student.weight.data.fill_(2); shadow.update_parameters(student)
        self.assertEqual(float(shadow.module.weight),2.)
        student.weight.data.fill_(4); shadow.update_parameters(student)
        self.assertAlmostEqual(float(shadow.module.weight),2.002,places=6)
        self.assertEqual(int(shadow.n_averaged),2)
        self.assertEqual(float(student.weight),4.)

    def test_ema_buffers_are_copied_instead_of_float_averaged(self):
        student=nn.BatchNorm1d(2)
        shadow=M.make_ema(student)
        for mean,counter in ((3.,7),(9.,13)):
            student.running_mean.fill_(mean);student.num_batches_tracked.fill_(counter)
            shadow.update_parameters(student)
        torch.testing.assert_close(shadow.module.running_mean,torch.full((2,),9.))
        self.assertEqual(int(shadow.module.num_batches_tracked),13)

    def test_bn_calibration_changes_only_candidate_buffers(self):
        student=nn.Sequential(nn.Linear(2,2,bias=False),nn.BatchNorm1d(2))
        shadow=M.make_ema(student);shadow.update_parameters(student)
        student_before=copy.deepcopy(student.state_dict());shadow_before=copy.deepcopy(shadow.state_dict())
        inputs=[torch.ones(4,2)*5,torch.ones(3,2)*7]
        candidate,record=M.calibrated_ema_candidate(shadow,inputs,role='fit_train')
        self.assertTrue(tensor_state_equal(student.state_dict(),student_before))
        self.assertTrue(tensor_state_equal(shadow.state_dict(),shadow_before))
        for p,q in zip(candidate.parameters(),shadow.module.parameters()):self.assertTrue(torch.equal(p,q))
        self.assertEqual(int(candidate[1].num_batches_tracked),2)
        self.assertFalse(torch.equal(candidate[1].running_mean,shadow.module[1].running_mean))
        self.assertFalse(candidate.training);self.assertEqual(record['samples'],7)

    def test_bn_calibration_restores_rng_and_rejects_nontrain_roles(self):
        random.seed(101);np.random.seed(102);torch.manual_seed(103)
        shadow=M.make_ema(nn.BatchNorm1d(2))
        python_before=random.getstate();numpy_before=np.random.get_state();torch_before=torch.get_rng_state().clone()
        def batches():
            for _ in range(2):
                random.random();np.random.random()
                yield torch.randn(4,2)
        first,_=M.calibrated_ema_candidate(shadow,batches(),role='fit_train')
        self.assertEqual(random.getstate(),python_before)
        np.testing.assert_array_equal(np.random.get_state()[1],numpy_before[1])
        self.assertTrue(torch.equal(torch.get_rng_state(),torch_before))
        second,_=M.calibrated_ema_candidate(shadow,batches(),role='fit_train')
        self.assertTrue(tensor_state_equal(first.state_dict(),second.state_dict()))
        for bad_role in ('checkpoint_val','source_selection','historical_held','target'):
            with self.assertRaisesRegex(ValueError,'fit_train only'):
                M.calibrated_ema_candidate(shadow,[],role=bad_role)

    def test_empty_bn_calibration_fails_without_modifying_shadow(self):
        shadow=M.make_ema(nn.BatchNorm1d(2));before=copy.deepcopy(shadow.state_dict())
        with self.assertRaisesRegex(ValueError,'calibration inputs'):
            M.calibrated_ema_candidate(shadow,[],role='fit_train')
        self.assertTrue(tensor_state_equal(shadow.state_dict(),before))

    def test_group_dro_matches_multiplicative_formula_and_gradient(self):
        losses=torch.tensor([1.,1.,3.],requires_grad=True)
        model=M.RxGroupDRO(2)
        objective,record=model.loss(losses,torch.tensor([0,0,1]))
        q=torch.softmax(torch.tensor([.01,.03],dtype=torch.float64),0)
        torch.testing.assert_close(record['q'],q)
        self.assertAlmostEqual(float(objective),float(q[0]+3*q[1]),places=6)
        objective.backward()
        torch.testing.assert_close(losses.grad,torch.tensor([float(q[0]/2),float(q[0]/2),float(q[1])]))
        self.assertFalse(record['q'].requires_grad)

    def test_missing_batch_groups_get_zero_loss_not_active_renormalization(self):
        dro=M.RxGroupDRO(3)
        loss,record=dro.loss(torch.tensor([2.,2.]),torch.tensor([0,0]))
        np.testing.assert_array_equal(record['group_count'].numpy(),[2,0,0])
        self.assertFalse(bool(record['present'][1]));self.assertEqual(float(record['group_loss'][1]),0)
        self.assertLess(float(record['q'][1]),1/3)
        self.assertAlmostEqual(float(record['q'].sum()),1.,places=12)
        self.assertLess(float(loss),2.)

    def test_group_dro_invalid_or_overflowing_loss_does_not_corrupt_q(self):
        for bad in (torch.tensor([float('nan')]),torch.tensor([-1.]),torch.tensor([3e38,3e38])):
            dro=M.RxGroupDRO(2);before=dro.log_q.clone()
            with self.assertRaises(ValueError):dro.loss(bad,torch.zeros(len(bad),dtype=torch.int64))
            self.assertTrue(torch.equal(before,dro.log_q))
        dro=M.RxGroupDRO(2)
        with self.assertRaisesRegex(ValueError,'Unregistered'):
            dro.loss(torch.tensor([1.]),torch.tensor([2]))

    def test_group_dro_state_roundtrip_preserves_next_update(self):
        dro=M.RxGroupDRO(2)
        dro.loss(torch.tensor([1.,2.]),torch.tensor([0,1]))
        other=M.RxGroupDRO(2);other.load_state_dict(dro.state_dict())
        first,_=dro.loss(torch.tensor([3.,1.]),torch.tensor([0,1]))
        second,_=other.loss(torch.tensor([3.,1.]),torch.tensor([0,1]))
        self.assertTrue(torch.equal(first,second));self.assertTrue(torch.equal(dro.log_q,other.log_q))

    def test_source_cv_preserves_old_membership_and_disjoint_roles(self):
        args=toy_source();views,arrays=P.source_views(*args)
        y,rx,day,train,val,names,legal=args
        self.assertFalse(set(views[0]['source_selection_rx'])&set(views[1]['source_selection_rx']))
        for view in views:
            ids=arrays[view['id']]
            self.assertEqual(view['fit_train']['rows'],480)
            self.assertEqual(view['checkpoint_val']['rows'],240)
            self.assertEqual(view['source_selection']['rows'],360)
            self.assertTrue(set(ids['fit_train'])<=set(train));self.assertTrue(set(ids['checkpoint_val'])<=set(val))
            self.assertEqual(set(np.r_[*ids.values()]),set(range(len(y))))
            self.assertFalse(set(ids['fit_train'])&set(ids['source_selection']))
            self.assertEqual(len(view['fit_rx']),12)
            self.assertEqual(view['gradient_updates_per_fit'],160)

    def test_source_cv_determinism_does_not_depend_on_role_list_order(self):
        args=list(toy_source());views,arrays=P.source_views(*args)
        args[-1]=list(reversed(args[-1]));again,second=P.source_views(*args)
        for i in range(2):
            self.assertEqual(views[i]['source_selection_rx'],again[i]['source_selection_rx'])
            for role in arrays[f'V{i}']:np.testing.assert_array_equal(arrays[f'V{i}'][role],second[f'V{i}'][role])

    def test_source_cv_rejects_extra_rx_invalid_dates_and_split_overlap(self):
        args=list(toy_source());args[-1]=args[-1][:-1]+['unregistered_target']
        with self.assertRaisesRegex(ValueError,'legal fold2'):P.source_views(*args)
        args=list(toy_source());args[2]=args[2].copy();args[2][0]=3
        with self.assertRaisesRegex(ValueError,'date'):P.source_views(*args)
        args=list(toy_source());args[4]=args[4].copy();args[4][0]=args[3][0]
        with self.assertRaisesRegex(ValueError,'partition'):P.source_views(*args)


if __name__=='__main__':unittest.main()
