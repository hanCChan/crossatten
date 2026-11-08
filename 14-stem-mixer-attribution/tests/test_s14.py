import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from s14_core import write,HERE,FROZEN
from s14_models import build,common_hash,MechanismBlock,costs,dense_mixer_flops
from models import build as original
from s14_stats import factorial,paired,mechanism_gate
import s14_data
from s14_cfo import estimate_window,masks,group_score
import s14_cfo
from wifi import rotate,FS
import s14_execute

class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_patch_preserves_non_tokenizer_and_mixer(self):
        for new,old in [('E1_patch_mha','A1_mha'),('E2_patch_hstu','A3_rfhstu')]:
            a,b=build(new,123),original(old,123)
            self.assertEqual(common_hash(a),common_hash(b))
            for k,v in a.state_dict().items():
                if not k.startswith('net.embedder.cnn_stem.tokenize.'):self.assertTrue(torch.equal(v,b.state_dict()[k]),k)
        a,b=build('E1_patch_mha',123),build('E2_patch_hstu',123)
        for k,v in a.state_dict().items():
            if not k.startswith('net.encoder.blocks.'):self.assertTrue(torch.equal(v,b.state_dict()[k]),k)
        self.assertIsInstance(a.net.embedder.cnn_stem.tokenize,torch.nn.Conv1d)
    def test_sdpa_backend_preserves_all_weights_and_mha_function(self):
        for arm in ['A1_mha','A1_mha_matched']:
            a,b=build(arm,7),build(arm+'_sdpa',7)
            for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]))
            x=torch.randn(3,32,64)
            for original_block,optimized_block in zip(a.net.encoder.blocks,b.net.encoder.blocks):
                torch.testing.assert_close(original_block(x),optimized_block(x),atol=2e-6,rtol=1e-5)
    def test_dense_flops_match_unfused_transformer_profiler(self):
        from torch.utils.flop_counter import FlopCounterMode
        model=build('A2_transformer',1).eval()
        expected=dense_mixer_flops(model);fast=torch.backends.mha.get_fastpath_enabled();torch.backends.mha.set_fastpath_enabled(False)
        originals=[]
        for block in model.net.encoder.blocks:
            method=block.self_attn.forward;originals.append(method)
            def explicit(*args,_method=method,**kwargs):kwargs['need_weights']=True;return _method(*args,**kwargs)
            block.self_attn.forward=explicit
        try:
            with torch.no_grad(),FlopCounterMode(display=False) as counter:
                x=torch.randn(1,32,64)
                for block in model.net.encoder.blocks:x=block(x)
        finally:torch.backends.mha.set_fastpath_enabled(fast)
        self.assertEqual(expected,counter.get_total_flops())
        self.assertEqual(dense_mixer_flops(build('A1_mha',1)),dense_mixer_flops(build('A1_mha_sdpa',1)))
    def test_new_models_forward_backward_and_single_factor_init(self):
        for arm in ['E1_patch_mha','E2_patch_hstu','G1_no_u','G2_no_rel']:
            m=build(arm,5);out=m(torch.randn(2,2,8192));self.assertEqual(out.shape,(2,10));out.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
            if arm.startswith('G'):
                ref=original('A3_rfhstu',5)
                for k,v in m.state_dict().items():
                    if 'running_' not in k and 'num_batches' not in k:self.assertTrue(torch.equal(v,ref.state_dict()[k]),k)
    def test_gate_and_relative_interventions_have_declared_unused_parameters(self):
        x=torch.randn(2,32,64)
        for arm in ['G1_no_u','G2_no_rel']:
            m=MechanismBlock(arm);m(x).square().mean().backward()
            if arm=='G1_no_u':self.assertTrue(torch.equal(m.qkvu.weight.grad[192:],torch.zeros_like(m.qkvu.weight.grad[192:])))
            else:self.assertIsNone(m.relative_bias.grad)
            self.assertGreater(costs(build(arm,1),arm)['effectively_unused_params'],0)
    def test_factorial_interaction_is_paired_difference_of_differences(self):
        value=factorial(80,81,83,86)
        self.assertEqual(value['hstem_effect_mha'],3);self.assertEqual(value['hstem_effect_hstu'],5)
        self.assertEqual(value['interaction'],2);self.assertEqual(value['stem_main'],4);self.assertEqual(value['mixer_main'],2)
        for r in range(3):self.assertEqual(factorial(80+r,81+r,83+r,86+r)['interaction'],2)
    def test_bootstrap_clusters_rx(self):
        rows=[dict(rx=r,fold=f,seed=s,delta_pp=v) for r,v in [('a',1),('b',3)] for f in [0,1] for s in [201,202,203]]
        self.assertEqual(paired(rows)['mean_delta_pp'],2)
        self.assertEqual(paired(rows)['ci95_pp'],paired(rows*10)['ci95_pp'])
    def test_mechanism_gate_requires_all4_attention_controls(self):
        value=dict(mean_delta_pp=1.5,positive_pairs=6,worst_receiver_delta_pp=0,ci95_pp=[.1,2])
        values={a:dict(value) for a in ['A1_mha','A1_mha_matched','A2_transformer','A2_transformer_matched']}
        self.assertTrue(mechanism_gate(values)['admitted']);values['A1_mha']['mean_delta_pp']=.9
        self.assertFalse(mechanism_gate(values)['admitted'])
    def test_forbidden_cells_rejected_before_io(self):
        from xa.wisig import receivers
        source,target=receivers('ManyRx',0)
        with patch.object(s14_data,'sha',side_effect=AssertionError('must check role first')):
            for rx,day in [(target[0],1),(source[0],3)]:
                with self.assertRaises(ValueError):s14_data.wifi_cell('ManyRx',0,rx,day)
        with patch.object(s14_data,'read',side_effect=AssertionError('must check role first')):
            with self.assertRaises(ValueError):s14_data.shen_cell('b200_2',0,'source')
    def test_readiness_requires_completion_and_exited_s13_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'execution.pid').write_text('123')
            write(p/'progress.json',dict(status='RUNNING',phases={}))
            with patch.object(s14_execute,'S13',p),patch.object(s14_execute,'process_alive',return_value=False):self.assertFalse(s14_execute.readiness()[0])
            write(p/'progress.json',dict(status='COMPLETE_BOUNDED_SOURCE_WORKFLOW',phases={}))
            with patch.object(s14_execute,'S13',p),patch.object(s14_execute,'process_alive',return_value=True):self.assertFalse(s14_execute.readiness()[0])
            with patch.object(s14_execute,'S13',p),patch.object(s14_execute,'process_alive',return_value=False):self.assertTrue(s14_execute.readiness()[0])
    def test_cfo_strata_cover_all_packets_without_deletion(self):
        y=np.arange(6);rx=np.array(['a','a','b','b','c','c']);day=np.array([1,2,1,2,1,2])
        v=masks(y,rx,day,np.array([0,2000,-5000,10000,50000,530000]),np.array([0,.7,.9,.1,.8,.95]))
        self.assertEqual(sum(m.sum() for k,m in v.items() if k.startswith('abs_cfo/')),6)
        self.assertEqual(v['coherence/high'].sum(),4)
    def test_group_accuracy_is_receiver_equal(self):
        y=np.zeros(10,dtype=int);pred=np.array([0,0]+[1]*8);rx=np.array(['a']*2+['b']*8)
        value=group_score(np.eye(2)[pred],y,rx,np.ones(10,dtype=bool))
        self.assertEqual(value['acc'],50)
    def test_long_lag_cfo_check_handles_frequency_alias(self):
        x=torch.randn(2,2,20).repeat(1,1,13)[:,:,:256];shift=rotate(x,530000.)
        a,_=estimate_window(shift,0,20);b,_=estimate_window(shift,0,40)
        self.assertTrue((a>500000).all());self.assertTrue((b<0).all())
        phase=(b-a)*2*torch.pi*40/FS
        self.assertTrue((torch.atan2(phase.sin(),phase.cos()).abs()<1e-4).all())
    def test_adapter_does_not_replace_training_function(self):
        import s14_train
        before=s14_train.frozen_s13_train.fit
        s14_train.configure();self.assertIs(s14_train.frozen_s13_train.fit,before)
        self.assertEqual(s14_train.frozen_s13_train.HERE,HERE)

    def test_cfo_prediction_preserves_s13_per_receiver_batch_boundaries(self):
        class Fake(torch.nn.Module):
            def __init__(self):super().__init__();self.batches=[]
            def forward(self,x):self.batches.append(len(x));return torch.stack([x[:,0,0],-x[:,0,0]],1)
        m=Fake();x=np.zeros((460,2,8),np.float32);x[:,0,0]=np.arange(460)
        rx=np.array(['b']*200+['a']*260)
        with patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self):out=s14_cfo.predict(m,x,rx)
        self.assertEqual(m.batches,[128,128,4,128,72]);np.testing.assert_array_equal(out[:,0],np.arange(460))

    def test_uncommitted_s13_summary_is_waited_for_not_treated_as_failed_fit(self):
        with patch.object(s14_execute,'committed',side_effect=ValueError('not committed yet')):
            self.assertFalse(s14_execute.evidence_committed(Path('pending.json')))

    def test_complete_reports_with_synthetic48_plus12_fits(self):
        import s14_reports as report
        from xa.evaluate import summarize
        arms=['A0_cnn','A1_mha','A1_mha_matched','A2_transformer','A2_transformer_matched','A3_rfhstu','A4_softmax_gate','A5_silu_gate']
        acc={a:.92 for a in arms};acc.update(A1_mha=.93,A3_rfhstu=.96,E1_patch_mha=.90,E2_patch_hstu=.91)
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'s13';stage=Path(tmp)/'s14';stage.mkdir()
            write(source/'registration.json',dict(arms={a:{} for a in arms}))
            summary=[]
            for arm in arms+['E1_patch_mha','E2_patch_hstu']:
                for f in [0,1]:
                    for s in [201,202,203]:
                        dest=(source if arm.startswith('A') else stage)/'runs'/f'p1__Shen__{arm}__f{f}__all__s{s}'
                        per={rx:dict(acc=acc[arm],macro_f1=acc[arm]-.01) for rx in ['a','b']}
                        write(dest/'target_metrics.json',{r:dict(summary=summarize(per),per_rx=per) for r in ['dev_val','confirm']})
                        if arm.startswith('A'):write(dest/'config.json',dict(job=dict(fold=f,seed=s),initial_shared_sha256=f'paired{f}-{s}',epochs_budget=200,batch=64))
                if arm.startswith('A'):
                    for role in ['dev_val','confirm']:summary.append(dict(arm=arm,role=role,acc=100*acc[arm],macro_f1=100*(acc[arm]-.01),worst_receiver_mean_acc=100*acc[arm],seed_sd_pp=0))
            write(source/'p1_summary.json',dict(arm_summary=summary))
            write(source/'costs.json',dict(rows=[dict(arm=a,params=100,counted_mflops=1,gpu_batch1_p50_ms=1) for a in arms]))
            jobs=[dict(arm=a,fold=f,seed=s) for a in ['E1_patch_mha','E2_patch_hstu'] for f in [0,1] for s in [201,202,203]]
            write(stage/'registration.json',dict(jobs=jobs,reused={}))
            with patch.object(report,'HERE',stage),patch.object(report,'S13',source),patch.object(report,'committed',return_value=None):
                a=report.s13_comparison();b=report.factorial_report()
            self.assertEqual(len(a['contrasts']),14);self.assertTrue(a['mechanism_gate']['admitted'])
            self.assertEqual(len(b['arm_summary']),8)
            for row in b['contrasts']:
                if row['contrast']=='interaction':self.assertAlmostEqual(row['mean_delta_pp'],2)

if __name__=='__main__':unittest.main()
