"""CPU structural/gradient/leakage checks; no experimental labels or GPU use."""
import unittest,copy,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from s16_core import HERE,read
from s16_models import build,tiny_config,small_match,trainable_state,load_trainable
from s16_data import cells,RFTokenizer,groups
from s16_reports import delta_stats,gate
from wifi import build_wifi
from xa.wisig import receivers

class EngineeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_causal_suffix_and_cls_readout(self):
        for arm in ['R1_qwen_small','R2_transformer']:
            m=build(arm,17001).eval();x=torch.randn(2,32,64);altered=x.clone();altered[:,31]+=5
            with torch.no_grad():a=m.hidden(x);b=m.hidden(altered)
            self.assertEqual(tuple(a.shape),(2,33,128))
            torch.testing.assert_close(a[:,:31],b[:,:31],rtol=0,atol=1e-6)
            self.assertGreater(float((a[:,-1]-b[:,-1]).abs().max()),1e-6)
            torch.testing.assert_close(m(x),m.head(a[:,-1]))

    def test_frozen_backbone_still_trains_rf_bridge(self):
        for arm in ['R3_pretrained_frozen','R4_random_frozen']:
            m=build(arm,17001,engineering_config=tiny_config());x=torch.randn(2,32,64)
            torch.nn.functional.cross_entropy(m(x),torch.tensor([0,1])).backward()
            self.assertTrue(all(p.grad is None for p in m.backbone.parameters()))
            self.assertGreater(float(m.project.weight.grad.abs().sum()),0)
            self.assertGreater(float(m.cls.grad.abs().sum()),0)
            self.assertGreater(float(m.head.weight.grad.abs().sum()),0)
            self.assertIsNone(m.backbone.get_input_embeddings())

    def test_paired_heads_and_lora_initialization(self):
        for a,b in [('R3_pretrained_frozen','R4_random_frozen'),('R5_pretrained_lora','R6_random_lora')]:
            x=build(a,17001,engineering_config=tiny_config());y=build(b,17001,engineering_config=tiny_config())
            sx=trainable_state(x);sy=trainable_state(y);self.assertEqual(set(sx),set(sy))
            for key in sx:torch.testing.assert_close(sx[key],sy[key],rtol=0,atol=0)
            if 'lora' in a:
                self.assertTrue(any('lora_A' in k for k in sx))
                self.assertTrue(all('lora_' in k for k,p in x.backbone.named_parameters() if p.requires_grad))
                self.assertTrue(all(p.dtype==torch.float32 for p in x.parameters() if p.requires_grad))
        x=build('R0_mha',17001);y=build('R0_hstu',17001)
        for key,value in x.state_dict().items():
            if not key.startswith('encoder.blocks.'):torch.testing.assert_close(value,y.state_dict()[key],rtol=0,atol=0)

    def test_native_rf_tokenizer_equivalence(self):
        base=build_wifi('ManyRx','B0',17001).eval();rf=RFTokenizer(base);x=torch.randn(2,2,256)
        with torch.no_grad():z=rf(x);old=base.tokens(x)
        self.assertEqual(tuple(z.shape),(2,32,64));torch.testing.assert_close(z,old,rtol=0,atol=0)
        self.assertFalse(any(p.requires_grad for p in rf.parameters()))

    def test_parameter_matching_and_checkpoint(self):
        result=small_match();self.assertEqual(result,read(HERE/'small_architecture_match.json'))
        self.assertLessEqual(result['relative_error'],.02)
        m=build('R1_qwen_small',17001);state=copy.deepcopy(trainable_state(m));x=torch.randn(2,32,64)
        before=m(x).detach();opt=torch.optim.AdamW(m.parameters(),lr=.01)
        m(x).sum().backward();opt.step();self.assertFalse(torch.equal(before,m(x).detach()))
        load_trainable(m,state);torch.testing.assert_close(before,m(x))

    def test_forbidden_role_rejected_before_io(self):
        for dataset in ['ManyRx','ManySig']:
            source,target=receivers(dataset,0)
            for kwargs in [dict(names=target),dict(names=source[:1],days=(3,4)),dict(names=source[:1],role='target')]:
                with self.assertRaises(ValueError):cells(dataset,0,**kwargs)
            a,b=groups(dataset,0,'last');self.assertFalse(set(a)&set(b));self.assertEqual(set(a+b),set(source))

    def test_receiver_cluster_gate_and_complete_folds(self):
        rows=[dict(fold=f,rx=f'r{r}',delta_pp=2.) for f in range(4) for r in range(6)]
        stats=delta_stats(rows);duplicated=delta_stats(rows+rows)
        self.assertEqual(stats,duplicated);self.assertTrue(gate(dict(ManyRx=stats))['admitted'])
        bad=copy.deepcopy(stats);bad['worst_rx_pp']=-3
        self.assertFalse(gate(dict(ManyRx=bad))['admitted'])
        with self.assertRaises(ValueError):delta_stats(rows[:6])

    def test_confirmation_controls_cannot_be_promoted(self):
        import s16_reports as reports
        from s16_core import write,sha
        from s16_train import run_id
        with tempfile.TemporaryDirectory(prefix='s16_gate_') as directory:
            here=Path(directory);jobs=[]
            for dataset in ['ManyRx','ManySig']:
                for fold in range(4):
                    for arm,acc in [('R1_qwen_small',.99),('R2_transformer',.95),('R0_mha',.5),('R0_hstu',.5)]:
                        job=dict(phase='confirm1',dataset=dataset,arm=arm,fold=fold,seed=17011,group='last');jobs.append(job)
                        out=here/'runs'/run_id(job);out.mkdir(parents=True);(out/'trainable.pt').write_bytes(b'synthetic test')
                        score=dict(summary=dict(acc_mean=acc,f1_mean=acc,worst_acc=acc),per_rx={f'r{r}':dict(acc=acc,macro_f1=acc) for r in range(2)})
                        write(out/'metrics.json',dict(trainable_sha256=sha(out/'trainable.pt'),held_source=score,source_validation=score,source_train=score,
                            source_to_held_gap_pp=0,active_params=1,trainable_params=1,peak_allocated_gib=0,wall_seconds=1))
            write(here/'jobs_confirm1.json',jobs);write(here/'registration_confirm1.json',dict(candidates=['R1_qwen_small']))
            with patch.object(reports,'HERE',here),patch.object(reports,'validate',lambda:None):result=reports.report('confirm1')
            self.assertEqual(result['selected'],['R1_qwen_small']);self.assertNotIn('R2_transformer',result['decisions'])

if __name__=='__main__':unittest.main(verbosity=2)
