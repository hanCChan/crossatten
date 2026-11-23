"""No-RF structure, interaction algebra, features and exact-state recovery gate."""
import ast
import gc
import importlib
from pathlib import Path
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from s27b_common import HERE,P,write,now
from s27b_models import build,TokenMLP,state_hash,surviving_hash,geometry
from s27b_training import setup,save_resume,restore,learning_rate
from s27b_features import extract,geometry_record
from s27b_stats import contrast,interaction,ARMS
from s27b_data import authorize
from xa.frontends import awgn
from xa.train import stable_seed
import train as canonical


def main():
    for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    for n in ['s27b_evaluate','s27b_report','s27b_execute','s27b_register','s27b_source_check']:importlib.import_module(n)
    setup(27028);audits={};peaks=[];feature_equality=[]
    for arm in ARMS:
        model=build(arm,27028).eval()
        with torch.no_grad(),FlopCounterMode(display=False) as counter:assert model(torch.randn(2,2,8192)).shape==(2,10)
        audits[arm]=geometry(model);audits[arm]['partial_macs_per_packet']=counter.get_total_flops()/4
    assert [audits[a]['params'] for a in ARMS]==[133690,138346,133562,138218]
    assert all(audits[a]['tokens']==32 and audits[a]['cnn_receptive_field']==350 for a in ARMS)
    assert audits['H0_P']['partial_macs_per_packet']==94765696 and audits['LP_P']['partial_macs_per_packet']==55214720
    for fold in (0,1):
        for seed in (201,202,203):
            data_seed=stable_seed('P-H5x160',fold,seed)
            for base in ['H0','LP']:
                original=build(base,data_seed);control=build(base+'_P',data_seed)
                assert surviving_hash(original)==surviving_hash(control)
                assert all(isinstance(b,TokenMLP) and sum(p.numel() for p in b.parameters())==16704 for b in control.net.encoder.blocks)
                assert len(control.net.encoder.blocks)==2
            assert state_hash(build('H0_P',data_seed).net.encoder.blocks)==state_hash(build('LP_P',data_seed).net.encoder.blocks)
    denied=0
    for rx,fold,role in [(P['confirm_receivers'][0],0,'source'),(P['blind_receivers_never_open'][0],0,'source'),
        (P['folds'][0]['dev_val'][0],0,'source'),(P['folds'][0]['dev_val'][0],0,'dev_report_only')]:
        try:authorize(rx,fold,role)
        except PermissionError:denied+=1
    assert denied==4
    for arm in ['H0_P','LP_P']:
        setup(27028);model=build(arm,27028).cuda().train();x=torch.randn(8,2,8192,device='cuda');y=torch.arange(8,device='cuda')
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=5e-4)
        order=torch.Generator(device='cuda').manual_seed(27028);noise=torch.Generator(device='cuda').manual_seed(27029)
        def update(m,opt,og,ng,step):
            ids=torch.randperm(8,generator=og,device='cuda');xx=awgn(x[ids],0,50,ng)
            for g in opt.param_groups:g['lr']=learning_rate(step,125)
            loss=F.cross_entropy(m(xx),y[ids]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();assert torch.isfinite(loss)
        update(model,optimizer,order,noise,0)
        with tempfile.TemporaryDirectory(prefix='s27b-resume-') as directory:
            path=Path(directory)/'resume.pt';save_resume(path,model,optimizer,order,noise)
            update(model,optimizer,order,noise,1);expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            alternative=build(arm,27028).cuda().train();other=torch.optim.AdamW(alternative.parameters(),lr=1e-3,weight_decay=5e-4)
            og=torch.Generator(device='cuda');ng=torch.Generator(device='cuda');restore(path,alternative,other,og,ng);update(alternative,other,og,ng,1)
            assert all(torch.equal(v,alternative.state_dict()[k].cpu()) for k,v in expected.items())
        del model,alternative,optimizer,other;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        model=build(arm,27028).cuda().train();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=5e-4)
        loss=F.cross_entropy(model(torch.randn(64,2,8192,device='cuda')),torch.arange(64,device='cuda')%10)
        loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());opt.step()
        with torch.no_grad():assert model.eval()(torch.randn(128,2,8192,device='cuda')).shape==(128,10)
        peaks.append(torch.cuda.max_memory_allocated()/2**30)
        del model,opt;gc.collect();torch.cuda.empty_cache()
    for arm in ARMS:
        model=build(arm,27028).cuda().eval();x=np.random.default_rng(27028).normal(size=(8,2,8192)).astype('float32')
        plain=canonical.predict(model,x).cpu().numpy();features=extract(model,x)
        assert np.array_equal(plain,features['logits']);feature_equality.append(arm)
        del model;torch.cuda.empty_cache()
    fixture=[]
    for f in (0,1):
        for seed in (201,202,203):
            for rx in range(5):
                for arm,value in dict(H0=.9,LP=.8,H0_P=.85,LP_P=.80).items():fixture.append(dict(arm=arm,fold=f,seed=seed,rx=f'synthetic{f}_{rx}',acc=value))
    effect=contrast(fixture,dict(H0=1,LP=-1,H0_P=-1,LP_P=1))
    assert abs(effect['mean_delta_pp']-5)<1e-10 and effect['practical_rule_met']
    assert abs(interaction(.9,.8,.85,.8)-.05)<1e-12
    write(HERE/'GATE.json',dict(status='PASS',time=now(),architecture=audits,peak_gpu_gib=max(peaks),
        synthetic_train_batch=64,synthetic_eval_batch=128,synthetic_optimizer_steps=8,RF_reads=0,scientific_fits=0,
        original_plan_readout_preserved=True,shared_initialization_checks=6,exact_optimizer_rng_resume_both_arms=True,
        illegal_role_checks=denied,feature_forward_bit_exact_arms=feature_equality,factorial_interaction_synthetic_check='PASS',
        synthetic_fixture_scope='Only arithmetic/engineering check; not experimental results or scientific fits.'))
    print('PASS: frozenreadout,4cellgeometry/MAC,all6initialpairs,gradients,RNGresume,resources,forbiddenroles,features andinteractionalgebra.',flush=True)


if __name__=='__main__':main()
