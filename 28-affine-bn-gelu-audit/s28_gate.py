"""Synthetic-only mathematical and reserved-arm engineering gate."""
import ast
import copy
import gc
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from s28_core import HERE,P,read,write,now
from s28_models import build,geometry,state_hash,shared_hash,surviving_equal
from s28_affine import MaterializedTokenizer,error_record
from s28_data import authorize,c_role_permitted
from s27b_training import setup,save_resume,restore,learning_rate
from xa.frontends import awgn
from xa.train import stable_seed

def same(a,b):
    if isinstance(a,torch.Tensor):return a.dtype==b.dtype and torch.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def main():
    assert not (HERE/'GATE.json').exists(),'Preserve a completed Gate; do not repeat synthetic updates.'
    torch.set_num_threads(2);setup(28028)
    for file in HERE.glob('*.py'):ast.parse(file.read_text(encoding='utf-8'))
    architecture={};pairs=[]
    for arm in ['H0','LA','H2','H2_LA','LP']:
        model=build(arm,28028).eval()
        with torch.no_grad(),FlopCounterMode(display=False) as count:assert model(torch.randn(2,2,8192)).shape==(2,10)
        architecture[arm]=geometry(model);architecture[arm]['partial_macs_per_packet']=count.get_total_flops()/4
    assert [architecture[a]['params'] for a in ['H0','LA','H2','H2_LA','LP']]==[133690,133690,133338,133338,138346]
    assert all(q['cnn_receptive_field']==350 and q['tokens']==32 and q['token_stride']==256 for q in architecture.values())
    for fold in (0,1):
        for seed in (201,202,203):
            data_seed=stable_seed('P-H5x160',fold,seed)
            a,b,c,d=[build(arm,data_seed) for arm in ['H0','LA','H2','H2_LA']]
            assert state_hash(a)==state_hash(b) and state_hash(c)==state_hash(d)
            assert surviving_equal(a,c) and surviving_equal(a,d)
            assert all(shared_hash(m)==shared_hash(a) for m in [b,c,d])
            c_seed=stable_seed('S27C-H4x160',fold,seed)
            assert c_seed!=data_seed and shared_hash(build('H0',c_seed))==shared_hash(build('LP',c_seed))
            pairs.append(dict(fold=fold,seed=seed,A_data_seed=data_seed,C_data_seed=c_seed,
                H0_LA_full_initial_state_equal=True,H2_H2LA_full_initial_state_equal=True,all_surviving_initial_parameters_equal=True))
    m=build('LA',28028).double().eval().net.embedder.cnn_stem.tokenize
    operator=MaterializedTokenizer(m)
    g=torch.Generator().manual_seed(28028);x=torch.randn(6,32,8192,generator=g,dtype=torch.float64)
    x[0].zero_();x[2]*=100;x[3:].zero_();x[3,:,0]=1;x[4,:,-1]=1;x[5,:,256]=1
    with torch.no_grad():
        expected=m(x);correct=operator(x);plain=operator.naive(x)
        math_check=error_record(expected,correct,1e-9,1e-9)
        naive_check=error_record(expected,plain,1e-9,1e-9)
        interior=error_record(expected[:,:,1:31],plain[:,:,1:31],1e-9,1e-9)
    assert math_check['allclose'] and interior['allclose'] and not naive_check['allclose']
    denied=0
    for rx,fold,role in [(P['confirm_receivers'][0],0,'audit_source_val'),(P['blind_receivers_never_open'][0],0,'audit_source_val'),
        (P['folds'][0]['dev_val'][0],0,'audit_source_val'),(P['folds'][0]['source'][0],0,'fit')]:
        try:authorize(rx,fold,role)
        except PermissionError:denied+=1
    assert denied==4
    plan=read(HERE/'PLAN_C.original.json')
    for split in plan['splits']:
        f=split['fold'];held=split['held_source_receiver']
        assert not c_role_permitted(held,f,'fit') and not c_role_permitted(held,f,'selection') and not c_role_permitted(held,f,'held_report')
        assert all(c_role_permitted(rx,f,'fit') for rx in split['training_receivers'])
    assert learning_rate(0,100)==1e-3/500 and learning_rate(499,100)==1e-3
    assert learning_rate(0,125)==1e-3/625 and learning_rate(624,125)==1e-3
    del a,b,c,d,m,operator,x,expected,correct,plain;gc.collect()
    setup(28028);model=build('H2_LA',28028).cuda().train()
    x=torch.randn(8,2,8192,device='cuda');y=torch.arange(8,device='cuda')
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=5e-4)
    order=torch.Generator(device='cuda').manual_seed(28028);noise=torch.Generator(device='cuda').manual_seed(28029)
    def update(m,o,og,ng):
        draws=(float(np.random.rand()),random.random(),float(torch.rand(())))
        ids=torch.randperm(8,generator=og,device='cuda');xx=awgn(x[ids],0,50,ng)
        loss=F.cross_entropy(m(xx),y[ids]);o.zero_grad(set_to_none=True);loss.backward();o.step()
        record_update('resume_equivalence')
        assert torch.isfinite(loss)
        return draws
    update(model,opt,order,noise)
    path=HERE/'local/gate/resume.pt';path.parent.mkdir(parents=True,exist_ok=True)
    save_resume(path,model,opt,order,noise,scope='Synthetic gate only')
    expected_draws=update(model,opt,order,noise)
    expected_state=copy.deepcopy(model.state_dict());expected_optimizer=copy.deepcopy(opt.state_dict())
    alternative=build('H2_LA',28028).cuda().train();other=torch.optim.AdamW(alternative.parameters(),lr=1e-3,weight_decay=5e-4)
    og=torch.Generator(device='cuda');ng=torch.Generator(device='cuda');restore(path,alternative,other,og,ng)
    assert update(alternative,other,og,ng)==expected_draws
    assert same(expected_state,alternative.state_dict()) and same(expected_optimizer,other.state_dict())
    del model,alternative,opt,other,expected_state,expected_optimizer,x,y;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    model=build('H2_LA',28028).cuda().train();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=5e-4)
    loss=F.cross_entropy(model(torch.randn(64,2,8192,device='cuda')),torch.arange(64,device='cuda')%10)
    loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());opt.step();record_update('batch64_resource')
    with torch.no_grad():assert model.eval()(torch.randn(128,2,8192,device='cuda')).shape==(128,10)
    peak=torch.cuda.max_memory_allocated()/2**30
    assert abs(((.95-.94)-(.93-.90))+.02)<1e-12
    write(HERE/'GATE.json',dict(time=now(),status='PASS',scope='Synthetic architecture/math/RNG/resource checks; no RF or scientific fitting.',
        architecture=architecture,pairing=pairs,affine_corrected=math_check,naive_shared_conv=naive_check,naive_interior=interior,
        optimizer_model_all_rng_resume_exact=True,optimizer_state_values_exact=True,optimizer_step_scalar_device_may_differ=True,
        synthetic_optimizer_updates=read(HERE/'ENGINEERING_UPDATES.json')['updates'],scientific_fits=0,
        batch64_backward=True,batch128_forward=True,peak_gpu_gib=peak,denied_audit_roles=denied,C_held_never_fit_or_select=True,
        C_learning_rate_100_steps=True,A_learning_rate_125_steps=True,factorial_arithmetic=True,RF_reads=0))
    print('PASS: affine boundaries, all initial pairs, exact optimizer/RNG recovery, resources and forbidden roles.',flush=True)

def record_update(kind):
    path=HERE/'ENGINEERING_UPDATES.json'
    ledger=read(path) if path.exists() else dict(updates=0,events=[])
    ledger['updates']+=1;ledger['events'].append(dict(time=now(),kind=kind,RF=False,scientific_fit=False))
    write(path,ledger)

if __name__=='__main__':main()
