"""Data-free checks of the actual execution adapter, not scientific fits."""
import ast
import copy
import gc
import random
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from ac_common import HERE,P,C,read,write,now
from ac_train import update
from ac_data import authorize
from ac_stats import contrast,c_pilot
from s28_models import build,geometry,state_hash,shared_hash,surviving_equal
from s27b_training import setup,save_resume,restore,learning_rate
from s27b_features import extract
from xa.frontends import awgn
from xa.train import stable_seed

def same(a,b):
    if isinstance(a,torch.Tensor):return a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def counted(kind):
    p=HERE/'ENGINEERING_UPDATES.json';q=read(p) if p.exists() else dict(updates=0,events=[])
    q['updates']+=1;q['events'].append(dict(time=now(),kind=kind,synthetic_only=True));write(p,q)

def main():
    assert not (HERE/'GATE.json').exists()
    for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    setup(28029);pairing=[];architecture={};peaks=[]
    for f in (0,1):
        for s in (201,202,203):
            a=stable_seed('P-H5x160',f,s);c=stable_seed('S27C-H4x160',f,s)
            x=build('H2_LA',a);assert state_hash(x)==state_hash(build('H2',a)) and surviving_equal(build('H0',a),x)
            assert shared_hash(build('H0',c))==shared_hash(build('LP',c)) and c!=a
            pairing.append(dict(fold=f,seed=s,A_seed=a,C_seed=c,initial_surviving_state_exact=True,fresh_C_shared_initialization=True))
    denied=0
    cases=[('C',C['splits'][f]['held_source_receiver'],f,role) for f in (0,1) for role in ['source','held_report']]
    cases += [('A',P['folds'][0]['dev_val'][0],0,'dev_report'),('A',P['confirm_receivers'][0],0,'source'),
              ('C',P['blind_receivers_never_open'][0],0,'source'),('A',P['folds'][0]['source'][0],0,'held_report')]
    for args in cases:
        try:authorize(*args)
        except PermissionError:denied+=1
    assert denied==8
    for arm,steps in [('H2_LA',125),('H0',100),('LP',100)]:
        setup(28029);m=build(arm,28029).cuda().train();architecture[arm]=geometry(m)
        x=torch.randn(8,2,8192,device='cuda');y=torch.arange(8,device='cuda')
        opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0005)
        order=torch.Generator(device='cuda').manual_seed(28029);noise=torch.Generator(device='cuda').manual_seed(28030)
        ids=torch.randperm(8,generator=order,device='cuda');update(m,opt,x[ids],y[ids],noise,0,steps);counted(arm+'_first_update')
        best={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        folder=HERE/'local/gate'/arm;folder.mkdir(parents=True,exist_ok=True);path=folder/'resume.pt'
        save_resume(path,m,opt,order,noise,best_model=best,history=[dict(epoch=1)],step=steps)
        draws=(float(np.random.rand()),random.random(),float(torch.rand(())))
        ids=torch.randperm(8,generator=order,device='cuda');aug=awgn(x[ids],0,50,noise)
        for group in opt.param_groups:group['lr']=learning_rate(1,steps)
        # Literal original update order, compared to resumed ac_train.update.
        loss=F.cross_entropy(m(aug),y[ids]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();counted(arm+'_literal_original_update')
        expected=copy.deepcopy(m.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
        second=build(arm,28029).cuda().train();other=torch.optim.AdamW(second.parameters(),lr=.001,weight_decay=.0005)
        og=torch.Generator(device='cuda');ng=torch.Generator(device='cuda');state=restore(path,second,other,og,ng)
        assert all(torch.equal(v,state['best_model'][k].cpu()) for k,v in best.items())
        assert draws==(float(np.random.rand()),random.random(),float(torch.rand(())))
        ids=torch.randperm(8,generator=og,device='cuda');update(second,other,x[ids],y[ids],ng,1,steps);counted(arm+'_resumed_update')
        assert same(expected,second.state_dict()) and same(expected_opt,other.state_dict())
        del m,second,opt,other,x,y,expected,expected_opt,state;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        m=build(arm,28029).cuda().train();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0005)
        resident=torch.zeros(10000,2,8192,device='cuda') # Include the largest full Source resident-data memory budget.
        update(m,opt,torch.randn(64,2,8192,device='cuda'),torch.arange(64,device='cuda')%10,
            torch.Generator(device='cuda').manual_seed(28030),0,steps);counted(arm+'_batch64_resource')
        with torch.no_grad():assert m.eval()(torch.randn(128,2,8192,device='cuda')).shape==(128,10)
        probe=np.random.default_rng(28029).normal(size=(8,2,8192)).astype('float32')
        with torch.no_grad():plain=m(torch.as_tensor(probe,device='cuda')).cpu().numpy()
        assert np.array_equal(plain,extract(m,probe)['logits'])
        peaks.append(torch.cuda.max_memory_allocated()/2**30)
        del m,opt,resident;gc.collect();torch.cuda.empty_cache()
    rows=[];sr=[];hr=[]
    for f in (0,1):
        for s in (201,202,203):
            for r in range(5):
                for arm,value in {'H0':.95,'LA':.94,'H2':.93,'H2_LA':.90}.items():rows.append(dict(arm=arm,fold=f,seed=s,rx=f'r{f}_{r}',acc=value))
            for arm,sa,ha in [('H0',.96,.93),('LP',.95,.90)]:
                hr.append(dict(arm=arm,fold=f,seed=s,rx=f'held{f}',acc=ha))
                sr += [dict(arm=arm,fold=f,seed=s,rx=f'r{f}_{r}',acc=sa) for r in range(4)]
    a=contrast(rows,{'H0':1,'LA':-1,'H2':-1,'H2_LA':1});assert abs(a['mean_delta_pp']+2)<1e-10 and a['two_sided_practical_rule_met']
    c=c_pilot(sr,hr);assert c['consistent_pilot_signal'] and abs(c['means']['delta_gap_pp']+2)<1e-10
    for q in hr:
        if q['fold']==1 and q['arm']=='H0':q['acc']=.89
    assert not c_pilot(sr,hr)['consistent_pilot_signal']
    write(HERE/'GATE.json',dict(time=now(),status='PASS',RF_reads=0,scientific_fits=0,synthetic_optimizer_updates=read(HERE/'ENGINEERING_UPDATES.json')['updates'],
        architecture=architecture,pairing=pairing,peak_gpu_gib=max(peaks),largest_resident_packet_budget=10000,
        actual_adapter_matches_original_update_and_full_resume=True,best_checkpoint_in_same_atomic_snapshot=True,
        feature_hook_forward_bit_exact=True,forbidden_role_checks=denied,A_negative_interaction_fixture=True,C_two_RX_signal_and_counterexample=True,
        scope='Engineering checks only. Original S28-0 seven synthetic updates remain separate historical evidence.'))
    print('PASS: actual adapter/recovery, three arms, all6 initial pairs, Source memory, roles, A/C decision fixtures; no RF reads.',flush=True)

if __name__=='__main__':main()
