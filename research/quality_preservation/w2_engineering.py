"""Synthetic CUDA acceptance exercises the exact inherited fit loop for 40 epochs."""
from pathlib import Path
import copy
import json
import numpy as np
import torch
from w2_models import A,C,QualityGate,packed,derangement
from run_w1 import digest,write,HERE


def main():
    A.setup(24101);device=torch.device('cuda',0)
    rng=np.random.default_rng(28901)
    data={}
    for role,n in [('train',256),('val',64)]:
        data[role]=dict(x=rng.normal(size=(n,2,256)).astype('float32'),y=np.arange(n,dtype='int64')%10,
            rx=np.arange(n,dtype='int64')%2,day=np.ones(n,dtype='int64'),keys=np.array([f'{role}_{i}' for i in range(n)]),pool_ids=np.arange(n,dtype='int64'))
    checks={};results={};initials={}
    for arm in ('B1','B2','N0'):
        A.setup(24101);m=QualityGate(arm,24101).to(device).eval()
        tr=copy.deepcopy(data['train']);va=copy.deepcopy(data['val'])
        tr['x'],p=packed(tr['x'],arm,29001);va['x'],_=packed(va['x'],arm,29002)
        inputs=torch.as_tensor(tr['x'][:16],device=device)
        with torch.no_grad():
            baseline=m.backbone(inputs[:,:512].reshape(-1,2,256));out=m(inputs)
        checks[arm+'_identity_initialization']=torch.equal(baseline,out)
        checks[arm+'_all_trainable']=all(p.requires_grad for p in m.parameters())
        checks[arm+'_same_parameter_count']=sum(p.numel() for p in m.parameters())==83554
        initials[arm]=A.semantic_hash(m.state_dict())
        # Full40-epoch scheduling, BN calibration, EMA and checkpoint replay use
        # the same unchanged C kernel as scientific fits, with synthetic data only.
        job=dict(id='synthetic_'+arm,family='ERM_EMA',readouts=['C0','C1'],stochastic_seed=24301,
            order_seed=24201,weight_decay=.001,epochs=40,batch=128,max_gradient_updates=80)
        outdir=HERE/'artifacts'/('synthetic_v2_'+arm)
        if outdir.exists():raise FileExistsError(outdir)
        results[arm]=C.fit_loop(job,tr,va,m,device,outdir,['r0','r1'])
        checks[arm+'_80_updates']=results[arm]['updates']==80
        checks[arm+'_10240_BN_samples']=results[arm]['costs']['BN_calibration_samples']==10240
        checks[arm+'_EMA_replay']=all(v['exact'] for v in results[arm]['val_inference_replay'].values())
    checks['common_initial_parameters']=len(set(initials.values()))==1
    p=derangement(257,29001)
    checks['negative_control_no_self_pair']=bool(np.all(p!=np.arange(257)))
    checks['negative_control_is_permutation']=np.array_equal(np.sort(p),np.arange(257))
    checks['negative_control_deterministic']=np.array_equal(p,derangement(257,29001))
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,synthetic_fits=3,
        synthetic_updates=240,scientific_fits=0,source_IQ_reads=0,torch=str(torch.__version__),numpy=np.__version__,
        device=torch.cuda.get_device_name(),results=results,
        code_sha256={n:digest(HERE/n) for n in ['w2_models.py','w2_engineering.py']})
    write('W2_ENGINEERING.json',result)
    if not all(checks.values()):raise ValueError('Synthetic acceptance failed')
    print('PASS',len(checks),'checks',flush=True)


if __name__=='__main__':main()
