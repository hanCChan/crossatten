"""Failure-path and exact recovery tests, sandbox technical steps only."""
import sys,tempfile,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from s24_common import P,write_json
from data import load_cell,preprocess
from models import build,forward
from execution import atomic_torch,setup

def main():
    setup(2402);torch.set_num_threads(4);checks=[]
    for rx in P['blind_receivers_never_open']+P['confirm_receivers']+P['folds'][0]['dev_val']:
        with patch('numpy.load',side_effect=AssertionError('IO happened before rejection')):
            try:load_cell(rx,0,'source','engineering')
            except PermissionError:pass
            else:raise AssertionError(rx)
    checks.append('blind/confirm/wrong-fold source rejected before waveform IO')
    try:load_cell('rtl_2',0,'dev','engineering',seal=None)
    except PermissionError:pass
    else:raise AssertionError('Dev opened without seal')
    checks.append('Dev denied without global seal')
    setup(2402);ce=build('ce').cuda();setup(2402);dann=build('dann').cuda()
    assert all(torch.equal(v,dann.state_dict()[k]) for k,v in ce.state_dict().items())
    assert sum(p.numel() for p in ce.parameters())==13442954
    x=torch.randn(2,2,8192,device='cuda')
    a=forward(ce,'ce',x)[0];b=forward(dann,'dann',x)[0]
    assert torch.equal(a,b)
    checks.append('CE and DANN share exact encoder/TX initialization and forward')
    # Recover optimizer + RNG and compare the next update, without any dataset fitting.
    with tempfile.TemporaryDirectory(dir=HERE/'local') as directory:
        torch.manual_seed(41);m=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.3),torch.nn.Linear(4,2))
        opt=torch.optim.SGD(m.parameters(),lr=.01,momentum=.9)
        inputs=torch.randn(4,3);target=torch.tensor([0,1,0,1])
        def step():
            opt.zero_grad();loss=F.cross_entropy(m(inputs),target);loss.backward();opt.step();return m.state_dict()
        step();p=Path(directory)/'checkpoint.pt'
        atomic_torch(p,dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()))
        expected={k:v.clone() for k,v in step().items()}
        saved=torch.load(p,weights_only=False);m.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
        assert all(torch.equal(v,expected[k]) for k,v in step().items())
    checks.append('atomic model+momentum+dropout RNG restore gives bit-identical next update')
    # Exact source projection budget, after arbitrary source AWGN, then method normalization.
    z=preprocess(x,'strict_ib','fan');s=torch.fft.fft(torch.complex(z[:,0],z[:,1]))
    mask=torch.fft.fftfreq(8192,d=1e-6,device=x.device).abs()>62500
    ratio=float(s[:,mask].abs().square().sum()/s.abs().square().sum())
    assert ratio<1e-12
    checks.append('strict pipeline retains only common input information budget')
    write_json(HERE/'ENGINEERING_TESTS.json',dict(status='PASS',checks=checks,scientific_fits=0,
                 sandbox_tiny_optimizer_steps=3,strict_projection_residual_ratio=ratio))
    print('PASS',len(checks),'permission, shared-control and recovery tests.')

if __name__=='__main__':main()
