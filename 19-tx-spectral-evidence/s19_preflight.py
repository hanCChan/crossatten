"""Synthetic GPU and metadata-only initialization checks; never read weights/IQ."""
import platform
import sys
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from s19_core import HERE,read,write,now
from s19_data import locations
from s19_models import build_model
from s19_train import shared_hash


def run():
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    rows=[]
    for fold in range(4):
        cfg=read(locations(fold)[1]/'config.json')
        for arm in ['B0','B1','B2','B3','B4']:
            m=build_model(arm,'ManyRx',cfg['data_seed']).cuda()
            if shared_hash(m)!=cfg['initial_state_sha256']:raise ValueError('Initial state mismatch')
            logits=m(torch.randn(4,2,256,device='cuda'))
            loss=logits.square().mean()
            if arm=='B4':loss=loss+.1*F.log_softmax(m.last_evidence_logits,1).square().mean()
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in m.parameters()):raise ValueError('Nonfinite synthetic gradient')
            rows.append(dict(fold=fold,arm=arm,parameters=sum(p.numel() for p in m.parameters()),backbone_initial_sha256=cfg['initial_state_sha256']))
            del m
    write(HERE/'ENGINEERING_TESTS.json',dict(time=now(),status='PASS',cpu_synthetic_tests=19,gpu_random_model_checks=20,
          historical_initial_state_checks=20,source_weights_read=False,source_iq_read=False,optimizer_steps=0,
          python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,
          gpu=torch.cuda.get_device_name(),checks=rows))
    print('PASS 19 CPU tests and20 synthetic CUDA/init checks; no weights or IQ read',flush=True)


if __name__=='__main__':run()
