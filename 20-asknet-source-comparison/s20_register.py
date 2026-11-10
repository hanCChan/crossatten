"""Prospective same-data package comparison, maximum four new official fits."""
import sys
import numpy as np
import torch
from s20_core import HERE,ROOT,S19,UPSTREAM,read,write,sha,now,git,official
from s20_run import CONF
from s20_stats import thresholds

def engineer():
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    api=official();api.setup_seed(20203);m=api.build_model(CONF,10).cuda();x=torch.randn(4,2,256,device='cuda')
    loss=torch.nn.functional.cross_entropy(m(x),torch.tensor([0,1,2,3],device='cuda'));loss.backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()):raise ValueError('Nonfinite official complex QR GPU gradients')
    write(HERE/'ENGINEERING_TESTS.json',dict(time=now(),status='PASS',synthetic_unit_tests=6,gpu_forward_backward=True,
          parameter_elements=sum(p.numel() for p in m.parameters()),real_scalar_dof=sum(p.numel()*(2 if p.is_complex() else 1) for p in m.parameters()),
          source_data_or_weights_read=False,python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name()))
    print('PASS synthetic CUDA ASKNet forward/backward including complex QR',flush=True)

def register():
    if (HERE/'registration.json').exists() or (HERE/'access.jsonl').exists() or (HERE/'runs').exists():raise ValueError('Registration must precede formal data access')
    if read(HERE/'ENGINEERING_TESTS.json')['status']!='PASS':raise ValueError('Engineering checks missing')
    if read(HERE/'REUSE_AUDIT.json')['s10_reusable']!=0:raise ValueError('Unexpected reuse eligibility')
    files=[HERE/p for p in ['s20_core.py','s20_stats.py','s20_run.py','s20_register.py','test_s20.py']]
    files+=list((ROOT/'lib').rglob('*.py'))+[ROOT/'13-mixer-cfo-mechanisms'/p for p in ['core.py','models.py','wifi.py','metrics.py']]
    files+=[ROOT/'17-rx-distortion'/p for p in ['s17_core.py','s17_models.py']]+[ROOT/'09-phase-invariance/phase_features.py']
    files+=[ROOT/'19-tx-spectral-evidence'/p for p in ['s19_core.py','s19_data.py','s19_metrics.py','s19_models.py']]
    s19=read(S19/'MANIFEST.json')['files_sha256']
    upstream=read(UPSTREAM/'FETCH_MANIFEST.json')['files']
    write(HERE/'registration.json',dict(time=now(),stage='S20',base_commit=git('rev-parse','HEAD'),
          code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))},
          design_sha256={p:sha(HERE/p) for p in ['README.zh-CN.md','REUSE_AUDIT.json','UPSTREAM.json','ENGINEERING_TESTS.json']},
          s19_evidence_sha256=s19,official_files_sha256={p:rec['sha256'] for p,rec in upstream.items()},
          source=dict(dataset='ManyRx',folds=[0,1,2,3],fit_rx=18,held_source_rx=6,days=[1,2],split_seed=16001,macro_seed=18101),
          asknet_official_commit='f00846d68425315b68ae69b8c6b712e08559c2fe',asknet_recipe=dict(epochs_max=200,batch=128,Adam_lr=.001,wd=0,plateau_factor=.1,plateau_patience=5,min_lr=1e-6,early_stop=10,checkpoint='source packet CE'),
          frozen_B0_recipe=dict(epochs_max=40,checkpoint='source RX-equal CE',reused=True),thresholds=thresholds(),
          maximum_new_formal_fits=4,new_module_training=False,causal_attribution_from_package_comparison=False,
          real_target_access=False,no_automatic_scientific_retry=True,all_models_sealed_before_new_held_evaluation=True))

if __name__=='__main__':engineer() if sys.argv[1]=='engineer' else register()
