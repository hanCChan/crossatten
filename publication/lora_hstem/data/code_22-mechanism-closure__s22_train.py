"""Use the unchanged S13 Shen200 training loop with isolated S22 output."""
import argparse
import torch
from s22_core import HERE, read, write, sha, validate, baseline_path
from s22_models import build, shared_hash
import s22_data
import train as previous
from xa.train import stable_seed

def configure():
    previous.HERE=HERE; previous.validate_registration=validate; previous.build=build
    previous.shen_train=s22_data.shen_train; previous.shen_cell=s22_data.shen_cell
    previous.shen_roles=s22_data.roles; previous.PROTOCOL=s22_data.PROTOCOL

def fit(job):
    validate(); seed=stable_seed('P-H5x160',job['fold'],job['seed'])
    reference=baseline_path('P0' if job['arm']=='S1' else 'H0',job['fold'],job['seed'])
    cfg=read(reference/'config.json')
    assert cfg['data_seed']==seed and cfg['batch']==64 and cfg['epochs_budget']==200
    a,b=build(job['arm'],seed),build('P0' if job['arm']=='S1' else 'H0',seed)
    assert shared_hash(a)==shared_hash(b)
    for key,v in a.state_dict().items(): assert torch.equal(v,b.state_dict()[key]), key
    from s22_core import job_id
    write(HERE/'runs'/job_id(job)/'pairing.json',dict(baseline=str(reference),baseline_config_sha256=sha(reference/'config.json'),initial_shared_sha256=shared_hash(a),same_surviving_initial_state=True))
    del a,b; configure(); return previous.fit(job)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--index',type=int,required=True)
    args=parser.parse_args(); fit(read(HERE/'jobs_p1.json')[args.index])
