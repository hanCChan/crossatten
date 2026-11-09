"""Reuse the byte-frozen S13 optimization loop; only the tokenizer changes."""
import argparse
from s15_core import HERE,S13,read,write,sha,validate,job_id,s13_id
from s15_models import build,BASE,shared_hash
import s15_data
from xa.train import stable_seed
import train as original_train

def configure():
    original_train.HERE=HERE;original_train.validate_registration=validate;original_train.build=build
    original_train.shen_train=s15_data.shen_train;original_train.shen_cell=s15_data.shen_cell
    original_train.shen_roles=s15_data.roles;original_train.PROTOCOL=s15_data.PROTOCOL

def fit(job):
    reg=validate();seed=stable_seed('P-H5x160',job['fold'],job['seed']);base=BASE[job['arm']]
    ref=S13/'runs'/s13_id(base,job['fold'],job['seed']);cfg=read(ref/'config.json')
    assert cfg['batch']==64 and cfg['epochs_budget']==200 and cfg['data_seed']==seed
    from models import build as original
    a,b=build(job['arm'],seed),original(base,seed)
    if shared_hash(a)!=shared_hash(b):raise ValueError('Shared initialization is not paired')
    out=HERE/'runs'/job_id(job)
    if not (out/'pairing.json').exists():write(out/'pairing.json',dict(baseline=base,initial_shared_sha256=shared_hash(a),source_config_sha256=sha(ref/'config.json')))
    del a,b;configure();return original_train.fit(job)

def evaluate(job):configure();return original_train.evaluate_shen(job)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    j=read(HERE/'jobs_p1.json')[a.index];evaluate(j) if a.evaluate else fit(j)
