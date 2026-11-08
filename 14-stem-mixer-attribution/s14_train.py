"""Explicit adapter around the byte-frozen S13 training/evaluation functions."""
import argparse
from s14_core import HERE,read,validate,write,job_id,sha,S13,s13_id
import s14_data
from s14_models import build,BASE,common_hash,costs
from xa.train import stable_seed
import train as frozen_s13_train

def configure():
    # No optimizer, RNG, forward/loss, step order or checkpoint logic is replaced.
    frozen_s13_train.HERE=HERE
    frozen_s13_train.validate_registration=validate
    frozen_s13_train.build=build
    frozen_s13_train.shen_train=s14_data.shen_train
    frozen_s13_train.shen_cell=s14_data.shen_cell
    frozen_s13_train.shen_roles=s14_data.roles
    frozen_s13_train.PROTOCOL=s14_data.PROTOCOL

def fit(job):
    reg=validate();seed=stable_seed('P-H5x160',job['fold'],job['seed'])
    baseline=BASE[job['arm']];ref=S13/'runs'/s13_id(baseline,job['fold'],job['seed'])
    cfg=read(ref/'config.json');assert cfg['data_seed']==seed and cfg['epochs_budget']==200 and cfg['batch']==64
    from models import build as build_s13
    a,b=build(job['arm'],seed),build_s13(baseline,seed)
    if common_hash(a)!=common_hash(b) and job['arm'].startswith('E'):raise ValueError('Non-tokenizer initialization is not paired')
    out=HERE/'runs'/job_id(job)
    if (out/'attempt.json').exists() and not (out/'metrics.json').exists():raise RuntimeError('Failed/interrupted fit retained; no retry')
    if not (out/'pairing.json').exists():
        write(out/'pairing.json',dict(baseline=baseline,baseline_config_sha256=sha(ref/'config.json'),
            baseline_model_sha256=reg['reused'][s13_id(baseline,job['fold'],job['seed'])]['model_sha256'],
            shared_initialization_sha256=common_hash(a),**costs(a,job['arm'])))
    del a,b
    configure();return frozen_s13_train.fit(job)

def evaluate(job):configure();return frozen_s13_train.evaluate_shen(job)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--evaluate',action='store_true');args=ap.parse_args()
    job=read(HERE/'jobs_p1.json')[args.index]
    evaluate(job) if args.evaluate else fit(job)
