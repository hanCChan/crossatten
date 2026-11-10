"""Redirect the frozen S17 small-model loop; no changes to historical files."""
import argparse
from s17b_core import HERE,read,validate,committed,access
import s17_data as data
import s17_jobs as jobs
import s17_train as engine
from s17b_augment import apply

def configure(job=None):
    data.HERE=HERE;data.access=access;jobs.HERE=HERE
    engine.HERE=HERE;engine.validate=validate;engine.committed=committed
    if job is not None:
        mode=job.get('control','physical')
        def transform(x,family,tau,profiles,generator,noise_generator=None,control=False,probability=.5):
            if family!='phase':raise ValueError('S17B only permits phase family')
            return apply(x,tau,generator,noise_generator,mode,probability)
        engine.perturb=transform
    return data,engine

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--index',required=True,type=int);p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    job=read(HERE/f'jobs_{a.phase}.json')[a.index];_,runner=configure(job)
    runner.evaluate(job) if a.evaluate else runner.fit(job)
