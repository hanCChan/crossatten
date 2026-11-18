import argparse
from common import PHASE,policy,authorized_training,jobid,atomic_json,digest,HERE
from runner import fit

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);args=parser.parse_args()
    a,r=policy();job=next(j for j in r['jobs'] if jobid(j)==args.job)
    authorized_training(job)
    atomic_json(PHASE/'runs'/args.job/'budget_authorization.json',dict(status='AUTHORIZED_RETAINED_JOB',
                job=args.job,amendment_sha256=digest(HERE/'AMENDMENT.json'),original_training_configuration_unchanged=True))
    fit(job)
