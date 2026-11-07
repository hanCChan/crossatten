"""Exclusive, resumable Windows/Linux single-GPU workflow; never retries attempts."""
import argparse
import os
import subprocess
import sys
import traceback
from core import HERE, ROOT, read, write, once, now, sha, exclusive, commit, commit_changed, validate_registration
from train import job_id
from reports import progress


def child(*args,log=None):
    env=dict(os.environ,PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    cmd=[sys.executable,'-X','utf8','-u',str(HERE/args[0]),*args[1:]]
    if log:
        with open(log,'a',encoding='utf-8') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    else:subprocess.run(cmd,cwd=ROOT,env=env,check=True)


def matrix(phase,evaluate=False):
    jobs=read(HERE/f'jobs_{phase}.json')
    for i,j in enumerate(jobs):
        out=HERE/'runs'/job_id(j);out.mkdir(parents=True,exist_ok=True)
        destination=out/('target_metrics.json' if evaluate else 'metrics.json')
        if destination.exists():continue
        if (HERE/'STOP_REQUESTED').exists():raise RuntimeError('STOP_REQUESTED: no further jobs dispatched')
        print(now(), 'EVAL' if evaluate else 'FIT',job_id(j),flush=True)
        args=['train.py','--matrix',f'jobs_{phase}.json','--index',str(i)]
        if evaluate:args.append('--evaluate')
        child(*args,log=out/('evaluation.stdout.log' if evaluate else 'stdout.log'))
        progress()


def register_p2():
    if (HERE/'registration_p2.json').exists():return
    reg=validate_registration();audit=read(HERE/'cfo_audit.json')
    value=dict(time=now(),diagnostic_decision_sha256=sha(HERE/'decision_p0.json'),
        bounds_hz={d:{f:v['prospective_randomization_bound_hz'] for f,v in folds.items()} for d,folds in audit['folds'].items()},
        bound_rule='q95 absolute source RX/day median CFO minus pooled median, clipped500..10000Hz; not pure RX CFO',
        variants=reg['p2']['variants'],pilot_cap=40,confirmation_cap=64,
        paired_controls='B1 vs B0; B2_dual must pass vs both B0 and parameter-identical B2_dual_raw',
        target_access=False,epochs_cap=40,batch=128)
    once(HERE/'registration_p2.json',value)
    jobs=[dict(phase='p2',dataset=d,fold=f,seed=16001,group='first',arm=a) for d in ['ManyRx','ManySig'] for f in range(4) for a in value['variants']]
    once(HERE/'jobs_p2.json',jobs)
    commit([HERE/'registration_p2.json',HERE/'jobs_p2.json'],'S13 preregister source-derived CFO pilot after P0 diagnostics')


def register_confirmation(selected):
    p=HERE/'jobs_p2_confirmation.json'
    if p.exists():return
    arms=['B0']+selected+(['B2_dual_raw'] if 'B2_dual' in selected else [])
    jobs=[dict(phase='p2_confirmation',dataset=d,fold=f,seed=s,group='last',arm=a)
          for d in ['ManyRx','ManySig'] for f in range(4) for s in [16011,16012] for a in arms]
    if len(jobs)>64:raise ValueError('Confirmation cap exceeded')
    once(p,jobs);commit([p,HERE/'decision_p2.json',HERE/'P2_CFO_RESULTS.md'],'S13 freeze independent paired source confirmation after CFO pilot gate')


def workflow():
    validate_registration();progress('RUNNING')
    if not (HERE/'costs.json').exists():child('reports.py','cost')
    matrix('smoke')
    if not (HERE/'cfo_audit.json').exists():child('diagnostics.py','cfo')
    matrix('p0')
    if not (HERE/'decision_p0.json').exists():child('diagnostics.py','report')
    commit_changed([HERE/p for p in ['decision_p0.json','P0_MECHANISM_REPORT.md','P0_LAYER_PROBES.png','cfo_audit.json','layer_probes.csv','source_cfo_packets.csv','p0_diagnostic_summary.json','costs.json','costs.csv']],
           'S13 freeze source-only WiSig mechanism and CFO estimator evidence')
    matrix('p1')
    sealed=HERE/'decision_p1_source.json'
    if not sealed.exists():
        checks={job_id(j):read(HERE/'runs'/job_id(j)/'metrics.json')['model_sha256'] for j in read(HERE/'jobs_p1.json')}
        once(sealed,dict(time=now(),checkpoints=checks,rule='source-CE selection fixed; all8arms reported; no target-based selection',target_access_before_seal=False))
        commit([sealed],'S13 freeze all48 source-selected LoRa mixer checkpoints before receiver evaluation')
    matrix('p1',evaluate=True)
    if not (HERE/'p1_summary.json').exists():child('reports.py','p1')
    commit_changed([HERE/p for p in ['P1_MIXER_RESULTS.md','p1_summary.json','p1_fit_metrics.csv','p1_receiver_metrics.csv','p1_paired_comparisons.csv']],
           'S13 report complete paired LoRa mixer comparisons')
    if read(HERE/'decision_p0.json')['cfo_pilot_allowed']:
        register_p2();matrix('p2')
        if not (HERE/'decision_p2.json').exists():child('reports.py','p2')
        selected=read(HERE/'decision_p2.json')['selected']
        if selected:
            register_confirmation(selected);matrix('p2_confirmation')
            if not (HERE/'decision_p2_confirmation.json').exists():child('reports.py','p2_confirmation')
        else:
            once(HERE/'jobs_p2_confirmation.json',[])
    progress('COMPLETE_BOUNDED_SOURCE_WORKFLOW')
    final=[p for p in HERE.iterdir() if p.is_file() and p.suffix in ['.md','.json','.csv']]
    commit_changed(final,'S13 finalize bounded source workflow, stops and complete mixer evidence')


if __name__=='__main__':
    with exclusive('gpu0_workflow'):
        try:workflow()
        except Exception:
            write(HERE/'execution_failure.json',dict(time=now(),traceback=traceback.format_exc(),automatic_retry=False))
            progress('STOP_EXECUTION_ERROR');raise
