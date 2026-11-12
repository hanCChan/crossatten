"""One locked worker; twelve fits then one seal, evaluation and a fixed stop."""
import os
import subprocess
import sys
import traceback
from s21_core import HERE, ROOT, read, write, sha, now, job_id, validate, commit
from s15_core import lock as old_lock
import s15_core

def child(script,*args,log):
    env=dict(os.environ,PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as f:
        subprocess.run([sys.executable,'-X','utf8','-u',str(HERE/script),*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

def run():
    reg=validate();jobs=reg['jobs']
    if not (HERE/'BASELINE_SOURCE_RECOVERY.json').exists():
        child('s21_evaluate.py','--restore-source',log=HERE/'baseline_recovery.log')
        commit([HERE/'BASELINE_SOURCE_RECOVERY.json'],'S21 verify exact source recovery of twelve historical controls')
    assert read(HERE/'BASELINE_SOURCE_RECOVERY.json')['status']=='PASS'
    if not (HERE/'costs.json').exists():child('s21_report.py','cost',log=HERE/'cost.log')
    for i,j in enumerate(jobs):
        if (HERE/'STOP_REQUESTED').exists():raise RuntimeError('User stop at fit boundary')
        out=HERE/'runs'/job_id(j)
        if (out/'metrics.json').exists():continue
        write(HERE/'STATUS.json',dict(time=now(),status='TRAINING_SOURCE',job=j,completed=sum((HERE/'runs'/job_id(k)/'metrics.json').exists() for k in jobs),registered=12))
        print(now(),'FIT',job_id(j),flush=True)
        child('s21_train.py','--index',str(i),log=out/'stdout.log')
    seal=HERE/'decision_p1_source.json'
    if not seal.exists():
        checkpoints={}
        for j in jobs:
            out=HERE/'runs'/job_id(j);m=read(out/'metrics.json')
            assert m['epochs']==200 and m['model_sha256']==sha(out/'model.pt')
            checkpoints[out.name]=m['model_sha256']
        write(seal,dict(time=now(),checkpoints=checkpoints,rule='All12 minimum source-CE checkpoints sealed before target reads'))
        commit([seal],'S21 seal all twelve source-selected checkpoints before receiver evaluation')
    for arm in ['H0','P0','H1','H2']:
        for fold in [0,1]:
            for seed in [201,202,203]:
                write(HERE/'STATUS.json',dict(time=now(),status='EVALUATING_SEALED',arm=arm,fold=fold,seed=seed,completed=12,registered=12))
                child('s21_evaluate.py','--arm',arm,'--fold',str(fold),'--seed',str(seed),log=HERE/'evaluation.log')
    child('s21_report.py','report',log=HERE/'report.log')
    child('s21_report.py','audit',log=HERE/'audit.log')
    write(HERE/'STATUS.json',dict(time=now(),status='COMPLETE_STOP',formal_fits=12,baseline_fits_reused=12,H3_executed=False))
    commit([p for p in HERE.iterdir() if p.is_file() and p.suffix in ['.json','.md','.csv']]+[HERE/'evidence'],'S21 finalize bounded HStem overlap and normalization mechanism evidence')

if __name__=='__main__':
    # Redirect the historical cross-platform lock helper to this stage only.
    s15_core.HERE=HERE
    with old_lock():
        try:run()
        except Exception:
            write(HERE/'execution_failure.json',dict(time=now(),traceback=traceback.format_exc(),automatic_retry=False))
            write(HERE/'STATUS.json',dict(time=now(),status='STOP_TECHNICAL_REVIEW'));raise
