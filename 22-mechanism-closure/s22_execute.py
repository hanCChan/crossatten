"""Persistent locked worker; predefined fits, publication artifacts, fixed stop."""
import os
import subprocess
import sys
import traceback
from s22_core import HERE,ROOT,read,write,sha,now,job_id,validate,commit
import s15_core
from s15_core import lock
from s22_models import ARMS

def child(script,*args,log):
    env=dict(os.environ,PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as f:subprocess.run([sys.executable,'-X','utf8','-u',str(HERE/script),*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def status(phase,**data):write(HERE/'STATUS.json',dict(time=now(),status=phase,**data))

def run():
    reg=validate();jobs=reg['jobs']
    if not (HERE/'BASELINE_SOURCE_RECOVERY.json').exists():
        status('RECOVERING24_SOURCE_CONTROLS',completed=0,registered=12)
        child('s22_evaluate.py','--restore-source',log=HERE/'baseline_recovery.log')
        assert len(read(HERE/'BASELINE_SOURCE_RECOVERY.json')['models'])==24
        commit([HERE/'BASELINE_SOURCE_RECOVERY.json'],'S22 recover24 historical controls on source validation before training')
    if not (HERE/'costs.json').exists():
        status('BENCHMARKING',completed=0,registered=12);child('s22_cost.py',log=HERE/'cost.log')
    for i,j in enumerate(jobs):
        if (HERE/'STOP_REQUESTED').exists():raise RuntimeError('Stop requested at scientific-fit boundary')
        out=HERE/'runs'/job_id(j)
        if (out/'metrics.json').exists():continue
        status('TRAINING_SOURCE',completed=sum((HERE/'runs'/job_id(k)/'metrics.json').exists() for k in jobs),registered=12,job=j)
        print(now(),'FIT',job_id(j),flush=True);child('s22_train.py','--index',str(i),log=out/'stdout.log')
    if not (HERE/'decision_p1_source.json').exists():
        checkpoints={}
        for j in jobs:
            out=HERE/'runs'/job_id(j);m=read(out/'metrics.json');assert m['epochs']==200 and m['model_sha256']==sha(out/'model.pt');checkpoints[out.name]=m['model_sha256']
        write(HERE/'decision_p1_source.json',dict(time=now(),checkpoints=checkpoints,rule='All12 source-CE-selected checkpoints sealed before any target RX read'))
        commit([HERE/'decision_p1_source.json'],'S22 seal all12 selected source checkpoints before target evaluation')
    for arm in ARMS:
        for fold in [0,1]:
            for seed in [201,202,203]:
                status('EVALUATING_SEALED',completed=12,registered=12,arm=arm,fold=fold,seed=seed)
                child('s22_evaluate.py','--arm',arm,'--fold',str(fold),'--seed',str(seed),log=HERE/'evaluation.log')
    status('REPORTING',completed=12,registered=12)
    child('s22_report.py',log=HERE/'report.log')
    if not (HERE/'ROBUSTNESS.json').exists():child('s22_robustness.py',log=HERE/'robustness.log')
    child('s22_paper.py',log=HERE/'paper.log');child('s22_audit.py',log=HERE/'audit.log')
    status('COMPLETE_STOP',formal_fits=12,reused_fits=24,models_evaluated=36,blind_evaluated=False)
    commit([p for p in HERE.iterdir() if p.is_file() and p.suffix in ['.json','.md','.csv']]+[HERE/'evidence',HERE/'paper'],'S22 finalize mechanism closure, source stress tests and paper evidence; stop at12 fits')

if __name__=='__main__':
    s15_core.HERE=HERE
    with lock():
        try:run()
        except Exception:
            write(HERE/'execution_failure.json',dict(time=now(),traceback=traceback.format_exc(),automatic_retry=False));status('STOP_TECHNICAL_REVIEW');raise
