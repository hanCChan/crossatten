"""Wait for the existing S13 worker; never stop it, restart it, or compete for GPU."""
import argparse
import os
import sys
import subprocess
import time
import traceback
from s14_core import HERE,ROOT,LIVE,S13,read,write,sha,now,git,committed,commit,validate,job_id,lock,process_alive

def child(script,*args,log=None):
    env=dict(os.environ,PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    cmd=[sys.executable,'-X','utf8','-u',str(HERE/script),*args]
    if log:
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a',encoding='utf-8') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    else:subprocess.run(cmd,cwd=ROOT,env=env,check=True)

def readiness():
    p=S13/'progress.json'
    if not p.exists():return False,dict(reason='S13 progress missing')
    status=read(p);pid=int((S13/'execution.pid').read_text().strip())
    alive=process_alive(pid)
    return status['status']=='COMPLETE_BOUNDED_SOURCE_WORKFLOW' and not alive,dict(s13_status=status['status'],s13_pid=pid,s13_alive=alive,phases=status['phases'])

def check_stop():
    if (HERE/'STOP_REQUESTED').exists():raise RuntimeError('S14 STOP_REQUESTED; S13 is untouched')

def evidence_committed(path):
    try:committed(path,LIVE);return True
    except (ValueError,subprocess.CalledProcessError):return False

def execute():
    validate('design.json')
    while True:
        check_stop();ready,detail=readiness()
        write(HERE/'STATUS.json',dict(time=now(),status='WAITING_FOR_S13' if not ready else 'S13_COMPLETE',**detail))
        if (S13/'execution_failure.json').exists():raise RuntimeError('S13 has a recorded technical error; review it, never restart automatically')
        # CPU-only report, only after all48 sealed target evaluations exist and are committed.
        if (S13/'p1_summary.json').exists() and not (HERE/'attention_comparison_s13.json').exists() and evidence_committed(S13/'p1_summary.json'):
            child('s14_reports.py','s13',log=HERE/'runs/attention_report.log')
            commit([HERE/n for n in ['ATTENTION_COMPARISON_S13.md','ATTENTION_PAIRED_DELTAS.png','attention_comparison_s13.json','attention_paired_receiver_deltas.csv']],
                'S14 review all48 frozen S13 HSTU versus ordinary attention results')
        if ready:break
        time.sleep(30)
    # Import the final S13 commits into this *isolated* branch, never switching the live tree.
    if git('branch','--show-current',cwd=LIVE)!='exp/s13-mixer-cfo':raise ValueError('Unexpected live S13 branch')
    subprocess.run(['git','-C',str(ROOT),'-c','user.name=Codex','-c','user.email=codex@local.invalid','merge','--no-edit','exp/s13-mixer-cfo'],check=True)
    validate('design.json')
    from s14_register import finalize
    reg=finalize();jobs=reg['jobs']
    if not (HERE/'costs.json').exists():child('s14_reports.py','cost',log=HERE/'runs/costs.log')
    for evaluate in [False,True]:
        for index,job in enumerate(jobs):
            check_stop();out=HERE/'runs'/job_id(job);result=out/('target_metrics.json' if evaluate else 'metrics.json')
            if result.exists():continue
            write(HERE/'STATUS.json',dict(time=now(),status='EVALUATING' if evaluate else 'TRAINING_SOURCE',job=job,
                fits_complete=sum((HERE/'runs'/job_id(j)/'metrics.json').exists() for j in jobs),fits_registered=len(jobs)))
            args=['--index',str(index)]+(['--evaluate'] if evaluate else [])
            print(now(),'EVAL' if evaluate else 'FIT',job_id(job),flush=True)
            child('s14_train.py',*args,log=out/('evaluation.log' if evaluate else 'stdout.log'))
        if not evaluate and not (HERE/'decision_p1_source.json').exists():
            checks={job_id(j):read(HERE/'runs'/job_id(j)/'metrics.json')['model_sha256'] for j in jobs}
            write(HERE/'decision_p1_source.json',dict(time=now(),checkpoints=checks,rule='All new source-only selected weights fixed before target evaluation; every registered arm reported'))
            commit([HERE/'decision_p1_source.json'],'S14 seal every source-selected checkpoint before receiver evaluation')
    child('s14_reports.py','factorial',log=HERE/'runs/reports.log');child('s14_reports.py','mechanism',log=HERE/'runs/reports.log')
    if reg['cfo_models']:child('s14_cfo.py',log=HERE/'runs/cfo.log')
    else:write(HERE/'cfo_classifier_summary.json',dict(status='NOT_RUN_NO_S13_CFO_MODELS',new_fits=0))
    child('s14_reports.py','conclude',log=HERE/'runs/reports.log')
    write(HERE/'STATUS.json',dict(time=now(),status='COMPLETE',source_fits=len(jobs),tta='FEASIBILITY_ONLY_NO_ADAPTATION_EXECUTED'))
    files=[p for p in HERE.iterdir() if p.is_file() and p.suffix in ['.json','.md','.csv','.png']]
    commit(files,'S14 finalize paired stem/mixer evidence, source CFO diagnostics and contribution limits')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');a=ap.parse_args()
    if a.check:
        ready,detail=readiness();print(dict(ready=ready,**detail));sys.exit(0)
    with lock():
        try:execute()
        except Exception:
            write(HERE/'execution_failure.json',dict(time=now(),traceback=traceback.format_exc(),automatic_retry=False))
            write(HERE/'STATUS.json',dict(time=now(),status='STOP_TECHNICAL_REVIEW_REQUIRED'));raise
