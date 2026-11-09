import os,sys,subprocess,traceback
from s15_core import HERE,ROOT,read,write,sha,now,commit,validate,job_id,lock

def child(script,*args,log):
    env=dict(os.environ,PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as f:subprocess.run([sys.executable,'-X','utf8','-u',str(HERE/script),*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def run():
    reg=validate();jobs=reg['jobs']
    if not (HERE/'costs.json').exists():child('s15_reports.py','cost',log=HERE/'runs/cost.log')
    for evaluate in [False,True]:
        for i,j in enumerate(jobs):
            if (HERE/'STOP_REQUESTED').exists():raise RuntimeError('STOP_REQUESTED at job boundary')
            out=HERE/'runs'/job_id(j)
            if (out/('target_metrics.json' if evaluate else 'metrics.json')).exists():continue
            write(HERE/'STATUS.json',dict(time=now(),status='EVALUATING' if evaluate else 'TRAINING_SOURCE',job=j,completed=sum((HERE/'runs'/job_id(k)/'metrics.json').exists() for k in jobs),registered=12))
            print(now(),'EVAL' if evaluate else 'FIT',job_id(j),flush=True)
            child('s15_train.py','--index',str(i),*(['--evaluate'] if evaluate else []),log=out/('evaluation.log' if evaluate else 'stdout.log'))
        if not evaluate and not (HERE/'decision_p1_source.json').exists():
            write(HERE/'decision_p1_source.json',dict(time=now(),checkpoints={job_id(j):read(HERE/'runs'/job_id(j)/'metrics.json')['model_sha256'] for j in jobs},rule='All12 source-CE checkpoints sealed before target reads'))
            commit([HERE/'decision_p1_source.json'],'S15 seal all12 matched-parameter source checkpoints before receiver evaluation')
    child('s15_reports.py','report',log=HERE/'runs/report.log')
    write(HERE/'STATUS.json',dict(time=now(),status='COMPLETE',formal_fits=12,evaluated=12))
    commit([p for p in HERE.iterdir() if p.is_file() and p.suffix in ['.json','.md','.csv','.png']],'S15 finalize matched-parameter attribution and all paired outcomes')
if __name__=='__main__':
    with lock():
        try:run()
        except Exception:
            write(HERE/'execution_failure.json',dict(time=now(),traceback=traceback.format_exc(),automatic_retry=False));write(HERE/'STATUS.json',dict(time=now(),status='STOP_TECHNICAL_REVIEW'));raise
