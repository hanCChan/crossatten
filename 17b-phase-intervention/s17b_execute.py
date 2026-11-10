import os,sys,subprocess,traceback
from s17b_core import HERE,ROOT,read,write,sha,now,commit,validate,lock
from s17_jobs import run_id,profiles_path

def status(state,**kwargs):write(HERE/'STATUS.json',dict(time=now(),status=state,**kwargs))
def child(script,*args,log):
    if (HERE/'STOP_REQUESTED').exists():raise RuntimeError('STOP_REQUESTED at job boundary')
    env=dict(os.environ,PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as f:subprocess.run([sys.executable,'-X','utf8','-u',str(HERE/script),*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

def matrix(phase,evaluate=False):
    jobs=read(HERE/f'jobs_{phase}.json')
    for i,j in enumerate(jobs):
        out=HERE/'runs'/run_id(j)
        if (out/('held_metrics.json' if evaluate else 'metrics.json')).exists():continue
        if j.get('family'):
            key=f"{j['dataset']}_f{j['fold']}_{j['group']}_s{j.get('split_seed',j['seed'])}"
            p=HERE/'profiles'/f'{key}.json'
            if not p.exists():
                write(p,dict(profiles={},phase_sampler='Fixed bounded FIR; no learned compensation/profile parameters',tau=j['tau'],source_only=True))
                commit([p],'S17B freeze analytic intervention configuration')
        status('EVALUATING_SOURCE_HOLDOUT' if evaluate else 'TRAINING_SMALL_MHA',phase=phase,job=j,
            completed=sum((HERE/'runs'/run_id(k)/'metrics.json').exists() for k in jobs),registered=len(jobs))
        child('s17b_train.py','--phase',phase,'--index',str(i),*(['--evaluate'] if evaluate else []),log=out/('evaluation.log' if evaluate else 'stdout.log'))

def outcome(phase):
    jobs=read(HERE/f'jobs_{phase}.json');write(HERE/f'seal_{phase}.json',dict(time=now(),checkpoints={run_id(j):read(HERE/'runs'/run_id(j)/'metrics.json')['model_sha256'] for j in jobs}))
    commit([HERE/f'seal_{phase}.json'],f'S17B seal {phase} checkpoints before source-holdout evaluation')
    matrix(phase,True);child('s17b_reports.py',phase,log=HERE/'runs'/f'report_{phase}.log')
    commit([HERE/f'decision_{phase}.json',HERE/f'{phase.upper()}_RESULTS.zh-CN.md']+list(HERE.glob(f'{phase}_*.csv')),f'S17B freeze {phase} outcomes and gate')
    return read(HERE/f'decision_{phase}.json')

def finish():
    child('s17b_reports.py','final',log=HERE/'runs/final_report.log');r=read(HERE/'summary.json')
    status('COMPLETE',research_decision=r['status'],formal_fits=r['formal_fits'])
    commit([HERE/'summary.json',HERE/'RESULTS.zh-CN.md',HERE/'STATUS.json'],'S17B finalize bounded phase-mechanism follow-up')

def run():
    validate();matrix('pilot');status('INTERVENING_D1');child('s17b_intervene.py',log=HERE/'runs/interventions.log')
    child('s17b_reports.py','d1',log=HERE/'runs/report_d1.log');commit([HERE/'D1_RESULTS.zh-CN.md',HERE/'D1_interventions.csv'],'S17B report fixed phase probes and candidate decision')
    if read(HERE/'decision_d1.json')['selected_tau'] is None:finish();return
    child('s17b_register.py','pilot',log=HERE/'runs/registration.log');matrix('pilot');decision=outcome('pilot')
    for phase in ['confirm1','confirm2']:
        if not decision['admitted']:break
        child('s17b_register.py',phase,log=HERE/'runs/registration.log');matrix(phase);decision=outcome(phase)
    finish()

if __name__=='__main__':
    with lock():
        try:run()
        except Exception:
            write(HERE/'execution_failure.json',dict(time=now(),traceback=traceback.format_exc(),automatic_retry=False));status('STOP_TECHNICAL_REVIEW');raise
