"""Bounded D0 -> D1 -> small Pilot -> independent source confirmation."""
import os,sys,subprocess,traceback
from s17_core import HERE,ROOT,read,write,sha,now,commit,validate,lock

def child(script,*args,log):
    if (HERE/'STOP_REQUESTED').exists():raise RuntimeError('STOP_REQUESTED at job boundary')
    env=dict(os.environ,PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as f:subprocess.run([sys.executable,'-X','utf8','-u',str(HERE/script),*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def status(state,**kwargs):write(HERE/'STATUS.json',dict(time=now(),status=state,**kwargs))

def matrix(phase,evaluate=False):
    from s17_jobs import run_id,profiles_path
    from s17_diagnostics import profile_only
    jobs=read(HERE/f'jobs_{phase}.json')
    for i,j in enumerate(jobs):
        out=HERE/'runs'/run_id(j)
        if (out/('held_metrics.json' if evaluate else 'metrics.json')).exists():continue
        if j.get('family') and not profiles_path(j).exists():
            profile_only(j);commit([profiles_path(j)],'S17 freeze new confirmation source-only nuisance profiles')
        status('EVALUATING_SOURCE_HOLDOUT' if evaluate else 'TRAINING_SOURCE_BASELINE' if j['arm']=='M0' else 'TRAINING_SOURCE_PILOT',
            phase=phase,job=j,completed=sum((HERE/'runs'/run_id(k)/'metrics.json').exists() for k in jobs),registered=len(jobs))
        child('s17_train.py','--phase',phase,'--index',str(i),*(['--evaluate'] if evaluate else []),log=out/('evaluation.log' if evaluate else 'stdout.log'))

def outcome(phase):
    from s17_jobs import run_id
    jobs=read(HERE/f'jobs_{phase}.json')
    write(HERE/f'seal_{phase}.json',dict(time=now(),checkpoints={run_id(j):read(HERE/'runs'/run_id(j)/'metrics.json')['model_sha256'] for j in jobs}))
    commit([HERE/f'seal_{phase}.json'],f'S17 seal all {phase} source checkpoints before held-source evaluation')
    matrix(phase,True);child('s17_reports.py',phase,log=HERE/'runs'/f'report_{phase}.log')
    commit([HERE/f'decision_{phase}.json',HERE/f'{phase.upper()}_RESULTS.zh-CN.md']+list(HERE.glob(f'{phase}_*.csv')),f'S17 archive {phase} source results and gate')
    return read(HERE/f'decision_{phase}.json')

def finish():
    child('s17_reports.py','final',log=HERE/'runs/final_report.log');result=read(HERE/'summary.json')
    status('COMPLETE',research_decision=result['status'],formal_fits=result['formal_fits'])
    commit([HERE/'summary.json',HERE/'RESULTS.zh-CN.md',HERE/'STATUS.json'],'S17 finalize bounded receiver mechanism investigation')

def run():
    validate();status('DIAGNOSING_D0')
    child('s17_diagnostics.py',log=HERE/'runs/diagnostics.log');child('s17_reports.py','d0',log=HERE/'runs/report_d0.log')
    commit([HERE/'D0_RESULTS.zh-CN.md',HERE/'D0_metrics.csv'],'S17 report source mechanism audit and confounding limits')
    if not read(HERE/'decision_d0.json')['eligible']:finish();return
    if read(HERE/'ENGINEERING_TESTS.json')['status']!='PASS':
        status('BLOCKED_ENVIRONMENT',completed_stage='D0',reason='PyTorch blocked by Windows CodeIntegrity4551; CPU diagnosis complete; no GPU fits attempted');return
    matrix('pilot')
    status('INTERVENING_D1');child('s17_intervene.py',log=HERE/'runs/interventions.log')
    if not read(HERE/'decision_d1.json')['selected']:finish();return
    child('s17_register.py','pilot',log=HERE/'runs/registration.log');matrix('pilot');decision=outcome('pilot')
    for phase in ['confirm1','confirm2']:
        if not decision['selected']:break
        child('s17_register.py',phase,log=HERE/'runs/registration.log');matrix(phase);decision=outcome(phase)
    finish()

if __name__=='__main__':
    with lock():
        try:run()
        except Exception:
            write(HERE/'execution_failure.json',dict(time=now(),traceback=traceback.format_exc(),automatic_retry=False));status('STOP_TECHNICAL_REVIEW');raise
