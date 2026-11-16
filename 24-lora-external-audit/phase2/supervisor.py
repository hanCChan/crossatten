"""Sequential durable source training queue; per-method freezes and globally gated Dev."""
import os,sys,subprocess,time,traceback
from pathlib import Path
from execution import HERE,ROOT,read,atomic_json,validate_registry,jobid,freeze_method,commit

def publish(message):
    files=[p for p in HERE.rglob('*') if p.is_file() and not any(v in p.parts for v in ['local','__pycache__'])
           and p.suffix not in ('.pt','.npz','.tmp','.lock')]
    commit(files,message)
    try:
        result=subprocess.run(['git','-C',str(ROOT),'push','origin','exp/s24-lora-external-audit'],capture_output=True,text=True,encoding='utf-8',timeout=90)
        atomic_json(HERE/'PUBLICATION.json',dict(returncode=result.returncode,output=result.stdout+result.stderr))
    except (subprocess.TimeoutExpired,OSError) as exc:
        atomic_json(HERE/'PUBLICATION.json',dict(status='PUSH_DEFERRED',exception=repr(exc),local_commit_saved=True))

def main():
    registry=validate_registry();lock=HERE/'supervisor.lock'
    # An exclusive file prevents concurrent GPU workers. Stale lock requires an explicit liveness check.
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    failed=[]
    def counts():
        return sum((HERE/'runs'/jobid(j)/'metrics.json').exists() and read(HERE/'runs'/jobid(j)/'metrics.json')['status']=='COMPLETE' for j in registry['jobs'])
    def run(j):
        jid=jobid(j);out=HERE/'runs'/jid;out.mkdir(parents=True,exist_ok=True)
        if (out/'metrics.json').exists() and read(out/'metrics.json')['status']=='COMPLETE':return True
        atomic_json(HERE/'LIVE_STATUS.json',dict(status='RUNNING_SOURCE_ONLY',pid=os.getpid(),active_job=jid,completed=counts(),registered=60,failed_methods=failed,dev_reads=0))
        attempt=1+len(list(out.glob('stdout_attempt*.log')))
        with (out/f'stdout_attempt{attempt}.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(HERE/'runner.py'),'--job',jid],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            atomic_json(HERE/'WORKER.json',dict(pid=proc.pid,job=jid,log=str(log.name)))
            rc=proc.wait()
        if rc:
            failed.append(j['method']);atomic_json(HERE/'FAILURES.json',dict(methods=failed,job=jid,returncode=rc))
            publish('S24 retain source-only failure evidence and continue independent methods');return False
        publish(f'S24 archive completed source-only fit {jid}')
        return True
    try:
        # First cheap and paired source pilots; ASK is scheduled after lighter methods because QR dominates time.
        for method in ['ib','ce','dann']:
            group=['fan','ib'] if method=='ib' else [method]
            okay=True
            for m in group:
                j=next(j for j in registry['jobs'] if j['method']==m and j['view']=='native' and j['fold']==0 and j['seed']==201)
                if not run(j):okay=False;break
            if okay:freeze_method(method);publish(f'S24 publish frozen {method} source-pilot registry')
        for group in [['fan','ib'],['ce'],['dann']]:
            if any(m in failed for m in group):continue
            for view in ['native','strict_ib']:
                for fold in [0,1]:
                    for seed in [201,202,203]:
                        for m in group:
                            if not run(next(j for j in registry['jobs'] if j['method']==m and j['view']==view and j['fold']==fold and j['seed']==seed)):break
                        if any(m in failed for m in group):break
                    if any(m in failed for m in group):break
                if any(m in failed for m in group):break
        j=next(j for j in registry['jobs'] if j['method']=='ask' and j['view']=='native' and j['fold']==0 and j['seed']==201)
        if run(j):
            freeze_method('ask');publish('S24 freeze full ASKNet after source-only pilot')
            for j in registry['jobs']:
                if j['method']=='ask' and not run(j):break
        if counts()==60 and not failed:
            result=subprocess.run([sys.executable,'-u',str(HERE/'report.py')],cwd=ROOT)
            if result.returncode:raise RuntimeError('sealed evaluation failed')
            result=subprocess.run([sys.executable,'-u',str(HERE/'bench.py')],cwd=ROOT)
            if result.returncode:raise RuntimeError('cost benchmark failed')
            atomic_json(HERE/'LIVE_STATUS.json',dict(status='COMPLETE',completed=60,registered=60,confirm_blind_reads=0))
            publish('S24 publish all paired full-method comparisons after global checkpoint seal')
        else:
            atomic_json(HERE/'LIVE_STATUS.json',dict(status='METHOD_FAILURES_PENDING',completed=counts(),registered=60,failed_methods=failed,dev_reads=0))
            publish('S24 record completed source matrix and unresolved method failures; Dev remains locked')
    except BaseException as exc:
        atomic_json(HERE/'SUPERVISOR_FAILURE.json',dict(exception=repr(exc),traceback=traceback.format_exc(),completed=counts()))
        raise
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__':main()
