"""Adopt the existing GPU worker; defer whole Strict ASK arm; report sealed baselines."""
import ctypes,json,os,subprocess,sys,time,traceback
from pathlib import Path
from common import HERE,PHASE,ROOT,read,digest,atomic_json,policy,jobid,authorized_training,create_seal,access_events,commit,now

def running(pid):
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32];k.OpenProcess.restype=ctypes.c_void_p
    k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];k.WaitForSingleObject.restype=ctypes.c_uint32
    k.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=k.OpenProcess(0x00100000,False,int(pid))
    if not handle:return False
    try:return k.WaitForSingleObject(handle,0)==258
    finally:k.CloseHandle(handle)

def publish(message):
    # Only stage-local evidence; never checkpoint/waveform/prediction binaries or partial lock files.
    files=[p for p in PHASE.rglob('*') if p.is_file() and not any(v in p.parts for v in ['local','__pycache__','predictions'])
           and p.suffix not in ('.pt','.npz','.tmp','.lock')]
    commit(files,message)
    try:
        result=subprocess.run(['git','-C',str(ROOT),'-c','http.version=HTTP/1.1','push','origin','exp/s24-lora-external-audit'],capture_output=True,text=True,encoding='utf-8',timeout=90)
        atomic_json(HERE/'PUBLICATION.json',dict(returncode=result.returncode,output=result.stdout+result.stderr))
    except (subprocess.TimeoutExpired,OSError) as exc:
        atomic_json(HERE/'PUBLICATION.json',dict(status='PUSH_DEFERRED',exception=repr(exc),local_commit_saved=True))

def main():
    a,r=policy();handoff=read(HERE/'HANDOFF.json')
    if running(handoff['old_supervisor_pid']):raise RuntimeError('Old supervisor still running; refuse two schedulers')
    lock=PHASE/'supervisor.lock'
    if lock.exists():
        if running(int(lock.read_text().strip())):raise RuntimeError('Another scheduler owns lock')
        lock.unlink()
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(descriptor,str(os.getpid()).encode());os.close(descriptor)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.SetThreadExecutionState.argtypes=[ctypes.c_uint32];kernel.SetThreadExecutionState.restype=ctypes.c_uint32
    if not kernel.SetThreadExecutionState(0x80000001):raise ctypes.WinError(ctypes.get_last_error())
    atomic_json(HERE/'KEEP_AWAKE.json',dict(status='ACTIVE',controller_pid=os.getpid(),release='On controller exit',display_required=False))
    cpu=None;cpu_log=None;cpu_published=False
    def completed():
        return sum((PHASE/'runs'/jid/'metrics.json').exists() and read(PHASE/'runs'/jid/'metrics.json')['status']=='COMPLETE' for jid in a['active_job_ids'])
    def state(active,status='RUNNING_RETAINED54'):
        events=access_events()
        atomic_json(PHASE/'LIVE_STATUS.json',dict(status=status,pid=os.getpid(),active_job=active,completed=completed(),registered=54,
                    original_registered=60,deferred=6,deferred_scope='ASK Strict-IB all six jobs',dev_reads=sum(e['role']=='dev_report_only' for e in events),
                    confirm_reads=0,blind_reads=0,baseline_evaluation=read(HERE/'EVALUATION_baseline48.json') if (HERE/'EVALUATION_baseline48.json').exists() else 'STARTING',
                    amendment_sha256=digest(HERE/'AMENDMENT.json')))
    def summarize():
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1')
        result=subprocess.run([sys.executable,'-u',str(HERE/'summarize.py')],cwd=ROOT,env=env)
        if result.returncode:raise RuntimeError('cohort summary failed')
    def monitor_cpu():
        nonlocal cpu_published
        if cpu is not None and cpu.poll() is not None and not cpu_published:
            cpu_published=True
            if cpu.returncode:
                atomic_json(HERE/'EVALUATION_FAILURE.json',dict(cohort='baseline48',returncode=cpu.returncode,training_continues=True))
                publish('S24 retain CPU report failure for technical repair; Native training continues')
            else:
                summarize();publish('S24 publish sealed CE/DANN/FAN/IB and frozen H0 Dev comparisons under user budget amendment')
    try:
        create_seal('baseline48')
        if not (HERE/'EVALUATION_baseline48.json').exists() or read(HERE/'EVALUATION_baseline48.json')['status']!='COMPLETE':
            attempt=1+len(list(HERE.glob('baseline48_attempt*.log')))
            cpu_log=(HERE/f'baseline48_attempt{attempt}.log').open('w',encoding='utf-8')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            cpu=subprocess.Popen([sys.executable,'-u',str(HERE/'evaluate.py'),'--cohort','baseline48','--device','cpu'],stdout=cpu_log,stderr=subprocess.STDOUT,cwd=ROOT,env=env)
            atomic_json(HERE/'EVALUATION_WORKER.json',dict(pid=cpu.pid,cohort='baseline48',device='cpu',log=str(cpu_log.name)))
        else:summarize();cpu_published=True
        # Keep the worker that was already running. Parent scheduler replacement does not restart it.
        current=read(PHASE/'WORKER.json')
        attached_job=handoff['worker_job'];attached_pid=handoff['worker_pid']
        # Manual recovery also adopts any later Native worker that survived its controller.
        if running(current['pid']) and not (PHASE/'runs'/current['job']/'metrics.json').exists():
            attached_job=current['job'];attached_pid=current['pid']
        authorized_training(next(j for j in r['jobs'] if jobid(j)==attached_job))
        while running(attached_pid):
            state(attached_job,'PRESERVING_ATTACHED_NATIVE_WORKER');monitor_cpu();time.sleep(10)
        attached=PHASE/'runs'/attached_job/'metrics.json'
        if not attached.exists() or read(attached)['status']!='COMPLETE':raise RuntimeError('Attached worker did not complete; preserve its recovery checkpoint')
        publish('S24 finish pre-amendment active ASK Native fit without interruption')
        for j in r['jobs']:
            if j['method']!='ask' or j['view']!='native':continue
            authorized_training(j);jid=jobid(j);out=PHASE/'runs'/jid;out.mkdir(parents=True,exist_ok=True)
            if (out/'metrics.json').exists() and read(out/'metrics.json')['status']=='COMPLETE':
                if digest(out/'best.pt')!=read(out/'metrics.json')['checkpoint_sha256']:raise ValueError('completed retained checkpoint changed')
                continue
            attempt=1+len(list(out.glob('stdout_budget_attempt*.log')))
            state(jid)
            with (out/f'stdout_budget_attempt{attempt}.log').open('w',encoding='utf-8') as log:
                worker=subprocess.Popen([sys.executable,'-u',str(HERE/'train_next.py'),'--job',jid],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                atomic_json(PHASE/'WORKER.json',dict(pid=worker.pid,job=jid,log=str(log.name),budget_amendment_sha256=digest(HERE/'AMENDMENT.json')))
                while worker.poll() is None:
                    state(jid);monitor_cpu();time.sleep(10)
            if worker.returncode:raise RuntimeError('Retained Native worker failed; do not change configuration or auto-reinitialize')
            publish(f'S24 archive retained Native fit {jid}; Strict-IB ASK remains deferred')
        if cpu is not None:cpu.wait();monitor_cpu()
        if not (HERE/'EVALUATION_baseline48.json').exists() or read(HERE/'EVALUATION_baseline48.json')['status']!='COMPLETE':raise RuntimeError('Baseline report needs technical recovery')
        create_seal('native_ask6');create_seal('retained54')
        with (HERE/'native_ask6.log').open('a',encoding='utf-8') as log:
            rc=subprocess.run([sys.executable,'-u',str(HERE/'evaluate.py'),'--cohort','native_ask6','--device','cuda'],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT).returncode
        if rc:raise RuntimeError('ASK Native report failed')
        summarize()
        with (HERE/'costs.log').open('a',encoding='utf-8') as log:
            rc=subprocess.run([sys.executable,'-u',str(HERE/'costs.py')],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT).returncode
        if rc:raise RuntimeError('retained budget cost benchmark failed')
        state(None,'COMPLETE_RETAINED54_STRICT_ASK_DEFERRED')
        publish('S24 complete retained54 matrix, full Native ASK comparison and partial Strict-IB table; deferred cells NA')
    except BaseException as exc:
        atomic_json(HERE/'CONTROLLER_FAILURE.json',dict(time=now(),exception=repr(exc),traceback=traceback.format_exc(),completed=completed()))
        if cpu is not None and cpu.poll() is None:
            cpu.wait()
            if cpu.returncode==0:
                try:summarize();publish('S24 preserve completed baseline report despite Native controller failure')
                except BaseException:pass
        raise
    finally:
        if cpu_log is not None:cpu_log.close()
        kernel.SetThreadExecutionState(0x80000000)
        atomic_json(HERE/'KEEP_AWAKE.json',dict(status='RELEASED',controller_pid=os.getpid()))
        if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()

if __name__=='__main__':main()
