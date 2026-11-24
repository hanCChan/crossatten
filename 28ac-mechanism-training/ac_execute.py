"""A6 then C12, with no result-conditioned transitions or extra fits."""
import argparse
import ctypes
import gc
import shutil
import subprocess
import sys
import time
import torch
from ac_common import HERE,read,write,sha,now,validate,identity,commit

def seal(stage,reg):
    models={}
    for j in [j for j in reg['jobs'] if j['stage']==stage]:
        name=identity(j);p=HERE/'runs'/name;metric=read(p/'metrics.json')
        assert metric['status']=='COMPLETE' and metric['epochs']==200
        assert metric['optimizer_steps']==(25000 if stage=='A' else 20000)
        models[name]={n:sha(p/n) for n in ['model.pt','config.json','metrics.json','history.csv']}
    path=HERE/f'seal_{stage}.json'
    if path.exists():assert read(path)['models']==models
    else:
        write(path,dict(time=now(),stage=stage,models=models,registration_sha256=sha(HERE/'registration.json')))
        commit('S28AC seal full '+stage+' source-selected cohort before target inference')

def copy_evidence(name):
    source=HERE/'runs'/name;target=HERE/'evidence'/name;target.mkdir(parents=True,exist_ok=True)
    for file in ['config.json','metrics.json','history.csv','pairing.json','streams.json']:
        shutil.copyfile(source/file,target/file)

def main():
    p=argparse.ArgumentParser();p.add_argument('--fit-index',type=int);args=p.parse_args();reg=validate()
    if args.fit_index is not None:
        from ac_train import fit
        raise SystemExit(fit(reg['jobs'][args.fit_index]))
    if (HERE/'RESULTS.json').exists():print('All18 fits and common evaluations already complete; no refit.');return
    if (HERE/'PAUSE_REQUESTED').exists():print('User pause retained; no jobs launched.');return
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        workers=2 if 2*(read(HERE/'GATE.json')['peak_gpu_gib']+.4)+.8<7.5 else 1
        write(HERE/'EXECUTION.json',dict(time=now(),workers=workers,ordering='A6 thenC12; alltargets afterbothseals; unconditional C design and launch.'))
        for stage in ['A','C']:
            queue=[i for i,j in enumerate(reg['jobs']) if j['stage']==stage and not (HERE/'runs'/identity(j)/'metrics.json').exists()]
            active={};handles=[]
            while queue or active:
                while queue and len(active)<workers and not (HERE/'PAUSE_REQUESTED').exists():
                    i=queue.pop(0);j=reg['jobs'][i];name=identity(j);folder=HERE/'local';folder.mkdir(exist_ok=True)
                    stream=(folder/(name+'.log')).open('ab');handles.append(stream)
                    process=subprocess.Popen([sys.executable,'-u',str(__file__),'--fit-index',str(i)],stdout=stream,stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));active[process]=name
                    print('Started',name,'PID',process.pid,flush=True)
                if (HERE/'PAUSE_REQUESTED').exists() and not active:print('Paused at complete epoch checkpoints.');return
                time.sleep(5)
                for process,name in list(active.items()):
                    code=process.poll()
                    if code is None:continue
                    del active[process]
                    if code==75:
                        for other in active:other.wait()
                        print('User pause: all active jobs saved complete epochs.');return
                    if code:
                        write(HERE/'CONTROLLER_FAILURE.json',dict(time=now(),job=name,exit_code=code,new_jobs='STOPPED'))
                        for other in active:other.wait()
                        raise RuntimeError('Training engineering failure; original checkpoint and log retained: '+name)
                    copy_evidence(name);commit('S28AC complete source-only fit '+name+'; no target evaluation')
                    print('Completed',name,flush=True)
            for stream in handles:stream.close()
            seal(stage,reg)
            print('Sealed',stage,'; no A result used to decide C execution.',flush=True)
        gc.collect();torch.cuda.empty_cache()
        write(HERE/'LIVE_STATUS.json',dict(time=now(),status='ALL18_SEALED_COMMON_EVALUATION',completed_fits=18,total_fits=18))
        from ac_evaluate import main as evaluate
        evaluate();write(HERE/'STATUS.json',dict(time=now(),status='TRAINING_AND_REGISTERED_EVALUATION_COMPLETE',new_fits=18,report='PENDING_INDEPENDENT_AUDIT'))
        commit('S28AC report all18 fits and both registered questions without extra training')
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main()
