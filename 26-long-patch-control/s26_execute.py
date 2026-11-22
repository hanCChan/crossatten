"""Source-only fits, complete-cohort seal, then report-only evaluation."""
import argparse
import ctypes
import gc
import shutil
import subprocess
import sys
import torch
from s26_common import HERE,read,write,sha,now,validate,commit,identity


def seal():
    r=validate();models={}
    for job in r['jobs']:
        name=identity(job['fold'],job['seed']);p=HERE/'runs'/name
        assert read(p/'metrics.json')['status']=='COMPLETE'
        assert read(p/'metrics.json')['epochs']==200
        models[name]={n:sha(p/n) for n in ['model.pt','config.json','metrics.json','history.csv']}
        out=HERE/'evidence'/name;out.mkdir(parents=True,exist_ok=True)
        for n in ['config.json','metrics.json','history.csv','pairing.json','streams.json']:
            shutil.copyfile(p/n,out/n)
    if not (HERE/'seal.json').exists():
        write(HERE/'seal.json',dict(time=now(),registration_sha256=sha(HERE/'registration.json'),models=models,
            rule='All six source-CE-selected models and complete configurations frozen before any S26 Dev read.'))
        commit('S26 seal all six source-selected Long-Patch checkpoints before Dev inference')
    else:assert read(HERE/'seal.json')['models']==models


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fit-index',type=int);args=parser.parse_args()
    r=validate()
    if args.fit_index is not None:
        from s26_training import fit
        fit(r['jobs'][args.fit_index]);return
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        if not (HERE/'SOURCE_RECOVERY.json').exists():
            from s26_source_check import main as recover_source
            recover_source();commit('S26 record read-only archived source recovery before new fitting')
        # Two isolated CUDA contexts; batch/RNG/update rules unchanged. Selected
        # only from the data-free memory gate, not from source/Dev accuracy.
        peak=read(HERE/'GATE.json')['peak_gpu_gib']
        workers=2 if 2*(peak+.7)+.8<7.5 else 1
        write(HERE/'EXECUTION.json',dict(time=now(),workers=workers,gate_peak_gpu_gib=peak,
            rule='Max2 independent jobs; each retains batch64 and its own generators. Memory gate only. No target-score scheduling.'))
        queue=[i for i,j in enumerate(r['jobs']) if not (HERE/'runs'/identity(j['fold'],j['seed'])/'metrics.json').exists()]
        active={};handles=[]
        while queue or active:
            while queue and len(active)<workers:
                i=queue.pop(0);j=r['jobs'][i];name=identity(j['fold'],j['seed'])
                p=HERE/'local';p.mkdir(exist_ok=True)
                stream=(p/(name+'.log')).open('ab');handles.append(stream)
                process=subprocess.Popen([sys.executable,'-u',str(__file__),'--fit-index',str(i)],stdout=stream,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                active[process]=name;print('Started',name,'PID',process.pid,flush=True)
            import time
            time.sleep(5)
            for process,name in list(active.items()):
                code=process.poll()
                if code is None:continue
                del active[process]
                if code:
                    write(HERE/'CONTROLLER_FAILURE.json',dict(time=now(),job=name,exit_code=code,other_jobs='Allowed to finish; no new jobs launched.'))
                    for other in active:other.wait()
                    raise RuntimeError('Fit failed; see preserved local log: '+name)
                commit('S26 completed source-only fit '+name+'; no Dev evaluation')
                print('Completed',name,flush=True)
        for stream in handles:stream.close()
        gc.collect();torch.cuda.empty_cache();seal()
        from s26_evaluate import evaluate,costs
        from s26_report import main as report
        evaluate();costs();report()
        write(HERE/'STATUS.json',dict(status='TRAINING_AND_UNIFIED_EVALUATION_COMPLETE',time=now(),completed_new_fits=6,
            reused_models=12,evaluated_predictions=180000,confirm_blind_reads=0,manuscript_revision='PENDING'))
        commit('S26 report all six Long-Patch controls and common GPU paired results')
        print('All6 fits and unified evaluation complete.',flush=True)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
