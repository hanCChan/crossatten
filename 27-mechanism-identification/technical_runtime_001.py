"""Registered Windows atomic-rename retry; frozen scientific files untouched."""
import argparse
import ctypes
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
import s27_common as common

_serial=itertools.count()


def replace_with_retry(source,target):
    for attempt in range(201):
        try:os.replace(source,target);return
        except PermissionError as error:
            if getattr(error,'winerror',None) not in [5,32,33] or attempt==200:raise
            time.sleep(.01)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f'{path.name}.tmp.{os.getpid()}.{next(_serial)}')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    replace_with_retry(temporary,path)


def atomic_torch(value,path):
    # Preserve the original Torch archive basename/serialization content.
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    torch.save(value,temporary);replace_with_retry(temporary,path)


def wait_pid(pid):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.restype=ctypes.c_void_p
    handle=kernel.OpenProcess(0x00100000,False,pid)
    if handle:
        try:
            while kernel.WaitForSingleObject(ctypes.c_void_p(handle),0)==258:time.sleep(5)
        finally:kernel.CloseHandle(ctypes.c_void_p(handle))


def main():
    p=argparse.ArgumentParser();p.add_argument('--fit-index',type=int);p.add_argument('--wait-original-pid',type=int)
    args=p.parse_args();registration=common.validate()
    record=common.HERE/'TECHNICAL_FIX_001.json';common.committed(record);fix=common.read(record)
    assert common.sha(__file__)==fix['runtime_sha256']
    common.write=write
    import s27_training
    s27_training.write=write;s27_training.atomic_torch=atomic_torch
    if args.fit_index is not None:
        s27_training.fit(registration['jobs'][args.fit_index]);return
    assert args.wait_original_pid is not None
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_wchar_p]
    kernel.CreateMutexW.restype=ctypes.c_void_p
    mutex_name='Local\\crossatten_s27_'+hashlib.sha256(str(common.ROOT).encode()).hexdigest()[:16]
    # Retain the named guard across the original controller's exit.
    mutex=kernel.CreateMutexW(None,False,mutex_name)
    if not mutex:raise ctypes.WinError(ctypes.get_last_error())
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        q=common.HERE/'runs/LA_f0_s203';failure=common.read(q/'failure.json')
        assert '[WinError 5]' in failure['traceback'] and 'live.json.tmp' in failure['traceback']
        assert common.sha(q/'failure.json')==fix['original_fit_failure_sha256']
        write(common.HERE/'TECHNICAL_FIX_001_RUNTIME.json',dict(time=common.now(),status='RECOVERING_SAME_EPOCH_STREAM',extra_fits=0))
        log=(common.HERE/'local/LA_f0_s203.log').open('ab')
        process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--fit-index','2'],stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        print('Recovering existing LA_f0_s203 from complete epoch',fix['completed_resume_epoch'],'PID',process.pid,flush=True)
        wait_pid(args.wait_original_pid)
        assert process.wait()==0,'Preserve new recovery failure; do not reset a seed'
        log.close()
        for job in registration['jobs'][:4]:
            folder=common.HERE/'runs'/common.identity(job['fold'],job['seed'])
            assert common.read(folder/'metrics.json')['status']=='COMPLETE'
        import s27_execute
        # The original coordinator uses __file__ only for its child CLI path;
        # route remaining identical --fit-index jobs through this IO wrapper.
        s27_execute.__file__=str(Path(__file__).resolve());sys.argv=[s27_execute.__file__]
        write(common.HERE/'TECHNICAL_FIX_001_RUNTIME.json',dict(time=common.now(),status='CONTINUING_ORIGINAL_QUEUE',extra_fits=0))
        s27_execute.main()
        write(common.HERE/'TECHNICAL_FIX_001_RUNTIME.json',dict(time=common.now(),status='COMPLETE',extra_fits=0,
            original_code_unchanged=True,optimizer_seed_or_data_changes=False))
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        kernel.CloseHandle(ctypes.c_void_p(mutex))


if __name__=='__main__':main()
