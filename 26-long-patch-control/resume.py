"""Idempotent user entry: observe active jobs and never duplicate a cohort."""
import ctypes
import hashlib
import json
import os
import subprocess
import sys
from s26_common import HERE,ROOT,read,validate


def main():
    validate()
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_wchar_p]
    kernel.CreateMutexW.restype=ctypes.c_void_p
    name='Local\\crossatten_s26_'+hashlib.sha256(str(ROOT).encode()).hexdigest()[:16]
    handle=kernel.CreateMutexW(None,True,name)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    already=ctypes.get_last_error()==183
    try:
        if already:
            print('S26 coordinator already active; no duplicate job launched.');return
        # Only return PIDs matching this experiment; do not expose unrelated
        # process command lines in logs.
        workspace=str(ROOT).replace("'","''")
        command=("Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" | "
                 "Where-Object { $_.CommandLine -like '*"+workspace+"*' -and "
                 "($_.CommandLine -like '*s26_execute.py*' -or $_.CommandLine -like '*technical_resume_001.py*') } | "
                 "Select-Object -ExpandProperty ProcessId | ConvertTo-Json -Compress")
        process=subprocess.run(['powershell.exe','-NoProfile','-Command',command],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
        value=process.stdout.decode('utf-8-sig').strip();active=json.loads(value) if value else []
        if isinstance(active,int):active=[active]
        active=[p for p in active if p!=os.getpid()]
        if active:
            print('Existing S26 processes:',active,'; no duplicate training or evaluation launched.')
            if (HERE/'LIVE_STATUS.json').exists():print(json.dumps(read(HERE/'LIVE_STATUS.json')))
            return
        if (HERE/'RESULTS.json').exists() and (HERE/'EVALUATION.json').exists():
            print('All6 fits and report-only evaluation already complete. Rebuild documents; do not rerun the cohort.');return
        if (HERE/'EVALUATION.json').exists():
            raise RuntimeError('Saved evaluation exists: recover analysis from it instead of re-reading Dev. Inspect the preserved runtime failure.')
        import technical_resume_001
        sys.argv=[str(HERE/'technical_resume_001.py')]
        technical_resume_001.main()
    finally:
        if not already:kernel.ReleaseMutex(ctypes.c_void_p(handle))
        kernel.CloseHandle(ctypes.c_void_p(handle))


if __name__=='__main__':main()
