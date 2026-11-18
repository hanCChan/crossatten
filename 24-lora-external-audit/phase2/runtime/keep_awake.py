"""Bounded Windows idle-sleep inhibition for the existing training supervisor."""
import ctypes
import datetime
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
PID=int(sys.argv[1])
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]
kernel.OpenProcess.restype=ctypes.c_void_p
kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
kernel.WaitForSingleObject.restype=ctypes.c_uint32
kernel.CloseHandle.argtypes=[ctypes.c_void_p]
kernel.SetThreadExecutionState.argtypes=[ctypes.c_uint32]
kernel.SetThreadExecutionState.restype=ctypes.c_uint32

def status(state,**details):
    record=dict(state=state,helper_pid=os.getpid(),supervisor_pid=PID,
                time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                system_idle_sleep_inhibition=True,display_required=False,
                automatic_release='When the held supervisor process handle signals exit',**details)
    temporary=HERE/'KEEP_AWAKE_STATUS.json.tmp'
    temporary.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    os.replace(temporary,HERE/'KEEP_AWAKE_STATUS.json')

handle=kernel.OpenProcess(0x00100000,False,PID)
if not handle:raise ctypes.WinError(ctypes.get_last_error())
try:
    if kernel.WaitForSingleObject(handle,0)!=258:raise RuntimeError('Supervisor already exited')
    if not kernel.SetThreadExecutionState(0x80000001):raise ctypes.WinError(ctypes.get_last_error())
    status('ACTIVE')
    while kernel.WaitForSingleObject(handle,0)==258:
        if not kernel.SetThreadExecutionState(0x80000001):raise ctypes.WinError(ctypes.get_last_error())
        time.sleep(30)
    status('RELEASED',system_idle_sleep_inhibition_ended=True)
finally:
    kernel.SetThreadExecutionState(0x80000000)
    kernel.CloseHandle(handle)
