"""Idempotent local execution entry protected by a named coordinator mutex."""
import ctypes
import hashlib
import json
import subprocess
from ac_common import HERE,validate

def main():
    validate();k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_wchar_p];k.CreateMutexW.restype=ctypes.c_void_p
    handle=k.CreateMutexW(None,True,'Local\\S28AC_'+hashlib.sha256(str(HERE).encode()).hexdigest()[:20])
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    already=ctypes.get_last_error()==183
    try:
        if already:print('S28AC coordinator already active; no duplicate jobs.');return
        query=("Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" | Where-Object { $_.CommandLine -like '*"+
            str(HERE).replace("'","''")+"*' -and $_.CommandLine -like '*ac_execute.py*' } | Select-Object -ExpandProperty ProcessId | ConvertTo-Json -Compress")
        output=subprocess.check_output(['powershell.exe','-NoProfile','-Command',query]).decode('utf-8-sig').strip()
        if output and json.loads(output):print('Existing S28AC training workers detected; no duplicates launched.');return
        from ac_execute import main as execute
        execute()
    finally:
        if not already:k.ReleaseMutex(ctypes.c_void_p(handle))
        k.CloseHandle(ctypes.c_void_p(handle))

if __name__=='__main__':main()
