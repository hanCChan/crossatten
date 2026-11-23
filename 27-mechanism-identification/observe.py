"""Read progress with Windows FILE_SHARE_DELETE; never obstruct atomic writers."""
import ctypes
import hashlib
import json
import msvcrt
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent


def read_text(path):
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p]
    k.CreateFileW.restype=ctypes.c_void_p
    h=k.CreateFileW(str(path),0x80000000,7,None,3,128,None)
    if h==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    fd=msvcrt.open_osfhandle(h,os.O_RDONLY|os.O_BINARY)
    with os.fdopen(fd,'rb') as stream:raw=stream.read()
    try:return raw.decode('utf-8')
    except UnicodeDecodeError:return raw.decode('gb18030')


def read(path):return json.loads(read_text(path))


def main():
    live=read(HERE/'LIVE_STATUS.json');active=[];completed=0;failure=False
    fix=read(HERE/'TECHNICAL_FIX_001.json')
    for folder in sorted((HERE/'runs').glob('LA_*')):
        if (folder/'metrics.json').exists():completed+=1
        elif (folder/'live.json').exists():active.append([folder.name,read(folder/'live.json')['epoch']])
        if (folder/'failure.json').exists():
            text=read_text(folder/'failure.json')
            if folder.name!='LA_f0_s203' or hashlib.sha256(text.encode()).hexdigest()!=fix['original_fit_failure_sha256']:failure=True
    technical=read(HERE/'TECHNICAL_FIX_001_RUNTIME.json')
    stderr=read_text(HERE/'local/technical_001.stderr.log')
    failure=failure or 'Traceback (most recent call last):' in stderr
    print(json.dumps(dict(time=live['updated_at'],completed=completed,active=active,
        eta_minutes=round((live.get('remaining_fit_seconds_estimate') or 0)/60,1),
        technical_status=technical['status'],failure=failure)))


if __name__=='__main__':main()
