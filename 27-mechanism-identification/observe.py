"""Read progress with Windows FILE_SHARE_DELETE; never obstruct atomic writers."""
import ctypes
import hashlib
import json
import msvcrt
import os
import math
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
    eta=(live.get('remaining_fit_seconds_estimate') or 0)/60
    pending=6-completed-len(active)
    paired_period=sum(read(HERE/'runs'/name/'metrics.json')['wall_seconds']/200 for name in ['LA_f0_s201','LA_f0_s202'])/2
    if eta<=0 and active:
        remaining=[200-q[1] for q in active]+[200]*pending
        eta=max(max(remaining),sum(remaining)/2)*paired_period/60
    if technical['status']=='RECOVERING_SAME_EPOCH_STREAM' and len(active)==1 and pending:
        remaining=200-active[0][1]
        scheduling=max(200, (remaining+pending*200)/2)
        current_period=eta*60/scheduling
        eta=(remaining*current_period+math.ceil(pending/2)*200*paired_period)/60
    elif not active and completed<6:eta=math.ceil((6-completed)/2)*200*paired_period/60
    print(json.dumps(dict(time=live['updated_at'],completed=completed,active=active,
        eta_minutes=round(eta,1),eta_scope='Pendingwaves use measured firstpaired-job throughput, not temporary singleworker rate.',
        technical_status=technical['status'],failure=failure)))


if __name__=='__main__':main()
