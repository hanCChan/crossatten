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
    live=read(HERE/'LIVE_STATUS.json') if (HERE/'LIVE_STATUS.json').exists() else {}
    active=[];completed=0;failure=False;durations=[];finished=set();arm_durations={}
    if (HERE/'runs').exists():
        for folder in sorted((HERE/'runs').iterdir()):
            if not folder.is_dir():continue
            if (folder/'metrics.json').exists():
                completed+=1;finished.add(folder.name)
                duration=read(folder/'metrics.json')['wall_seconds']/200
                durations.append(duration);arm_durations.setdefault(folder.name.rsplit('_f',1)[0],[]).append(duration)
            elif (folder/'live.json').exists():active.append([folder.name,read(folder/'live.json')['epoch']])
            if (folder/'failure.json').exists():failure=True
    stage=read(HERE/'STATUS.json').get('status') if (HERE/'STATUS.json').exists() else None
    stderr='\n'.join(read_text(p) for p in sorted((HERE/'local').glob('controller*.stderr.log')))
    failure=failure or 'Traceback (most recent call last):' in stderr
    eta=live.get('remaining_fit_seconds_estimate')
    if len(durations)>=2 and completed<12:
        average=sum(durations)/len(durations)
        per_epoch={arm:sum(values)/len(values) for arm,values in arm_durations.items()}
        workers=read(HERE/'EXECUTION.json')['workers']
        loads=[(200-epoch)*per_epoch.get(name.rsplit('_f',1)[0],average) for name,epoch in active]
        loads += [0.] * max(0,workers-len(loads))
        known=finished|{name for name,_ in active}
        for job in read(HERE/'registration.json')['jobs']:
            name=f"{job['arm']}_f{job['fold']}_s{job['seed']}"
            if name in known:continue
            first=min(range(len(loads)),key=loads.__getitem__)
            loads[first]+=200*per_epoch.get(job['arm'],average)
        eta=max(loads) if loads else 0.
    print(json.dumps(dict(time=live.get('updated_at'),completed=completed,total=12,active=active,
        eta_minutes=round(eta/60,1) if eta is not None and eta>0 else None,status=stage,failure=failure)))

if __name__=='__main__':main()
