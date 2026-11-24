"""Read-only Windows progress with shared-delete access; no Torch or RF IO."""
import ctypes
import json
import msvcrt
import os
from pathlib import Path
from statistics import median

HERE=Path(__file__).resolve().parent
def read_text(path):
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p]
    k.CreateFileW.restype=ctypes.c_void_p;h=k.CreateFileW(str(path),0x80000000,7,None,3,128,None)
    if h==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    with os.fdopen(msvcrt.open_osfhandle(h,os.O_RDONLY|os.O_BINARY),'rb') as f:raw=f.read()
    for encoding in ['utf-8','utf-16','gb18030']:
        try:return raw.decode(encoding)
        except UnicodeError:pass
    raise UnicodeError(str(path))
def read(path):return json.loads(read_text(path))

def main():
    jobs=read(HERE/'registration.json')['jobs'];finished=set();active=[];per_step=[];specific={};failure=False
    for j in jobs:
        name=f"{j['stage']}_{j['arm']}_f{j['fold']}_s{j['seed']}";p=HERE/'runs'/name;steps=125 if j['stage']=='A' else 100
        if (p/'metrics.json').exists():
            m=read(p/'metrics.json');finished.add(name);speed=m['wall_seconds']/m['optimizer_steps']
            per_step.append(speed);specific.setdefault((j['stage'],j['arm']),[]).append(speed)
        elif (p/'live.json').exists():
            live=read(p/'live.json');active.append(dict(id=name,stage=j['stage'],arm=j['arm'],epoch=live['epoch']))
            history=p/'history.csv'
            if history.exists():
                import csv,io
                rows=list(csv.DictReader(io.StringIO(read_text(history))))
                rows=[q for q in rows if q.get('seconds') not in [None,'']]
                if len(rows)>=11:
                    speed=median([float(b['seconds'])-float(a['seconds']) for a,b in zip(rows[-11:-1],rows[-10:])])/steps
                    per_step.append(speed);specific.setdefault((j['stage'],j['arm']),[]).append(speed)
        failure|=(p/'failure.json').exists()
    errors='\n'.join(read_text(p) for p in (HERE/'local').glob('controller*.stderr.log'))
    failure|='Traceback (most recent call last):' in errors or (HERE/'CONTROLLER_FAILURE.json').exists()
    eta=None
    if per_step:
        avg=sum(per_step)/len(per_step);workers=read(HERE/'EXECUTION.json')['workers'];cursor=0.
        for stage in ['A','C']:
            loads=[cursor]*workers
            for a in [q for q in active if q['stage']==stage]:
                index=min(range(workers),key=loads.__getitem__);q=specific.get((stage,a['arm']),[avg]);loads[index]+=(200-a['epoch'])*(125 if stage=='A' else 100)*sum(q)/len(q)
            known=finished|{a['id'] for a in active}
            for j in [j for j in jobs if j['stage']==stage]:
                name=f"{stage}_{j['arm']}_f{j['fold']}_s{j['seed']}"
                if name in known:continue
                q=specific.get((stage,j['arm']),[avg]);index=min(range(workers),key=loads.__getitem__)
                loads[index]+=200*(125 if stage=='A' else 100)*sum(q)/len(q)
            cursor=max(loads)
        eta=round(cursor/60,1)
    status=read(HERE/'STATUS.json').get('status') if (HERE/'STATUS.json').exists() else ('PAUSED_USER' if (HERE/'PAUSE_REQUESTED').exists() else None)
    print(json.dumps(dict(completed=len(finished),total=18,A_completed=sum(n.startswith('A_') for n in finished),C_completed=sum(n.startswith('C_') for n in finished),active=active,
        eta_minutes=eta,status=status,failure=failure,scope='Fits only; final common evaluation, independent verification and report extra.')))

if __name__=='__main__':main()
