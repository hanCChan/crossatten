import sys, json, time, traceback,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from s24_common import HERE as STAGE, verified_source, write_json
import numpy as np
import torch

OUT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',type=int,default=1);args=parser.parse_args()
    start=time.perf_counter()
    report=dict(status='RUNNING',batch=args.batch,scientific_fits=0,technical_optimizer_steps=0,events=[],gpu=torch.cuda.get_device_name(0),candidates=[128,256,512,1024,2048,4096])
    def event(name, **data):
        report['events'].append(dict(name=name,elapsed=time.perf_counter()-start,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak=torch.cuda.max_memory_allocated(),**data))
        write_json(OUT/f'ASK_GATE_b{args.batch}.json',report);print(name,flush=True)
    torch.set_num_threads(4);torch.manual_seed(2401);torch.cuda.reset_peak_memory_stats()
    qr=torch.linalg.qr
    def timed_qr(*args,**kwargs):
        torch.cuda.synchronize();t=time.perf_counter()
        result=qr(*args,**kwargs)
        torch.cuda.synchronize();event('QR',seconds=time.perf_counter()-t,shape=list(args[0].shape))
        return result
    torch.linalg.qr=timed_qr
    try:
        ns={};exec(compile(verified_source('BeechburgPieStar/ASKNet-RFF','Lora/backbones/ResCISNet.py'),'official_ResCISNet.py','exec'),ns)
        event('before_constructor')
        with torch.device('cuda'):
            model=ns['ResCISNet'](num_classes=10)
        torch.cuda.synchronize();event('constructed',params=sum(p.numel() for p in model.parameters()))
        model.train();optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
        x=torch.randn(args.batch,2,8192,device='cuda');x=x/torch.sqrt((x*x).sum(1).mean(-1))[:,None,None]
        t=time.perf_counter();logits=model(x,tau=5.0);torch.cuda.synchronize();event('forward',seconds=time.perf_counter()-t)
        loss=torch.nn.functional.cross_entropy(logits,torch.zeros(args.batch,dtype=torch.long,device='cuda'))
        t=time.perf_counter();loss.backward();torch.cuda.synchronize();event('backward',seconds=time.perf_counter()-t,finite=all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None))
        t=time.perf_counter();optimizer.step();torch.cuda.synchronize();report['technical_optimizer_steps']=1;event('Adam_step',seconds=time.perf_counter()-t)
        report['status']='PASS'
    except BaseException as exc:
        report['status']='RESOURCE_INFEASIBLE' if isinstance(exc,torch.OutOfMemoryError) else 'TECHNICAL_FAILURE'
        report['exception']=repr(exc);report['traceback']=traceback.format_exc()
        print(report['traceback'],flush=True)
    finally:
        report['wall_seconds']=time.perf_counter()-start
        report['peak_bytes']=torch.cuda.max_memory_allocated()
        write_json(OUT/f'ASK_GATE_b{args.batch}.json',report)

if __name__=='__main__':main()
