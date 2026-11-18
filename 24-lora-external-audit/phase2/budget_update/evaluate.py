"""Report-only cohort inference. CPU baselines can run beside the preserved GPU fit."""
import argparse,importlib.util,json,os,sys,time
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--cohort',choices=['baseline48','native_ask6'],required=True)
parser.add_argument('--device',choices=['cpu','cuda'],required=True);parser.add_argument('--verify-only',action='store_true')
args=parser.parse_args()
if args.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from common import HERE,PHASE,ROOT,STAGE,P,read,digest,atomic_json,policy,verify_seal,dev_cell,cohort_jobs,jobid,h0_records
import numpy as np
import torch
from torch.nn import functional as F
from models import build,forward
from data import preprocess
from inference import h0_model,metrics

@torch.no_grad()
def main():
    a,r=policy();s=verify_seal(args.cohort);torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.verify_only:print('Committed cohort and checkpoint hashes verified. No Dev IO.');return
    jobs=cohort_jobs(args.cohort,r)
    if args.cohort=='baseline48':jobs += [dict(method='h0',view='native',fold=f,seed=seed) for f in (0,1) for seed in (201,202,203)]
    total_started=time.perf_counter()
    for index,j in enumerate(jobs):
        jid=jobid(j);dest=HERE/'evaluations'/jid;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'metrics.json').exists():
            if read(dest/'metrics.json')['seal_sha256']!=digest(HERE/f'seal_{args.cohort}.json'):raise ValueError('evaluation seal changed')
            continue
        started=time.perf_counter();torch.manual_seed(j['seed']+10000*j['fold'])
        if j['method']=='h0':model=h0_model(jid,args.device)
        else:
            model=build(j['method']).to(args.device).eval()
            p=PHASE/'runs'/jid/'best.pt'
            if digest(p)!=s['jobs'][jid]['checkpoint_sha256']:raise ValueError('checkpoint changed')
            model.load_state_dict(torch.load(p,map_location=args.device,weights_only=True),strict=True)
        rows=[]
        for rx in P['folds'][j['fold']]['dev_val']:
            x,y,raw_rows=dev_cell(jid,rx,args.cohort);outputs=[]
            for start in range(0,len(x),64 if args.device=='cpu' else 128):
                batch=64 if args.device=='cpu' else 128
                z=torch.from_numpy(np.asarray(x[start:start+batch]).copy()).to(args.device)
                logits=model(z) if j['method']=='h0' else forward(model,j['method'],preprocess(z,j['view'],j['method']))[0]
                outputs.append(logits.cpu().numpy())
            logits=np.concatenate(outputs)
            if not np.isfinite(logits).all():raise FloatingPointError('nonfinite report logits')
            pred_dir=HERE/'predictions'/jid;pred_dir.mkdir(parents=True,exist_ok=True)
            np.savez(pred_dir/f'{rx}.npz',logits=logits,y=y,rows=raw_rows)
            row=dict(method=j['method'],view=j['view'],fold=j['fold'],seed=j['seed'],rx=rx,**metrics(logits,y))
            rows.append(row);print(jid,rx,row['acc'],row['macro_f1'],flush=True)
        result=dict(status='COMPLETE_REPORT_ONLY',job=j,rows=rows,device=args.device,network_precision='float32',
                    original_frontend_precision_preserved=True,seal_sha256=digest(HERE/f'seal_{args.cohort}.json'),
                    amendment_sha256=digest(HERE/'AMENDMENT.json'),wall_seconds=time.perf_counter()-started,
                    target_adaptation=False,new_training_steps=0)
        atomic_json(dest/'metrics.json',result)
        atomic_json(HERE/f'EVALUATION_{args.cohort}.json',dict(status='RUNNING',completed=index+1,total=len(jobs),active_job=jid,device=args.device))
        del model
        if args.device=='cuda':torch.cuda.empty_cache()
    atomic_json(HERE/f'EVALUATION_{args.cohort}.json',dict(status='COMPLETE',completed=len(jobs),total=len(jobs),device=args.device,
                wall_seconds=time.perf_counter()-total_started,all_target_inference_report_only=True))

if __name__=='__main__':main()
