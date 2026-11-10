"""Four official ASKNet fits on exact frozen S19 source train/val samples."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from s20_core import HERE,ROOT,S19,HISTORY,UPSTREAM,read,write,sha,now,validate,committed,access,official,configure_data,git
from s20_stats import score,decision

ARMS=['B0','ASK']
CONF=argparse.Namespace(candidates=[8,16,24,32,48,64],patch_size=64,embed_dim=128,mlp_ratio=2.)
def outdir(fold,arm):return HERE/'runs'/f'{arm}__f{fold}__first__s18101'
def olddir(fold):return S19/'runs'/f'ManyRx__B0__f{fold}__first__s18101'
def save_predictions(p,logits,data):np.savez_compressed(p,logits=logits,**{k:v for k,v in data.items() if k!='x'})
@torch.no_grad()
def predict(model,x):
    model.eval()
    return np.concatenate([model(torch.as_tensor(np.array(x[i:i+128],copy=True),device='cuda')).cpu().numpy() for i in range(0,len(x),128)])

def preparation():
    if (HERE/'REUSE_AUDIT.json').exists():raise ValueError('Preparation already recorded')
    historical=read(S19/'MANIFEST.json')
    for rel,h in historical['files_sha256'].items():
        if sha(S19/rel)!=h:raise ValueError('S19 archive not intact: '+rel)
    data=configure_data();old_configs=[]
    for p in sorted((ROOT/'10-tifs-validation/runs/ask').glob('ManyRx*/config.json')):
        c=read(p);fit,held,target=data.roles(c['job']['fold'])
        reasons=[]
        if c['roles']['source']!=fit:reasons.append('S10 trained all24 source RX, including the6 required held-source RX')
        if set(c['roles']['source'])&set(held):reasons.append('S20 held-source contamination in S10 training')
        if c['job']['seed']!=18101:reasons.append('Different train/validation indices and seed')
        old_configs.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),job=c['job'],source_rx=len(c['roles']['source']),reusable=not reasons,reasons=reasons))
    if len(old_configs)!=12 or any(c['reusable'] for c in old_configs):raise ValueError('Unexpected S10 reuse audit')
    baselines=[]
    seal=read(S19/'seal_pilot.json')
    for f in range(4):
        p=olddir(f);ref=seal['checkpoints'][f'ManyRx__B0__f{f}__first__s18101'];cfg=read(p/'config.json')
        baselines.append(dict(fold=f,checkpoint=ref,config_sha256=sha(p/'config.json'),metrics_sha256=sha(p/'metrics.json'),
                              held_metrics_sha256=sha(p/'held_metrics.json'),source_predictions_sha256=read(p/'metrics.json')['prediction_sha256'],
                              held_predictions_sha256=read(p/'held_metrics.json')['prediction_sha256'],data_seed=cfg['data_seed']))
    central=[]
    for f in range(4):central.extend(read(S19/f'evidence_f{f}.json')['summary']['bar_q'])
    q=np.asarray(central)
    write(HERE/'REUSE_AUDIT.json',dict(time=now(),s19_commit='9641b7b',s19_archived_files_verified=len(historical['files_sha256']),
          s19_baselines=baselines,s10_asknet_candidates=old_configs,s10_reusable=0,new_asknet_fits_required=4,
          weights_or_iq_read=False,central_evidence_check=dict(rows=len(q),max_in_central_two=int(np.isin(q.argmax(1),[3,4]).sum()),mean_central_mass=float(q[:,3:5].sum(1).mean()))))
    official_manifest=read(UPSTREAM/'FETCH_MANIFEST.json')
    write(HERE/'UPSTREAM.json',dict(official_manifest,source_directory=str(UPSTREAM),source_redistributed=False,
          invoked='build_model/setup_seed/train_epoch/evaluate_epoch/train_and_evaluate; no official.main or dataset loading',
          model_path='PatchNet(L256,6cutoff candidates,complexQR127,patch64stride32,dim128,MLP ratio2,10classes)'))
    print('PASS S19 full archive;0/12 S10 ASKNet checkpoints eligible;4 new fits required',flush=True)

class RecordedDataset(TensorDataset):
    def __init__(self,*args):super().__init__(*args);self.order=[]
    def __getitem__(self,index):self.order.append(int(index));return super().__getitem__(index)

def fit(fold):
    validate();out=outdir(fold,'ASK')
    if (out/'metrics.json').exists():return
    if (out/'attempt.json').exists():raise RuntimeError('Retained failed/interrupted scientific attempt; no automatic retry')
    write(out/'attempt.json',dict(time=now(),fold=fold,status='RUNNING'))
    try:
        data_module=configure_data();data,meta,_,ids=data_module.load_fit(fold);tr,va=data['train'],data['val']
        seed=read(olddir(fold)/'config.json')['data_seed'];api=official();torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        api.setup_seed(seed)
        train_ds=RecordedDataset(torch.from_numpy(tr['x']),torch.from_numpy(tr['y']))
        train_loader=DataLoader(train_ds,batch_size=128,shuffle=True,drop_last=False)
        val_loader=DataLoader(TensorDataset(torch.from_numpy(va['x']),torch.from_numpy(va['y'])),batch_size=128,shuffle=False,drop_last=False)
        model=api.build_model(CONF,10)
        total=sum(p.numel() for p in model.parameters());real=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())
        initial=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.state_dict().values())).hexdigest()
        cfg=dict(fold=fold,arm='ASK',macro_seed=18101,data_seed=seed,split_seed=16001,train_packets=len(tr['y']),val_packets=len(va['y']),
                 train_indices_sha256=hashlib.sha256(ids['train'].tobytes()).hexdigest(),val_indices_sha256=hashlib.sha256(ids['val'].tobytes()).hexdigest(),
                 official_commit='f00846d68425315b68ae69b8c6b712e08559c2fe',official_defaults=vars(CONF),parameter_elements=total,real_scalar_dof=real,
                 epochs_max=200,batch=128,optimizer='official Adam1e-3 wd0;Plateau factor.1 patience5 min1e-6;earlystop10',precision='FP32',
                 selection='official packet-equal source validation CE',shuffle='official CPU DataLoader/global Torch RNG',
                 initial_state_sha256=initial,git_sha=git('rev-parse','HEAD'),fit_rx=meta['fit_rx'],held_rx=meta['held_rx'])
        write(out/'config.json',cfg)
        history=[];train_fn,val_fn=api.train_epoch,api.evaluate_epoch
        def train_trace(*args,**kwargs):
            train_ds.order=[];loss,acc=train_fn(*args,**kwargs);order=np.asarray(train_ds.order,dtype=np.int64)
            if not np.array_equal(np.sort(order),np.arange(len(tr['y']))):raise ValueError('Training sample exposure differs')
            history.append(dict(epoch=int(args[4]),train_ce=loss,train_acc=acc/100,lr=args[3].param_groups[0]['lr'],order_sha256=hashlib.sha256(order.tobytes()).hexdigest()))
            return loss,acc
        def val_trace(*args,**kwargs):
            batches=[]
            handle=model.register_forward_hook(lambda mod,ins,result:batches.append(result.detach().cpu().numpy().copy()))
            try:loss,acc=val_fn(*args,**kwargs)
            finally:handle.remove()
            logits=np.concatenate(batches);scores=score(logits,va,meta['rx_names']);stats=api.get_k_stats(model)
            history[-1].update(val_packet_ce=loss,val_packet_acc=acc/100,val_rx_ce=scores['summary']['ce'],val_rx_acc=scores['summary']['acc'],argmax_k=stats[0],soft_k=stats[1],top_prob=stats[2])
            with (out/'history.csv').open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(history[-1]),lineterminator='\n');writer.writeheader();writer.writerows(history)
            write(out/'live.json',dict(time=now(),status='RUNNING',**history[-1]))
            print('S20 ASK fold',fold,'epoch',history[-1]['epoch'],'source_packet_CE',round(loss,5),flush=True)
            return loss,acc
        api.train_epoch,api.evaluate_epoch=train_trace,val_trace
        started=time.perf_counter()
        with (out/'stdout.log').open('w',encoding='utf-8',newline='\n') as log,redirect_stdout(log),redirect_stderr(log):
            best,trace=api.train_and_evaluate(model,train_loader,val_loader,epochs=200,save_path=str(out/'model.pt'),
                                            log_path=str(out/'k_trace.txt'),device='cuda')
        model.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True));model.cuda().eval()
        logits=predict(model,va['x']);save_predictions(out/'source_val_predictions.npz',logits,va)
        bestrow=min(history,key=lambda r:r['val_packet_ce']);rxrow=min(history,key=lambda r:r['val_rx_ce'])
        result=dict(time=now(),status='COMPLETE',fold=fold,arm='ASK',epochs=len(history),best_epoch=bestrow['epoch'],best_source_packet_ce=best,
                    diagnostic_rx_ce_best_epoch=rxrow['epoch'],source_validation=score(logits,va,meta['rx_names']),
                    model_sha256=sha(out/'model.pt'),source_predictions_sha256=sha(out/'source_val_predictions.npz'),
                    seconds=time.perf_counter()-started,k_history=trace,selected_checkpoint_cutoff=api.get_k_stats(model),
                    historical_checkpoint_or_held_data_used=False)
        write(out/'metrics.json',result);write(out/'attempt.json',dict(time=now(),status='COMPLETE',fold=fold))
        write(out/'live.json',dict(time=now(),status='COMPLETE',epochs=len(history),best_epoch=bestrow['epoch']))
        print('S20 ASK fold',fold,'COMPLETE epochs',len(history),'seconds',round(result['seconds'],1),flush=True)
    except Exception:
        write(out/'failure.json',dict(time=now(),traceback=traceback.format_exc(),automatic_retry=False));raise

def seal():
    validate();p=HERE/'seal_pilot.json'
    if p.exists():committed(p);return
    refs={};metadata={}
    reuse=read(HERE/'REUSE_AUDIT.json')['s19_baselines']
    for fold in range(4):
        record=reuse[fold];old=olddir(fold);dest=outdir(fold,'B0');dest.mkdir(parents=True,exist_ok=True)
        for name,h in [('config.json',record['config_sha256']),('metrics.json',record['metrics_sha256'])]:
            if sha(old/name)!=h:raise ValueError('Frozen baseline metadata changed')
        source=record['checkpoint']
        if sha(source['path'])!=source['sha256']:raise ValueError('Frozen HStem weights changed')
        write(dest/'reference.json',dict(fold=fold,arm='B0',reused=True,**record))
        refs[f'B0_f{fold}']=source
        for name in ['config.json','metrics.json','reference.json']:
            path=dest/name if name=='reference.json' else old/name
            metadata[str(path)]=sha(path)
        ask=outdir(fold,'ASK');metric=read(ask/'metrics.json')
        if metric['status']!='COMPLETE' or sha(ask/'model.pt')!=metric['model_sha256']:raise ValueError('Incomplete ASK checkpoint')
        refs[f'ASK_f{fold}']=dict(fold=fold,arm='ASK',path=str(ask/'model.pt'),sha256=metric['model_sha256'])
        for name in ['config.json','metrics.json']:metadata[str(ask/name)]=sha(ask/name)
    write(p,dict(time=now(),all_checkpoints_complete=True,new_fits=4,reused_B0=4,checkpoints=refs,metadata_sha256=metadata,
                 baseline_held_predictions_preexisting=True,new_ask_held_source_read=False,real_target_access=False))
    subprocess.run(['git','-C',str(ROOT),'add','--',HERE.relative_to(ROOT).as_posix()],check=True)
    subprocess.run(['git','-C',str(ROOT),'-c','user.name=Codex','-c','user.email=codex@local.invalid','commit','-m','S20 seal four official ASKNet source models before paired held-source evaluation','--',HERE.relative_to(ROOT).as_posix()],check=True)

def evaluate():
    validate();committed(HERE/'seal_pilot.json');sealed=read(HERE/'seal_pilot.json')
    marker=HERE/'held_attempt.json'
    if marker.exists():
        if read(marker)['status']=='COMPLETE':return
        raise RuntimeError('Interrupted held evaluation retained; no automatic retry')
    for path,h in sealed['metadata_sha256'].items():
        if sha(path)!=h:raise ValueError('Sealed metadata changed')
    write(marker,dict(time=now(),status='RUNNING'))
    data_module=configure_data()
    for fold in range(4):
        held,names,sources=data_module.load_held(fold)
        old=olddir(fold);previous=read(old/'held_metrics.json');p=old/'held_predictions.npz'
        if sha(p)!=previous['prediction_sha256']:raise ValueError('Frozen baseline prediction mismatch')
        access(fold=fold,arm='B0',kind='baseline_cached_predictions',path=str(p),sha256=sha(p),seal_sha256=sha(HERE/'seal_pilot.json'))
        with np.load(p,allow_pickle=False) as cache:
            for key in ['y','rx','day']:
                if not np.array_equal(cache[key],held[key]):raise ValueError('Held input/label order differs from frozen B0')
            baseline=score(cache['logits'],held,names)
        if baseline!=previous['held_source'] or sources!=previous['held_sources']:raise ValueError('Frozen B0 source/metric parity failed')
        write(outdir(fold,'B0')/'held_metrics.json',dict(time=now(),fold=fold,arm='B0',held_source=baseline,reused=True,
              prediction_path=str(p),prediction_sha256=previous['prediction_sha256'],rx_names=names,held_sources=sources))
        api=official();api.setup_seed(read(olddir(fold)/'config.json')['data_seed']);model=api.build_model(CONF,10).cuda().eval()
        ref=sealed['checkpoints'][f'ASK_f{fold}']
        if sha(ref['path'])!=ref['sha256']:raise ValueError('Unsealed ASK weight')
        access(fold=fold,arm='ASK',kind='sealed_checkpoint',path=ref['path'],sha256=ref['sha256'],seal_sha256=sha(HERE/'seal_pilot.json'))
        model.load_state_dict(torch.load(ref['path'],map_location='cpu',weights_only=True));logits=predict(model,held['x'])
        out=outdir(fold,'ASK');save_predictions(out/'held_predictions.npz',logits,held)
        write(out/'held_metrics.json',dict(time=now(),fold=fold,arm='ASK',held_source=score(logits,held,names),
              model_sha256=ref['sha256'],prediction_path=str(out/'held_predictions.npz'),prediction_sha256=sha(out/'held_predictions.npz'),rx_names=names,held_sources=sources))
        print('S20 held-source fold',fold,'paired comparison complete',flush=True)
    write(marker,dict(time=now(),status='COMPLETE'))

def decide():
    validate();p=HERE/'decision.json'
    if p.exists():return
    rows=[read(outdir(f,a)/'held_metrics.json') for f in range(4) for a in ARMS]
    write(p,dict(time=now(),**decision(rows),new_asknet_fits=4,reused_hstem_models=4))
    print(read(p)['status'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','run','fit','seal','held','decide']);args=parser.parse_args()
    if args.phase=='prepare':preparation()
    else:
        validate()
        if args.phase in ['run','fit']:
            for f in range(4):fit(f)
        if args.phase in ['run','seal']:seal()
        if args.phase in ['run','held']:evaluate()
        if args.phase in ['run','decide']:decide()
