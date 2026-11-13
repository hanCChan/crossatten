"""Prespecified W3-B. Frozen inference and shallow statistics; no neural fit entry."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime,timezone
import numpy as np
from selective_pickle import read_rows
from transport import METHODS,pca_basis,fit,predict,transport_error,separation_ratio,prototype_metrics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
ART=HERE/'artifacts'
SIMPLE=('identity','shift','diagonal','rank2')
CHECKPOINT_DIR=Path('D:/crossatten/worktrees/training-instance-attribution/research/standard_method_competition_c/artifacts/V0_S0_wd1e3_ERM_EMA')
CHECKPOINTS={
    'EMA':dict(path=str(CHECKPOINT_DIR/'C1_selected.pt'),sha256='8aa97fb78d5a7992b081c274911644d02b51e66b7385959ab76e9e63e9eafb79'),
    'ERM':dict(path=str(CHECKPOINT_DIR/'C0_selected.pt'),sha256='8503320e833d30c3f9ab565846faf228ab4231b9f4c737648ccc41cdbb592145')}


def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(16<<20):h.update(b)
    return h.hexdigest()
def csv_write(p,rows):
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def roles(manifest):
    tx=sorted({c['tx'] for c in manifest['cells']})
    ordered=sorted(tx,key=lambda t:hashlib.sha256(('W3-TX-v1|'+t).encode()).hexdigest())
    assert len(ordered)==96
    a,b=ordered[:48],ordered[48:];contexts=[]
    for direction,(f,h) in enumerate(((a,b),(b,a))):
        for fd,ed in ((1,2),(2,1)):
            contexts.append(dict(id=f'T{direction}_D{fd}toD{ed}',fit_TX=f,held_TX=h,fit_day=fd,eval_day=ed))
    return dict(TX=tx,RX=manifest['source_RX'],days=[1,2],contexts=contexts,query_rows=[16,32],prototype_rows=[0,16])


def helper():
    spec=importlib.util.spec_from_file_location('W3_frozen_C',ROOT/'research/standard_method_competition_c/engine.py')
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    return c.A


def features(model,x):
    h=model.pre_ln(model.encoder.prepare_tokens(model.tokens(x)))
    for block in model.encoder.blocks:h=block(h)
    return model.encoder.final_norm(h).mean(1)


def load_model(a,name):
    import torch
    item=CHECKPOINTS[name]
    if sha(item['path'])!=item['sha256']:raise ValueError('Checkpoint is not the frozen reference')
    model=a.build_model(24101)
    ck=torch.load(item['path'],map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model']);model.eval();model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters())==82426
    return model


def preflight():
    import torch
    a=helper();a.setup(24101);out={}
    for name in CHECKPOINTS:
        model=load_model(a,name);before=a.semantic_hash(model.state_dict())
        x=torch.randn(7,2,256,device='cuda')
        with torch.inference_mode():
            z=features(model,x);logits=model.classifier(z);reference=model(x)
        assert torch.equal(logits,reference) and list(z.shape)==[7,64]
        assert a.semantic_hash(model.state_dict())==before
        out[name]=dict(checkpoint=CHECKPOINTS[name],feature_head_equals_original_forward=True,
                       shape=list(z.shape),state_sha256=before,unchanged=True)
        del model
    write(HERE/'B_FEATURE_PREFLIGHT.json',dict(status='PASS',only_synthetic_IQ=True,official_IQ_rows_decoded=0,models=out))
    print('PASS synthetic feature extraction and checkpoint identity',flush=True)


def validate_registration():
    reg=read(HERE/'B_EXECUTION_REGISTRATION.json')
    paths=[]
    for rel,digest in reg['files_sha256'].items():
        p=ROOT/rel
        if sha(p)!=digest:raise ValueError('Registered file differs: '+rel)
        paths.append(rel)
    paths.append((HERE/'B_EXECUTION_REGISTRATION.json').relative_to(ROOT).as_posix())
    subprocess.run(['git','-C',str(ROOT),'ls-files','--error-unmatch','--',*paths],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(ROOT),'diff','--quiet','HEAD','--',*paths],check=True)
    assert reg['maximum_new_neural_fits']==0 and reg['W3_C_authorized'] is False
    return reg


def extract():
    import torch
    reg=validate_registration();manifest=read(HERE/'B_SOURCE_MANIFEST.json');rr=read(HERE/'B_ROLES.json')
    if rr!=roles(manifest):raise ValueError('Committed roles do not match manifest')
    if (HERE/'B_FEATURE_ATTEMPT.json').exists():raise RuntimeError('An IQ feature attempt already exists; no silent rerun')
    if Path(manifest['path']).stat().st_size!=manifest['bytes'] or sha(manifest['path'])!=manifest['sha256']:
        raise ValueError('Official payload differs')
    a=helper();a.setup(24101)
    # Load and check both weights before admitting official waveform values.
    models={n:load_model(a,n) for n in CHECKPOINTS}
    states={n:a.semantic_hash(m.state_dict()) for n,m in models.items()}
    write(HERE/'B_FEATURE_ATTEMPT.json',dict(started=now(),registration_sha256=sha(HERE/'B_EXECUTION_REGISTRATION.json'),
        registered_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        neural_fits=0,authorized_IQ_rows=36864,status='OFFICIAL_SOURCE_FEATURE_READOUT_STARTED'))
    start=time.perf_counter();cells=manifest['cells'];allowed={c['key'] for c in cells}
    txid={t:i for i,t in enumerate(rr['TX'])};rxid={r:i for i,r in enumerate(rr['RX'])}
    x=np.empty((len(cells)*32,2,256),np.float32);access=[]
    for i,c in enumerate(cells):
        assert c['eq']==0 and c['day'] in (1,2) and c['tx'] not in manifest['excluded_TX']
        assert c['rx'] in rr['RX'] and c['rows']==[0,32]
        raw=read_rows(manifest['path'],c,allowed)
        v=raw.transpose(0,2,1);power=(v[:,0]**2+v[:,1]**2).mean(-1)
        if not np.isfinite(v).all() or not np.isfinite(power).all() or np.any(power<=0):raise ValueError('Invalid official waveform')
        x[i*32:(i+1)*32]=(v/np.sqrt(power)[:,None,None]).astype(np.float32)
        access.append(dict(key=c['key'],rows=c['rows'],offset=c['offset'],bytes=raw.nbytes,
                           selected_raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest()))
    decoded=time.perf_counter()-start;result={};proof={}
    for name,model in models.items():
        t=time.perf_counter();chunks=[]
        with torch.inference_mode():
            for i in range(0,len(x),128):chunks.append(features(model,torch.from_numpy(x[i:i+128]).cuda()).cpu().numpy())
        flat=np.concatenate(chunks);assert flat.shape==(36864,64) and np.isfinite(flat).all()
        arranged=np.empty((96,6,2,32,64),np.float32)
        for i,c in enumerate(cells):arranged[txid[c['tx']],rxid[c['rx']],c['day']-1]=flat[i*32:(i+1)*32]
        after=a.semantic_hash(model.state_dict());assert states[name]==after
        assert sha(CHECKPOINTS[name]['path'])==CHECKPOINTS[name]['sha256']
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        assert all(not m.training for m in model.modules())
        result[name]=arranged
        proof[name]=dict(before_state_sha256=states[name],after_state_sha256=after,unchanged=True,
            feature_seconds=time.perf_counter()-t,feature_shape=list(arranged.shape),feature_finite=True,
            checkpoint_sha256=CHECKPOINTS[name]['sha256'],no_gradients=True,every_module_eval=True)
        print(name+' frozen features complete',flush=True)
    np.savez_compressed(ART/'B_FEATURES.npz',TX=np.array(rr['TX']),RX=np.array(rr['RX']),days=np.array([1,2]),**result)
    write(HERE/'B_IQ_ACCESS_LOG.json',dict(status='PASS',cells=access,cells_read=len(access),
        rows_read=len(x),raw_numeric_bytes_read=sum(v['bytes'] for v in access),
        forbidden_RX_day_TX_rows_read=0,model_passes_per_source_packet=2,
        normalization='Original float64 per-packet IQ power normalization, then float32'))
    write(HERE/'B_FEATURE_RECEIPT.json',dict(status='PASS_FROZEN_SOURCE_ONLY_FEATURES',completed=now(),
        feature_cache_sha256=sha(ART/'B_FEATURES.npz'),source_manifest_sha256=sha(HERE/'B_SOURCE_MANIFEST.json'),
        registration_sha256=sha(HERE/'B_EXECUTION_REGISTRATION.json'),neural_fits=0,W3_C_authorized=False,
        IQ_decode_seconds=decoded,elapsed_seconds=time.perf_counter()-start,models=proof))
    print('PASS source-only features; new neural fits=0',flush=True)


def bootstrap(control,candidate):
    # One cluster per held-out TX, after averaging all its RX pairs and both days.
    control=np.asarray(control);candidate=np.asarray(candidate)
    rng=np.random.default_rng(31013);idx=rng.integers(len(control),size=(20000,len(control)))
    diff=(control-candidate)[idx].mean(1)
    rel=1-candidate[idx].mean(1)/control[idx].mean(1)
    return dict(TX_clusters=len(control),replicates=20000,seed=31013,
        mean_paired_error_reduction=float((control-candidate).mean()),
        paired_error_reduction_CI95=np.quantile(diff,[.025,.975]).tolist(),
        relative_error_reduction=float(1-candidate.mean()/control.mean()),
        relative_error_reduction_CI95=np.quantile(rel,[.025,.975]).tolist(),
        conditional_on='Frozen extractor, TX split, source RX and two observed source dates; not new-RX uncertainty')


def evaluate_arrays(arrays,rr):
    txid={t:i for i,t in enumerate(rr['TX'])};r=rr['RX']
    pairs=[(i,j) for i in range(6) for j in range(6) if i!=j]
    records=[];geometry=[];operators={};summaries={};fits=0;pca_fits=0
    for name,z in arrays.items():
        mu=z.astype(np.float64).mean(3)
        errs=np.empty((4,30,6,48));accs=np.empty_like(errs);margins=np.empty_like(errs);geom=np.empty((4,30,6))
        for ci,context in enumerate(rr['contexts']):
            fi=[txid[t] for t in context['fit_TX']];hi=[txid[t] for t in context['held_TX']]
            fd,ed=context['fit_day']-1,context['eval_day']-1
            basis=pca_basis(mu[fi,:,fd].reshape(-1,64));pca_fits+=1
            operators[f'{name}_{ci}_basis']=basis
            for pi,(s,d) in enumerate(pairs):
                xf,yf=mu[fi,s,fd],mu[fi,d,fd];xh,yh=mu[hi,s,ed],mu[hi,d,ed]
                prototypes=z[hi,d,ed,:16].astype(np.float64).mean(1)
                query=z[hi,s,ed,16:].astype(np.float64)
                for mi,method in enumerate(METHODS):
                    op=fit(method,xf,yf,basis,31013+ci*100+pi)
                    if method!='identity':fits+=1
                    pred=predict(op,xh);errors=transport_error(pred,yh)
                    accuracies,margin=prototype_metrics(predict(op,query.reshape(-1,64)).reshape(48,16,64),prototypes)
                    ratio=separation_ratio(xh,pred)
                    errs[ci,pi,mi]=errors;accs[ci,pi,mi]=accuracies;margins[ci,pi,mi]=margin;geom[ci,pi,mi]=ratio
                    prefix=f'{name}_{ci}_{pi}_{method}'
                    for key,val in op.items():
                        if isinstance(val,np.ndarray):operators[prefix+'_'+key]=val
                    geometry.append(dict(model=name,context=context['id'],source_RX=r[s],destination_RX=r[d],
                        method=method,median_TX_squared_distance_ratio=ratio))
                    for ti,tx in enumerate(context['held_TX']):
                        records.append(dict(model=name,context=context['id'],source_RX=r[s],destination_RX=r[d],method=method,
                            held_TX=tx,transport_error=float(errors[ti]),labeled_prototype_accuracy=float(accuracies[ti]),
                            normalized_margin=float(margin[ti])))
            print(name+' '+context['id']+' complete',flush=True)
        control_ids=errs[:,:,:4].mean((1,3)).argmin(1)
        candidate=METHODS.index('rank8');null=METHODS.index('shuffled_rank8')
        ctrl=np.stack([errs[c,:,control_ids[c],:] for c in range(4)])
        ctrl_acc=np.stack([accs[c,:,control_ids[c],:] for c in range(4)])
        ctrl_margin=np.stack([margins[c,:,control_ids[c],:] for c in range(4)])
        cand=errs[:,:,candidate];null_err=errs[:,:,null]
        clustered={};contexts=[]
        for ci,c in enumerate(rr['contexts']):
            for ti,t in enumerate(c['held_TX']):
                clustered.setdefault(t,[]).append([float(ctrl[ci,:,ti].mean()),float(cand[ci,:,ti].mean()),float(null_err[ci,:,ti].mean())])
            contexts.append(dict(id=c['id'],strongest_simple_control=METHODS[int(control_ids[ci])],
                control_error=float(ctrl[ci].mean()),rank8_error=float(cand[ci].mean()),
                relative_error_reduction=float(1-cand[ci].mean()/ctrl[ci].mean()),
                prototype_accuracy_delta_pp=float(100*(accs[ci,:,candidate]-ctrl_acc[ci]).mean()),
                normalized_margin_delta=float((margins[ci,:,candidate]-ctrl_margin[ci]).mean())))
        clusters=np.array([np.mean(clustered[t],axis=0) for t in rr['TX']])
        boot=bootstrap(clusters[:,0],clusters[:,1]);boot_null=bootstrap(clusters[:,2],clusters[:,1])
        pairrel=1-cand.mean((0,2))/ctrl.mean((0,2))
        geom_c=geom[:,:,candidate]
        summary=dict(method_means={m:dict(transport_error=float(errs[:,:,mi].mean()),
            labeled_prototype_accuracy=float(accs[:,:,mi].mean()),normalized_margin=float(margins[:,:,mi].mean()),
            median_pair_context_TX_squared_distance_ratio=float(np.median(geom[:,:,mi])),
            minimum_pair_context_TX_squared_distance_ratio=float(geom[:,:,mi].min())) for mi,m in enumerate(METHODS)},
            contexts=contexts,strongest_simple_control_error=float(ctrl.mean()),rank8_error=float(cand.mean()),
            relative_error_reduction=float(1-cand.mean()/ctrl.mean()),
            positive_directed_RX_pairs=int((pairrel>0).sum()),total_directed_RX_pairs=30,
            pair_relative_error_reductions={r[s]+'->'+r[d]:float(pairrel[p]) for p,(s,d) in enumerate(pairs)},
            vs_strongest_simple_control=boot,vs_shuffled_rank8=boot_null,
            median_TX_squared_distance_ratio=float(np.median(geom_c)),minimum_TX_squared_distance_ratio=float(geom_c.min()),
            labeled_prototype_accuracy_delta_pp=float(100*(accs[:,:,candidate]-ctrl_acc).mean()),
            worst_context_prototype_accuracy_delta_pp=min(c['prototype_accuracy_delta_pp'] for c in contexts),
            normalized_margin_delta=float((margins[:,:,candidate]-ctrl_margin).mean()))
        summaries[name]=summary
    ema=summaries['EMA'];erm=summaries['ERM']
    gates=dict(relative_improvement_ge_15pct=ema['relative_error_reduction']>=.15,
        all_four_contexts_positive=all(c['relative_error_reduction']>0 for c in ema['contexts']),
        at_least_20_of_30_RX_pairs_positive=ema['positive_directed_RX_pairs']>=20,
        paired_TX_CI_vs_simple_lower_positive=ema['vs_strongest_simple_control']['paired_error_reduction_CI95'][0]>0,
        paired_TX_CI_vs_shuffled_lower_positive=ema['vs_shuffled_rank8']['paired_error_reduction_CI95'][0]>0,
        median_geometry_ratio_ge_0p9=ema['median_TX_squared_distance_ratio']>=.9,
        minimum_geometry_ratio_ge_0p8=ema['minimum_TX_squared_distance_ratio']>=.8,
        prototype_accuracy_delta_ge_minus1pp=ema['labeled_prototype_accuracy_delta_pp']>=-1,
        every_context_accuracy_delta_ge_minus5pp=ema['worst_context_prototype_accuracy_delta_pp']>=-5,
        normalized_margin_delta_ge_minus0p01=ema['normalized_margin_delta']>=-.01,
        ERM_sensitivity_error_improvement_positive=erm['relative_error_reduction']>0)
    result=dict(status='PASS_W3B_SOURCE_STATISTICAL_TRANSFER' if all(gates.values()) else 'STOP_W3B_LINEAR_TRANSPORT_HYPOTHESIS',
        neural_fits=0,linear_operator_fits=fits,PCA_fits=pca_fits,identity_evaluations=240,
        W3_C_authorized=False,gates=gates,models=summaries,
        claim_boundary='Source-only receiver-associated changes, not identified hardware, deployable accuracy, algorithm novelty, or unseen-RX improvement')
    return result,records,geometry,operators


def analyze():
    validate_registration();receipt=read(HERE/'B_FEATURE_RECEIPT.json')
    if sha(ART/'B_FEATURES.npz')!=receipt['feature_cache_sha256']:raise ValueError('Frozen features differ')
    if (HERE/'B_RESULT.json').exists():raise RuntimeError('Scientific analysis already exists; use audit to replay')
    rr=read(HERE/'B_ROLES.json')
    with np.load(ART/'B_FEATURES.npz',allow_pickle=False) as f:
        assert f['TX'].tolist()==rr['TX'] and f['RX'].tolist()==rr['RX'] and f['days'].tolist()==rr['days']
        arrays={name:f[name] for name in ('EMA','ERM')}
    result,records,geometry,operators=evaluate_arrays(arrays,rr)
    csv_write(HERE/'B_HELD_TX_METRICS.csv',records);csv_write(HERE/'B_PAIR_GEOMETRY.csv',geometry)
    np.savez_compressed(ART/'B_OPERATORS.npz',**operators)
    result.update(completed=now(),feature_cache_sha256=receipt['feature_cache_sha256'],
        registration_sha256=sha(HERE/'B_EXECUTION_REGISTRATION.json'),
        output_sha256={p:sha(HERE/p) for p in ('B_HELD_TX_METRICS.csv','B_PAIR_GEOMETRY.csv','artifacts/B_OPERATORS.npz')})
    write(HERE/'B_RESULT.json',result)
    print(json.dumps(dict(status=result['status'],gates=result['gates'],EMA_relative_error_reduction=result['models']['EMA']['relative_error_reduction']),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('preflight','extract','analyze'));args=parser.parse_args()
    {'preflight':preflight,'extract':extract,'analyze':analyze}[args.command]()
