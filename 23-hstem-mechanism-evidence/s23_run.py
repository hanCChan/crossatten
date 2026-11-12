"""Zero-fit checkpoint analysis on authorized source-validation packets only."""
import csv
import itertools
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from s23_core import HERE,ROOT,read,write,sha,now,validate,baseline_path
from s23_data import source_val,probe_ids,roles,PROTOCOL
from s23_math import build,operator,materialize_s1,fold_h0,shifted,difference,equivalence
from xa.frontends import band_limit,rms_normalize
from xa.models3 import legacy5_views,IB
from xa.evaluate import summarize
from metrics import rx_metrics

ARMS=['P0','S1','H1','H0','S2','H2']

def save_csv(name,rows):
    if not rows:return
    with (HERE/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');writer.writeheader();writer.writerows(rows)

def setup(seed):
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True);torch.manual_seed(seed)

def frontend(x):return band_limit(rms_normalize(x),IB)

@torch.no_grad()
def predict(model,x,fft_lock=None):
    model.eval();offset=0;handle=None
    if fft_lock is not None:
        def override(module,args):
            nonlocal offset
            v=args[0].clone();batch=len(v);v[:,2:3]=fft_lock[offset:offset+batch];offset+=batch;return (v,)
        handle=model.net.embedder.cnn_stem.register_forward_pre_hook(override)
    try:return torch.cat([model(x[i:i+128]) for i in range(0,len(x),128)])
    finally:
        if handle is not None:handle.remove()

def score(logits,y,rx):
    a=logits.detach().cpu().numpy();labels=np.asarray(y)
    per={str(r):rx_metrics(a[np.asarray(rx)==r],labels[np.asarray(rx)==r]) for r in np.unique(rx)}
    return dict(summary=summarize(per),per_rx=per)

def emit_metrics(rows,model_tag,fold,seed,condition,logits,y,rx):
    for receiver in range(5):
        ids=np.asarray(rx)==receiver;a=logits[torch.as_tensor(ids,device=logits.device)];labels=torch.as_tensor(np.asarray(y)[ids],device=a.device)
        m=rx_metrics(a.cpu().numpy(),labels.cpu().numpy());other=a.clone();other[torch.arange(len(a),device=a.device),labels]=-torch.inf
        rows.append(dict(arm=model_tag,fold=fold,seed=seed,condition=condition,rx=roles(fold)['source'][receiver],accuracy_pct=100*m['acc'],macro_f1_pct=100*m['macro_f1'],ce=float(F.cross_entropy(a,labels)),margin=float((a[torch.arange(len(a),device=a.device),labels]-other.max(1).values).mean()),packets=len(a)))

def spectral_rows(arm,fold,seed,model):
    layer=model.net.embedder.cnn_stem.tokenize[0]
    if arm=='S1':weight=layer.smoothed_weight()
    elif arm=='H1':weight=layer.expanded_weight()
    else:weight=layer.weight
    power=torch.fft.rfft(weight.detach().double(),n=1024).abs().square().mean((0,1)).cpu().numpy()
    normalized=power/power.sum();freq=np.fft.rfftfreq(1024)
    spectrum=[dict(arm=arm,fold=fold,seed=seed,frequency_cycles_per_frontend_sample=float(f),mean_power=float(p),normalized_power=float(n)) for f,p,n in zip(freq,power,normalized)]
    bandrow=dict(arm=arm,fold=fold,seed=seed,kernel_size=weight.shape[-1],weight_rms=float(weight.detach().square().mean().sqrt()),low_energy=float(normalized[freq<.125].sum()),mid_energy=float(normalized[(freq>=.125)&(freq<.25)].sum()),high_energy=float(normalized[freq>=.25].sum()))
    return spectrum,bandrow

def local_gradients(model,x,y,arm,fold,seed):
    model.eval();layer=model.net.embedder.cnn_stem.tokenize[0];box={};original=None
    if arm=='S1':
        original=layer.smoothed_weight
        def capture():
            value=original();box['effective']=value;return value
        layer.smoothed_weight=capture
    names=[];params=[]
    for name,p in model.named_parameters():names.append(name);params.append(p)
    logits=model(x);loss=F.cross_entropy(logits,y)
    gradients=torch.autograd.grad(loss,params+([box['effective']] if arm=='S1' else []),allow_unused=True)
    if original is not None:layer.smoothed_weight=original
    rows=[]
    for name,p,g in zip(names,params,gradients):
        rows.append(dict(arm=arm,fold=fold,seed=seed,parameter=name,gradient_rms=0.0 if g is None else float(g.square().mean().sqrt()),weight_rms=float(p.detach().square().mean().sqrt()),probe_ce=float(loss.detach()),scope='Checkpoint-local eval CE sensitivity;not historical training gradients'))
    chain=None
    if arm=='S1':
        index=names.index('net.embedder.cnn_stem.tokenize.0.weight');actual=gradients[index].detach().double()
        expected=gradients[-1].detach().double().reshape(-1,256)@operator().to(x.device)
        chain=dict(arm=arm,fold=fold,seed=seed,**difference(actual,expected.reshape_as(actual)),scope='grad_w = grad_v @ T;row convention;not an AdamW update')
    del gradients,logits,loss
    return rows,chain

@torch.no_grad()
def snapshot(model,x,arm,fft_lock=None):
    values={};handles=[]
    def feature_hook(name):
        def capture(module,args,output):values[name]=output.detach().clone()
        return capture
    stem=model.net.embedder.cnn_stem
    def input_capture(module,args):
        v=args[0]
        if fft_lock is not None:v=v.clone();v[:,2:3]=fft_lock
        values['views']=v.detach().clone()
        if fft_lock is not None:return (v,)
    handles.append(stem.register_forward_pre_hook(input_capture))
    taps={'stem':stem.stem,'tokens':stem.tokenize,'pre_ln':model.pre_ln,'final_norm':model.net.encoder.final_norm}
    if arm=='H0':taps.update({'stage1':stem.tokenize[2],'stage2':stem.tokenize[5],'stage3':stem.tokenize[8]})
    for name,module in taps.items():handles.append(module.register_forward_hook(feature_hook(name)))
    try:values['logits']=model(x).detach()
    finally:
        for handle in handles:handle.remove()
    return values

def layer_rows(base,changed,arm,fold,seed,condition,shift,kind):
    rows=[]
    for name,value in changed.items():
        b=base[name];metrics=difference(value,b);aligned=None
        stride={'stem':1,'stage1':4,'stage2':16,'stage3':64}.get(name)
        if kind=='circular' and stride and shift%stride==0:aligned=difference(value.roll(-shift//stride,-1),b)['relative_l2']
        rows.append(dict(arm=arm,fold=fold,seed=seed,condition=condition,layer=name,**metrics,integer_stride_aligned_relative_l2=aligned))
    for channel,label in [(0,'I'),(1,'Q'),(2,'FFT_magnitude'),(3,'amplitude'),(4,'time_phase')]:
        a,b=changed['views'][:,channel],base['views'][:,channel]
        align=0 if channel==2 else -shift
        aligned=difference(a.roll(align,-1),b)['relative_l2'] if kind=='circular' else None
        rows.append(dict(arm=arm,fold=fold,seed=seed,condition=condition,layer='view_'+label,**difference(a,b),integer_stride_aligned_relative_l2=aligned))
    return rows

@torch.no_grad()
def latency(model,x,arm,fold,seed,variant):
    rows=[]
    for batch in [1,128]:
        probe=x[:batch]
        for repeat in range(3):
            for _ in range(10):model(probe)
            times=[]
            for _ in range(50):
                a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)];a.record();model(probe);b.record();b.synchronize();times.append(a.elapsed_time(b))
            rows.append(dict(arm=arm,fold=fold,seed=seed,variant=variant,batch=batch,repeat=repeat,p50_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95))))
    return rows

def trajectories(reg):
    raw=[];summaries=[]
    for key,ref in reg['references'].items():
        path=Path(ref['path']);m=read(path/'metrics.json')
        with (path/'history.csv').open(encoding='utf-8') as f:history=list(csv.DictReader(f))
        assert len(history)==200
        best=min(history,key=lambda r:float(r['source_val_ce']))
        assert int(best['epoch'])==m['best_epoch'] and float(best['source_val_ce'])==m['best_source_ce']
        for row in history:raw.append(dict(model=key,**row))
        summaries.append(dict(model=key,best_epoch=m['best_epoch'],best_source_ce=m['best_source_ce'],final_source_ce=float(history[-1]['source_val_ce']),first_ce_le_point2=next((int(r['epoch']) for r in history if float(r['source_val_ce'])<=.2),None),last10_source_ce_sd=float(np.std([float(r['source_val_ce']) for r in history[-10:]])),wall_seconds=m['wall_seconds']))
    save_csv('archived_trajectories.csv',raw);save_csv('convergence_summary.csv',summaries)

@torch.no_grad()
def signal_audit(data,fold):
    rows=[];frontrows=[]
    x=torch.as_tensor(data['x'],device='cuda');base_front=frontend(x)
    for amount,kind in itertools.product([-4,-1,1,4],['zero','circular']):
        changed=shifted(x,amount,kind);power=x.square().sum((1,2));loss=1-changed.square().sum((1,2))/power
        a=frontend(changed);spectrum=torch.fft.fft(torch.complex(x[:,0],x[:,1]),dim=-1)
        newspec=torch.fft.fft(torch.complex(changed[:,0],changed[:,1]),dim=-1)
        freq=torch.fft.fftfreq(8192,device='cuda');expected=spectrum*torch.exp(-2j*torch.pi*freq*amount)
        for rx in range(5):
            ids=torch.as_tensor(data['rx']==rx,device='cuda')
            seam=(x[ids,:,0]-x[ids,:,-1]).square().sum(1).sqrt();internal=(x[ids,:,1:]-x[ids,:,:-1]).square().sum(1).sqrt()
            rows.append(dict(fold=fold,rx=roles(fold)['source'][rx],kind=kind,shift=amount,mean_energy_fraction_lost=float(loss[ids].mean()),max_energy_fraction_lost=float(loss[ids].max()),boundary_jump_median=float(seam.median()),internal_step_median=float(internal.median()),within_packet_original_rows='source-val only',adjacent_capture_context_available=False))
        frontrows.append(dict(fold=fold,kind=kind,shift=amount,bandlimit_commutation_error=difference(a,shifted(base_front,amount,kind))['relative_l2'],fft_magnitude_change=difference(newspec.abs(),spectrum.abs())['relative_l2'],fourier_shift_theorem_error=difference(newspec,expected)['relative_l2'] if kind=='circular' else None))
    del x,base_front
    return rows,frontrows

def run():
    reg=validate();setup(23023)
    # Runtime guard: any accidental optimizer construction fails immediately.
    def forbidden_optimizer(*args,**kwargs):raise RuntimeError('S23 forbids optimizer creation and new training')
    torch.optim.Optimizer.__init__=forbidden_optimizer
    results={k:[] for k in ['source_recovery','equivalence','conversion_metrics','kernel_spectra','kernel_bands','local_gradients','gradient_chain','timing_metrics','layer_responses','bn_statistics','activation_distributions','conversion_latency','boundary_audit','frontend_audit','tx_rx_timing','fft_lock_effects']}
    trajectories(reg)
    T=operator();singular=torch.linalg.svdvals(T);M=T@T.T;eigen=torch.linalg.eigvalsh(M).flip(0)
    n=torch.arange(256,dtype=torch.float64);U=torch.exp(2j*torch.pi*n[:,None]*n[None,:]/256)/16;FM=U.conj().T@M.to(torch.complex128)@U
    write(HERE/'OPERATOR.json',dict(rank=int(torch.linalg.matrix_rank(T)),singular_min=float(singular.min()),singular_max=float(singular.max()),tt_eigen_min=float(eigen.min()),tt_eigen_max=float(eigen.max()),tt_fourier_offdiagonal_energy_fraction=float((FM-torch.diag(FM.diag())).abs().square().sum()/FM.abs().square().sum()),scope='v=T w;SGD preconditioner only,not exact AdamW update'))
    save_csv('operator_spectrum.csv',[dict(index=i,singular_value=float(singular[i]),tt_eigenvalue=float(eigen[i]),fourier_direction_gain=float(FM.diag()[i].real),cycles_per_weight_index=i/256) for i in range(256)])
    for fold in [0,1]:
        data=source_val(fold);ids=probe_ids(data);x=torch.as_tensor(data['x'],device='cuda');y=torch.as_tensor(data['y'],device='cuda');rx=torch.as_tensor(data['rx'],device='cuda');probe=x[ids];probe_y=y[ids]
        br,fr=signal_audit(data,fold);results['boundary_audit']+=br;results['frontend_audit']+=fr
        for arm,seed in itertools.product(ARMS,[201,202,203]):
            write(HERE/'STATUS.json',dict(time=now(),status='ANALYZING_FROZEN_SOURCE',arm=arm,fold=fold,seed=seed,new_fits=0))
            path=baseline_path(arm,fold,seed);key=f'{arm}_f{fold}_s{seed}';ref=reg['references'][key]
            for name,digest in ref['sha256'].items():assert sha(path/name)==digest
            cfg=read(path/'config.json');metrics=read(path/'metrics.json');setup(cfg['data_seed']);m=build(arm,cfg['data_seed']).cuda().eval();m.load_state_dict(torch.load(path/'model.pt',map_location='cuda',weights_only=True),strict=True)
            logits=predict(m,x);ce=float(torch.stack([F.cross_entropy(logits[rx==r],y[rx==r]) for r in range(5)]).mean())
            assert abs(ce-metrics['best_source_ce'])<1e-7 and score(logits,data['y'],data['rx'])==metrics['source_validation']
            results['source_recovery'].append(dict(arm=arm,fold=fold,seed=seed,source_metrics_exact=True,source_ce_delta=ce-metrics['best_source_ce']))
            if arm in ['P0','S1','H1']:
                sr,band=spectral_rows(arm,fold,seed,m);results['kernel_spectra']+=sr;results['kernel_bands'].append(band)
            gr,chain=local_gradients(m,probe,probe_y,arm,fold,seed);results['local_gradients']+=gr
            if chain:results['gradient_chain'].append(chain)
            if arm in ['S1','H0']:
                converted=materialize_s1(m) if arm=='S1' else fold_h0(m)
                new_logits=predict(converted,x);eq=equivalence(new_logits,logits)
                results['equivalence'].append(dict(arm=arm,fold=fold,seed=seed,**eq,original_params=sum(p.numel() for p in m.parameters()),converted_params=sum(p.numel() for p in converted.parameters())))
                emit_metrics(results['conversion_metrics'],arm,fold,seed,'original',logits,data['y'],data['rx']);emit_metrics(results['conversion_metrics'],arm,fold,seed,'converted',new_logits,data['y'],data['rx'])
                if eq['allclose']:
                    out=HERE/'converted'/f'{key}.pt';out.parent.mkdir(exist_ok=True);torch.save({k:v.detach().cpu() for k,v in converted.state_dict().items()},out)
                results['conversion_latency']+=latency(m,x,arm,fold,seed,'original')+latency(converted,x,arm,fold,seed,'converted')
                del converted,new_logits
            if arm in ['H0','S2','H2']:
                snapshots=snapshot(m,probe,'H0' if arm=='H0' else arm)
                for name,value in snapshots.items():results['activation_distributions'].append(dict(arm=arm,fold=fold,seed=seed,layer=name,mean=float(value.mean()),rms=float(value.square().mean().sqrt()),std=float(value.std()),scope='Balanced50 source-val eval-mode packets'))
                del snapshots
                if arm=='H0':
                    for name,bn in m.net.embedder.cnn_stem.tokenize.named_modules():
                        if isinstance(bn,torch.nn.BatchNorm1d):results['bn_statistics'].append(dict(fold=fold,seed=seed,layer=name,running_mean_rms=float(bn.running_mean.square().mean().sqrt()),running_var_mean=float(bn.running_var.mean()),gamma_rms=float(bn.weight.square().mean().sqrt()),beta_rms=float(bn.bias.square().mean().sqrt()),num_batches_tracked=int(bn.num_batches_tracked)))
            if arm in ['H0','P0']:
                emit_metrics(results['timing_metrics'],arm,fold,seed,'clean',logits,data['y'],data['rx']);base=snapshot(m,probe,arm)
                with torch.no_grad():clean_fft=torch.cat([legacy5_views(frontend(x[i:i+128]))[:,2:3] for i in range(0,len(x),128)])
                for kind,amount in itertools.product(['zero','circular'],[-4,-1,1,4]):
                    condition=f'{kind}_{amount:+d}';changed=shifted(x,amount,kind);lg=predict(m,changed)
                    emit_metrics(results['timing_metrics'],arm,fold,seed,condition,lg,data['y'],data['rx'])
                    for receiver,label in itertools.product(range(5),range(10)):
                        mask=(data['rx']==receiver)&(data['y']==label);pred=lg[torch.as_tensor(mask,device='cuda')].argmax(1)
                        results['tx_rx_timing'].append(dict(arm=arm,fold=fold,seed=seed,condition=condition,rx=roles(fold)['source'][receiver],tx=label,errors=int((pred!=label).sum()),packets=int(mask.sum())))
                    changed_snapshot=snapshot(m,changed[ids],arm);results['layer_responses']+=layer_rows(base,changed_snapshot,arm,fold,seed,condition,amount,kind);del changed_snapshot
                    if kind=='circular' and abs(amount)==4:
                        locked=predict(m,changed,clean_fft);emit_metrics(results['timing_metrics'],arm,fold,seed,condition+'_fft_locked',locked,data['y'],data['rx']);results['fft_lock_effects'].append(dict(arm=arm,fold=fold,seed=seed,condition=condition,**equivalence(locked,lg)));del locked
                    del changed,lg
                del clean_fft,base
            del m,logits
            # Persist bounded progress so a technical failure keeps completed evidence.
            for name,rows in results.items():save_csv(name+'.csv',rows)
            print(now(),key,'complete;zero training',flush=True)
        del x,y,rx,probe,probe_y,data;torch.cuda.empty_cache()
    for key,ref in reg['references'].items():
        for name,digest in ref['sha256'].items():assert sha(Path(ref['path'])/name)==digest
    write(HERE/'ANALYSIS_COMPLETE.json',dict(time=now(),models_recovered=len(results['source_recovery']),equivalence_checks=len(results['equivalence']),equivalence_all_pass=all(r['allclose'] for r in results['equivalence']),prediction_disagreements=sum(r['packets']-r['predictions_equal'] for r in results['equivalence']),new_fits=0,optimizer_constructions=0,all_reference_files_unchanged=True))

if __name__=='__main__':run()
