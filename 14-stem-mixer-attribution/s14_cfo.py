"""Fixed, source-only classifier diagnostics of S13 CFO models; no fitting."""
import numpy as np
import torch
from s14_core import HERE,S13,read,write,sha,now,validate
from s14_data import wifi_cell
from s14_reports import csv_write
from wifi import build_wifi,cfo,rotate,FS
from metrics import rx_metrics

def estimate_window(x,start=0,lag=20):
    z=torch.complex(x[:,0],x[:,1])[:,start:start+160]
    a,b=z[:,lag:],z[:,:-lag]
    p=(a*b.conj()).sum(-1);q=p.abs()/(a.abs().square().sum(-1)*b.abs().square().sum(-1)).sqrt().clamp_min(1e-12)
    return p.angle()*FS/(2*torch.pi*lag),q.clamp(0,1)

def masks(y,rx,day,estimated,quality):
    result={'all':np.ones(len(y),bool),'coherence/low':quality<.7,'coherence/high':quality>=.7}
    for r in np.unique(rx):
        for d in [1,2]:result[f'rxday/{r}/{d}']=(rx==r)&(day==d)
    for tx in np.unique(y):result[f'tx/{tx}']=y==tx
    absolute=np.abs(estimated)
    bounds=[0,2000,5000,10000,50000,float('inf')]
    for a,b in zip(bounds[:-1],bounds[1:]):result[f'abs_cfo/{a:g}/{b:g}']=(absolute>=a)&(absolute<b)
    if sum(m.sum() for name,m in result.items() if name.startswith('abs_cfo/'))!=len(y):raise ValueError('CFO strata lost packets')
    if result['coherence/low'].sum()+result['coherence/high'].sum()!=len(y):raise ValueError('Confidence strata lost packets')
    return result

def group_score(logits,y,rx,mask,tx_only=False):
    per=[]
    for name in np.unique(rx[mask]):
        selected=mask&(rx==name)
        per.append(rx_metrics(logits[selected],y[selected]))
    if not per:return dict(packets=0,receivers=0,acc=None,macro_f1=None,classes_present=0)
    return dict(packets=int(mask.sum()),receivers=len(per),acc=100*float(np.mean([v['acc'] for v in per])),
        macro_f1=None if tx_only else 100*float(np.mean([v['macro_f1'] for v in per])),classes_present=len(np.unique(y[mask])))

@torch.no_grad()
def predict(model,x,rx=None):
    if rx is not None:
        output=None
        for name in np.unique(rx):
            mask=rx==name;value=predict(model,x[mask])
            if output is None:output=np.empty((len(x),value.shape[1]),dtype=value.dtype)
            output[mask]=value
        return output
    model.eval();return torch.cat([model(torch.from_numpy(np.ascontiguousarray(x[i:i+128])).cuda()) for i in range(0,len(x),128)]).cpu().numpy()

def run():
    reg=validate();torch.set_num_threads(2);torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    shifts=reg['cfo_diagnostics']['shifts_hz'];rows=[];paired=[];windows=[]
    from xa.wisig import receivers
    for record in reg['cfo_models']:
        job=record['job'];name=record['run'];out=S13/'runs'/name
        if sha(out/'model.pt')!=record['model_sha256']:raise ValueError('CFO checkpoint changed')
        if sha(out/'metrics.json')!=record['metrics_sha256']:raise ValueError('Frozen CFO metrics changed')
        source,_=receivers(job['dataset'],job['fold']);n=6 if job['dataset']=='ManyRx' else 2
        held=source[:n] if job['group']=='first' else source[-n:]
        xs=[];ys=[];rs=[];ds=[]
        for rx in held:
            for day in [1,2]:
                d=wifi_cell(job['dataset'],job['fold'],rx,day)
                xs.append(d['x']);ys.append(d['y']);rs.extend([rx]*len(d['y']));ds.extend([day]*len(d['y']))
        x=np.concatenate(xs);y=np.concatenate(ys);rx=np.asarray(rs);day=np.asarray(ds)
        estimate,quality=cfo(torch.from_numpy(x));estimate=estimate.numpy();quality=quality.numpy()
        group_masks=masks(y,rx,day,estimate,quality)
        cfg=read(out/'config.json');model=build_wifi(job['dataset'],job['arm'],cfg['data_seed']).cuda()
        model.load_state_dict(torch.load(out/'model.pt',map_location='cuda',weights_only=True))
        baseline=None
        for offset in shifts:
            signal=x if offset==0 else rotate(torch.from_numpy(x),offset).numpy()
            logits=predict(model,signal,rx)
            if offset==0:
                baseline=logits
                # The preserved S13 metrics must be reproduced with the same batch and inputs.
                old=read(out/'metrics.json')['held_source']['per_rx']
                for name_rx in held:
                    selected=rx==name_rx;actual=rx_metrics(logits[selected],y[selected])
                    if abs(actual['acc']-old[name_rx]['acc'])>1e-10 or actual['confusion']!=old[name_rx]['confusion']:
                        raise ValueError('Classifier diagnostic does not reproduce S13 held-source decisions')
            for group,mask in group_masks.items():
                rows.append(dict(run=name,dataset=job['dataset'],phase=job['phase'],fold=job['fold'],seed=job['seed'],group=job['group'],
                    arm=job['arm'],stratum=group,added_cfo_hz=offset,**group_score(logits,y,rx,mask,group.startswith('tx/'))))
        dest=HERE/'runs/cfo_predictions'/name;dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'unshifted.npz',logits=baseline,y=y,rx=rx,day=day,estimated_hz=estimate,coherence=quality)
        del model;torch.cuda.empty_cache()
        print('CFO classifier diagnostics',name,flush=True)
    # Compare every candidate with its *same* phase/fold/seed/group baseline.
    lookup={(r['dataset'],r['phase'],r['fold'],r['seed'],r['group'],r['arm'],r['stratum'],r['added_cfo_hz']):r for r in rows}
    for r in rows:
        controls=['B0','B2_dual_raw'] if r['arm']=='B2_dual' else ['B0'] if r['arm'] in ['B1_cfo','B2_corrected'] else []
        for control in controls:
            key=(r['dataset'],r['phase'],r['fold'],r['seed'],r['group'],control,r['stratum'],r['added_cfo_hz'])
            b=lookup.get(key)
            if b and r['acc'] is not None and b['acc'] is not None:
                paired.append(dict(**r,control=control,delta_acc_pp=r['acc']-b['acc'],delta_f1_pp=None if r['macro_f1'] is None else r['macro_f1']-b['macro_f1']))
    csv_write(HERE/'cfo_classifier_groups.csv',rows);csv_write(HERE/'cfo_paired_groups.csv',paired)
    # Audit all source anomalous cells named by S13; never remove any from scoring.
    import pandas as pd
    extreme=pd.read_csv(S13/'cfo_extreme_cells.csv')
    for entry in extreme.to_dict('records'):
        d=wifi_cell(entry['dataset'],entry['fold'],entry['rx'],entry['day'])
        x=torch.from_numpy(d['x'][np.flatnonzero(d['y']==entry['tx'])[:16]])
        ref,_=estimate_window(x)
        for start in [0,10,20,40,60,80]:
            hz,q=estimate_window(x,start)
            windows.append(dict(dataset=entry['dataset'],fold=entry['fold'],rx=entry['rx'],day=entry['day'],tx=entry['tx'],
                diagnostic='start_window',start=start,lag=20,packets=len(x),median_hz=float(hz.median()),median_coherence=float(q.median())))
        for lag in [40,60,80]:
            hz,q=estimate_window(x,0,lag)
            difference=(hz-ref)*2*torch.pi*lag/FS
            residual=torch.atan2(difference.sin(),difference.cos()).abs()
            windows.append(dict(dataset=entry['dataset'],fold=entry['fold'],rx=entry['rx'],day=entry['day'],tx=entry['tx'],
                diagnostic='lag_phase_consistency',start=0,lag=lag,packets=len(x),median_hz=float(hz.median()),median_coherence=float(q.median()),
                median_modulo_phase_residual_rad=float(residual.median())))
    csv_write(HERE/'cfo_anomalous_window_audit.csv',windows)
    # Fold/seed curves are descriptive, not packet-independent confidence intervals.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,dataset in zip(axes,['ManyRx','ManySig']):
        for arm in ['B0','B1_cfo','B2_corrected','B2_dual','B2_dual_raw']:
            means=[];std=[]
            for offset in shifts:
                values=[r['acc'] for r in rows if r['dataset']==dataset and r['phase']=='p2' and r['arm']==arm and r['stratum']=='all' and r['added_cfo_hz']==offset]
                means.append(np.mean(values));std.append(np.std(values))
            ax.errorbar(np.asarray(shifts)/1000,means,yerr=std,label=arm,marker='o',capsize=2)
        ax.set(title=dataset,xlabel='Added CFO (kHz)',ylabel='Held-source RX-equal accuracy (%)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(HERE/'CFO_ROBUSTNESS.png',dpi=160);plt.close(fig)
    write(HERE/'cfo_classifier_summary.json',dict(time=now(),models=len(reg['cfo_models']),stratum_rows=len(rows),paired_rows=len(paired),
        anomalies=len(extreme),samples_removed=0,target_access=False,fit_performed=False,s13_prediction_confusion_reproduced=True,
        limitations=['CFO magnitude is observed total CFO, not true estimator error or pure RX error.',
            'Fixed lag/window changes are diagnostic only; no best window chosen and no model recalibrated.',
            'Strata may have different TX mixtures; TX rows report recall/accuracy, not misleading K-class macroF1.']))
    lines=['# WiSig CFO分类器级机制诊断','',f"复用{len(reg['cfo_models'])}个S13源域模型，没有新增训练。所有未扰动混淆矩阵与原记录一致。",'',
        '完整逐RX/day、TX、coherence和观测CFO分组见cfo_classifier_groups.csv；同fold/seed/group的配对变化见cfo_paired_groups.csv。',
        '组内先按RX等权，空组标为NA；TX单类组报告准确率/召回率，不把缺失TX计入一个误导性的Macro-F1。',
        '重点判断raw+corrected是否同时优于raw-only和双raw参数控制；若双raw也同样改善，不能将参数收益归给CFO校正。','',
        '![冻结CFO扰动曲线](CFO_ROBUSTNESS.png)','',
        '异常cell未删除，窗口/延迟诊断不用于修改校正起点、阈值或候选。长延迟估计存在不同无歧义范围，因此按模2π的相位一致性检查，不能直接比较Hz之差当作真误差。',
        '没有天然CFO真值；观测CFO分组不能称为估计误差真值分组。RX/day、TX混合与包间相关性限制因果解读。',
        '沿用S13既有STOP/GO，诊断不救活未通过的候选，不读取真实目标CRD。']
    (HERE/'CFO_MECHANISM_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')

if __name__=='__main__':run()
