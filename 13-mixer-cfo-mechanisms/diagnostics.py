"""Source-only CFO validation and layer diagnostics; no target adaptation."""
import argparse
import copy
import csv
import math
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.optim.swa_utils import update_bn
from core import HERE, read, write, now, sha, array_sha, validate_registration
from data import wisig_cells, flatten, source_groups
from wifi import cfo, rotate, FS, LAG, build_wifi
from train import predict, job_id, setup
from xa.evaluate import summarize
from metrics import rx_metrics


def csv_write(path, rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)


def sample_cells(cells,cap=16):
    out={}
    for key,d in cells.items():
        ids=np.concatenate([np.flatnonzero(d['y']==tx)[:cap] for tx in np.unique(d['y'])])
        out[key]={k:v[ids] for k,v in d.items()}
    return flatten(out)


def cfo_audit():
    reg=validate_registration();rows=[];summaries={}
    for dataset in reg['p0']['datasets']:
        summaries[dataset]={}
        for fold in range(4):
            from xa.wisig import receivers
            source,_=receivers(dataset,fold)
            d=sample_cells(wisig_cells(dataset,fold,source,role='diagnostic'))
            x=torch.from_numpy(d['x']);est,q=cfo(x);est=est.numpy();quality=q.numpy()
            for i in range(len(est)):
                rows.append(dict(dataset=dataset,fold=fold,rx=d['rx_names'][d['rx'][i]],day=int(d['day'][i]),tx=int(d['y'][i]),
                                 packet_sample=i,estimated_hz=float(est[i]),coherence=float(quality[i])))
            rng=torch.Generator().manual_seed(13000+fold)
            errors={}
            for snr in reg['cfo']['added_noise_snr_db']:
                values=[]
                for offset in reg['cfo']['shifts_hz']:
                    shifted=rotate(x,float(offset));power=shifted.square().sum(1).mean(-1)
                    noise=torch.randn(shifted.shape,generator=rng)*torch.sqrt(power/10**(snr/10)/2)[:,None,None]
                    changed,_=cfo(shifted+noise)
                    delta=changed.numpy()-est-offset
                    period=FS/LAG;delta=(delta+period/2)%period-period/2
                    values.extend(abs(delta[quality>=.7]).tolist())
                errors[str(snr)]=dict(median_abs_error_hz=float(np.median(values)) if values else None,
                                      p95_abs_error_hz=float(np.quantile(values,.95)) if values else None)
            reliable=quality>=.7
            groups=[]
            for ri in np.unique(d['rx']):
                for day in (1,2):
                    mask=(d['rx']==ri)&(d['day']==day)&reliable
                    if mask.any():groups.append(float(np.median(est[mask])))
            centered=np.asarray(groups)-np.median(groups)
            # A bounded source observation, never declared pure RX oscillator error.
            bound=float(np.clip(np.quantile(np.abs(centered),.95),500,10000)) if len(groups) else 500.
            entry=dict(packets=len(x),reliable_fraction=float(reliable.mean()),median_coherence=float(np.median(quality)),
                       observed_cfo_quantiles_hz=np.quantile(est,[.01,.25,.5,.75,.99]).tolist(),
                       synthetic_added_cfo_error=errors,prospective_randomization_bound_hz=bound,
                       sample_sha256=array_sha(d['x']),interpretation='incremental synthetic CFO error; true native CFO is unknown')
            summaries[dataset][str(fold)]=entry
            print('CFO',dataset,fold,'reliable',round(entry['reliable_fraction'],3),'20dB',errors['20'],flush=True)
    csv_write(HERE/'source_cfo_packets.csv',rows)
    write(HERE/'cfo_audit.json',dict(created_at=now(),source_dates=[1,2],target_read=False,folds=summaries))
    return summaries


def probe_data(d):
    calibration=np.zeros(len(d['y']),bool)
    for tx in np.unique(d['y']):
        for rx in np.unique(d['rx']):
            for day in (1,2):
                ids=np.flatnonzero((d['y']==tx)&(d['rx']==rx)&(d['day']==day))
                calibration[ids[::2]]=True
    return calibration


@torch.no_grad()
def activations(model,x):
    locations={'stem1':model.frequency.stem[2],'stem2':model.frequency.stem[5],
               'down1':model.frequency.tokenize[2],'down2':model.frequency.tokenize[5],'down3':model.frequency.tokenize[8],
               'tokens':model.pre_ln,'mixer1':model.encoder.blocks[0],
               'mixer2':model.encoder.blocks[1],'final':model.encoder.final_norm}
    outputs={k:[] for k in locations}
    def hook(name):
        def save(module,args,out):
            if name in ['tokens','mixer1','mixer2','final']:out=out.transpose(1,2)
            pooled=torch.nn.functional.adaptive_avg_pool1d(out,8).flatten(1)
            outputs[name].append(pooled.cpu().numpy())
        return save
    handles=[m.register_forward_hook(hook(k)) for k,m in locations.items()]
    model.eval()
    try:
        for i in range(0,len(x),128):model(torch.from_numpy(x[i:i+128]).cuda())
    finally:
        for handle in handles:handle.remove()
    return {k:np.concatenate(v) for k,v in outputs.items()}


def layer_statistics(value,d,fit_names):
    cal=probe_data(d);test=~cal
    fit_mask=np.isin(d['rx'],[d['rx_names'].index(n) for n in fit_names])
    # TX transfer probe never fits held-source receivers.
    scale=StandardScaler().fit(value[cal&fit_mask]);v=scale.transform(value)
    tx=LogisticRegression(C=1,max_iter=300,class_weight='balanced',random_state=13).fit(v[cal&fit_mask],d['y'][cal&fit_mask])
    tx_acc=float((tx.predict(v[test&~fit_mask])==d['y'][test&~fit_mask]).mean())
    conditional=v.copy()
    for c in np.unique(d['y']):conditional[d['y']==c]-=v[cal&(d['y']==c)].mean(0)
    rx=LogisticRegression(C=1,max_iter=300,class_weight='balanced',random_state=13).fit(conditional[cal],d['rx'][cal])
    rx_acc=float((rx.predict(conditional[test])==d['rx'][test]).mean())
    unit=value/(np.linalg.norm(value,axis=1,keepdims=True)+1e-12)
    tx_means=[];rx_var=[];day_var=[]
    for c in np.unique(d['y']):
        cells=[]
        for ri in np.unique(d['rx']):
            pair=[]
            for day in (1,2):
                ids=(d['y']==c)&(d['rx']==ri)&(d['day']==day)
                if ids.any():pair.append(unit[ids].mean(0))
            if len(pair)==2:cells.append(pair)
        if len(cells)>=2:
            cells=np.asarray(cells);tx_means.append(cells.mean((0,1)))
            rx_var.append(np.var(cells.mean(1),axis=0).sum())
            day_var.append(np.var(cells,axis=1).mean(0).sum())
    b=float(np.var(tx_means,axis=0).sum());r=float(np.mean(rx_var));day=float(np.mean(day_var))
    return dict(tx_probe_held_rx_acc=tx_acc,tx_conditioned_rx_probe_acc=rx_acc,rx_chance=1/len(d['rx_names']),
                tx_probe_iterations=int(tx.n_iter_.max()),rx_probe_iterations=int(rx.n_iter_.max()),
                D_TX=b,D_RX=r,D_day=day,rho=r/(b+1e-12),feature_dim=value.shape[1],
                normalization='per-packet unit L2 for scatter; fit-calibration standardization for probes',
                calibration_samples=int(cal.sum()),probe_test_samples=int(test.sum()))


def held_accuracy(model,cells):
    per={}
    for rx in sorted({k[0] for k in cells}):
        x=np.concatenate([cells[rx,d]['x'] for d in (1,2)]);y=np.concatenate([cells[rx,d]['y'] for d in (1,2)])
        per[rx]=rx_metrics(predict(model,x).cpu().numpy(),y)
    return dict(summary=summarize(per),per_rx=per)


def trained_diagnostics(job):
    out=HERE/'runs'/job_id(job);destination=out/'diagnostics.json'
    if destination.exists():return read(destination)
    metrics=read(out/'metrics.json');cfg=read(out/'config.json')
    if sha(out/'model.pt')!=metrics['model_sha256']:raise ValueError('diagnostic checkpoint differs')
    setup(cfg['data_seed']);model=build_wifi(job['dataset'],'B0',cfg['data_seed']).cuda()
    model.load_state_dict(torch.load(out/'model.pt',map_location='cuda',weights_only=True))
    fit_rx,held_rx=source_groups(job['dataset'],job['fold'],'first')
    all_cells=wisig_cells(job['dataset'],job['fold'],fit_rx+held_rx,role='diagnostic')
    d=sample_cells(all_cells)
    features=activations(model,d['x'])
    rows=[]
    for layer,value in features.items():
        stat=layer_statistics(value,d,fit_rx);rows.append(dict(dataset=job['dataset'],fold=job['fold'],layer=layer,**stat))
        print('Probe',job['dataset'],job['fold'],layer,'rho',round(stat['rho'],3),flush=True)
    held={k:v for k,v in all_cells.items() if k[0] in held_rx}
    original=held_accuracy(model,held)
    state=copy.deepcopy(model.state_dict());bn_before={k:v.cpu().numpy().tolist() for k,v in state.items() if 'running_' in k}
    # Recreate exactly the legal fit partition; packet-val is excluded from BN calibration.
    from sklearn.model_selection import train_test_split
    pool=flatten({k:v for k,v in all_cells.items() if k[0] in fit_rx})
    train_ids,_=train_test_split(np.arange(len(pool['y'])),test_size=.3,random_state=job['seed'])
    update_bn((torch.from_numpy(pool['x'][ids]).cuda() for ids in np.array_split(train_ids,math.ceil(len(train_ids)/128))),model)
    interventions={'BN_recal_fit_only':held_accuracy(model,held)}
    generator=torch.Generator(device='cuda').manual_seed(13013)
    for label,prefix in [('stem_1pct','frequency.stem.0.weight'),('mixer_1pct','encoder.blocks.0.qkvu.weight')]:
        model.load_state_dict(state)
        with torch.no_grad():
            for name,p in model.named_parameters():
                if name==prefix:
                    noise=torch.randn(p.shape,generator=generator,device='cuda');p.add_(noise/noise.norm()*p.norm()*.01)
        interventions[label]=held_accuracy(model,held)
    model.load_state_dict(state)
    shifts={}
    for hz in [-10000,-5000,5000,10000]:
        changed={k:dict(x=rotate(torch.from_numpy(v['x']),hz).numpy(),y=v['y']) for k,v in held.items()}
        shifts[str(hz)]=held_accuracy(model,changed)
    result=dict(job=job,layers=rows,bn_running_statistics=bn_before,baseline=original,interventions=interventions,
                synthetic_cfo=shifts,target_read=False,adaptation_run=False,
                limitation='RX probes and sensitivity are descriptive; no claim of shallow-only adaptation benefit or natural-CFO causation')
    write(destination,result)
    return result


def p0_report():
    reg=validate_registration();c=read(HERE/'cfo_audit.json');results=[]
    for job in read(HERE/'jobs_p0.json'):results.append(trained_diagnostics(job))
    rows=[row for result in results for row in result['layers']];csv_write(HERE/'layer_probes.csv',rows)
    checks={}
    for dataset,folds in c['folds'].items():
        for fold,v in folds.items():
            errors=v['synthetic_added_cfo_error']['20'];gate=reg['cfo']['estimator_gate']
            checks[f'{dataset}/fold{fold}']=dict(reliable=v['reliable_fraction']>=gate['min_reliable_fraction'],
                median=errors['median_abs_error_hz'] is not None and errors['median_abs_error_hz']<=gate['max_added20db_median_error_hz'],
                p95=errors['p95_abs_error_hz'] is not None and errors['p95_abs_error_hz']<=gate['max_added20db_p95_error_hz'])
    allowed=all(all(x.values()) for x in checks.values())
    decision=dict(time=now(),status='GO_CFO_SOURCE_PILOT' if allowed else 'STOP_ESTIMATOR_GATE',
                  cfo_pilot_allowed=allowed,checks=checks,source_fits_complete=len(results),
                  shallow_adaptation_claim='NOT_ESTABLISHED: P3 not executed',target_access=False,
                  evidence_sha256={p.name:sha(p) for p in [HERE/'cfo_audit.json',HERE/'layer_probes.csv',HERE/'source_cfo_packets.csv']})
    write(HERE/'decision_p0.json',decision)
    lines=['# P0 WiSig 源域机制诊断','',f"状态：{decision['status']}。完成{len(results)}/8个新源域诊断fit。",'',
           '只使用各折合法源RX的第1–2天；原始pickle按字节跨度索引，仅解码获准cell，未载入目标RX/第3–4天IQ。','',
           '层级TX跨RX探针、控制TX后的RX探针和统一尺度rho见layer_probes.csv。','',
           'BN仅用70%fit训练分区重新校准；权重扰动是敏感性诊断，不是浅层无监督适配。没有验证师弟的微调等价性。','',
           '| 数据 | fold | STF可靠比例 | +CFO/20dB中位误差Hz | P95 Hz |','|---|---:|---:|---:|---:|']
    for dataset,folds in c['folds'].items():
        for f,v in folds.items():
            e=v['synthetic_added_cfo_error']['20']
            med='NA' if e['median_abs_error_hz'] is None else f"{e['median_abs_error_hz']:.2f}"
            p95='NA' if e['p95_abs_error_hz'] is None else f"{e['p95_abs_error_hz']:.2f}"
            lines.append(f"|{dataset}|{f}|{v['reliable_fraction']:.3f}|{med}|{p95}|")
    lines+=['','以下为4折描述性均值；TX probe只在fit-RX拟合、在留出源RX测试。RX probe使用各源RX互不重叠的包分区，控制TX均值后分类RX。',
            '','| 数据 | 层 | TX跨RX probe % | TX条件RX probe % | rho |','|---|---|---:|---:|---:|']
    layers=list(dict.fromkeys(r['layer'] for r in rows))
    for dataset in reg['p0']['datasets']:
        for layer in layers:
            subset=[r for r in rows if r['dataset']==dataset and r['layer']==layer]
            v={k:np.mean([r[k] for r in subset]) for k in ['tx_probe_held_rx_acc','tx_conditioned_rx_probe_acc','rho']}
            lines.append(f"|{dataset}|{layer}|{100*v['tx_probe_held_rx_acc']:.2f}|{100*v['tx_conditioned_rx_probe_acc']:.2f}|{v['rho']:.3f}|")
    lines+=['','| 数据 | 干预 | 留出源RX准确率变化 pp，4折均值 |','|---|---|---:|']
    for dataset in reg['p0']['datasets']:
        subset=[r for r in results if r['job']['dataset']==dataset]
        for name in subset[0]['interventions']:
            delta=np.mean([100*(r['interventions'][name]['summary']['acc_mean']-r['baseline']['summary']['acc_mean']) for r in subset])
            lines.append(f'|{dataset}|{name}|{delta:+.3f}|')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for dataset in reg['p0']['datasets']:
        for ax,key in zip(axes,['rho','tx_probe_held_rx_acc','tx_conditioned_rx_probe_acc']):
            values=np.asarray([[r[key] for r in rows if r['dataset']==dataset and r['layer']==layer] for layer in layers])
            ax.errorbar(np.arange(len(layers)),values.mean(1),yerr=values.std(1),marker='o',capsize=2,label=dataset)
            ax.set_xticks(np.arange(len(layers)),layers,rotation=60);ax.set_title(key);ax.grid(alpha=.25);ax.legend()
    fig.suptitle('S13 source-only layer probes: mean +/- fold SD (descriptive)')
    fig.savefig(HERE/'P0_LAYER_PROBES.png',dpi=160);plt.close(fig)
    lines+=['','合成CFO增量误差不等于天然CFO绝对误差；观测频偏不能唯一拆成TX和RX。',
            '估计器门槛仅决定是否值得进行受控源域候选试验，不证明自然跨RX错误由CFO导致。',
            'P2训练预算和分布界限需另行冻结提交；P3未启动。',
            '','![归一化层级诊断](P0_LAYER_PROBES.png)',
            '这些结果不自动推出浅层适配有效。探针测试包可能与骨干fit包重合，跨RX的TX probe才是未参与骨干训练的源RX；连续采集包也不视为独立接收机重复。']
    (HERE/'P0_MECHANISM_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    write(HERE/'p0_diagnostic_summary.json',dict(decision=decision,results=results))
    return decision


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['cfo','report']);a=ap.parse_args()
    cfo_audit() if a.phase=='cfo' else p0_report()
