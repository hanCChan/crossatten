"""Read-only result aggregation and prospective source gates."""
import argparse
import time
import numpy as np
import torch
from core import HERE, read, write, once, now, sha
from models import ARMS, build, nparams
from train import job_id
from diagnostics import csv_write


def benchmark():
    if (HERE/'costs.json').exists():return read(HERE/'costs.json')
    from torch.utils.flop_counter import FlopCounterMode
    torch.set_num_threads(2);rows=[]
    for arm in ARMS:
        model=build(arm,13).eval()
        # Count explicit Transformer ops; fused inference is used again for timing.
        fastpath=torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.no_grad(),FlopCounterMode(display=False) as counter:model(torch.randn(1,2,8192))
        torch.backends.mha.set_fastpath_enabled(fastpath)
        flops=counter.get_total_flops();model=model.cuda();x=torch.randn(1,2,8192,device='cuda')
        with torch.no_grad():
            for _ in range(20):model(x)
            torch.cuda.synchronize();times=[]
            for _ in range(100):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();model(x);end.record();end.synchronize();times.append(start.elapsed_time(end))
            xb=torch.randn(64,2,8192,device='cuda')
            for _ in range(5):model(xb)
            torch.cuda.synchronize();t=time.perf_counter()
            for _ in range(10):model(xb)
            torch.cuda.synchronize();rate=640/(time.perf_counter()-t)
        row=dict(arm=arm,params=nparams(model),counted_mflops=flops/1e6,gpu_batch1_p50_ms=float(np.median(times)),
                 gpu_batch1_p95_ms=float(np.quantile(times,.95)),gpu_batch64_packets_s=rate)
        rows.append(row);print('Cost',row,flush=True);del model,x,xb;torch.cuda.empty_cache()
    result=dict(time=now(),gpu=torch.cuda.get_device_name(0),torch=torch.__version__,precision='FP32',
                scope='resident raw IQ to logits; no host transfer; no concurrent S13 training; desktop GPU use not isolated',
                flops='torch counted convolution/matrix operations; FFT/elementwise incomplete',rows=rows)
    write(HERE/'costs.json',result);csv_write(HERE/'costs.csv',rows);return result


def paired_stats(values):
    # values: receiver-clustered fit pairs, never packet-independent uncertainty.
    grouped={}
    for v in values:grouped.setdefault(v['rx'],[]).append(v['delta_pp'])
    rx=np.asarray([np.mean(v) for v in grouped.values()]);rng=np.random.default_rng(13013)
    draws=rng.choice(rx,size=(5000,len(rx)),replace=True).mean(1)
    fit={}
    for v in values:fit.setdefault((v['fold'],v['seed']),[]).append(v['delta_pp'])
    means=np.asarray([np.mean(v) for v in fit.values()])
    return dict(mean_delta_pp=float(means.mean()),positive_pairs=int((means>0).sum()),pairs=len(means),
                worst_receiver_delta_pp=float(rx.min()),rx_cluster_ci95_pp=np.quantile(draws,[.025,.975]).tolist(),
                uncertainty='conditional on trained seeds; no multiplicity correction')


def p1_report():
    jobs=read(HERE/'jobs_p1.json');metrics={};rows=[];receivers=[]
    for j in jobs:
        out=HERE/'runs'/job_id(j);m=read(out/'metrics.json');ev=read(out/'target_metrics.json')
        metrics[j['arm'],j['fold'],j['seed']]=ev
        for role in ['dev_val','confirm']:
            s=ev[role]['summary'];rows.append(dict(arm=j['arm'],fold=j['fold'],seed=j['seed'],role=role,
                acc=s['acc_mean']*100,macro_f1=s['f1_mean']*100,worst_rx=s['worst_rx'],worst_acc=s['worst_acc']*100,
                params=m['params'],seconds=m['wall_seconds']))
            for rx,v in ev[role]['per_rx'].items():receivers.append(dict(arm=j['arm'],fold=j['fold'],seed=j['seed'],role=role,rx=rx,
                acc=v['acc']*100,macro_f1=v['macro_f1']*100))
    comparisons=[]
    for other in ARMS:
        if other=='A3_rfhstu':continue
        for role in ['dev_val','confirm']:
            pairs=[]
            for fold in [0,1]:
                for seed in [201,202,203]:
                    a=metrics['A3_rfhstu',fold,seed][role]['per_rx'];b=metrics[other,fold,seed][role]['per_rx']
                    for rx in a:pairs.append(dict(rx=rx,fold=fold,seed=seed,delta_pp=100*(a[rx]['acc']-b[rx]['acc'])))
            s=paired_stats(pairs);s.update(reference='A3_rfhstu',control=other,role=role)
            s['verdict']='improves' if s['mean_delta_pp']>=1 and s['positive_pairs']>=5 and s['worst_receiver_delta_pp']>=-2 else 'unclear_or_not_superior'
            comparisons.append(s)
    csv_write(HERE/'p1_fit_metrics.csv',rows);csv_write(HERE/'p1_receiver_metrics.csv',receivers);csv_write(HERE/'p1_paired_comparisons.csv',comparisons)
    arm_summary=[]
    for arm in ARMS:
        for role in ['dev_val','confirm']:
            local=[r for r in rows if r['arm']==arm and r['role']==role]
            per=[r for r in receivers if r['arm']==arm and r['role']==role]
            seeds=[np.mean([r['acc'] for r in local if r['seed']==seed]) for seed in [201,202,203]]
            rx_means={rx:float(np.mean([r['acc'] for r in per if r['rx']==rx])) for rx in set(r['rx'] for r in per)}
            worst=min(rx_means,key=rx_means.get)
            arm_summary.append(dict(arm=arm,role=role,acc=float(np.mean([r['acc'] for r in local])),
                macro_f1=float(np.mean([r['macro_f1'] for r in local])),seed_sd_pp=float(np.std(seeds,ddof=1)),
                worst_receiver=worst,worst_receiver_mean_acc=rx_means[worst],minimum_single_fit_receiver_acc=min(r['acc'] for r in per)))
    write(HERE/'p1_summary.json',dict(fits=len(jobs),rows=rows,arm_summary=arm_summary,comparisons=comparisons,confirm='previously used receivers; report only'))
    lines=['# P1 LoRa Mixer 同协议结果','',f'完成{len(jobs)}/48个200-epoch正式fit；所有checkpoint先在源域冻结提交后，才读取dev_val/已使用confirm。','',
           '| Mixer | 参数 | dev_val均值±fit SD % | 已使用confirm均值±fit SD % |','|---|---:|---:|---:|']
    for arm in ARMS:
        means=[]
        for role in ['dev_val','confirm']:
            a=[r['acc'] for r in rows if r['arm']==arm and r['role']==role];means.append(f'{np.mean(a):.3f}±{np.std(a,ddof=1):.3f}')
        lines.append(f"|{arm}|{next(r['params'] for r in rows if r['arm']==arm)}|{means[0]}|{means[1]}|")
    lines+=['','| Mixer | 角色 | Macro-F1 % | seed SD pp | 最差RX均值 % | 最差单次RX % |',
            '|---|---|---:|---:|---:|---:|']
    for r in arm_summary:lines.append(f"|{r['arm']}|{r['role']}|{r['macro_f1']:.3f}|{r['seed_sd_pp']:.3f}|{r['worst_receiver_mean_acc']:.3f}|{r['minimum_single_fit_receiver_acc']:.3f}|")
    lines+=['','配对方向是RF-HSTU减去对照；完整逐折/seed、逐RX、F1和接收机聚类区间见CSV。',
            '参数匹配Transformer采用FFN32瓶颈，不能替代自然FFN256对照；匹配MHA内部投影80维，输入输出仍32×64。',
            'A3/A4/A5初始完整state逐项相同，唯一结构开关为Sigmoid/Softmax/SiLU打分。A5不是完整原版HSTU复现。',
            '多项比较未做多重校正；区间条件于已拟合种子。若普通MHA/Transformer接近或更好，应收缩HSTU特定优越性。',
            '成本见costs.csv：不能因参数更少就宣称更快，也不能使用A100历史速度替代4060测量。']
    (HERE/'P1_MIXER_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


def source_gate(rows):
    details={};checks={}
    for dataset in ['ManyRx','ManySig']:
        local=[r for r in rows if r['dataset']==dataset]
        if not local or set(r['fold'] for r in local)!=set(range(4)):
            raise ValueError('Gate requires both datasets and all4 source folds')
        folds={f:np.mean([r['delta_pp'] for r in local if r['fold']==f]) for f in range(4)}
        rx={}
        for r in local:rx.setdefault(r['rx'],[]).append(r['delta_pp'])
        details[dataset]=dict(mean_pp=float(np.mean(list(folds.values()))),positive_folds=int(sum(v>0 for v in folds.values())),
                              folds={str(k):float(v) for k,v in folds.items()},worst_rx_pp=float(min(np.mean(v) for v in rx.values())))
    a,b=details['ManyRx'],details['ManySig']
    checks=dict(manyrx_mean=a['mean_pp']>=1,manyrx_positive_folds=a['positive_folds']>=3,
                worst_rx=min(a['worst_rx_pp'],b['worst_rx_pp'])>=-2,manysig_mean=b['mean_pp']>=0,
                manysig_folds=sum(v>=-1 for v in b['folds'].values())>=3)
    return dict(admitted=all(checks.values()),checks=checks,details=details)


def p2_report(phase='p2'):
    jobs=read(HERE/f'jobs_{phase}.json');available={}
    for j in jobs:available[j['dataset'],j['fold'],j['seed'],j['arm']]=read(HERE/'runs'/job_id(j)/'metrics.json')['held_source']['per_rx']
    decisions={}
    for candidate in ['B1_cfo','B2_dual']:
        comparisons=[]
        if not any(j['arm']==candidate for j in jobs):continue
        for baseline in (['B0'] if candidate=='B1_cfo' else ['B0','B2_dual_raw']):
            pairs=[]
            for j in jobs:
                if j['arm']!=candidate:continue
                a=available[j['dataset'],j['fold'],j['seed'],candidate];b=available[j['dataset'],j['fold'],j['seed'],baseline]
                for rx in a:pairs.append(dict(dataset=j['dataset'],fold=j['fold'],seed=j['seed'],rx=rx,delta_pp=100*(a[rx]['acc']-b[rx]['acc'])))
            comparisons.append(dict(control=baseline,**source_gate(pairs)))
        decisions[candidate]=dict(admitted=all(c['admitted'] for c in comparisons),comparisons=comparisons)
    selected=[k for k,v in decisions.items() if v['admitted']]
    result=dict(time=now(),phase=phase,selected=selected,status='GO_SOURCE_CONFIRMATION' if selected and phase=='p2' else 'SOURCE_CONFIRMATION_PASSED' if selected else 'STOP_SOURCE_GATE',
                decisions=decisions,target_access=False)
    write(HERE/f'decision_{phase}.json',result)
    (HERE/f'{phase.upper()}_CFO_RESULTS.md').write_text('# WiSig CFO 源域结果\n\n'+json_block(result)+'\n\n这些是源域held-RX结果，不是目标CR/CRD。\n',encoding='utf-8',newline='\n')
    return result


def json_block(value):
    import json
    return '```json\n'+json.dumps(value,ensure_ascii=False,indent=2)+'\n```'


def progress(status=None):
    phases={}
    for phase in ['smoke','p0','p1','p2','p2_confirmation']:
        p=HERE/f'jobs_{phase}.json'
        if not p.exists():continue
        jobs=read(p);done=sum((HERE/'runs'/job_id(j)/'metrics.json').exists() for j in jobs)
        phases[phase]=dict(done=done,total=len(jobs))
    active=[]
    for p in (HERE/'runs').glob('*/attempt.json'):
        if (p.parent/'metrics.json').exists():continue
        value=read(p.parent/'live.json') if (p.parent/'live.json').exists() else read(p)
        if (p.parent/'failure.json').exists():value['status']='FAILED'
        active.append(dict(run=p.parent.name,**value))
    result=dict(time=now(),status=status or 'RUNNING',phases=phases,active_or_interrupted=active)
    write(HERE/'progress.json',result)
    lines=['# S13 执行状态','',f"更新时间：{result['time']}；状态：{result['status']}。",'', '| 阶段 | 完成 | 注册 |','|---|---:|---:|']
    for p,v in phases.items():lines.append(f"|{p}|{v['done']}|{v['total']}|")
    lines+=['','P0和P2只读WiSig各折源RX日期1–2；P1只按源验证CE选checkpoint。',
            'P1最终模型清单提交后才允许读取Shen dev_val/已使用confirm；blind始终禁止。',
            'P3未启动；历史S00–S12文件与STOP保持。未完成实验不得解读为负面或正面科学结果。']
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['cost','p1','p2','p2_confirmation','progress']);a=ap.parse_args()
    {'cost':benchmark,'p1':p1_report,'p2':p2_report,'p2_confirmation':lambda:p2_report('p2_confirmation'),'progress':progress}[a.phase]()
