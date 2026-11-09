"""Paired source-only gate decisions, never target CRD selection."""
import csv,argparse
from pathlib import Path
import numpy as np
from s16_core import HERE,S13,read,write,sha,now,validate
from s16_train import run_id

def csv_write(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows)

CONTROLS={
    'R1_qwen_small':['R2_transformer','R0_mha','R0_hstu'],
    'R2_transformer':['R0_mha','R0_hstu'],
    'R3_pretrained_frozen':['R4_random_frozen','R0_mha','R0_hstu'],
    'R5_pretrained_lora':['R3_pretrained_frozen','R0_mha','R0_hstu'],
}

def delta_stats(rows):
    if not rows:raise ValueError('Empty contrast')
    if set(r['fold'] for r in rows)!=set(range(4)):raise ValueError('All four paired folds are required')
    folds={f:np.mean([r['delta_pp'] for r in rows if r['fold']==f]) for f in range(4)}
    rx={}
    for r in rows:rx.setdefault(r['rx'],[]).append(r['delta_pp'])
    # Resample whole RX clusters including their fold occurrences. RXs occur in
    # unequal numbers of folds; averaging cluster means would change the estimand.
    if len({sum(r['fold']==f for r in rows) for f in range(4)})!=1:raise ValueError('Fold receiver counts must match')
    values=np.asarray([np.mean(v) for v in rx.values()]);rng=np.random.default_rng(16016)
    sums=np.asarray([sum(v) for v in rx.values()]);counts=np.asarray([len(v) for v in rx.values()])
    index=rng.integers(len(rx),size=(5000,len(rx)))
    draws=sums[index].sum(1)/counts[index].sum(1)
    return dict(mean_pp=float(np.mean(list(folds.values()))),positive_folds=int(sum(v>0 for v in folds.values())),folds={str(k):float(v) for k,v in folds.items()},
        worst_rx_pp=float(values.min()),ci95_pp=np.quantile(draws,[.025,.975]).tolist(),receiver_count=len(values),uncertainty='conditional RX bootstrap, no multiplicity correction')

def gate(stats,confirm=False):
    a=stats['ManyRx'];checks=dict(manyrx_mean=a['mean_pp']>=1,manyrx_positive=a['positive_folds']>=3,worst_rx=a['worst_rx_pp']>=-2,manyrx_ci=a['ci95_pp'][0]>0)
    if confirm:
        b=stats['ManySig'];checks.update(worst_rx=checks['worst_rx'] and b['worst_rx_pp']>=-2,manysig_mean=b['mean_pp']>=0,
            manysig_folds=sum(v>=-1 for v in b['folds'].values())>=3)
    return dict(admitted=all(checks.values()),checks=checks,stats=stats)

def report(phase):
    validate();jobs=read(HERE/f'jobs_{phase}.json');metrics={};rows=[];receiver_rows=[]
    for j in jobs:
        out=HERE/'runs'/run_id(j);m=read(out/'metrics.json')
        if sha(out/'trainable.pt')!=m['trainable_sha256']:raise ValueError('Selected head checkpoint changed')
        metrics[j['dataset'],j['fold'],j['seed'],j['arm']]=m
        s=m['held_source']['summary']
        rows.append(dict(phase=phase,dataset=j['dataset'],fold=j['fold'],seed=j['seed'],arm=j['arm'],acc=100*s['acc_mean'],macro_f1=100*s['f1_mean'],
            worst_rx_acc=100*s['worst_acc'],source_val_acc=100*m['source_validation']['summary']['acc_mean'],source_train_acc=100*m['source_train']['summary']['acc_mean'],
            generalization_gap_pp=m['source_to_held_gap_pp'],active_params=m['active_params'],trainable_params=m['trainable_params'],
            peak_allocated_gib=m['peak_allocated_gib'],seconds=m['wall_seconds']))
        for rx,v in m['held_source']['per_rx'].items():receiver_rows.append(dict(dataset=j['dataset'],fold=j['fold'],seed=j['seed'],arm=j['arm'],rx=rx,acc=100*v['acc'],macro_f1=100*v['macro_f1']))
    contrast_rows=[];decisions={};all_arms=set(j['arm'] for j in jobs)
    # R5-R6 is mandatory in confirmation to test pretrained weights under LoRA.
    controls={k:list(v) for k,v in CONTROLS.items()}
    if phase!='pilot' and 'R6_random_lora' in all_arms:controls['R5_pretrained_lora'].append('R6_random_lora')
    candidates=read(HERE/f'registration_{phase}.json')['candidates']
    for arm,baselines in controls.items():
        if arm not in all_arms:continue
        if arm not in candidates:continue
        comparisons=[]
        for baseline in baselines:
            if baseline not in all_arms:raise ValueError('Paired control missing')
            by_dataset={}
            for dataset in sorted(set(j['dataset'] for j in jobs)):
                pairs=[]
                for j in jobs:
                    if j['dataset']!=dataset or j['arm']!=arm:continue
                    a=metrics[dataset,j['fold'],j['seed'],arm]['held_source']['per_rx'];b=metrics[dataset,j['fold'],j['seed'],baseline]['held_source']['per_rx']
                    if set(a)!=set(b):raise ValueError('Paired receiver sets differ')
                    for rx in a:
                        row=dict(dataset=dataset,fold=j['fold'],seed=j['seed'],rx=rx,candidate=arm,control=baseline,
                            delta_pp=100*(a[rx]['acc']-b[rx]['acc']),delta_f1_pp=100*(a[rx]['macro_f1']-b[rx]['macro_f1']))
                        pairs.append(row);contrast_rows.append(row)
                by_dataset[dataset]=delta_stats(pairs)
            comparisons.append(dict(control=baseline,**gate(by_dataset,phase!='pilot')))
        decisions[arm]=dict(admitted=all(c['admitted'] for c in comparisons),comparisons=comparisons)
    selected=[a for a,d in decisions.items() if d['admitted'] and a in candidates]
    # Bound costly confirmation: at most one architecture branch and one pretrained branch.
    if phase=='pilot':
        chosen=[]
        for category in [['R1_qwen_small','R2_transformer'],['R3_pretrained_frozen','R5_pretrained_lora']]:
            options=[a for a in category if a in selected]
            if options:
                chosen.append(max(options,key=lambda a:(min(c['stats']['ManyRx']['mean_pp'] for c in decisions[a]['comparisons']),-category.index(a))))
        selected=chosen
    result=dict(time=now(),phase=phase,selected=selected,status='GO_SOURCE_CONFIRMATION' if selected and phase=='pilot' else 'SOURCE_CONFIRMATION_PASSED' if selected else 'STOP_SOURCE_GATE',
        decisions=decisions,real_target_access=False,scope='Frozen source-pretrained RF tokenizer; same20-epoch tail adaptation, not independent end-to-end optimum')
    write(HERE/f'decision_{phase}.json',result);csv_write(HERE/f'{phase}_fit_metrics.csv',rows);csv_write(HERE/f'{phase}_receiver_metrics.csv',receiver_rows);csv_write(HERE/f'{phase}_paired_deltas.csv',contrast_rows)
    lines=['# S16 '+phase+'：RF-Qwen3源域结果','',f"状态：{result['status']}；完成{len(jobs)}个新尾部训练。",'',
        '所有模型共用同一合法源RX训练并冻结的RF tokenizer。R0两种尾部配对重训；S13端到端B0仅为历史背景。',
        'R1/R2深度与参数近似匹配、均因果CLS末尾。R3/R4固定骨干配对；R5/R6若确认则相同LoRA策略配对。','',
        '| 数据 | 模型 | 留出源RX % | Macro-F1 % | 源验证RX % | 源到留出差距 pp | 活跃参数 | 可训练参数 | 峰值显存GiB | 每fit分钟 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for dataset in sorted(set(r['dataset'] for r in rows)):
        for arm in sorted(set(r['arm'] for r in rows)):
            r=[v for v in rows if v['dataset']==dataset and v['arm']==arm]
            if r:lines.append(f"|{dataset}|{arm}|{np.mean([v['acc'] for v in r]):.3f}|{np.mean([v['macro_f1'] for v in r]):.3f}|{np.mean([v['source_val_acc'] for v in r]):.3f}|{np.mean([v['generalization_gap_pp'] for v in r]):.3f}|{r[0]['active_params']}|{r[0]['trainable_params']}|{max(v['peak_allocated_gib'] for v in r):.3f}|{np.mean([v['seconds'] for v in r])/60:.2f}|")
    lines+=['','逐fold、最差RX、条件区间和所有负收益见对应CSV与decision JSON。源域Pilot不是目标CRD；不读真实目标RX/date3–4。',
        'R3−R4只说明在冻结骨干及共同RF前端配方下预训练权重的价值；不能唯一归因于语言知识。R5−R6也不等价于对比全参从零训练。',
        '相对R0的增益混合深度、维数、参数规模与结构，R1−R2才是较受控的结构组合比较；本轮不声称单独隔离深度或RoPE。',
        '词嵌入已删除，LM head未实例化；总参数/活跃参数/训练参数及checkpoint序列化参数另行审计。',
        '没有使用Held RX标签改学习率/轮数/LoRA配置；候选不满足固定门槛就STOP，不能追加训练或转目标调参。']
    (HERE/f'{phase.upper()}_RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase');a=p.parse_args();report(a.phase)
