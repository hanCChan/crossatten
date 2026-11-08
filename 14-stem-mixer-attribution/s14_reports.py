"""HSTU versus ordinary attention, then the paired Stem x Mixer interaction."""
import argparse
import csv
import time
import numpy as np
from s14_core import HERE,ROOT,S13,read,write,sha,now,committed,s13_id,job_id
from s14_stats import paired,factorial,mechanism_gate

def csv_write(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        out=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');out.writeheader();out.writerows(rows)

def s13_comparison():
    committed(S13/'decision_p1_source.json',ROOT if S13.is_relative_to(ROOT) else S13.parent)
    committed(S13/'p1_summary.json',S13.parent)
    arms=list(read(S13/'registration.json')['arms'])
    ev={(a,f,s):read(S13/'runs'/s13_id(a,f,s)/'target_metrics.json') for a in arms for f in [0,1] for s in [201,202,203]}
    configs=[read(S13/'runs'/s13_id(a,f,s)/'config.json') for a in arms for f in [0,1] for s in [201,202,203]]
    for fold in [0,1]:
        for seed in [201,202,203]:
            group=[c for c in configs if c['job']['fold']==fold and c['job']['seed']==seed]
            if len({c['initial_shared_sha256'] for c in group})!=1:raise ValueError('S13 shared initialization not paired')
            if any(c['epochs_budget']!=200 or c['batch']!=64 for c in group):raise ValueError('S13 recipe drift')
    contrasts=[];observations=[];gate_stats={}
    for role in ['dev_val','confirm']:
        for other in [a for a in arms if a!='A3_rfhstu']:
            local=[]
            for fold in [0,1]:
                for seed in [201,202,203]:
                    a,b=[ev[x,fold,seed][role]['per_rx'] for x in ['A3_rfhstu',other]]
                    if set(a)!=set(b):raise ValueError('Unpaired receivers')
                    for rx in a:
                        local.append(dict(role=role,control=other,fold=fold,seed=seed,rx=rx,
                            delta_pp=100*(a[rx]['acc']-b[rx]['acc']),delta_f1_pp=100*(a[rx]['macro_f1']-b[rx]['macro_f1'])))
            observations+=local;s=paired(local)
            if role=='dev_val':gate_stats[other]=s
            label='有正向配对证据' if s['ci95_pp'][0]>0 else '对照更好（条件区间低于0）' if s['ci95_pp'][1]<0 else '差异方向尚不确定'
            contrasts.append(dict(role=role,control=other,**s,f1=paired(local,'delta_f1_pp'),interpretation=label))
    result=dict(time=now(),status='COMPLETE_S13_PAIRED_ATTENTION_REVIEW',contrasts=contrasts,
        mechanism_gate=mechanism_gate(gate_stats),source_sha256=sha(S13/'p1_summary.json'),all48_fits_checked=True)
    write(HERE/'attention_comparison_s13.json',result)
    csv_write(HERE/'attention_paired_receiver_deltas.csv',observations)
    lines=['# RF-HSTU与普通注意力：完整S13配对审查','',
        '48/48源CE选模模型；2折×3种子；先冻结全部checkpoint再统一评估。以下方向统一为RF-HSTU减对照。',
        '源验证精度不用于判断跨RX优劣。confirm为历史已用接收机，只作报告，不用于S14门控消融准入。','',
        '| 角色 | 对照 | ΔRX准确率 pp | 条件RX95%区间 pp | 正向配对 | 最差RX平均Δ pp | 判断 |',
        '|---|---|---:|---|---:|---:|---|']
    for r in contrasts:lines.append(f"|{r['role']}|{r['control']}|{r['mean_delta_pp']:+.3f}|[{r['ci95_pp'][0]:+.3f},{r['ci95_pp'][1]:+.3f}]|{r['positive_pairs']}/6|{r['worst_receiver_delta_pp']:+.3f}|{r['interpretation']}|")
    summary=read(S13/'p1_summary.json');costs={r['arm']:r for r in read(S13/'costs.json')['rows']}
    lines+=['','| 模型 | dev RX准确率 % | dev Macro-F1 % | 最差RX均值 % | seed SD pp | 参数 | 计数MFLOPs | 4060单包P50 ms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
        r=next(r for r in summary['arm_summary'] if r['arm']==arm and r['role']=='dev_val');c=costs[arm]
        lines.append(f"|{arm}|{r['acc']:.3f}|{r['macro_f1']:.3f}|{r['worst_receiver_mean_acc']:.3f}|{r['seed_sd_pp']:.3f}|{c['params']}|{c['counted_mflops']:.3f}|{c['gpu_batch1_p50_ms']:.3f}|")
    lines+=['','A1/A2自然配置与参数匹配配置分别报告。FFN32匹配Transformer是瓶颈控制，不能替代FFN256。A3/A4只换打分方式，A5不是原始HSTU完整复现。',
        '所有区间条件于现有种子，未校正多重比较；6个fold/seed配对不是6个独立RX。区间跨0不能宣称等效，区间偏正也不等于跨所有数据集普遍领先。',
        'FLOPs不完整覆盖FFT/逐元素运算；推理延迟来自同一4060。训练wall_seconds可能含系统暂停，不用于归因每步网络效率。',
        '上表保留S13原始计数；融合SDPA矩阵运算可能未被该计数器覆盖。S14 costs.csv会统一使用共同前端计数+Mixer稠密矩阵/FFN公式，两个口径不能混用。',
        f"门控/位置追加预算门槛：{'通过，允许新增12 fit' if result['mechanism_gate']['admitted'] else '未通过，不启动G1/G2'}。主表必须保留普通MHA/Transformer更好的情况。"]
    (HERE/'ATTENTION_COMPARISON_S13.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,role in zip(axes,['dev_val','confirm']):
        subset=[r for r in contrasts if r['role']==role]
        for i,r in enumerate(subset):
            mean=r['mean_delta_pp'];lo,hi=r['ci95_pp']
            # Conditional receiver cluster mean can differ from fold-equal point estimate.
            ax.plot([lo,hi],[i,i],color='tab:blue');ax.plot(mean,i,'o',color='tab:blue')
        ax.axvline(0,color='black',linewidth=1);ax.set_yticks(range(len(subset)),[r['control'] for r in subset])
        ax.set_title(role);ax.set_xlabel('RF-HSTU minus control (pp); conditional RX95% interval');ax.grid(alpha=.2)
    fig.savefig(HERE/'ATTENTION_PAIRED_DELTAS.png',dpi=160);plt.close(fig)
    return result

def factorial_report():
    committed(HERE/'decision_p1_source.json');reg=read(HERE/'registration.json')
    for name,record in reg['reused'].items():
        if sha(S13/'runs'/name/'target_metrics.json')!=record['target_metrics_sha256']:raise ValueError('Reused S13 receiver outcomes changed')
    evals={}
    for a in ['A1_mha','A3_rfhstu','E1_patch_mha','E2_patch_hstu']:
        for f in [0,1]:
            for s in [201,202,203]:
                root=S13 if a.startswith('A') else HERE
                evals[a,f,s]=read(root/'runs'/s13_id(a,f,s)/'target_metrics.json')
    rows=[];raw=[];details=[]
    for role in ['dev_val','confirm']:
        for fold in [0,1]:
            for seed in [201,202,203]:
                cells={a:evals[a,fold,seed][role]['per_rx'] for a in ['A1_mha','A3_rfhstu','E1_patch_mha','E2_patch_hstu']}
                for a,per in cells.items():
                    for rx,v in per.items():raw.append(dict(arm=a,role=role,fold=fold,seed=seed,rx=rx,acc=100*v['acc'],macro_f1=100*v['macro_f1']))
                for rx in cells['A1_mha']:
                    for metric in ['acc','macro_f1']:
                        pm,ph,hm,hh=[100*cells[a][rx][metric] for a in ['E1_patch_mha','E2_patch_hstu','A1_mha','A3_rfhstu']]
                        for contrast,value in factorial(pm,ph,hm,hh).items():rows.append(dict(role=role,metric=metric,contrast=contrast,fold=fold,seed=seed,rx=rx,delta_pp=value))
        for metric in ['acc','macro_f1']:
            for contrast in factorial(0,0,0,0):
                local=[r for r in rows if r['role']==role and r['metric']==metric and r['contrast']==contrast]
                details.append(dict(role=role,metric=metric,contrast=contrast,**paired(local)))
    csv_write(HERE/'factorial_receiver_metrics.csv',raw);csv_write(HERE/'factorial_paired_deltas.csv',rows);csv_write(HERE/'factorial_contrasts.csv',details)
    summaries=[]
    for role in ['dev_val','confirm']:
        for arm in ['E1_patch_mha','E2_patch_hstu','A1_mha','A3_rfhstu']:
            group=[r for r in raw if r['role']==role and r['arm']==arm]
            rxmeans={rx:np.mean([r['acc'] for r in group if r['rx']==rx]) for rx in set(r['rx'] for r in group)}
            seedmeans=[np.mean([r['acc'] for r in group if r['seed']==seed]) for seed in [201,202,203]]
            summaries.append(dict(role=role,arm=arm,acc=float(np.mean([r['acc'] for r in group])),macro_f1=float(np.mean([r['macro_f1'] for r in group])),
                worst_rx_mean_acc=float(min(rxmeans.values())),seed_sd_pp=float(np.std(seedmeans,ddof=1))))
    result=dict(time=now(),new_fits=sum(j['arm'].startswith('E') for j in reg['jobs']),reused_fits=12,arm_summary=summaries,contrasts=details)
    write(HERE/'factorial_summary.json',result)
    lines=['# Stem × Mixer配对消融','',
        'Patch+MHA、Patch+RF-HSTU为本阶段新增12 fit；HStem两格复用S13相同fold/seed的12个模型。所有非tokenizer初始权重逐项配对。',
        '交互差值=(HStem+HSTU−Patch+HSTU)−(HStem+MHA−Patch+MHA)，先在同一RX/fold/seed内作差，再按RX聚类估计区间。','',
        '| 角色 | 实验臂 | RX准确率 % | Macro-F1 % | 最差RX均值 % | seed SD pp |','|---|---|---:|---:|---:|---:|']
    for r in summaries:lines.append(f"|{r['role']}|{r['arm']}|{r['acc']:.3f}|{r['macro_f1']:.3f}|{r['worst_rx_mean_acc']:.3f}|{r['seed_sd_pp']:.3f}|")
    lines+=['',
        '| 角色 | 指标 | 效应 | 均值 pp | 条件95%区间 pp | 正向配对 | 最差RX平均Δ pp |','|---|---|---|---:|---|---:|---:|']
    for r in details:lines.append(f"|{r['role']}|{r['metric']}|{r['contrast']}|{r['mean_delta_pp']:+.3f}|[{r['ci95_pp'][0]:+.3f},{r['ci95_pp'][1]:+.3f}]|{r['positive_pairs']}/6|{r['worst_receiver_delta_pp']:+.3f}|")
    interaction=next(r for r in details if r['role']=='dev_val' and r['metric']=='acc' and r['contrast']=='interaction')
    lines+=['','交互结论：'+('出现正向交互的条件区间证据，仍需结合有限RX和多重比较限制。' if interaction['ci95_pp'][0]>0 else '当前条件区间不支持确定的正向互补性，不能只凭两个均值宣布协同。'),
        '参数、FLOPs、延迟见costs.csv。Patch与HStem参数不同；本12-fit矩阵未提供Stem参数匹配控制，不能把容量、感受野和层级结构效应完全分离。',
        'confirm为历史已用接收机；不用于本阶段选模或选择最优结构。LoRa与WiSig前端不同，不据此宣称统一最优无线模型。']
    (HERE/'STEM_MIXER_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    return result

def mechanism_report():
    jobs=read(HERE/'jobs_p1.json');rows=[]
    for arm in ['G1_no_u','G2_no_rel']:
        for role in ['dev_val','confirm']:
            local=[]
            for j in jobs:
                if j['arm']!=arm:continue
                a=read(S13/'runs'/s13_id('A3_rfhstu',j['fold'],j['seed'])/'target_metrics.json')[role]['per_rx']
                b=read(HERE/'runs'/job_id(j)/'target_metrics.json')[role]['per_rx']
                for rx in a:local.append(dict(rx=rx,fold=j['fold'],seed=j['seed'],delta_pp=100*(a[rx]['acc']-b[rx]['acc'])))
            if local:rows.append(dict(arm=arm,role=role,**paired(local)))
    write(HERE/'mechanism_summary.json',dict(status='CONDITIONAL_FITS_COMPLETE' if rows else 'NOT_ADMITTED',rows=rows,
        caveat='No-U retains unused U projection; No-Rel retains allocated bias for exact one-switch controls. Report active and allocated parameters separately.'))

def benchmark():
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from s14_models import build,costs,dense_mixer_flops
    torch.set_num_threads(2)
    arms=list(read(S13/'registration.json')['arms'])+['E1_patch_mha','E2_patch_hstu','A1_mha_sdpa','A1_mha_matched_sdpa']
    if read(HERE/'registration.json')['mechanism_gate']['admitted']:arms+=['G1_no_u','G2_no_rel']
    rows=[]
    for arm in arms:
        m=build(arm,14014).eval();mixer_flops=dense_mixer_flops(m)
        blocks=m.net.encoder.blocks;m.net.encoder.blocks=torch.nn.ModuleList()
        fast=torch.backends.mha.get_fastpath_enabled();torch.backends.mha.set_fastpath_enabled(False)
        try:
            with torch.no_grad(),FlopCounterMode(display=False) as count:m(torch.randn(1,2,8192))
        finally:torch.backends.mha.set_fastpath_enabled(fast);m.net.encoder.blocks=blocks
        common_flops=count.get_total_flops();flops=common_flops+mixer_flops
        m=m.cuda();x=torch.randn(1,2,8192,device='cuda');xb=torch.randn(64,2,8192,device='cuda')
        for repeat in range(3):
            with torch.no_grad():
                for _ in range(20):m(x)
                times=[]
                for _ in range(100):
                    a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    a.record();m(x);b.record();b.synchronize();times.append(a.elapsed_time(b))
                for _ in range(5):m(xb)
                torch.cuda.synchronize();start=time.perf_counter()
                for _ in range(10):m(xb)
                torch.cuda.synchronize();rate=640/(time.perf_counter()-start)
            rows.append(dict(arm=arm,repeat=repeat,**costs(m,arm),counted_mflops=flops/1e6,common_mflops=common_flops/1e6,mixer_mflops=mixer_flops/1e6,
                p50_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),batch64_packets_s=rate))
        del m,x,xb;torch.cuda.empty_cache()
    csv_write(HERE/'costs.csv',rows)
    write(HERE/'costs.json',dict(time=now(),gpu=torch.cuda.get_device_name(0),precision='FP32',batch1_warmup=20,trials=100,repeats=3,
        scope='Resident raw IQ to logits; no host copy; no other registered training. Desktop GPU not isolated. FFT/elementwise FLOPs incomplete.',
        flops_method='FMA=2; common conv/linear profiler with mixers bypassed, plus explicit dense QKV/AV/FFN math. Avoid fused-SDPA undercount; identical math for manual/SDPA backends.',
        implementation_control='SDPA suffix is an inference-only backend using identical MHA parameters, not a new trained accuracy arm; actual selected backend depends on installed torch/device/dtype.',rows=rows))

def conclude():
    import json
    from s14_core import access as log_access
    from s14_data import roles,PROTOCOL
    from xa.wisig import receivers
    reg=read(HERE/'registration.json');factor=read(HERE/'factorial_summary.json')
    comparison=read(HERE/'attention_comparison_s13.json');violations=[]
    for line in (HERE/'access.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        if r['dataset']=='Shen':
            allowed=roles(r['fold'])['source'] if r['role']=='source' else roles(r['fold'])['dev_val'] if r['role']=='dev_val' else PROTOCOL['confirm_receivers'] if r['role']=='confirm' else []
            if r['rx'] not in allowed or r['rx'] in PROTOCOL['blind_receivers_never_open']:violations.append(r)
            if r['role']!='source' and r['time']<read(HERE/'decision_p1_source.json')['time']:violations.append(r)
        else:
            source,_=receivers(r['dataset'],r['fold'])
            if r['rx'] not in source or r['day'] not in [1,2]:violations.append(r)
    if violations:raise ValueError('Data access audit found a violation')
    from s14_core import git
    historical=git('diff','--name-only',reg['s13_final_git_sha'],'HEAD','--','.',':!'+HERE.name).splitlines()
    if historical:raise ValueError('Historical tracked content changed')
    write(HERE/'FINAL_ACCESS_AUDIT.json',dict(time=now(),violations=violations,historical_tracked_content_changed=historical,
        data_access_log_sha256=sha(HERE/'access.jsonl'),s13_checkpoint_references='immutable registered hashes',all_outputs='registered fits only'))
    effects={r['contrast']:r for r in factor['contrasts'] if r['role']=='dev_val' and r['metric']=='acc'}
    stem_positive=all(effects[k]['ci95_pp'][0]>0 for k in ['hstem_effect_mha','hstem_effect_hstu'])
    interaction_positive=effects['interaction']['ci95_pp'][0]>0
    cfo={}
    for name in ['decision_p2.json','decision_p2_confirmation.json']:
        if (S13/name).exists():cfo[name]=read(S13/name)['status']
    lines=['# 最终研究判断与贡献边界','',
        '完整对照见ATTENTION_COMPARISON_S13.md、STEM_MIXER_RESULTS.md、costs.csv和CFO_MECHANISM_REPORT.md。',
        '所有判断条件于这套LoRa/WiSig协议、有限接收机和既有种子；未做多重比较校正。','',
        '## HSTU与普通注意力',
        'RF-HSTU'+('满足事前登记的四种普通注意力对照门槛；可以进一步报告G1/G2机制对照。' if comparison['mechanism_gate']['admitted'] else '未满足事前登记的独立优势门槛；不追加G1/G2，不预设HSTU更优。'),
        '自然MHA/Transformer、参数匹配控制和同门控换打分全部保留。精度方向、最差RX与参数/延迟要分别解读，不能从其中一个数概括全面优越。','',
        '## 分层RF Token与交互',
        'HStem在两种Mixer上'+('均有正向条件区间证据。' if stem_positive else '没有同时取得正向条件区间证据，应按两种Mixer分别报告收益与失败。'),
        '交互差值'+('有正向条件区间证据，可有限讨论互补性。' if interaction_positive else '不支持确定的正向互补性；不能只凭均值说有协同。'),
        'Patch/HStem容量和感受野也不同，未做Stem参数匹配；不能宣称已唯一分离层级结构成因。','',
        '## CFO路线',
        f'沿用S13决策：{cfo}。',
        '分类器级分组和异常诊断不重新选择候选、不删除样本、不修改门槛。未通过源域确认就保持STOP；即使源域确认通过，也只构成冻结独立CRD的前提，本阶段没有新增真实WiSig CRD证据。','',
        '## 创新边界',
        'RF-HSTU承接第一篇；本轮训练沿用CE、AdamW和预热/余弦，没有新训练机制。第二篇的新增价值要由受控HStem效果和结构交互证明，不能把已有HSTU首次引入重新作为贡献。',
        'LoRa与WiSig前端不同，不声称一个跨无线体制统一最优模型。浅层无标签适配仅有可行性审计，训练预算0，不能报告适配优势或与零样本DG混排。',
        '审计：没有Shen blind访问、没有WiSig目标RX/date3–4访问，没有修改S00–S13历史。']
    (HERE/'RESEARCH_CONCLUSION.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')

if __name__=='__main__':
    from s14_core import validate
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['s13','factorial','mechanism','cost','conclude']);a=ap.parse_args()
    validate('design.json' if a.mode=='s13' else 'registration.json')
    {'s13':s13_comparison,'factorial':factorial_report,'mechanism':mechanism_report,'cost':benchmark,'conclude':conclude}[a.mode]()
