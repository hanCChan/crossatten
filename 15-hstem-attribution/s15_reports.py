import csv
import argparse
from pathlib import Path
import numpy as np
from s15_core import HERE,ROOT,read,write,sha,now,validate,committed
from s15_stats import paired

def csv_write(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        out=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');out.writeheader();out.writerows(rows)

def report():
    reg=validate();committed(HERE/'decision_p1_source.json');metrics={};rows=[];summaries=[];contrasts=[]
    cells=['E1_patch_mha','E2_patch_hstu','A1_mha','A3_rfhstu','C2_lite_mha','C2_lite_hstu']
    for arm in cells:
        for fold in [0,1]:
            for seed in [201,202,203]:
                name=f'p1__Shen__{arm}__f{fold}__all__s{seed}'
                if arm.startswith('C2'):root=HERE
                else:
                    ref=reg['reused'][name];root=Path(ref['root'])
                    if sha(root/'runs'/name/'target_metrics.json')!=ref['target_metrics_sha256']:raise ValueError('Frozen reference outcomes changed')
                e=read(root/'runs'/name/'target_metrics.json');metrics[arm,fold,seed]=e
                for role in ['dev_val','confirm']:
                    for rx,v in e[role]['per_rx'].items():rows.append(dict(arm=arm,fold=fold,seed=seed,role=role,rx=rx,acc=100*v['acc'],macro_f1=100*v['macro_f1']))
    for arm in cells:
        for role in ['dev_val','confirm']:
            r=[v for v in rows if v['arm']==arm and v['role']==role]
            rxmeans=[np.mean([v['acc'] for v in r if v['rx']==rx]) for rx in set(v['rx'] for v in r)]
            seedmeans=[np.mean([v['acc'] for v in r if v['seed']==s]) for s in [201,202,203]]
            summaries.append(dict(arm=arm,role=role,acc=float(np.mean([v['acc'] for v in r])),macro_f1=float(np.mean([v['macro_f1'] for v in r])),worst_rx_acc=float(min(rxmeans)),seed_sd_pp=float(np.std(seedmeans,ddof=1))))
    pairs=[('A1_mha','C2_lite_mha'),('A3_rfhstu','C2_lite_hstu'),('C2_lite_mha','E1_patch_mha'),('C2_lite_hstu','E2_patch_hstu')]
    delta_rows=[]
    for a,b in pairs:
        for role in ['dev_val','confirm']:
            local=[]
            for fold in [0,1]:
                for seed in [201,202,203]:
                    x,y=[metrics[c,fold,seed][role]['per_rx'] for c in [a,b]]
                    for rx in x:local.append(dict(contrast=a+'-'+b,role=role,fold=fold,seed=seed,rx=rx,delta_pp=100*(x[rx]['acc']-y[rx]['acc']),delta_f1_pp=100*(x[rx]['macro_f1']-y[rx]['macro_f1'])))
            delta_rows+=local;contrasts.append(dict(contrast=a+'-'+b,role=role,**paired(local),f1=paired(local,'delta_f1_pp')))
    csv_write(HERE/'receiver_metrics.csv',rows);csv_write(HERE/'paired_deltas.csv',delta_rows);csv_write(HERE/'paired_contrasts.csv',contrasts)
    write(HERE/'summary.json',dict(time=now(),new_fits=12,reused_fits=24,arm_summary=summaries,contrasts=contrasts))
    selected=[r for r in contrasts if r['role']=='dev_val' and r['contrast'].startswith('A')]
    supported=all(r['ci95_pp'][0]>0 for r in selected)
    write(HERE/'decision.json',dict(time=now(),status='HSTEM_ADVANTAGE_REMAINS_UNDER_MATCHED_PARAMETER_CONTROL' if supported else 'NO_UNIFORM_HSTEM_ADVANTAGE_UNDER_MATCHED_CONTROL',
        criterion='Both dev_val conditional RX95% intervals strictly above0; descriptive only, no multiplicity correction; confirm reporting only',new_architecture_search=False))
    audit=read(HERE/'architecture_audit.json')
    lines=['# S15 HStem参数匹配归因结果','',
        '12个新模型全部源CE选模，统一提交checkpoint SHA后才评估dev_val和历史confirm。旧24个匹配控制直接复用。','',
        '| 角色 | Stem/Mixer | RX准确率 % | Macro-F1 % | 最差RX % | seed SD pp |','|---|---|---:|---:|---:|---:|']
    for r in summaries:lines.append(f"|{r['role']}|{r['arm']}|{r['acc']:.3f}|{r['macro_f1']:.3f}|{r['worst_rx_acc']:.3f}|{r['seed_sd_pp']:.3f}|")
    lines+=['','| 对比方向 | 角色 | 差值 pp | 条件RX95%区间 | 正向配对 | 最差RX差值 pp |','|---|---|---:|---|---:|---:|']
    for r in contrasts:lines.append(f"|{r['contrast']}|{r['role']}|{r['mean_delta_pp']:+.3f}|[{r['ci95_pp'][0]:+.3f},{r['ci95_pp'][1]:+.3f}]|{r['positive_pairs']}/6|{r['worst_receiver_delta_pp']:+.3f}|")
    lines+=['','| 架构 | 参数 | CNN局部感受野 | Token步长 | 首Token中心 | BN层数 |','|---|---:|---:|---:|---:|---:|']
    for arm,a in audit.items():lines.append(f"|{arm}|{a['params']}|{a['cnn_receptive_field']}|{a['token_stride']}|{a['first_token_center']}|{a['bn_layers']}|")
    lines+=['','结论：'+('在两个Mixer上，HStem相对参数匹配Patch-Lite仍有正向条件区间证据。' if supported else '当前条件区间不支持两个Mixer上统一的HStem优势；应报告差异消失或不确定，不能继续预设层级结构获胜。'),
        '分组/低秩Patch-Lite同时改变归纳偏置，参数匹配没有唯一隔离层级性、感受野、BN和算力。去中间BN消融本阶段未运行。',
        'CNN感受野只对应卷积坐标；TrueIB/FFT全局前端使它不等于端到端物理感受野。Token形状、中心和步长保持匹配。',
        '成本见costs.csv，统一统计卷积/线性/稠密注意力；FFT及逐元素不完全覆盖。参数少不等于更快。',
        '所有区间条件于现有种子，RX是统计单位；未做多重比较校正。Confirm为历史已用接收机，不是新盲测。']
    (HERE/'RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')

def benchmark():
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from s15_models import build
    torch.set_num_threads(2);rows=[]
    for arm in read(HERE/'architecture_audit.json'):
        m=build(arm,123).eval();params=sum(p.numel() for p in m.parameters());blocks=m.net.encoder.blocks
        dense=0
        for b in blocks:
            if hasattr(b,'qkv'):dense+=2*32*(3*64*b.inner+b.inner*64)+4*32*32*b.inner
            else:dense+=2*32*(4*64*64+64*64)+4*32*32*64
        m.net.encoder.blocks=torch.nn.ModuleList()
        with torch.no_grad(),FlopCounterMode(display=False) as c:m(torch.randn(1,2,8192))
        flops=c.get_total_flops()+dense;m.net.encoder.blocks=blocks;m=m.cuda();x=torch.randn(1,2,8192,device='cuda')
        for repeat in range(3):
            with torch.no_grad():
                for _ in range(20):m(x)
                times=[]
                for _ in range(100):
                    a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)];a.record();m(x);b.record();b.synchronize();times.append(a.elapsed_time(b))
            rows.append(dict(arm=arm,repeat=repeat,params=params,counted_mflops=flops/1e6,p50_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95))))
        del m,x;torch.cuda.empty_cache()
    csv_write(HERE/'costs.csv',rows);write(HERE/'costs.json',dict(time=now(),scope='FP32 resident IQ to logits;3 repeats;20 warmup/100timings;FMA2,FFT/elementwise incomplete',rows=rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['report','cost']);a=p.parse_args();validate()
    report() if a.mode=='report' else benchmark()
