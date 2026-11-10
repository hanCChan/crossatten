"""Full source outcomes, conditional uncertainty and explicit STOP decisions."""
import csv,sys
import numpy as np
from s17_core import HERE,read,write,sha,now,validate
from s17_jobs import run_id
from s17_stats import delta_stats,gate

def csv_write(path,rows):
    if rows:
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows)

def d0_report():
    rows=[];lines=['# S17 D0：源RX机制诊断','',
        '仅在每fold合法源fit RX的训练70%估计描述子（split16001），TX/day条件化。错误关联复用冻结S16 MHA源验证混淆矩阵的TX宏平均错误；该错误合并日期1–2，不能单独控制逐日错误。没有逐包跨RX配对或纯接收机传递函数估计。','',
        '| fold | 描述子 | 部分RX eta² | 跨日相关 | 跨TX相关 | 留TX识别RX % | 置乱p | 错误关联rho | 稳定 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for p in sorted(HERE.glob('D0_fold*.json')):
        d=read(p)
        for family,v in d['families'].items():
            r=dict(fold=d['job']['fold'],family=family,**{k:v[k] for k in ['partial_rx_eta2','cross_day_correlation','cross_tx_half_correlation','leave_tx_out_rx_accuracy','rx_permutation_p','source_val_error_spearman','stable']});rows.append(r)
            lines.append(f"|{r['fold']}|{family}|{r['partial_rx_eta2']:.3f}|{r['cross_day_correlation']:.3f}|{r['cross_tx_half_correlation']:.3f}|{100*r['leave_tx_out_rx_accuracy']:.2f}|{r['rx_permutation_p']:.3f}|{r['source_val_error_spearman']:.3f}|{r['stable']}|")
    decision=read(HERE/'decision_d0.json');lines+=['',f"固定出口：{decision['status']}；可进入D1候选：{decision['eligible']}。",'',
        '接收机标签置乱在每个TX/day块内执行；错误关联以RX为单位，避免把同RX的数据包误当成独立接收机样本。阈值是预注册的探索筛选标准，不是机制因果证明。',
        'RX相关差异仍可能包含固定传播路径、天线与接收机硬件；跨日和跨TX复现降低部分混杂，不能消除全部混杂。',
        '相位描述子为FFT相位二阶差分的圆统计，消除常相位/线性相位的理想项，但仍受CFO、噪声和信道影响；本轮不据此直接构造接收机相位补偿器。']
    (HERE/'D0_RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n');csv_write(HERE/'D0_metrics.csv',rows)

def phase_report(phase):
    validate();jobs=read(HERE/f'jobs_{phase}.json');reg=read(HERE/f'registration_{phase}.json');metrics={};rows=[];receiver_rows=[]
    for j in jobs:
        out=HERE/'runs'/run_id(j);m=read(out/'metrics.json');e=read(out/'held_metrics.json')
        if sha(out/'model.pt')!=m['model_sha256'] or e['model_sha256']!=m['model_sha256']:raise ValueError('Selected checkpoint changed')
        metrics[j['dataset'],j['fold'],j['arm']]=e['held_source'];s=e['held_source']['summary']
        rows.append(dict(dataset=j['dataset'],fold=j['fold'],seed=j['seed'],arm=j['arm'],acc=100*s['acc_mean'],macro_f1=100*s['f1_mean'],
            worst_rx_acc=100*s['worst_acc'],source_val_acc=100*m['source_validation']['summary']['acc_mean'],
            source_to_held_gap_pp=100*(m['source_validation']['summary']['acc_mean']-s['acc_mean']),seconds=m['wall_seconds'],best_epoch=m['best_epoch']))
        for rx,v in e['held_source']['per_rx'].items():receiver_rows.append(dict(dataset=j['dataset'],fold=j['fold'],seed=j['seed'],arm=j['arm'],rx=rx,acc=100*v['acc'],macro_f1=100*v['macro_f1']))
    decisions={};deltas=[]
    for family in reg['families']:
        comparisons=[]
        for baseline in ['M0','N_'+family]:
            stats={}
            for dataset in sorted({j['dataset'] for j in jobs}):
                r=[]
                for fold in range(4):
                    a=metrics[dataset,fold,'W_'+family]['per_rx'];b=metrics[dataset,fold,baseline]['per_rx']
                    if set(a)!=set(b):raise ValueError('Unpaired RX sets')
                    for rx in a:r.append(dict(dataset=dataset,fold=fold,rx=rx,candidate=family,control=baseline,delta_pp=100*(a[rx]['acc']-b[rx]['acc'])))
                deltas.extend(r);stats[dataset]=delta_stats(r)
            comparisons.append(dict(control=baseline,**gate(stats,phase!='pilot')))
        decisions[family]=dict(admitted=all(c['admitted'] for c in comparisons),comparisons=comparisons)
    selected=[f for f,v in decisions.items() if v['admitted']]
    result=dict(time=now(),phase=phase,selected=selected,decisions=decisions,status='GO_CONFIRMATION' if selected else 'STOP_SOURCE_GATE',real_target_access=False)
    write(HERE/f'decision_{phase}.json',result);csv_write(HERE/f'{phase}_metrics.csv',rows);csv_write(HERE/f'{phase}_receivers.csv',receiver_rows);csv_write(HERE/f'{phase}_paired_deltas.csv',deltas)
    lines=['# S17 '+phase+'：物理约束扰动Pilot','',f"结论：{result['status']}；准入：{selected}",'',
        '| 数据 | 模型 | 留出源RX Accuracy % | Macro-F1 % | 源到留出差距 pp |','|---|---|---:|---:|---:|']
    for dataset in sorted({r['dataset'] for r in rows}):
        for arm in sorted({r['arm'] for r in rows}):
            r=[v for v in rows if v['dataset']==dataset and v['arm']==arm]
            lines.append(f"|{dataset}|{arm}|{np.mean([v['acc'] for v in r]):.3f}|{np.mean([v['macro_f1'] for v in r]):.3f}|{np.mean([v['source_to_held_gap_pp'] for v in r]):.3f}|")
    lines+=['','M0为HStem+MHA清洁训练；W为RX相关源统计约束扰动；N为相同逐包EVM的复高斯噪声控制。全部源CE选模，无目标调参。',
        '相对M0和N均需满足门槛；不能把一般噪声正则化收益直接解释为所建模物理机制的贡献。',
        'RX聚类区间条件于本轮已训练种子，未经多重比较校正；两轮独立种子确认都通过后才可另行考虑真实目标评估。']
    (HERE/f'{phase.upper()}_RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n');return result

def final():
    decisions={p.stem:read(p) for p in sorted(HERE.glob('decision_*.json'))}
    last=read(HERE/'decision_confirm2.json') if (HERE/'decision_confirm2.json').exists() else read(HERE/'decision_confirm1.json') if (HERE/'decision_confirm1.json').exists() else read(HERE/'decision_pilot.json') if (HERE/'decision_pilot.json').exists() else read(HERE/'decision_d1.json') if (HERE/'decision_d1.json').exists() else read(HERE/'decision_d0.json')
    status='READY_TO_PREREGISTER_FINAL_CRD' if (HERE/'decision_confirm2.json').exists() and last['selected'] else last['status']
    fit_count=len(list((HERE/'runs').glob('*/metrics.json')));result=dict(time=now(),status=status,formal_fits=fit_count,decisions=decisions,real_target_access=False)
    write(HERE/'summary.json',result)
    lines=['# S17 接收机失真机制审计：最终结果','',f'完成{fit_count}个小型MHA训练，科研出口：{status}。','',
        '本轮不再优化HSTU或Qwen3。S15/S16结论及历史追踪文件保持冻结。','',
        '- [D0统计与混杂边界](D0_RESULTS.zh-CN.md)', '- [完整预注册协议](README.zh-CN.md)', '- [技术来源](SOURCES.md)']
    for phase in ['pilot','confirm1','confirm2']:
        if (HERE/f'{phase.upper()}_RESULTS.zh-CN.md').exists():lines.append(f'- [{phase}]({phase.upper()}_RESULTS.zh-CN.md)')
    if not (HERE/'decision_d1.json').exists():lines+=['','D0没有找到达到固定要求的稳定RX差异与模型错误关联，因此未建立增强候选，也未进入D1或方法Pilot。该出口符合诊断方案，不构成物理机制已被证伪。']
    lines+=['','全部结论限于源域RX及日期1–2；不读取真实目标RX、日期3–4或Shen blind。无科学失败自动重试，不扩大强度、种子或方法搜索来救结果。']
    (HERE/'RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n');return result

if __name__=='__main__':
    d0_report() if sys.argv[1]=='d0' else final() if sys.argv[1]=='final' else phase_report(sys.argv[1])
