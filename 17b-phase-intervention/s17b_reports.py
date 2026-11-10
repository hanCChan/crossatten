import csv,sys
import numpy as np
from s17b_core import HERE,read,write,sha,now,validate
from s17_jobs import run_id
from s17_stats import delta_stats,gate

def csv_write(path,rows):
    if rows:
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows)

def d1_report():
    d=read(HERE/'decision_d1.json');rows=read(HERE/'D1_interventions.json')['rows'];csv_write(HERE/'D1_interventions.csv',rows)
    lines=['# S17B：相位受控干预','',f"固定出口：{d['status']}；入选强度：{d['selected_tau']}。",'',
        '| 强度 | fold | Accuracy变化pp | Macro-F1变化pp | 最差TX变化pp | EVM q95 | 幅频纹波q95 dB | 相位子空间覆盖 | 相位对齐率 | 保持 | 物理约束 | 覆盖 | 比噪声更对齐 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|']
    for level in d['levels']:
        for r in level['folds']:
            lines.append(f"|{level['tau']}|{r['fold']}|{r['delta_acc_pp']:.3f}|{r['delta_f1_pp']:.3f}|{r['worst_tx_delta_pp']:.3f}|{r['evm_q95']:.4f}|{r['ripple_q95_db']:.3f}|{r['phase_subspace_coverage']:.3f}|{r['phase_subspace_alignment']:.3f}|{r['preserved']}|{r['physical_bounds']}|{r['coverage']}|{r['specificity']}|")
    lines+=['','幅频纹波为每个FIR全频段最大偏差的95%分位数；表中为两个draw平均。非物理对照的纹波不适用，留空。',
        '相位子空间来自已冻结训练RX代理profile，rank3；对干预前后的相位曲率使用同一清洁幅度支持mask。覆盖和对齐是观测几何代理，未识别纯接收机相位响应。',
        '一阶相位附加通道不变不代表整个模型不变，原始FFT实虚部被保留；全局相位因此是实测对照，不要求网络logits严格不变。',
        '若任一保持/物理条件失败，或不足3fold满足覆盖及相对噪声的对齐优势，则该强度不准入；禁止追加强度搜索。']
    (HERE/'D1_RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')

def phase_report(phase):
    validate();jobs=read(HERE/f'jobs_{phase}.json');metrics={};rows=[];deltas=[];init={}
    for j in jobs:
        out=HERE/'runs'/run_id(j);m=read(out/'metrics.json');e=read(out/'held_metrics.json');cfg=read(out/'config.json')
        if sha(out/'model.pt')!=m['model_sha256'] or e['model_sha256']!=m['model_sha256']:raise ValueError('Checkpoint changed')
        key=(j['dataset'],j['fold']);init.setdefault(key,set()).add(cfg['initial_state_sha256'])
        metrics[j['dataset'],j['fold'],j['arm']]=e['held_source'];s=e['held_source']['summary']
        rows.append(dict(dataset=j['dataset'],fold=j['fold'],seed=j['seed'],arm=j['arm'],acc=100*s['acc_mean'],macro_f1=100*s['f1_mean'],
            worst_rx_acc=100*s['worst_acc'],source_val_acc=100*m['source_validation']['summary']['acc_mean'],
            source_to_held_gap_pp=100*(m['source_validation']['summary']['acc_mean']-s['acc_mean']),seconds=m['wall_seconds'],best_epoch=m['best_epoch'],
            peak_allocated_gib=m['peak_allocated_gib']))
    if any(len(v)!=1 for v in init.values()):raise ValueError('Arms are not paired at initialization')
    comparisons=[]
    for control in ['M0','N_phase','G_phase']:
        stats={}
        for dataset in sorted({j['dataset'] for j in jobs}):
            local=[]
            for fold in range(4):
                a=metrics[dataset,fold,'W_phase']['per_rx'];b=metrics[dataset,fold,control]['per_rx']
                if set(a)!=set(b):raise ValueError('Unpaired receiver sets')
                for rx in a:local.append(dict(dataset=dataset,fold=fold,rx=rx,control=control,delta_pp=100*(a[rx]['acc']-b[rx]['acc'])))
            deltas.extend(local);stats[dataset]=delta_stats(local)
        comparisons.append(dict(control=control,**gate(stats,phase!='pilot')))
    admitted=all(c['admitted'] for c in comparisons)
    result=dict(time=now(),phase=phase,admitted=admitted,status='GO_SOURCE_CONFIRMATION' if admitted else 'STOP_SOURCE_GATE',comparisons=comparisons,real_target_access=False)
    write(HERE/f'decision_{phase}.json',result);csv_write(HERE/f'{phase}_metrics.csv',rows);csv_write(HERE/f'{phase}_paired_deltas.csv',deltas)
    lines=['# S17B '+phase+'：相位扰动对比','',f"出口：{result['status']}。",'',
        '| 数据 | 方法 | 留出源RX Accuracy % | Macro-F1 % | 最差fold RX % | 源到留出差距pp | 每fit分钟 |','|---|---|---:|---:|---:|---:|---:|']
    for dataset in sorted({r['dataset'] for r in rows}):
        for arm in ['M0','W_phase','N_phase','G_phase']:
            r=[v for v in rows if v['dataset']==dataset and v['arm']==arm]
            lines.append(f"|{dataset}|{arm}|{np.mean([v['acc'] for v in r]):.3f}|{np.mean([v['macro_f1'] for v in r]):.3f}|{min(v['worst_rx_acc'] for v in r):.3f}|{np.mean([v['source_to_held_gap_pp'] for v in r]):.3f}|{np.mean([v['seconds'] for v in r])/60:.2f}|")
    lines+=['','W_phase须同时优于清洁M0、同EVM噪声N_phase与同EVM全局相位G_phase。M0/W/N/G参数、初始化、数据顺序及源CE训练协议配对，无额外网络容量。',
        '留出源RX并非真实目标CRD；RX条件bootstrap没有全流程多重比较校正。此为D0后的新问题，不改写原S17的幅频/IQ STOP。']
    (HERE/f'{phase.upper()}_RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n');return result

def final():
    d1=read(HERE/'decision_d1.json');phases={p.stem:read(p) for p in sorted(HERE.glob('decision_*.json')) if p.name!='decision_d1.json'}
    latest=phases.get('decision_confirm2',phases.get('decision_confirm1',phases.get('decision_pilot')))
    state='STOP_D1_GATE' if d1['selected_tau'] is None else 'READY_TO_PREREGISTER_FINAL_CRD' if 'decision_confirm2' in phases and latest['admitted'] else latest['status']
    count=len(list((HERE/'runs').glob('*/metrics.json')));write(HERE/'summary.json',dict(time=now(),status=state,formal_fits=count,d1=d1,source_phases=phases,real_target_access=False))
    lines=['# S17B 相位补充验证：最终结果','',f'科研出口：{state}；完成{count}个小型MHA训练。','',
        '原S17 D0幅频/IQ STOP保持。相位3/4fold复现性是本补充验证动机；新门槛在读取任何合成干预结果前注册。','',
        '- [D1保持、覆盖与负对照](D1_RESULTS.zh-CN.md)', '- [完整协议](README.zh-CN.md)']
    for phase in ['pilot','confirm1','confirm2']:
        if (HERE/f'{phase.upper()}_RESULTS.zh-CN.md').exists():lines.append(f'- [{phase}]({phase.upper()}_RESULTS.zh-CN.md)')
    if d1['selected_tau'] is None:lines+=['','固定干预未同时满足所有保持、物理、覆盖及对齐条件，因此不进入增强方法训练。不能据此断言所有接收机相位机制不存在或不可利用。']
    lines+=['','仅访问合法源fit RX及日期1–2；后续源留出RX仅在相应阶段统一封存checkpoint后评估。真实目标RX/date3–4与Shen blind未打开。']
    (HERE/'RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n');return state,count

if __name__=='__main__':d1_report() if sys.argv[1]=='d1' else final() if sys.argv[1]=='final' else phase_report(sys.argv[1])
