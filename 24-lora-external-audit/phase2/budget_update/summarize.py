"""All registered comparisons are reported; deferred cells remain NA."""
import csv,json
from pathlib import Path
import numpy as np
from common import HERE,PHASE,read,digest,atomic_json,policy,verify_seal,jobid,cohort_jobs

def main():
    a,r=policy();verify_seal('baseline48')
    if read(HERE/'EVALUATION_baseline48.json')['status']!='COMPLETE':raise RuntimeError('baseline inference incomplete')
    records=[]
    for p in sorted((HERE/'evaluations').glob('*/metrics.json')):
        result=read(p)
        if result['status']!='COMPLETE_REPORT_ONLY':continue
        if result['amendment_sha256']!=digest(HERE/'AMENDMENT.json'):raise ValueError('amendment changed')
        # Incomplete ASK cohorts never contribute partial target results to a main table.
        if result['job']['method']=='ask':
            state=HERE/'EVALUATION_native_ask6.json'
            if not state.exists() or read(state)['status']!='COMPLETE':continue
        records+=result['rows']
    summary=[];pairs=[]
    for view in ['native','strict_ib']:
        for method in ['h0','ce','dann','fan','ib','ask']:
            rows=[v for v in records if v['method']==method and v['view']==('native' if method=='h0' else view)]
            if method=='ask' and view=='strict_ib':
                summary.append(dict(view=view,method=method,status='DEFERRED_BY_USER_BUDGET',fits=0));continue
            if not rows:
                summary.append(dict(view=view,method=method,status='PENDING_COMPLETE_NATIVE_COHORT',fits=0));continue
            keyset={(v['fold'],v['seed'],v['rx']) for v in rows}
            if len(keyset)!=30:raise ValueError('incomplete paired result group')
            fold_seed=[np.mean([v['acc'] for v in rows if v['fold']==f and v['seed']==s]) for f in (0,1) for s in (201,202,203)]
            seed_means=[np.mean([v['acc'] for v in rows if v['seed']==s]) for s in (201,202,203)]
            rx_means={rx:float(np.mean([v['acc'] for v in rows if v['rx']==rx])) for rx in sorted({v['rx'] for v in rows})}
            summary.append(dict(view=view,method=method,status='COMPLETE_REPORT_ONLY',fits=6,acc=float(np.mean(fold_seed)),
                                macro_f1=float(np.mean([v['macro_f1'] for v in rows])),fit_sd=float(np.std(fold_seed,ddof=1)),
                                seed_sd=float(np.std(seed_means,ddof=1)),worst_rx=min(rx_means,key=rx_means.get),worst_rx_acc=min(rx_means.values()),
                                per_fold_acc={str(f):float(np.mean([v['acc'] for v in rows if v['fold']==f])) for f in (0,1)},
                                within_fold_seed_sd={str(f):float(np.std([np.mean([v['acc'] for v in rows if v['fold']==f and v['seed']==s]) for s in (201,202,203)],ddof=1)) for f in (0,1)}))
        for aa,bb in [('h0','ce'),('h0','dann'),('h0','fan'),('h0','ib'),('h0','ask'),('dann','ce'),('ib','fan')]:
            ar={(v['fold'],v['seed'],v['rx']):v for v in records if v['method']==aa and v['view']==('native' if aa=='h0' else view)}
            br={(v['fold'],v['seed'],v['rx']):v for v in records if v['method']==bb and v['view']==view}
            if not br:continue
            if ar.keys()!=br.keys():raise ValueError('unpaired comparison')
            for metric in ['acc','macro_f1']:
                rx_deltas={rx:float(np.mean([ar[k][metric]-br[k][metric] for k in ar if k[2]==rx])) for rx in sorted({k[2] for k in ar})}
                values=np.array(list(rx_deltas.values()));rng=np.random.default_rng(24016)
                boot=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
                deltas=[np.mean([ar[k][metric]-br[k][metric] for k in ar if k[:2]==(f,s)]) for f in (0,1) for s in (201,202,203)]
                pairs.append(dict(view=view,a=aa,b=bb,metric=metric,delta=float(values.mean()),conditional_rx_cluster_ci=np.quantile(boot,[.025,.975]).tolist(),
                                  unique_rx=len(values),positive_fold_seed_pairs=sum(d>0 for d in deltas),total_fold_seed_pairs=6,
                                  worst_rx_delta=min(values),per_rx_delta=rx_deltas,no_multiple_comparison_adjustment=True))
    with (HERE/'PER_RX.csv').open('w',newline='',encoding='utf-8') as f:
        scalar=[{k:v for k,v in row.items() if k!='confusion'} for row in records]
        writer=csv.DictWriter(f,fieldnames=list(scalar[0]),lineterminator='\n');writer.writeheader();writer.writerows(scalar)
    atomic_json(HERE/'RESULTS.json',dict(summary=summary,paired_comparisons=pairs,amendment_sha256=digest(HERE/'AMENDMENT.json'),
                statistical_scope='Ten unique historically developed RX; seeds/folds are not additional devices. RX cluster intervals condition on three seeds, no multiplicity correction; report only, no selection.',
                strict_ask='Deferred, no full same-band comparison claim for ASKNet',new_independent_blind_test=False))
    text='# S24 预算修订后的统一 Dev 对比\n\nCE、DANN、FAN、IB共48个checkpoint与六个冻结H0在读取Dev前已整体封存。全部Dev推断无加噪、无适应、逐包分类；Native保留方法原生前处理，Strict-IB先统一投影。此为历史开发RX上的报告，不是新独立盲测。\n'
    text+='\n已完成的48个基线和H0采用CPU FP32网络推断（原CIS/投影float64保留），以便与当前GPU训练并行；ASK Native六模型完成后采用GPU FP32评估。硬件耗时另按同机GPU测量。\n'
    for view,title in [('native','Native完整方法'),('strict_ib','Strict-IB信息预算控制')]:
        text+=f'\n## {title}\n\n| 方法 | fit | RX等权Acc% ± fit SD | Macro-F1% | 最差RX均值% | 状态 |\n|---|---:|---:|---:|---:|---|\n'
        for row in summary:
            if row['view']!=view:continue
            if row['status']=='COMPLETE_REPORT_ONLY':text+=f"|{row['method']}|6|{row['acc']*100:.3f} ± {row['fit_sd']*100:.3f}|{row['macro_f1']*100:.3f}|{row['worst_rx_acc']*100:.3f}|完成|\n"
            else:text+=f"|{row['method']}|0|NA|NA|NA|{row['status']}|\n"
    text+='\n## 预登记配对差值\n\n下表为A−B的准确率差；按唯一RX聚类、先平均种子、10000次bootstrap；条件95%区间没有多重比较校正。F1配对差及区间同时保存在RESULTS.json。\n\n| 视图 | A−B | 差值pp | RX条件95%区间pp | 正向fold/seed对 |\n|---|---|---:|---|---:|\n'
    for p in pairs:
        if p['metric']=='acc':text+=f"|{p['view']}|{p['a']}−{p['b']}|{p['delta']*100:+.3f}|[{p['conditional_rx_cluster_ci'][0]*100:+.3f}, {p['conditional_rx_cluster_ci'][1]*100:+.3f}]|{p['positive_fold_seed_pairs']}/6|\n"
    text+='\nASK Strict-IB六项由用户按计算预算整体暂缓；不能据此声称完成ASK的同带内信息预算比较。没有根据Dev成绩改变Native六项的种子、epoch、模型、前处理或checkpoint选择。HStem历史机制消融、±4样点敏感性与独立泛化边界保持原证据。\n'
    (HERE/'RESULTS.zh-CN.md').write_text(text,encoding='utf-8',newline='\n')
    print(text,flush=True)

if __name__=='__main__':main()
