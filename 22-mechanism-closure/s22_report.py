"""All four fixed comparisons, with practical gates and explicit uncertainty."""
import argparse
import csv
import itertools
import numpy as np
from s22_core import HERE,read,write,now,job_id
from s22_models import ARMS,REFERENCES
from s15_stats import paired

PAIRS=[('H1','S1'),('S1','P0'),('H0','S2'),('S2','H2')]
def csv_write(name,rows):
    if not rows:return
    with (HERE/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows)
def outpath(arm,fold,seed):return HERE/'runs'/(f'reused__{arm}__f{fold}__s{seed}' if arm in REFERENCES else job_id(dict(arm=arm,fold=fold,seed=seed)))
def ce_value(logits,y):
    a=np.asarray(logits,dtype=np.float64);maximum=a.max(1)
    return float(np.mean(maximum+np.log(np.exp(a-maximum[:,None]).sum(1))-a[np.arange(len(y)),y]))
def gate(stats):
    return bool(stats['mean_delta_pp']>=.5 and stats['positive_pairs']>=5 and stats['ci95_pp'][0]>0 and all(v>0 for v in stats['fold_delta_pp'].values()) and stats['worst_receiver_delta_pp']>=-2)
def equivalent(stats):
    return bool(stats['ci95_pp'][0]>=-.5 and stats['ci95_pp'][1]<=.5 and all(abs(v)<=.5 for v in stats['fold_delta_pp'].values()))

def report():
    from s22_evaluate import sealed
    sealed();rows=[];tx=[];geometry=[]
    for arm,fold,seed in itertools.product(ARMS,[0,1],[201,202,203]):
        out=outpath(arm,fold,seed);e=read(out/'evaluation.json')
        for role in ['dev_val','confirm']:
            with np.load(out/f'diagnostics_{role}.npz') as z:
                distances=[]
                for rx,m in e['metrics'][role]['per_rx'].items():
                    y=z['y_'+rx];lg=z[rx];pred=lg.argmax(1);margin=z['margin_'+rx]
                    rows.append(dict(arm=arm,fold=fold,seed=seed,role=role,rx=rx,acc=100*m['acc'],macro_f1=100*m['macro_f1'],ce=ce_value(lg,y),margin=float(margin.mean()),errors=int((pred!=y).sum()),packets=len(y)))
                    for label in range(10):
                        ids=y==label
                        tx.append(dict(arm=arm,fold=fold,seed=seed,role=role,rx=rx,tx=label,packets=int(ids.sum()),errors=int((pred[ids]!=label).sum()),margin=float(margin[ids].mean())))
                    v=z['distances_'+rx][np.triu_indices(10,1)].astype(np.float64);distances.append((rx,v/max(v.mean(),1e-12)))
                for (r,v),(s,w) in itertools.combinations(distances,2):
                    geometry.append(dict(arm=arm,fold=fold,seed=seed,role=role,rx_a=r,rx_b=s,distance_correlation=float(np.corrcoef(v,w)[0,1]) if min(v.std(),w.std())>1e-12 else None,normalized_distance_rmse=float(np.sqrt(np.mean((v-w)**2)))))
    summaries=[]
    for arm,role in itertools.product(ARMS,['dev_val','confirm']):
        local=[r for r in rows if r['arm']==arm and r['role']==role]
        seedmeans=[np.mean([r['acc'] for r in local if r['seed']==s]) for s in [201,202,203]]
        rxmeans=[np.mean([r['acc'] for r in local if r['rx']==rx]) for rx in sorted(set(r['rx'] for r in local))]
        summaries.append(dict(arm=arm,role=role,**{k:float(np.mean([r[k] for r in local])) for k in ['acc','macro_f1','ce','margin']},seed_sd_pp=float(np.std(seedmeans,ddof=1)),worst_rx_acc=float(min(rxmeans))))
    lookup={(r['arm'],r['fold'],r['seed'],r['role'],r['rx']):r for r in rows}
    contrasts=[];deltas=[];txdeltas=[]
    for a,b in PAIRS:
        for role in ['dev_val','confirm']:
            local=[]
            for x in rows:
                if x['arm']!=a or x['role']!=role:continue
                y=lookup[b,x['fold'],x['seed'],role,x['rx']]
                local.append(dict(contrast=a+'-'+b,role=role,fold=x['fold'],seed=x['seed'],rx=x['rx'],delta_pp=x['acc']-y['acc'],delta_f1_pp=x['macro_f1']-y['macro_f1'],delta_ce=x['ce']-y['ce'],delta_margin=x['margin']-y['margin']))
            stats=paired(local);stats['fold_delta_pp']={str(f):float(np.mean([r['delta_pp'] for r in local if r['fold']==f])) for f in [0,1]}
            stats['practical_signal']=gate(stats);stats['conditional_equivalence']=equivalent(stats)
            contrasts.append(dict(contrast=a+'-'+b,role=role,**stats,f1=paired(local,'delta_f1_pp')));deltas+=local
            right={(r['fold'],r['seed'],r['rx'],r['tx']):r for r in tx if r['arm']==b and r['role']==role}
            for x in tx:
                if x['arm']!=a or x['role']!=role:continue
                y=right[x['fold'],x['seed'],x['rx'],x['tx']]
                txdeltas.append(dict(contrast=a+'-'+b,role=role,fold=x['fold'],seed=x['seed'],rx=x['rx'],tx=x['tx'],error_count_delta=x['errors']-y['errors'],error_rate_delta_pp=100*(x['errors']/x['packets']-y['errors']/y['packets']),margin_delta=x['margin']-y['margin']))
    for name,data in [('receiver_metrics.csv',rows),('tx_rx_metrics.csv',tx),('class_geometry.csv',geometry),('paired_deltas.csv',deltas),('paired_contrasts.csv',contrasts),('tx_rx_deltas.csv',txdeltas),('arm_summary.csv',summaries)]:csv_write(name,data)
    primary={r['contrast']:r for r in contrasts if r['role']=='dev_val'}
    decision=dict(time=now(),status='COMPLETE_STOP',formal_fits=12,reused_fits=24,main_model='Fixed H0 HStem+MHA 133690 parameters',signals={k:r['practical_signal'] for k,r in primary.items()},conditional_equivalence={k:r['conditional_equivalence'] for k,r in primary.items()},no_additional_training=True,blind_evaluated=False,S1_deviation='Full-rank round-trip smoothing, not center-crop')
    write(HERE/'summary.json',dict(time=now(),arms=summaries,contrasts=contrasts));write(HERE/'decision.json',decision)
    lines=['# S22 机制闭环结果','', '主模型预先固定为H0 HStem+MHA。新增12次拟合、复用24次；全部新权重封存后统一评价。S1采用训练前修订的256→340→256满秩平滑，不能写成原方案的中心裁剪。','', '|模型|角色|Accuracy %|Macro-F1 %|CE|margin|最差RX %|seed SD pp|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:lines.append(f"|{r['arm']}|{r['role']}|{r['acc']:.3f}|{r['macro_f1']:.3f}|{r['ce']:.4f}|{r['margin']:.3f}|{r['worst_rx_acc']:.3f}|{r['seed_sd_pp']:.3f}|")
    lines+=['','|比较|角色|ΔAccuracy pp|条件RX95%区间|正配对|最差RX差值 pp|实用信号|','|---|---|---:|---|---:|---:|---|']
    for r in contrasts:lines.append(f"|{r['contrast']}|{r['role']}|{r['mean_delta_pp']:+.3f}|[{r['ci95_pp'][0]:+.3f},{r['ci95_pp'][1]:+.3f}]|{r['positive_pairs']}/6|{r['worst_receiver_delta_pp']:+.3f}|{r['practical_signal']}|")
    lines+=['','实用信号预注册为：均值至少+0.5pp、至少5/6正向、两个开发折均为正、条件RX区间下界>0、最差RX不低于−2pp。未触发信号不等于无效或等价。仅当区间完全位于±0.5pp且两个折均在此范围内，才报告条件等价信号。历史confirm仅描述，不参与判定。','', 'S1-P0检验可逆平滑参数化及其优化效果；H1-S1仍混合局部支持与频响等因素。H0-S2检验完整BN相对仅保留仿射的增量，但不能分离批统计、运行统计和优化动力学。四个比较共同报告，未作多重比较校正，区间条件于既有种子，不能把seed当独立RX。','', '本轮到此停止，不追加H3、裁剪版本、kernel、损失或盲测。论文材料见paper/；编码修订仅改显示文字，不改历史科学结果。']
    (HERE/'RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')

if __name__=='__main__':report()
