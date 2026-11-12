"""Fixed comparisons, receiver risks, margins, geometry and computational cost."""
import argparse
import csv
import itertools
import numpy as np
from s21_core import HERE, read, write, now, job_id, validate
from s15_stats import paired

def csv_write(name,rows):
    if not rows:return
    with (HERE/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');writer.writeheader();writer.writerows(rows)

def recovery(h0,p0,h1):
    denominator=h0-p0
    return (h1-p0)/denominator if denominator>=.25 else None

def bn_signal(contrast):
    return bool(contrast['mean_delta_pp']<=-1 and contrast['negative_pairs']>=5 and all(v<0 for v in contrast['fold_delta_pp'].values()))

def outpath(arm,fold,seed):
    return HERE/'runs'/(f'reused__{arm}__f{fold}__s{seed}' if arm in ['H0','P0'] else job_id(dict(arm=arm,fold=fold,seed=seed)))

def report():
    from s21_evaluate import sealed
    sealed();rows=[];txrows=[];georows=[];lookup={}; summaries=[]
    for arm,fold,seed in itertools.product(['H0','P0','H1','H2'],[0,1],[201,202,203]):
        p=outpath(arm,fold,seed); e=read(p/'evaluation.json');lookup[arm,fold,seed]=e
        for role in ['dev_val','confirm']:
            with np.load(p/f'diagnostics_{role}.npz') as z:
                vectors=[]
                for rx,m in e['metrics'][role]['per_rx'].items():
                    y=z['y_'+rx]; lg=z[rx]; pred=lg.argmax(1); margin=z['margin_'+rx]
                    rows.append(dict(arm=arm,fold=fold,seed=seed,role=role,rx=rx,acc=100*m['acc'],macro_f1=100*m['macro_f1'],margin_mean=float(margin.mean()),margin_median=float(np.median(margin)),margin_p10=float(np.quantile(margin,.1)),errors=int((y!=pred).sum()),packets=len(y)))
                    for tx in range(10):
                        ids=y==tx
                        txrows.append(dict(arm=arm,fold=fold,seed=seed,role=role,rx=rx,tx=tx,packets=int(ids.sum()),errors=int((pred[ids]!=tx).sum()),margin_mean=float(margin[ids].mean())))
                    v=z['distances_'+rx][np.triu_indices(10,1)].astype(np.float64)
                    vectors.append((rx,v/max(v.mean(),1e-12)))
                for (r,v),(s,w) in itertools.combinations(vectors,2):
                    correlation=float(np.corrcoef(v,w)[0,1]) if min(v.std(),w.std())>1e-12 else None
                    georows.append(dict(arm=arm,fold=fold,seed=seed,role=role,rx_a=r,rx_b=s,distance_correlation=correlation,normalized_distance_rmse=float(np.sqrt(np.mean((v-w)**2)))))
    csv_write('receiver_metrics.csv',rows);csv_write('tx_rx_metrics.csv',txrows);csv_write('class_geometry.csv',georows)
    for arm,role in itertools.product(['H0','P0','H1','H2'],['dev_val','confirm']):
        local=[r for r in rows if r['arm']==arm and r['role']==role]
        seedmeans=[np.mean([r['acc'] for r in local if r['seed']==s]) for s in [201,202,203]]
        rxmeans=[np.mean([r['acc'] for r in local if r['rx']==rx]) for rx in sorted(set(r['rx'] for r in local))]
        summaries.append(dict(arm=arm,role=role,acc=float(np.mean([r['acc'] for r in local])),macro_f1=float(np.mean([r['macro_f1'] for r in local])),seed_sd_pp=float(np.std(seedmeans,ddof=1)),worst_rx_acc=float(min(rxmeans)),margin_mean=float(np.mean([r['margin_mean'] for r in local]))))
    contrasts=[];deltas=[];txdeltas=[]
    for a,b in [('H0','P0'),('H1','P0'),('H1','H0'),('H2','H0')]:
        for role in ['dev_val','confirm']:
            local=[]
            for fold,seed in itertools.product([0,1],[201,202,203]):
                x,y=[lookup[c,fold,seed]['metrics'][role]['per_rx'] for c in [a,b]]
                for rx in x:
                    local.append(dict(contrast=a+'-'+b,role=role,fold=fold,seed=seed,rx=rx,delta_pp=100*(x[rx]['acc']-y[rx]['acc']),delta_f1_pp=100*(x[rx]['macro_f1']-y[rx]['macro_f1'])))
            deltas+=local;stats=paired(local)
            stats['negative_pairs']=sum(np.mean([r['delta_pp'] for r in local if r['fold']==f and r['seed']==s])<0 for f,s in itertools.product([0,1],[201,202,203]))
            stats['fold_delta_pp']={str(f):float(np.mean([r['delta_pp'] for r in local if r['fold']==f])) for f in [0,1]}
            contrasts.append(dict(contrast=a+'-'+b,role=role,**stats,f1=paired(local,'delta_f1_pp')))
            right={(r['fold'],r['seed'],r['rx'],r['tx']):r for r in txrows if r['arm']==b and r['role']==role}
            for r in txrows:
                if r['arm']!=a or r['role']!=role:continue
                s=right[r['fold'],r['seed'],r['rx'],r['tx']]
                txdeltas.append(dict(contrast=a+'-'+b,role=role,fold=r['fold'],seed=r['seed'],rx=r['rx'],tx=r['tx'],error_count_delta=r['errors']-s['errors'],error_rate_delta_pp=100*(r['errors']/r['packets']-s['errors']/s['packets']),margin_delta=r['margin_mean']-s['margin_mean']))
    csv_write('paired_deltas.csv',deltas);csv_write('paired_contrasts.csv',contrasts);csv_write('tx_rx_deltas.csv',txdeltas)
    dev={r['arm']:r for r in summaries if r['role']=='dev_val'}
    rho=recovery(dev['H0']['acc'],dev['P0']['acc'],dev['H1']['acc'])
    h1=next(r for r in contrasts if r['contrast']=='H1-P0' and r['role']=='dev_val')
    h2=next(r for r in contrasts if r['contrast']=='H2-H0' and r['role']=='dev_val')
    overlap=bool(rho is not None and rho>=.5 and all(v>0 for v in h1['fold_delta_pp'].values()))
    bn=bn_signal(h2)
    conclusion=('Overlap-Patch recovered at least half the gap in both fold directions; filter/support confounds remain.' if overlap else 'This fixed overlap/interpolation intervention did not meet the registered recovery signal; it does not falsify all local-context hypotheses.')
    conclusion+=' '+('Removing tokenizer BN caused consistent material degradation; normalization/affine/optimization effects remain inseparable.' if bn else 'Removing tokenizer BN did not meet the registered consistent material-degradation signal; smaller effects are still possible.')
    decision=dict(time=now(),status='COMPLETE_STOP',formal_fits=12,reused_baselines=12,recovery_ratio=rho,overlap_followup_signal=overlap,bn_importance_signal=bn,conclusion=conclusion,H3_executed=False,additional_search_authorized=False)
    write(HERE/'summary.json',dict(time=now(),arms=summaries,contrasts=contrasts,decision=decision));write(HERE/'decision.json',decision)
    lines=['# S21：HStem 机制验证结果','', '12 次新增拟合；所有 checkpoint 先按源 RX 验证 CE 选择并统一封存，随后评价既有 dev RX 和历史 confirm。H0/P0 各复用 6 次训练，归档 float32 预测逐元素严格恢复。','', '| 模型 | 角色 | RX等权 Accuracy % | Macro-F1 % | 最差RX % | seed SD pp | 均值margin |','|---|---|---:|---:|---:|---:|---:|']
    for r in summaries:lines.append(f"|{r['arm']}|{r['role']}|{r['acc']:.3f}|{r['macro_f1']:.3f}|{r['worst_rx_acc']:.3f}|{r['seed_sd_pp']:.3f}|{r['margin_mean']:.3f}|")
    lines+=['','| 对比 | 角色 | ΔAccuracy pp | 条件RX95%区间 | 正配对 | 最差RX变化 pp |','|---|---|---:|---|---:|---:|']
    for r in contrasts:lines.append(f"|{r['contrast']}|{r['role']}|{r['mean_delta_pp']:+.3f}|[{r['ci95_pp'][0]:+.3f},{r['ci95_pp'][1]:+.3f}]|{r['positive_pairs']}/6|{r['worst_receiver_delta_pp']:+.3f}|")
    lines+=['',f"H1 恢复率（总体均值之比）：{rho:.3%}。" if rho is not None else 'H1 恢复率：分母不足，未定义。',f"H1-P0 两开发折变化：{h1['fold_delta_pp']}。",'',conclusion,'', '解释边界：H1 同时改变局部覆盖、滤波频响、平滑与计算量，不能隔离重叠的唯一因果作用。H2 同时去除 BN 的归一化与仿射作用，退化不等于证明 BN 去除了 RX 特征。类别几何、margin 为辅助描述证据。区间以 RX 为抽样单位、条件于既有种子，未做多重比较校正。历史 confirm 已使用，不是新盲测。','', '逐 RX、TX×RX 错误与 margin 变化见 CSV；类别距离结构见 class_geometry.csv，使用每 RX 十个 TX 的64维最终均值特征，45个类间距离除以自身均值后比较。模型成本见 costs.csv；计数不完全覆盖 FFT、插值、归一化和逐元素操作。','', '本轮结束：不自动启动 H3、抗混叠新训练、kernel 搜索或 WiSig 新模型。']
    (HERE/'RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    update_manuscript(dev,rho,h1,h2)

def update_manuscript(dev,rho,h1,h2):
    p=HERE/'MANUSCRIPT_DRAFT.zh-CN.md';text=p.read_text(encoding='utf-8')
    marker='<!-- S21_RESULTS -->'
    addition=f"S21 完成12次新增拟合。H0/P0/H1/H2 的 RX 等权开发准确率分别为 {dev['H0']['acc']:.3f}% / {dev['P0']['acc']:.3f}% / {dev['H1']['acc']:.3f}% / {dev['H2']['acc']:.3f}%。H1−P0 为 {h1['mean_delta_pp']:+.3f} pp；H2−H0 为 {h2['mean_delta_pp']:+.3f} pp。\n\n"+read(HERE/'decision.json')['conclusion']
    p.write_text(text.split(marker)[0]+marker+'\n\n'+addition+'\n',encoding='utf-8',newline='\n')

def cost():
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from s21_models import build
    validate();torch.set_num_threads(2);rows=[]
    for arm in ['H0','P0','H1','H2']:
        m=build(arm,123).eval(); params=sum(p.numel() for p in m.parameters());blocks=m.net.encoder.blocks
        dense=sum(2*32*(3*64*b.inner+b.inner*64)+4*32*32*b.inner for b in blocks)
        m.net.encoder.blocks=torch.nn.ModuleList()
        with torch.no_grad(),FlopCounterMode(display=False) as count:m(torch.randn(1,2,8192))
        flops=count.get_total_flops()+dense;m.net.encoder.blocks=blocks;m=m.cuda()
        for batch in [1,128]:
            x=torch.randn(batch,2,8192,device='cuda')
            for repeat in range(3):
                with torch.no_grad():
                    for _ in range(20):m(x)
                    times=[]
                    for _ in range(100):
                        a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)];a.record();m(x);b.record();b.synchronize();times.append(a.elapsed_time(b))
                rows.append(dict(arm=arm,repeat=repeat,batch=batch,params=params,counted_macs=flops/2,counted_mflops=flops/1e6,p50_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),packets_per_second=batch*1000/float(np.mean(times))))
            del x
        del m;torch.cuda.empty_cache()
    csv_write('costs.csv',rows);write(HERE/'costs.json',dict(time=now(),gpu=torch.cuda.get_device_name(0),torch=torch.__version__,scope='Resident FP32 IQ to logits;3 repeats;20warmup/100timings; FMA2; FFT/interpolation/elementwise incomplete; latency includes actual interpolation',rows=rows))

def audit():
    import json
    import shutil
    from s21_core import sha, baseline_path
    from s21_evaluate import sealed, reference_checks
    from s21_data import PROTOCOL,roles
    reg=sealed();reference_checks(reg);evidence=HERE/'evidence';evidence.mkdir(exist_ok=True);manifest={}
    for j in reg['jobs']:
        p=HERE/'runs'/job_id(j);m=read(p/'metrics.json');cfg=read(p/'config.json')
        with (p/'history.csv').open(encoding='utf-8') as f:history=list(csv.DictReader(f))
        assert len(history)==200 and m['epochs']==200 and cfg['epochs_budget']==200 and cfg['batch']==64
        best=min(history,key=lambda row:float(row['source_val_ce']))
        assert int(best['epoch'])==m['best_epoch'] and float(best['source_val_ce'])==m['best_source_ce']
        assert m['model_sha256']==sha(p/'model.pt')
        target=evidence/p.name;target.mkdir(exist_ok=True)
        for name in ['metrics.json','config.json','history.csv','pairing.json','evaluation.json']:
            shutil.copyfile(p/name,target/name)
            manifest[(target/name).relative_to(HERE).as_posix()]=sha(target/name)
    restored=0
    for arm,fold,seed in itertools.product(['H0','P0'],[0,1],[201,202,203]):
        p=outpath(arm,fold,seed);e=read(p/'evaluation.json')
        assert len(e['baseline_prediction_recovery'])==9 and all(v['float32_predictions_bit_exact'] for v in e['baseline_prediction_recovery'])
        restored+=1;target=evidence/p.name;target.mkdir(exist_ok=True)
        shutil.copyfile(p/'evaluation.json',target/'evaluation.json')
        manifest[(target/'evaluation.json').relative_to(HERE).as_posix()]=sha(target/'evaluation.json')
    entries=[json.loads(line) for line in (HERE/'access.jsonl').read_text(encoding='utf-8').splitlines()]
    seal=read(HERE/'decision_p1_source.json');first_target=None
    for e in entries:
        assert e['rx'] not in PROTOCOL['blind_receivers_never_open']
        allowed=roles(e['fold'])['source'] if e['role']=='source' else roles(e['fold'])['dev_val'] if e['role']=='dev_val' else PROTOCOL['confirm_receivers'] if e['role']=='confirm' else []
        assert e['rx'] in allowed
        if e['role']!='source':
            assert e['time']>=seal['time']
            first_target=e['time'] if first_target is None else min(first_target,e['time'])
    shutil.copyfile(HERE/'access.jsonl',evidence/'access.jsonl');manifest['evidence/access.jsonl']=sha(evidence/'access.jsonl')
    write(HERE/'FINAL_AUDIT.json',dict(time=now(),status='PASS',formal_fits=12,all_epochs_200=True,minimum_source_ce_checkpoints=True,baseline_models_exact_recovered=restored,baseline_receiver_prediction_arrays_exact=108,all_target_access_after_all12_sealed=True,first_target_access=first_target,blind_accesses=0,artifacts_sha256=manifest))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['report','cost','audit']);a=p.parse_args()
    dict(report=report,cost=cost,audit=audit)[a.mode]()
