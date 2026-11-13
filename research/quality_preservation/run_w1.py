"""One-shot, preregistered source-only W1 using frozen historical predictions."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import time
import numpy as np
from quality import FEATURE_NAMES, proxies, within_cell_low, conditional_effect, bootstrap_rx

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(name,value):
    (HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def table(name,rows):
    with (HERE/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    started=time.perf_counter()
    if (HERE/'W1_RESULT.json').exists() or (HERE/'W1_ATTEMPT.json').exists():
        raise FileExistsError('Preserve the first W1 attempt/results')
    rel=HERE.relative_to(ROOT).as_posix()
    subprocess.run(['git','-C',str(ROOT),'ls-files','--error-unmatch',rel+'/REGISTRATION.json'],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(ROOT),'diff','--exit-code','HEAD','--',rel],check=True)
    reg=json.loads((HERE/'REGISTRATION.json').read_text(encoding='utf8'))
    for name,h in reg['code_sha256'].items():
        if digest(HERE/name)!=h:raise ValueError('Registered code changed: '+name)
    write('W1_ATTEMPT.json',dict(status='STARTED',registration_sha256=digest(HERE/'REGISTRATION.json'),time=time.time()))
    log=[]
    def checked(record,action):
        p=Path(record['path']);actual=digest(p)
        if actual!=record['sha256']:raise ValueError('Pinned input mismatch: '+str(p))
        log.append(dict(path=str(p),action=action,sha256=actual));write('W1_ACCESS_LOG.json',log)
        return p
    pool={k:np.load(checked(reg['source_input_whitelist']['fit_'+k],'source_metadata'),allow_pickle=False) for k in ('y','rx','day')}
    assert all(v.shape==(34668,) for v in pool.values()) and set(pool['day'])=={1,2}
    names=np.asarray(reg['rx_names']);pool['rx_names']=names[pool['rx']]
    array=np.load(checked(reg['source_input_whitelist']['fit_x'],'hash_and_mmap_source_fit18'),mmap_mode='r',allow_pickle=False)
    if array.shape!=(34668,2,256) or array.dtype!=np.float32:raise ValueError('Source IQ schema differs')
    all_cells=[];all_receivers=[];per_view=[];packets={};examples=[];summary=[]
    for view in reg['views']:
        v=view['id'];ids=np.flatnonzero(np.isin(pool['rx_names'],view['source_selection_rx'])).astype(np.int64)
        if len(ids)!=view['source_selection']['rows'] or hashlib.sha256(ids.tobytes()).hexdigest()!=view['source_selection']['indices_array_sha256']:
            raise ValueError('Frozen selection role differs')
        log.append(dict(action='copy_source_selection_IQ',view=v,rows=len(ids),RX=view['source_selection_rx'],days=[1,2]));write('W1_ACCESS_LOG.json',log)
        f,pairs,valid=proxies(np.array(array[ids],copy=True))
        y,rx,day=pool['y'][ids],pool['rx_names'][ids],pool['day'][ids]
        errors={}
        for a in ('F0','E0'):
            values=[]
            for s in ('S0','S1','S2'):
                path=checked(reg['predictions'][v+'/'+s+'/'+a],'decode_frozen_selected_source_prediction')
                with np.load(path,allow_pickle=False) as p:
                    for k,expected in (('pool_ids',ids),('y',y),('rx',rx),('day',day)):
                        if not np.array_equal(p[k],expected):raise ValueError('Prediction packet identity differs: '+k)
                    if not np.isfinite(p['logits']).all():raise ValueError('Nonfinite historical predictions')
                    values.append((p['logits'].argmax(1)!=y).astype(float))
            errors[a]=np.stack(values)
        packets[v]=dict(ids=ids,y=y,rx=rx,day=day,features=f,pairs=pairs,valid=valid,**errors)
        for j,feature in enumerate(FEATURE_NAMES):
            low=within_cell_low(f[:,j],y,rx,day,ids)
            for a in ('F0','E0'):
                avg=errors[a].mean(0)
                cells,receivers=conditional_effect(avg,low,y,rx,day)
                all_cells.extend(dict(view=v,model=a,feature=feature,**r) for r in cells)
                all_receivers.extend(dict(view=v,model=a,feature=feature,**r) for r in receivers)
                stream_effects=[]
                for e in errors[a]:
                    _,r=conditional_effect(e,low,y,rx,day);stream_effects.append(float(np.mean([x['difference_pp'] for x in r])))
                per_view.append(dict(view=v,model=a,feature=feature,difference_pp=float(np.mean([r['difference_pp'] for r in receivers])),streams_pp=stream_effects,
                    all_rx_equal_accuracy_pct=float(np.mean([100*(1-avg[rx==r].mean()) for r in np.unique(rx)])),
                    low_packet_fraction=float(low.mean()),low_group_error_pct=float(100*avg[low].mean()),other_error_pct=float(100*avg[~low].mean()),
                    low_error_share=float(avg[low].sum()/avg.sum()),invalid_packets=int((~valid).sum())))
        main_low=within_cell_low(f[:,0],y,rx,day,ids)
        avg=errors['E0'].mean(0)
        # Reproducible counterexamples; no choice of a preferred TX or RX.
        for kind,mask in (('high_repeatability_all_three_wrong',(~main_low)&(avg==1)),('low_repeatability_all_three_correct',main_low&(avg==0))):
            pos=np.flatnonzero(mask)
            order=np.argsort((-f[pos,0] if kind.startswith('high') else f[pos,0]),kind='stable')[:12]
            for i in pos[order]:examples.append(dict(view=v,kind=kind,pool_id=int(ids[i]),rx=rx[i],tx=int(y[i]),day=int(day[i]),q20_mean=float(f[i,0])))
        # Fixed identity offsets versus within-cell variance; descriptive only.
        cell_groups=[f[(rx==r)&(y==t)&(day==d),0] for r,t,d in sorted(set(zip(rx,y,day)))]
        grand=f[:,0].mean();between=sum(len(g)*(g.mean()-grand)**2 for g in cell_groups)
        total=float(np.sum((f[:,0]-grand)**2))
        summary.append(dict(view=v,rows=len(ids),q20_quantiles=np.quantile(f[:,0],[0,.01,.1,.25,.5,.75,.9,1]).tolist(),
            within_cell_variance_fraction=float(1-between/total) if total else 0,
            warning='RX/TX/day means and absolute timing may be confounded; conditional association is not causal.'))
    for a in ('F0','E0'):
        for feature in FEATURE_NAMES:
            rows=[r for r in all_receivers if r['model']==a and r['feature']==feature]
            values=[r['difference_pp'] for r in rows]
            vv=[r for r in per_view if r['model']==a and r['feature']==feature]
            summary.append(dict(model=a,feature=feature,difference_pp=float(np.mean(values)),
                conditional_rx_cluster_ci95_pp=bootstrap_rx(values,reg['bootstrap_seed'],reg['bootstrap_count']),
                positive_rx=sum(x>0 for x in values),total_rx=len(rows),
                views_pp={r['view']:r['difference_pp'] for r in vv},
                paired_streams_pp=np.mean([r['streams_pp'] for r in vv],axis=0).tolist()))
    primary={r['model']:r for r in summary if r.get('feature')=='q20_mean'}
    gates={}
    for a,r in primary.items():
        gates[a+'_effect_at_least_2pp']=r['difference_pp']>=2
        gates[a+'_rx_ci_lower_positive']=r['conditional_rx_cluster_ci95_pp'][0]>0
    e=primary['E0'];gates['EMA_both_views_positive']=all(x>0 for x in e['views_pp'].values())
    gates['EMA_2of3_streams_positive']=sum(x>0 for x in e['paired_streams_pp'])>=2
    gates['EMA_8of12_rx_positive']=e['positive_rx']>=8
    passed=all(gates.values())
    artifact=HERE/'artifacts';artifact.mkdir(exist_ok=True)
    for v,p in packets.items():np.savez_compressed(artifact/(v+'_packets.npz'),**p)
    table('W1_CELLS.csv',all_cells);table('W1_RECEIVERS.csv',all_receivers);table('W1_COUNTEREXAMPLES.csv',examples)
    write('W1_VIEW_RESULTS.json',per_view)
    result=dict(status='W1_PASS_PROCEED_TO_FROZEN_W2_DESIGN' if passed else 'STOP_W1_PRIMARY_QUALITY_ASSOCIATION_NOT_STABLE',
        gates=gates,summary=summary,primary=primary,scientific_deep_fits=0,unique_IQ_packets=sum(len(p['ids']) for p in packets.values()),
        target_IQ_reads=0,days=[1,2],elapsed_seconds=time.perf_counter()-started,
        artifact_sha256={p.name:digest(p) for p in artifact.glob('*.npz')},
        limitations=['source-only retrospective conditional association, not causal','RX cluster interval conditions on these two existing development views',
                      'q20 measures repeatability, not SNR or confirmed preamble validity','No new algorithm accuracy result in W1'])
    write('W1_RESULT.json',result)
    lines=['# WiSig质量假设 W1 实测结果','',f"判定：`{result['status']}`。深度训练0次，合法源包{result['unique_IQ_packets']}个，目标RX/Day3–4访问0。",'',
        '|冻结模型|同TX/RX/Day内低一致性额外错误 pp|RX聚类95%条件区间 pp|正向RX|','|---|---:|---:|---:|']
    for a,r in primary.items():lines.append(f"|{'C0 普通频域' if a=='F0' else 'C1 EMA强参照'}|{r['difference_pp']:+.4f}|[{r['conditional_rx_cluster_ci95_pp'][0]:+.4f}, {r['conditional_rx_cluster_ci95_pp'][1]:+.4f}]|{r['positive_rx']}/12|")
    lines+=['','区间以RX为单位，先对三个随机流平均，不将包数量冒充独立接收机数量。两视图、每流和完整辅助代理结果见W1_RESULT.json / W1_VIEW_RESULTS.json，240个条件单元的每模型/代理结果见W1_CELLS.csv。', '',
        '门槛逐项：', '',*[f"- {k}: {'通过' if val else '未通过'}" for k,val in gates.items()], '',
        ('按原授权继续固定W2模型实现、合成验收及匹配训练，W1本身不证明算法收益。' if passed else '按预登记停止本质量条件化路线，不启动W2，不换窗口或改用辅助代理救结果。'),'',
        '低组是每个TX×RX×Day内部最低四分位，仅用于控制混杂的事后分层，不能作为推断阈值、删包策略或TX无关质量判定。全覆盖识别率保留，所有包进入分母。反例和固定条件均值混杂诊断同时交付，未证明TX硬件成因、B210硬件因果或完整前导码有效性。']
    (HERE/'RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(status=result['status'],primary=primary,gates=gates),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
