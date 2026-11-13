"""Score once after committed training seal; all preregistered conditions reported."""
import json
import subprocess
import time
import numpy as np
import torch
from sklearn.metrics import f1_score
from w2_execute import validate,metadata,materialize,checked,ROOT,A,C,QualityGate,packed
from run_w1 import HERE,digest,write,table
from quality import proxies,within_cell_low,bootstrap_rx


def metrics(logits,y):
    pred=logits.argmax(1);shift=logits-logits.max(1,keepdims=True)
    ce=np.log(np.exp(shift).sum(1))-shift[np.arange(len(y)),y]
    return dict(accuracy_pct=float(100*np.mean(pred==y)),macro_f1_pct=float(100*f1_score(y,pred,labels=np.arange(10),average='macro',zero_division=0)),ce=float(ce.mean()))


def main():
    started=time.perf_counter();reg=validate()
    A.require_committed(HERE/'W2_ANALYSIS_REGISTRATION.json')
    analysis=A.read(HERE/'W2_ANALYSIS_REGISTRATION.json')
    if digest(__file__)!=analysis['code_sha256']:raise ValueError('Frozen analysis code changed')
    A.require_committed(HERE/'W2_TRAINING_SEAL.json')
    seal=A.read(HERE/'W2_TRAINING_SEAL.json')
    if seal['new_scientific_fits']!=18 or seal['gradient_updates']!=91440:raise ValueError('Incomplete training budget')
    if (HERE/'W2_EVALUATION_ATTEMPT.json').exists():raise FileExistsError('Preserve first evaluation')
    # Verify every recorded checkpoint/state before any fresh selection scoring.
    for j in seal['completed']:
        folder=HERE/'artifacts'/j['id'];p=folder/'COMPLETE.json'
        if digest(p)!=j['complete_sha256']:raise ValueError('Training completion seal differs')
        for name,h in A.read(p)['files_sha256'].items():
            if digest(folder/name)!=h:raise ValueError('Sealed payload changed: '+name)
    write('W2_EVALUATION_ATTEMPT.json',dict(status='STARTED',seal_sha256=digest(HERE/'W2_TRAINING_SEAL.json'),code_sha256=digest(__file__),time=time.time()))
    pool=metadata(reg);rows=[];cells=[];pairing=[];scores={};costs={};artifacts={}
    for view in reg['views']:
        v=view['id'];data=materialize(reg,pool,view,('source_selection',))['source_selection']
        y,rx,day=data['y'],data['rx'],data['day'];names=np.asarray(reg['source']['rx_names'])[rx]
        features,_,_=proxies(data['x'])
        low=within_cell_low(features[:,0],y,names,day,data['pool_ids'])
        for condition,db in [('clean',None),('noise20',20),('noise10',10),('noise0',0)]:
            x=data['x'].copy()
            if db is not None:
                rng=np.random.default_rng(29401+(v=='V1')*100)
                noise=rng.normal(size=x.shape)
                # E[sum IQ-noise^2 /2] = observed complex power/(2*ratio).
                power=np.mean(x[:,0].astype('float64')**2+x[:,1].astype('float64')**2,axis=1)
                x+=(noise*np.sqrt(power/(2*10**(db/10)))[:,None,None]).astype('float32')
            for k in range(3):
                stream='S'+str(k)
                for arm in ('B0','B1','B2','N0'):
                    A.setup(24101+k)
                    if arm=='B0':
                        ref=reg['references'][v+'/'+stream]
                        model=A.build_model(24101+k).eval()
                        ck=torch.load(checked(ref['checkpoint'],'frozen_B0_inference',v+'/'+stream),map_location='cpu',weights_only=False)
                        model.load_state_dict(ck['model']);inputs=x
                    else:
                        model=QualityGate(arm,24101+k).cuda().eval()
                        folder=HERE/'artifacts'/f'{v}_{stream}_{arm}'
                        ck=torch.load(folder/'C1_selected.pt',map_location='cpu',weights_only=False)
                        model.load_state_dict(ck['model']);inputs,p=packed(x,arm,29301+k)
                        if condition=='clean':pairing.append(dict(view=v,stream=stream,arm=arm,no_self=bool(np.all(p!=np.arange(len(p)))) if arm=='N0' else None,permutation_sha256=A.semantic_hash(p)))
                    logits=C.predict_tensor(model,torch.as_tensor(inputs,device='cuda'),128)
                    if arm=='B0' and condition=='clean':
                        with np.load(checked(ref['selection_predictions'],'B0_clean_replay',v+'/'+stream),allow_pickle=False) as p:
                            if not np.array_equal(logits,p['logits']) or not np.array_equal(data['pool_ids'],p['pool_ids']):raise ValueError('B0 selection replay differs')
                    pred=logits.argmax(1);scores[(v,stream,arm,condition)]=pred
                    output=HERE/'artifacts'/f'{v}_{stream}_{arm}_{condition}_selection.npz'
                    np.savez_compressed(output,logits=logits,y=y,rx=names,day=day,pool_ids=data['pool_ids'],low_clean_quality=low)
                    artifacts[output.name]=digest(output)
                    for r in np.unique(names):
                        rr=names==r
                        for subset,mask in [('all',rr),('low_clean_q',rr&low),('other_clean_q',rr&~low),('day1',rr&(day==1)),('day2',rr&(day==2))]:
                            rows.append(dict(view=v,stream=stream,arm=arm,condition=condition,rx=r,subset=subset,packets=int(mask.sum()),**metrics(logits[mask],y[mask])))
                        for t in np.unique(y[rr]):
                            mask=rr&(y==t)
                            cells.append(dict(view=v,stream=stream,arm=arm,condition=condition,rx=r,tx=int(t),packets=int(mask.sum()),error_pct=float(100*np.mean(pred[mask]!=y[mask]))))
                    del model
            print('SCORED',v,condition,flush=True)
    # Complete numerical tables, including low/normal strata and every RX.
    table('W2_RX_METRICS.csv',rows);table('W2_TX_RX_RISKS.csv',cells)
    write('W2_NEGATIVE_CONTROL_PAIRING.json',pairing)
    totals=[]
    for cond in ('clean','noise20','noise10','noise0'):
        for arm in ('B0','B1','B2','N0'):
            for subset in ('all','low_clean_q','other_clean_q','day1','day2'):
                rr=[r for r in rows if r['arm']==arm and r['condition']==cond and r['subset']==subset]
                totals.append(dict(arm=arm,condition=cond,subset=subset,**{k:float(np.mean([r[k] for r in rr])) for k in ('accuracy_pct','macro_f1_pct','ce')}))
    comparisons=[];gates={}
    for control in ('B0','B1','N0'):
        deltas=[];by_view={};by_stream=[];local=[];corrections=[]
        for v in ('V0','V1'):
            rr=sorted({r['rx'] for r in rows if r['view']==v})
            view_effect=[]
            for r in rr:
                dif=[]
                for stream in ('S0','S1','S2'):
                    b=next(z for z in rows if z['view']==v and z['rx']==r and z['stream']==stream and z['condition']=='clean' and z['subset']=='all' and z['arm']=='B2')
                    c=next(z for z in rows if z['view']==v and z['rx']==r and z['stream']==stream and z['condition']=='clean' and z['subset']=='all' and z['arm']==control)
                    dif.append(b['accuracy_pct']-c['accuracy_pct'])
                d=float(np.mean(dif));deltas.append(d);view_effect.append(d)
                for t in range(10):
                    b=[z for z in cells if z['view']==v and z['rx']==r and z['tx']==t and z['arm']=='B2' and z['condition']=='clean']
                    c=[z for z in cells if z['view']==v and z['rx']==r and z['tx']==t and z['arm']==control and z['condition']=='clean']
                    if b and c and min(z['packets'] for z in b+c)>=50:
                        local.append(dict(view=v,rx=r,tx=t,control=control,error_increase_pp=float(np.mean([z['error_pct'] for z in b])-np.mean([z['error_pct'] for z in c]))))
            by_view[v]=float(np.mean(view_effect))
        for stream in ('S0','S1','S2'):
            b=[z['accuracy_pct'] for z in rows if z['stream']==stream and z['condition']=='clean' and z['subset']=='all' and z['arm']=='B2']
            c=[z['accuracy_pct'] for z in rows if z['stream']==stream and z['condition']=='clean' and z['subset']=='all' and z['arm']==control]
            by_stream.append(float(np.mean(b)-np.mean(c)))
        # Correction/additional-error rates use the same packet identities per stream.
        for v in reg['views']:
            ids=C.role_ids(reg['source'],v,pool)['source_selection'];y=pool['fit_y'][ids];rx=pool['fit_rx'][ids]
            for stream in ('S0','S1','S2'):
                b=scores[(v['id'],stream,'B2','clean')];c=scores[(v['id'],stream,control,'clean')]
                for r in np.unique(rx):
                    m=rx==r
                    corrections.append(dict(view=v['id'],stream=stream,rx=reg['source']['rx_names'][r],control=control,
                        corrected_pct=float(100*np.mean((b[m]==y[m])&(c[m]!=y[m]))),introduced_pct=float(100*np.mean((b[m]!=y[m])&(c[m]==y[m])))))
        ci=bootstrap_rx(deltas)
        comparison=dict(control=control,mean_delta_pp=float(np.mean(deltas)),conditional_rx_ci95_pp=ci,view_deltas_pp=by_view,
            stream_deltas_pp=by_stream,worst_rx_delta_pp=float(min(deltas)),maximum_supported_TX_RX_error_increase_pp=max(z['error_increase_pp'] for z in local),
            corrected_rx_equal_pct=float(np.mean([z['corrected_pct'] for z in corrections])),introduced_rx_equal_pct=float(np.mean([z['introduced_pct'] for z in corrections])))
        comparisons.append(comparison)
        threshold=0 if control=='N0' else 1
        gates[control+'_mean_effect']=comparison['mean_delta_pp']>threshold if control=='N0' else comparison['mean_delta_pp']>=threshold
        gates[control+'_cluster_ci_lower_positive']=ci[0]>0
        gates[control+'_both_views_positive']=all(x>0 for x in by_view.values())
        gates[control+'_2of3_streams_positive']=sum(x>0 for x in by_stream)>=2
        gates[control+'_worst_RX_not_below_minus2pp']=comparison['worst_rx_delta_pp']>=-2
        gates[control+'_local_risk_at_most10pp']=comparison['maximum_supported_TX_RX_error_increase_pp']<=10
        table('W2_LOCAL_RISK_VS_'+control+'.csv',local);table('W2_CORRECTIONS_VS_'+control+'.csv',corrections)
    for job in reg['jobs']:
        complete=A.read(HERE/'artifacts'/job['id']/'COMPLETE.json');costs[job['id']]=complete['costs']
    result=dict(status='PASS_QUALITY_GATE_WORTH_EXTERNAL_VALIDATION' if all(gates.values()) else 'STOP_W2_QUALITY_GATE_NOT_PROMOTED',
        totals=totals,comparisons=comparisons,gates=gates,new_scientific_fits=18,new_gradient_updates=91440,
        base_parameters=82426,gated_parameters=83554,training_costs=costs,
        quality_computation='CPU float64 proxies materialized before training/inference; proxy preprocessing time excluded from fit-loop costs; no latency advantage claim',
        total_fit_loop_seconds=sum(v['total_fit_loop_seconds'] for v in costs.values()),evaluation_seconds=time.perf_counter()-started,
        prediction_sha256=artifacts,source_only=True,target_IQ_reads=0,days=[1,2],RX_model_mapping='unknown; do not infer hardware models from IDs',
        limitations=['Development RX already used by W1 and historical selection; no independent blind confirmation',
            'Additional noise ratio is relative to observed packet power, not physical SNR','Low quality strata condition on TX/RX/day; not deployable rejection thresholds',
            'Packet-level quality conditions frequency gates; no direct mapping from time intervals to Fourier-bin reliability'])
    write('W2_RESULT.json',result)
    lines=['# WiSig质量条件化门控：W2完整执行结果','',f"最终判定：`{result['status']}`。18/18新拟合、720epoch、91,440更新；B0复用6份严格核验的EMA基线。",'',
        '|方法|clean Accuracy %|Macro-F1 %|CE|参数|','|---|---:|---:|---:|---:|']
    for arm in ('B0','B1','B2','N0'):
        t=next(t for t in totals if t['arm']==arm and t['condition']=='clean' and t['subset']=='all')
        lines.append(f"|{arm}|{t['accuracy_pct']:.4f}|{t['macro_f1_pct']:.4f}|{t['ce']:.5f}|{82426 if arm=='B0' else 83554:,}|")
    lines+=['','|B2相对|差值 pp|RX聚类95%条件区间 pp|最差RX pp|最严重TX×RX错误增加 pp|','|---|---:|---:|---:|---:|']
    for c in comparisons:lines.append(f"|{c['control']}|{c['mean_delta_pp']:+.4f}|[{c['conditional_rx_ci95_pp'][0]:+.4f}, {c['conditional_rx_ci95_pp'][1]:+.4f}]|{c['worst_rx_delta_pp']:+.4f}|{c['maximum_supported_TX_RX_error_increase_pp']:+.4f}|")
    lines+=['','以上为源域留RX、RX等权且对三个流平均后的结果，不是正式CRD或新RX盲测。全包覆盖，无删包。','','|名义新增噪声|B0 Accuracy %|B1 Accuracy %|B2 Accuracy %|N0 Accuracy %|','|---|---:|---:|---:|---:|']
    for cond in ('noise20','noise10','noise0'):
        values=[next(t['accuracy_pct'] for t in totals if t['arm']==arm and t['condition']==cond and t['subset']=='all') for arm in ('B0','B1','B2','N0')]
        lines.append('|'+cond+'|'+'|'.join(f'{x:.4f}' for x in values)+'|')
    lines+=['','质量代理随带噪IQ重算，N0在训练/校准/验证/评分均使用无自配对固定乱序。名义噪声比例相对观测总功率，不冒充真实空口SNR；噪声结果不修改clean主判定。', '',
        '完整正常/低一致性/日期/逐RX指标在W2_RX_METRICS.csv，TX×RX风险、纠错和新增错误分别单列。详细门槛、随机流/视图、成本和每份预测SHA在W2_RESULT.json。RX型号元数据不足，不能声称已分离B210/N210影响。', '',
        ('全部事前门满足，仍需合格的外部验证才能主张未知RX泛化。' if all(gates.values()) else '至少一项预设门未通过，本实现到此STOP；不追加调参、种子、Loss或架构补救。W1相关性仍保留，但不能冒充算法创新。')]
    (HERE/'W2_RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(status=result['status'],comparisons=comparisons),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
