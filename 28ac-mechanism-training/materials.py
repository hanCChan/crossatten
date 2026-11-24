"""Actual A/C results, complete paired figures and bounded mechanism judgments."""
import copy
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ac_common import HERE,ROOT,P,C,read,write,sha,now,identity

ARMS=['H0','LA','H2','H2_LA','LP']
LABEL={'H0':'HStem','LA':'No-GELU','H2':'NoMidBN','H2_LA':'NoBN-NoGELU','LP':'Long-Patch'}
COLOR={'H0':'#176b8a','LA':'#619aa8','H2':'#b17a3b','H2_LA':'#a64b46','LP':'#737c89'}

def main():
    integrity=read(HERE/'EXPERIMENT_INTEGRITY.json');assert integrity['status']=='PASS'
    result=read(HERE/'RESULTS.json');a=result['A'];c=result['C'];effect=a['primary_interaction'];pilot=c['pilot']
    means={q['arm']:q for q in a['dev_summary']};sources={q['arm']:q for q in a['source_summary']}
    supported=effect['two_sided_practical_rule_met'];value=effect['mean_delta_pp'];ci=effect['conditional_rx_ci95_pp']
    if supported and value>0:aj='达到预登记的正向交互规则：所选训练程序中，保留中间BN时GELU收益更大。'
    elif supported:aj='达到预登记的负向交互规则：所选训练程序中，移除中间BN时GELU收益更大；不能把它改写成正向BN×GELU协同。'
    else:aj='未达到预登记的双向实用交互规则；本轮没有建立稳定的大幅BN×GELU交互，也不证明两因素无作用或等效。'
    if pilot['consistent_pilot_signal']:cj='两台固定留出RX均满足较高留出准确率和较小退化的预登记Pilot规则；只支持这两台历史RX的条件性结果。'
    else:cj='未达到原C的共同Pilot规则，不能声称已经证明跨接收机退化专门减小。应分别查看留出绝对收益、Source收益及每台RX的退化方向。'
    write(HERE/'MECHANISM_DECISION.json',dict(time=now(),status='COMPLETE_STOP_NO_AUTOMATIC_EXTRA_FITS',scientific_fits=18,
        A=dict(judgment=aj,interaction_pp=value,conditional_rx_ci95_pp=ci,registered_rule_met=supported),
        C=dict(judgment=cj,pilot_rule_met=pilot['consistent_pilot_signal'],two_held_RX=pilot['fold_means']),
        remaining_authorized_fits=0,S28B_fits=0,IQ_only_fits=0,ASK_strict_fits=0,
        limitations=['BN/GELU effects concern a joint training procedure, not a unique factorization cause.',
            'TwohistoricalheldRX, not fresh acquisition orpopulationRX inference.','NativeS28-0 precisionfailures retained.','No automatic favorable-result rewrite of TIFS claims.']))
    write_tables(result)
    base=read(ROOT/'28-affine-bn-gelu-audit/EVIDENCE_MATRIX.json');rows=copy.deepcopy(base['rows'])
    for row in rows:
        if row['claim']=='BN moderates the usefulness of tokenizer GELU.':
            row['previous_status']=row['status'];row['status']='REGISTERED_SIGNED_INTERACTION_SUPPORTED' if supported else 'REGISTERED_INTERACTION_RULE_NOT_MET'
            row['result']=f"A18old+6new models: interaction{value:+.3f}pp, conditionalCI{ci};{aj}"
            row['evidence']=['28ac-mechanism-training/RESULTS.json','28ac-mechanism-training/EXPERIMENT_INTEGRITY.json']
        if row['claim']=='HStem specifically reduces source-to-heldRX degradation.':
            row['previous_status']=row['status'];row['status']='TWO_RX_PILOT_RULE_MET' if pilot['consistent_pilot_signal'] else 'TWO_RX_PILOT_RULE_NOT_MET'
            row['result']=cj;row['evidence']=['28ac-mechanism-training/RESULTS.json','28ac-mechanism-training/c_all_paired_gaps.csv']
    write(HERE/'EVIDENCE_MATRIX.json',dict(time=now(),rows=rows,prior_snapshot_sha256=sha(ROOT/'28-affine-bn-gelu-audit/EVIDENCE_MATRIX.json'),
        completed_A_C_rows_updated=True,other_historical_rows_retained=True,unique_optimization_orphysical_invariance_not_established=True))
    plots(result)
    text=f'''# S28-A 与原 S27-C：18次科学训练完整报告

**已完成A六次、C十二次，每次200轮；另复用24个冻结模型。** 保留的18条轨迹共390000次优化更新，工程合成更新另计为{integrity['synthetic_engineering_updates']}次。所有目标推理在A、C两组模型均封存并提交后统一执行。C没有依据A结果更改设计或跳过。新增S28-B、IQ-only、ASK Strict-IB与额外训练均为0。

## 1. A：BN×GELU的实际结果

{aj}

主交互 `(H0−LA)−(H2−H2_LA)` 为 **{value:+.3f} pp**，条件性RX 95%区间 **[{ci[0]:.3f}, {ci[1]:.3f}]**；与均值同向配对 {effect['same_direction_fit_pairs']}/6，两折为 {effect['fold_deltas_pp']} pp。规则是双向、幅度至少0.5pp、条件区间排零、至少5/6配对同向、两折同向，全部按原计划判定。负向交互也完整报告，不把正向作为成功的唯一方向。

|模型|参数|Dev Accuracy（%）|六拟合SD（pp）|Macro-F1（%）|最差RX（%）|Source Accuracy（%）|Source CE|
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    for arm,param in [('H0',133690),('LA',133690),('H2',133338),('H2_LA',133338),('LP',138346)]:
        q=means[arm];s=sources[arm]
        text+=f"|{LABEL[arm]}|{param}|{q['accuracy_pct']:.3f}|{q['fit_sd_pp']:.3f}|{q['macro_f1_pct']:.3f}|{q['worst_rx_accuracy_pct']:.3f}|{s['accuracy_pct']:.3f}|{s['CE']:.5f}|\n"
    text+='\n'+ '\n'.join(f"- {q['name']}：{q['mean_delta_pp']:+.3f} pp；条件区间 [{q['conditional_rx_ci95_pp'][0]:.3f}, {q['conditional_rx_ci95_pp'][1]:.3f}]。" for q in a['secondary'])
    text+=f'''

Source主交互为 {a['source_interaction']['mean_delta_pp']:+.3f} pp。Source已参与checkpoint选择，不能当作独立泛化测试。条件区间按十个唯一RX内先平均三个种子、10000次RX bootstrap、seed28028计算；不包括新种子、新采集、架构选择或多重比较校正后的确证不确定性。

仅Tokenizer中的三个中间BN与三个GELU被干预，前端两BN、两LeakyReLU和MHA保留。移除BN同时移除了352个仿射参数，并改变训练批统计、运行统计及优化；即使交互成立，也不是纯层级参数化的唯一因果证明。LP的分组、自由参数、初始化和算力仍不同。次要比较不能替代主交互，也不能把未达门槛解释成等效。

![A全部配对和逐RX交互](figures/A_interaction.png)

## 2. C：整体判别改善与接收机退化分开判断

{cj}

|留出RX|HStem−LP Source（pp）|HStem−LP Held（pp）|ΔG（pp）|
|---|---:|---:|---:|
'''
    for q in pilot['fold_means']:text+=f"|{q['held_rx']}|{q['source_delta_pp']:+.3f}|{q['held_delta_pp']:+.3f}|{q['delta_gap_pp']:+.3f}|\n"
    text+=f'''

六组均值：Held差 **{pilot['means']['held_delta_pp']:+.3f} pp**，Source差 **{pilot['means']['source_delta_pp']:+.3f} pp**，`ΔG=ΔSource−ΔHeld` 为 **{pilot['means']['delta_gap_pp']:+.3f} pp**。同时有利方向（Held差>0且ΔG<0）为 {pilot['favorable_fit_pairs']}/6；原规则还要求两台RX均满足Held差≥0.5pp、ΔG≤−0.5pp。两种要求共同判断，不能只看较小gap而忽略留出绝对准确率。

|折|种子|HStem Source（%）|HStem Held（%）|LP Source（%）|LP Held（%）|Held差（pp）|ΔG（pp）|
|---:|---:|---:|---:|---:|---:|---:|---:|
'''
    for q in pilot['pairs']:
        text+=f"|{q['fold']}|{q['seed']}|{q['H0']['source_accuracy_pct']:.3f}|{q['H0']['held_accuracy_pct']:.3f}|{q['LP']['source_accuracy_pct']:.3f}|{q['LP']['held_accuracy_pct']:.3f}|{q['held_delta_pp']:+.3f}|{q['delta_gap_pp']:+.3f}|\n"
    text+='''
Source Accuracy接近上限可能扭曲差距，Source与Held样本数也不同；两者及逐TX混淆均完整报告。两台固定历史RX不能变成六台独立RX，六组SD只描述训练重复，不做RX总体显著性或bootstrap。若Held更好而差距没有稳定变小，结论应收窄为整体TX判别更强；若两RX方向相反，须报告接收条件异质性。

![C全部重复的Source和Held](figures/C_source_held.png)

![C绝对收益与差距变化同时呈现](figures/C_gap_pairs.png)

## 3. 固定层几何与类别错误

C固定提取Tokenizer均值、后端前均值和最终分类器输入三种64维特征，先池化再逐包L2。四SourceRX验证每TX×RX40包，共1600包；留出几何仅使用原固定最后40包/TX，共400包，Held分类仍用全部2000包。全部72组层记录、72000个特征向量保留，不更换层、归一化或投影。

完整B_TX、D_RX|TX、W、Jw和Source Jc见C_geometry.csv。Held-only仅一台RX，D按构造为0，Jc不识别接收机不变性，显示NA并保留代数正则化比值；不把epsilon产生的大比值当作机制证据。Source已经用于选模，且Source/Held的RX数不同，几何只作描述。最终Jw不能作为跨RX性能的充分解释，S27-B反例仍有效。

逐RX/TX召回、混淆矩阵、CE与真实类margin见A_rx_tx.csv、C_rx_tx.csv及原始聚合JSON。所有困难TX、接收机和种子均保留；不通过删接收机或失败重复获得更好均值。

## 4. 训练配对、成本与完整性

基线de7d010；执行激活及科学适配器在首次RF读取前提交并推送。A沿用P-H5x160种子、8000/2000包、125更新/轮；C使用S27C-H4x160新种子、6400/1600包、100更新/轮，从头初始化H0/LP。所有组200轮、batch64、原AdamW/warmup/cosine/AWGN及Source RX等权CE选模。初始化、首个排列与噪声流、checkpoint选择和完整轨迹均经独立核验。

全部42模型统一产生324000条目标预测、79200条Source预测。24个旧模型的A Dev logits逐元素恢复；全部新模型的Source指标及最优CE恢复，旧权重/config/history哈希不变。C没有加载五RX权重；Source角色从缓存打开前就排除其留出RX。两个阶段的设计不互相适应，但它们共享数据研发历史，不声称是统计独立的数据集。

每轮原子断点含当前/最优权重、优化器、所有随机状态和历史；如有暂停或工程恢复，其原始日志和快照记录保留，390000指最终保留的完整轨迹，不代表可能被丢弃的部分轮物理执行。多进程wall时间会重叠，不能把拟合wall之和称作独占GPU小时。18次全部最佳轮次、Source CE、耗时与参数见all_new_fits.csv。

![全部18次200轮训练曲线](figures/all_training.png)

## 5. S28-0 数值限定与科研解释

S28-0六个冻结No-GELU Tokenizer在float64、显式首尾边界修正后具有有限仿射表示；原生TF32仍有1/12000个预测变化并未通过既定容差。关TF32的300包诊断通过，不覆盖原12000包失败，不替换历史模型或成绩。本轮使用原生科学模型与原精度训练，并未改用物化算子。

A回答BN/GELU对完整训练程序的联合影响，C回答两台固定历史留出RX上是否出现较小退化。这些结果不自动证明纯层级因果、物理接收机不变性、未知TX拒识、安全攻击防御或独立新采集泛化。创新证据矩阵更新A/C已完成条目，其余历史机制边界保留，不直接改写TIFS贡献。

## 6. 下一步建议与停止点

本阶段停止自动训练。先结合A的四格与带符号交互、C每RX的Source/Held/ΔG以及反例确定论文允许的措辞。C若未支持稳定较小退化，优先定位为低预算下更强的轻量TX判别学习，避免强称RX抑制。即使A/C规则满足，也仍只支持所选程序和这两台历史RX。

若未来要研究分层参数化与直接参数化，须先审计相同起始函数、边界约束、自由参数量、初始化和AdamW状态；S28-0表明函数表示相同不等于优化路径相同。S28-B没有授权预算，本轮不搜索或启动。没有新的独立采集，不宣称独立盲测已经完成。
'''
    (HERE/'RESEARCH_REPORT.zh-CN.md').write_text(text,encoding='utf-8',newline='\n')
    write(HERE/'REPORT_PROVENANCE.json',dict(time=now(),generator_sha256=sha(__file__),results_sha256=sha(HERE/'RESULTS.json'),integrity_sha256=sha(HERE/'EXPERIMENT_INTEGRITY.json')))
    print('Complete A/C mechanisms, all paired tables, figures and bounded decisions generated.',flush=True)

def table(file,rows):
    with (HERE/file).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def write_tables(result):
    fits=[]
    for j in read(HERE/'registration.json')['jobs']:
        m=read(HERE/'evidence'/identity(j)/'metrics.json');cfg=read(HERE/'evidence'/identity(j)/'config.json')
        fits.append(dict(**j,params=cfg['params'],epochs=200,updates=m['optimizer_steps'],best_epoch=m['best_epoch'],source_ce=m['best_source_ce'],
            source_accuracy_pct=m['source_validation']['summary']['acc_mean']*100,wall_seconds=m['wall_seconds'],peak_gpu_gib=m['peak_gpu_gib']))
    table('all_new_fits.csv',fits)
    rows=[]
    for q in result['C']['pilot']['pairs']:
        rows.append(dict(fold=q['fold'],seed=q['seed'],held_rx=q['H0']['held_rx'],
            H0_source_pct=q['H0']['source_accuracy_pct'],H0_held_pct=q['H0']['held_accuracy_pct'],H0_gap_pp=q['H0']['gap_pp'],
            LP_source_pct=q['LP']['source_accuracy_pct'],LP_held_pct=q['LP']['held_accuracy_pct'],LP_gap_pp=q['LP']['gap_pp'],
            held_delta_pp=q['held_delta_pp'],source_delta_pp=q['source_delta_pp'],delta_gap_pp=q['delta_gap_pp']))
    table('c_all_paired_gaps.csv',rows)
    for stage,file in [('A','EVALUATION_A.json'),('C','EVALUATION_C.json')]:
        d=read(HERE/file);rows=[]
        roles=[('Source',d['source_records']),('Dev',d['dev_records'])] if stage=='A' else [('Source',d['source_records']),('Held',d['held_records'])]
        for role,records in roles:
            for q in records:
                name=identity(dict(stage=stage,arm=q['arm'],fold=q['fold'],seed=q['seed']))
                file='SOURCE' if role=='Source' else q['rx'] if stage=='A' else 'HELD'
                with np.load(HERE/'predictions'/name/(file+'.npz'),allow_pickle=False) as z:
                    logits=z['logits'].astype(np.float64);y=z['y']
                    if role=='Source':
                        receivers=P['folds'][q['fold']]['source'] if stage=='A' else C['splits'][q['fold']]['training_receivers']
                        mask=z['rx']==receivers.index(q['rx']);logits=logits[mask];y=y[mask]
                ce,margin=packet_diagnostics(logits,y);cm=np.asarray(q['confusion'])
                for tx,recall in enumerate(q['per_tx_recall']):
                    mask=y==tx
                    rows.append(dict(role=role,arm=q['arm'],fold=q['fold'],seed=q['seed'],rx=q['rx'],TX=tx+31,packets=int(mask.sum()),
                        recall=recall,precision=float(cm[tx,tx]/max(cm[:,tx].sum(),1)),F1=float(2*cm[tx,tx]/max(cm[tx].sum()+cm[:,tx].sum(),1)),
                        TX_CE=float(ce[mask].mean()),TX_mean_trueclass_margin=float(margin[mask].mean()),
                        RX_accuracy=q['acc'],RX_CE=q['ce'],RX_macroF1=q['macro_f1']))
        table(stage+'_rx_tx.csv',rows)
    geometry=read(HERE/'EVALUATION_C.json')['geometry']
    keys=['arm','fold','seed','role','layer','status','between_tx','conditional_rx','within_tx_rx','tx_to_rx_centroid_ratio','tx_to_within_class_ratio']
    table('C_geometry.csv',[{k:q.get(k) for k in keys} for q in geometry])
    margins=[]
    for j in [j for j in read(HERE/'registration.json')['jobs'] if j['stage']=='C']:
        for role in ['SOURCE','HELD']:
            with np.load(HERE/'predictions'/identity(j)/(role+'_features.npz'),allow_pickle=False) as z:
                ce,margin=packet_diagnostics(z['logits'].astype(np.float64),z['y'])
                for ri in np.unique(z['rx']):
                    for tx in range(10):
                        mask=(z['rx']==ri)&(z['y']==tx);assert mask.sum()==40
                        rx=C['splits'][j['fold']]['training_receivers'][int(ri)] if role=='SOURCE' else C['splits'][j['fold']]['held_source_receiver']
                        margins.append(dict(role=role,arm=j['arm'],fold=j['fold'],seed=j['seed'],rx=rx,TX=tx+31,packets=40,
                            mean_trueclass_margin=float(margin[mask].mean()),CE=float(ce[mask].mean())))
    table('C_fixed40_margin.csv',margins)

def packet_diagnostics(logits,y):
    maximum=logits.max(1);ce=maximum+np.log(np.exp(logits-maximum[:,None]).sum(1))-logits[np.arange(len(y)),y]
    alternatives=logits.copy();alternatives[np.arange(len(y)),y]=-np.inf
    return ce,logits[np.arange(len(y)),y]-alternatives.max(1)

def plots(result):
    folder=HERE/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    a=result['A'];c=result['C']['pilot'];summaries={q['arm']:q for q in a['dev_summary']}
    fig,ax=plt.subplots(1,2,figsize=(8,3));arms=ARMS[:4]
    for i in range(6):ax[0].plot(range(4),[summaries[k]['fit_accuracies_pct'][i] for k in arms],color='#bbc1ca',lw=.65)
    for i,k in enumerate(arms):ax[0].scatter([i]*6,summaries[k]['fit_accuracies_pct'],s=16,color=COLOR[k])
    ax[0].set(xticks=range(4),xticklabels=[LABEL[k] for k in arms],ylabel='Historical Dev accuracy (%)',title='All six factorial repetitions');ax[0].tick_params(axis='x',rotation=15,labelsize=7)
    names=list(a['primary_interaction']['per_rx_delta_pp']);values=list(a['primary_interaction']['per_rx_delta_pp'].values())
    ax[1].bar(range(10),values,color=['#176b8a' if x>=0 else '#a64b46' for x in values]);ax[1].axhline(0,color='#666',lw=.6)
    ax[1].set(xticks=range(10),xticklabels=names,ylabel='BN by GELU interaction (pp)',title='Every RX; mean of three seeds');ax[1].tick_params(axis='x',rotation=35,labelsize=6)
    for t in ax[1].get_xticklabels():t.set_ha('right')
    fig.tight_layout();save(fig,folder/'A_interaction')
    fig,ax=plt.subplots(1,2,figsize=(8,3))
    for f in (0,1):
        for q in [q for q in c['pairs'] if q['fold']==f]:
            for arm in ['H0','LP']:ax[f].plot([0,1],[q[arm]['source_accuracy_pct'],q[arm]['held_accuracy_pct']],marker='o',ms=3,lw=.8,alpha=.8,color=COLOR[arm],label=LABEL[arm] if q['seed']==201 else None)
        ax[f].set(xticks=[0,1],xticklabels=['Source validation','Held RX'],ylabel='Accuracy (%)',title=c['fold_means'][f]['held_rx']);ax[f].legend(frameon=False,fontsize=7)
    fig.tight_layout();save(fig,folder/'C_source_held')
    fig,ax=plt.subplots(figsize=(6.4,3.3))
    for f,marker,color in [(0,'o','#176b8a'),(1,'s','#a64b46')]:
        rows=[q for q in c['pairs'] if q['fold']==f]
        ax.scatter([q['held_delta_pp'] for q in rows],[-q['delta_gap_pp'] for q in rows],s=28,marker=marker,color=color,label=c['fold_means'][f]['held_rx'])
        for q in rows:ax.annotate(str(q['seed']),(q['held_delta_pp'],-q['delta_gap_pp']),xytext=(4,3),textcoords='offset points',fontsize=6)
        m=c['fold_means'][f];ax.scatter([m['held_delta_pp']],[-m['delta_gap_pp']],s=90,marker=marker,facecolors='none',edgecolors=color,linewidths=1.3)
    ax.axhline(0,color='#777',lw=.6);ax.axvline(0,color='#777',lw=.6);ax.axhline(.5,color='#999',ls=':',lw=.6);ax.axvline(.5,color='#999',ls=':',lw=.6)
    ax.set(xlabel='Held HStem - LP (pp)',ylabel='Gap reduction: -delta G (pp)',title='Six repeats of two RXs; outlined markers are RX means');ax.legend(frameon=False,fontsize=7)
    fig.tight_layout();save(fig,folder/'C_gap_pairs')
    fig,ax=plt.subplots(3,2,figsize=(8,6.8))
    for row,(stage,arm) in enumerate([('A','H2_LA'),('C','H0'),('C','LP')]):
        for f in (0,1):
            for s in (201,202,203):
                with (HERE/'evidence'/identity(dict(stage=stage,arm=arm,fold=f,seed=s))/'history.csv').open(encoding='utf-8',newline='') as stream:h=list(csv.DictReader(stream))
                ax[row,f].plot([int(q['epoch']) for q in h],[float(q['source_val_ce']) for q in h],lw=.65,label=str(s))
            ax[row,f].set(yscale='log',xlabel='Epoch',ylabel='Clean source RX-equal CE',title=f'{stage}: {LABEL[arm]}, fold {f}');ax[row,f].legend(frameon=False,fontsize=6)
    fig.tight_layout();save(fig,folder/'all_training')

def save(fig,path):
    for ext in ['pdf','png']:fig.savefig(path.with_suffix('.'+ext),dpi=230,bbox_inches='tight',pad_inches=.035)
    plt.close(fig)

if __name__=='__main__':main()
