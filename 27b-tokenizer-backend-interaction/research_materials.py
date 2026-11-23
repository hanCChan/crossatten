"""Complete four-cell research record, all RX/fits and fixed-layer evidence."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from s27b_common import HERE,ROOT,read,write,sha,now

ARMS=['H0','LP','H0_P','LP_P']
LABEL={'H0':'HStem/MHA','LP':'LP/MHA','H0_P':'HStem/MLP','LP_P':'LP/MLP'}
COLOR={'H0':'#176b8a','LP':'#737c89','H0_P':'#b47730','LP_P':'#568f76'}


def geometry_summary(rows):
    result=[]
    for arm in ARMS:
        for layer in ['tokens','pre_backend','final']:
            q=[q for q in rows if q['arm']==arm and q['layer']==layer];assert len(q)==6
            valid=all(q['status']=='VALID' for q in q)
            result.append(dict(arm=arm,layer=layer,valid_fits=sum(q['status']=='VALID' for q in q),
                **{k:(float(np.mean([r[k] for r in q])) if valid else None) for k in
                    ['between_tx','conditional_rx','within_tx_rx','tx_to_rx_centroid_ratio','tx_to_within_class_ratio']}))
    return result


def main():
    result=read(HERE/'RESULTS.json');audit=read(HERE/'EXPERIMENT_INTEGRITY.json');assert audit['status']=='PASS' and result['new_fits']==12
    primary=result['primary_interaction'];s={q['arm']:q for q in result['summary']};src={q['arm']:q for q in result['source_summary']}
    source=read(HERE/'SOURCE_EVALUATION.json');geometry=geometry_summary(source['geometry'])
    write(HERE/'GEOMETRY_SUMMARY.json',dict(rows=geometry,scope='Mean of6perfitratios, not ratioofpooledfeatures; NA if anyfit invalid. Alllayers/arms retained.'))
    fits=[]
    for f in (0,1):
        for seed in (201,202,203):
            i=f*3+[201,202,203].index(seed)
            for arm in ARMS:
                new=arm.endswith('_P');m=read(HERE/'evidence'/f'{arm}_f{f}_s{seed}'/'metrics.json') if new else None
                fits.append(dict(arm=arm,fold=f,seed=seed,new_fit=new,epochs=200 if new else None,
                    best_epoch=m['best_epoch'] if new else None,source_ce=m['best_source_ce'] if new else None,
                    source_accuracy_pct=src[arm]['fit_accuracies_pct'][i],dev_accuracy_pct=s[arm]['fit_accuracies_pct'][i],
                    source_minus_dev_pp=src[arm]['fit_accuracies_pct'][i]-s[arm]['fit_accuracies_pct'][i],
                    fit_wall_seconds=m['wall_seconds'] if new else None))
    with (HERE/'all_fit_source_dev.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(fits[0]),lineterminator='\n');writer.writeheader();writer.writerows(fits)
    ci=primary['conditional_rx_ci95_pp'];value=primary['mean_delta_pp'];met=primary['practical_rule_met']
    if met:
        judgment='达到预登记的正交互门槛：HStem 相对 Long-Patch 的优势在所选 MHA 后端下更大。它支持这两个已训练系统在本协议中的条件性架构交互，不证明普遍注意力必要性。'
    elif value<0 and ci[1]<0:
        judgment='交互估计及条件区间均位于零以下：所选 MLP 下的 Tokenizer 差异更大，当前证据不支持“MHA 放大 HStem 相对收益”的假设。不能据此说所有注意力都会降低性能。'
    else:
        judgment='未达到预登记的正交互门槛：本次没有建立稳定的 MHA 放大作用；小差值或区间含零也不等于证明等效或注意力无用。'
    negative={k:v for k,v in primary['per_rx_delta_pp'].items() if v<0};positive=sum(v>0 for v in primary['per_rx_delta_pp'].values())
    write(HERE/'MECHANISM_DECISION.json',dict(time=now(),status='B_COMPLETE_STOP_AUTOMATIC_FITTING',new_fits=12,
        positive_interaction_rule_met=met,primary_interaction_pp=value,conditional_rx_ci95_pp=ci,judgment=judgment,
        positive_fit_pairs=primary['positive_fit_pairs'],positive_rx_means=positive,negative_rx=negative,
        C_needed_for_receiver_specific_claim=True,C_fits=0,IQ_only_fits=0,ASK_strict_fits=0,remaining_authorized_automatic_fits=0,
        C_reason='B changes a backend in historicallydeveloped5SourceRX fits. Source/Dev interaction differences do not replace the frozen4fitRX/1heldSourceRX pilot. C remains needed to investigate a receiver-specific gap claim; only2heldRX, not independent newcapture.',
        limit='Nonlinear form, expressivity and compute differ. Mean source geometry of different end-to-end trained models is descriptive, not causal information flow or mutual information.'))
    previous=read(ROOT/'26-long-patch-control/EVIDENCE_MATRIX.json')['rows']
    a=read(ROOT/'27-mechanism-identification/RESULTS.json')['paired_comparisons'][0]
    previous += [dict(claim='Three intermediateGELUs benefit the frozenHStem procedure.',definition='S27-A sameweights/init/BN/params, threeIdentity interventions.',
        evidence=['27-mechanism-identification/RESULTS.json'],result=f"ExistingA H0-LA {a['mean_delta_pp']:+.3f}pp; noArefit inB.",status='EXISTING_A_REUSED',limit='Optimization andBN remain coupled.'),
        dict(claim='MHA amplifies the HStem-versus-LP advantage.',definition='Fourcell difference of differences; frozenTokenMLP readout.',
        evidence=['27b-tokenizer-backend-interaction/RESULTS.json'],result=f"{value:+.3f}pp, conditional95%CI{ci}, {primary['positive_fit_pairs']}/6positive.",
        status='REGISTERED_POSITIVE_RULE_MET' if met else 'REGISTERED_POSITIVE_RULE_NOT_MET',limit=judgment),
        dict(claim='Pooled sourcefeature geometry uniquely explains the backend interaction.',definition='All4arms/all6fits/3fixed64Dtaps, unitL2,S25scatter definitions.',
        evidence=['27b-tokenizer-backend-interaction/SOURCE_EVALUATION.json'],result='72sourcegeometryrecords and all48k sourcepredictions archived.',
        status='DESCRIPTIVE_NOT_CAUSAL',limit='Sourceval already selectedcheckpoints; differentmodels jointlytrained; ratios are notMI.'),
        dict(claim='B establishes independentRX robustness or completes C.',definition='Independentacquisition/sourceLOSO permissions distinct.',
        evidence=['27b-tokenizer-backend-interaction/registration.json'],result='C0fits; newindependentcapture0; historicaldevelopmentRX only.',
        status='NOT_ESTABLISHED',limit='No newblindRX, unknownTX rejection orphysicalinvariance guarantee.')]
    write(HERE/'EVIDENCE_MATRIX.json',dict(time=now(),rows=previous,prior_S26_rows_retained=True,A_reused_not_renamed=True))
    plots(result,source)
    text=f'''# S27-B：Tokenizer × 后端交互科研报告

## 执行与主结论

严格复用 S27 既有 PLAN_B 的两层 TokenMLP（64→128→64，GELU、残差、LayerNorm），本轮仅新增 HStem/Long-Patch × TokenMLP 各六次，共12次，每次200轮；原12个MHA模型保留。全部新模型封存后统一评价四组，未启动 C、IQ-only、ASK Strict-IB 或新增后端搜索。正式 TIFS 全文未重写。

{judgment}

主交互为 **{value:+.3f} pp**，条件性 RX 95% 区间 **[{ci[0]:.3f}, {ci[1]:.3f}]**，{primary['positive_fit_pairs']}/6 配对正向，两折为 {primary['fold_deltas_pp']} pp。判定完全沿用均值≥0.5pp、条件下界>0、至少5/6正向、两折均正向。十个唯一RX内先平均三个种子，再固定10000次bootstrap（seed27028）；不覆盖新采集、架构选择或新种子不确定性，不作多重比较校正后的确证声明。

## 四组完整结果

|模型|参数|Dev Accuracy（%）|六组拟合SD（pp）|Macro-F1（%）|最差RX（%）|Source Accuracy（%）|
|---|---:|---:|---:|---:|---:|---:|
'''
    for arm,param in zip(ARMS,[133690,138346,133562,138218]):
        q=s[arm];text+=f"|{LABEL[arm]}|{param:,}|{q['accuracy_pct']:.3f}|{q['fit_sd_pp']:.3f}|{q['macro_f1_pct']:.3f}|{q['worst_rx_accuracy_pct']:.3f}|{src[arm]['accuracy_pct']:.3f}|\n"
    text+='\n'+ '\n'.join(f"- {q['name']}：{q['mean_delta_pp']:+.3f} pp，条件区间 [{q['conditional_rx_ci95_pp'][0]:.3f}, {q['conditional_rx_ci95_pp'][1]:.3f}]；仅作完整次要描述。" for q in result['secondary_contrasts'])
    text+=f'''

逐 RX 主交互正向 {positive}/10，反向为 {negative}；所有接收机和全部配对种子均在图表、CSV中保留。不能只比较两个后端绝对 Accuracy 就宣称协同，更不能因未得到正交互而更换MLP或种子。

## Source、Dev 与泛化差距

源域交互为 {result['source_interaction']['mean_delta_pp']:+.3f} pp，Dev 交互为 {value:+.3f} pp；二者之差为 {result['source_minus_dev_interaction']['mean_pp']:+.3f} pp，六组值为 {result['source_minus_dev_interaction']['fit_deltas_pp']}。Source CE、逐模型 Source/Dev 与全部12组完整训练记录见聚合JSON和 all_fit_source_dev.csv。

Source Validation 每TX×RX40包且已用于选模；Dev 每TX×RX200包且有研发历史。Source和Dev的变化不能构成C的重新训练留一RX实验。尤其当MLP源域学习不足时，退化也不能全部解释为跨RX失效。报告四格Source学习与绝对后端差异，而非仅挑有利交互。

## 固定层特征：描述性证据

全部24模型在同一合法Source验证包上提取三个64维特征：Tokenizer32Token均值、输入投影/位置项/pre-LN后的均值、最终分类器输入。先池化，再逐包L2；按TX×RX平衡分解 B_TX、D_RX|TX、W，并报告 Jc、Jw。每模型2000包、每TX×RX40包；共72组层统计、144000个64维特征向量。定义与S25一致，保持本轮原始推理精度，不与旧TF32-off数值直接拼接。

所有层、归一化和种子固定。若任一组有未定义特征，整组六拟合聚合指标标为NA，不改归一化或排除失败样本。不同已训练网络的前后层差异不能解释为因果信息流，更不能视为互信息或物理RX不变性。全指标与反例见 SOURCE_EVALUATION.json、GEOMETRY_SUMMARY.json、source_geometry.csv。

## 参数与计算成本

两种新MLP整网均比对应MHA少128参数，但保留Token接口并不等于表达能力相同。逐TokenMLP在pooling前不直接跨Token混合，也引入不同非线性；MHA与MLP因而不是纯单一算子因果干预。HStem的卷积开销仍显著高于LP。四组同会话GPU-resident延迟、batch1/128、三轮预热/15次测量及部分MAC在 COSTS.json 和GATE中报告；FFT与未支持算子未计入部分MAC，但包含在整网延迟中，主机传输另排除。不同会话不合并，拟合wall总和不等于独占GPU小时。

## 创新证据与后续边界

本轮更新既有创新证据矩阵，保留全部S15–S27-A已完成的证据及局限；不把旧结果改名为新实验。交互正向只支持所选架构与后端下的条件性协同；不满足门槛不证明等效；交互负向则必须据实收窄MHA放大收益的主张。无论结果如何，都不证明所有Attention的必要性、纯层级性因果或RF物理不变性。

若论文要进一步声称专门减小训练RX到未参与拟合RX的性能损失，C仍有必要；若只作当前架构/预算的经验结论，不能把未运行C写成已有证据。C的四Source/一HeldSource、b210_1/rtl_5固定划分与数学边界保持原样，仅两台留出RX不能提供总体泛化证明。本轮C拟合数为0，不自动启动。

## 科研完整性

科学代码与B激活登记在首次RF读取/拟合前提交。12个新checkpoint/config/history封存后才进行10RX的统一Dev推理，共24模型、120模型×RX单元、240000条预测。另保存48000条Source预测和72组层统计。独立核验混淆矩阵、Accuracy/F1/CE、交互条件区间、行标签身份、源域选模、完整初始化、首个排列/AWGN流、历史模型哈希与全部200轮/25000步记录。8次合成Gate更新与最终保留轨迹中的300000次科学优化更新分开计数。大波形、权重、恢复状态及逐包特征留在本地；代码、聚合表、真实stdout与审计提交GitHub。

用户暂停后，于2026年11月23日明确授权继续训练。恢复前，3个模型已完成，LP_P_f0_s202和H0_P_f0_s203分别保存至epoch 180和3；核验模型、优化器、随机状态、登记哈希、最优Source checkpoint一致性后，从相应完整轮快照恢复。RESUME_AUDIT_001.json与recovery_epoch181/4记录保留对应快照哈希。暂停时尚未封存的部分轮计算可能被丢弃，因此300000表示最终保留的12条完整训练轨迹更新数，不是所有物理执行的优化步骤总数；没有替换种子或新增拟合。原始与恢复控制器日志分别归档。
'''
    (HERE/'RESEARCH_REPORT.zh-CN.md').write_text(text,encoding='utf-8',newline='\n')
    write(HERE/'REPORT_PROVENANCE.json',dict(time=now(),status='FULL_COMPLETED_AND_INDEPENDENTLY_VERIFIED_COHORT',
        generator_sha256=sha(__file__),results_sha256=sha(HERE/'RESULTS.json'),integrity_sha256=sha(HERE/'EXPERIMENT_INTEGRITY.json')))
    print('Fourcell mechanism report, evidence matrix and4figures generated from actual complete results.',flush=True)


def plots(result,source):
    folder=HERE/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    primary=result['primary_interaction'];s={q['arm']:q for q in result['summary']}
    fig,ax=plt.subplots(1,2,figsize=(8,2.9),gridspec_kw={'width_ratios':[.85,1.25]})
    m=result['secondary_contrasts'][0]['fit_deltas_pp'];p=result['secondary_contrasts'][1]['fit_deltas_pp']
    for a,b in zip(m,p):ax[0].plot([0,1],[a,b],color='#b5bcc6',lw=.7)
    ax[0].scatter([0]*6,m,color=COLOR['H0'],s=18,label='Six MHA pairs');ax[0].scatter([1]*6,p,color=COLOR['H0_P'],s=18,label='Six MLP pairs')
    ax[0].axhline(0,color='#777777',lw=.6);ax[0].set(xticks=[0,1],xticklabels=['MHA','Token MLP'],ylabel='HStem - Long-Patch (pp)',title='All paired tokenizer contrasts')
    names=list(primary['per_rx_delta_pp']);values=list(primary['per_rx_delta_pp'].values())
    ax[1].bar(range(10),values,color=[COLOR['H0'] if x>=0 else COLOR['H0_P'] for x in values],width=.65)
    ax[1].axhline(0,color='#666666',lw=.6);ax[1].set(xticks=range(10),xticklabels=names,ylabel='Interaction (pp)',title='Every RX: mean of three seeds')
    ax[1].tick_params(axis='x',labelrotation=35,labelsize=6)
    for t in ax[1].get_xticklabels():t.set_ha('right')
    fig.tight_layout();save(fig,folder/'fig01_interaction')
    fig,ax=plt.subplots(1,2,figsize=(8,2.9))
    for j,summary in enumerate([result['source_summary'],result['summary']]):
        records={q['arm']:q for q in summary}
        for i in range(6):ax[j].plot(range(4),[records[a]['fit_accuracies_pct'][i] for a in ARMS],color='#b6bdc6',lw=.6)
        for k,a in enumerate(ARMS):ax[j].scatter([k]*6,records[a]['fit_accuracies_pct'],s=15,color=COLOR[a])
        ax[j].set(xticks=range(4),xticklabels=[LABEL[a] for a in ARMS],ylabel='Accuracy (%)',title='Source validation' if j==0 else 'Historical Dev')
        ax[j].tick_params(axis='x',labelrotation=15,labelsize=7)
    fig.tight_layout();save(fig,folder/'fig02_fourcells')
    fig,ax=plt.subplots(1,2,figsize=(8,2.9));layers=['tokens','pre_backend','final']
    for j,metric in enumerate(['tx_to_within_class_ratio','conditional_rx']):
        for arm in ARMS:
            groups=[[q for q in source['geometry'] if q['arm']==arm and q['layer']==layer] for layer in layers]
            ys=[np.mean([q[metric] for q in g]) if all(q['status']=='VALID' for q in g) else np.nan for g in groups]
            sd=[np.std([q[metric] for q in g],ddof=1) if all(q['status']=='VALID' for q in g) else 0 for g in groups]
            ax[j].errorbar(range(3),ys,yerr=sd,marker='o',ms=3,lw=.8,capsize=2,color=COLOR[arm],label=LABEL[arm])
        ax[j].set(xticks=range(3),xticklabels=['Tokenizer','Pre-backend','Final'],title='Final TX separation' if j==0 else 'RX-conditional scatter',ylabel='Jw (mean +/- fit SD)' if j==0 else 'D_RX|TX (unit features)')
        ax[j].legend(frameon=False,fontsize=6)
    fig.tight_layout();save(fig,folder/'fig03_source_geometry')
    fig,ax=plt.subplots(2,2,figsize=(8,4.8))
    for j,arm in enumerate(['H0_P','LP_P']):
        for f in (0,1):
            for seed in (201,202,203):
                name=f'{arm}_f{f}_s{seed}'
                with (HERE/'evidence'/name/'history.csv').open(encoding='utf-8',newline='') as stream:h=list(csv.DictReader(stream))
                ax[j,f].plot([int(q['epoch']) for q in h],[float(q['source_val_ce']) for q in h],lw=.65,label=str(seed))
            ax[j,f].set(yscale='log',xlabel='Epoch',ylabel='Clean source RX-equal CE',title=f'{LABEL[arm]}, fold {f}: all 200 epochs');ax[j,f].legend(frameon=False,fontsize=6)
    fig.tight_layout();save(fig,folder/'fig04_all_training')


def save(fig,path):
    for extension in ['pdf','png']:fig.savefig(path.with_suffix('.'+extension),dpi=230,bbox_inches='tight',pad_inches=.035)
    plt.close(fig)


if __name__=='__main__':main()
