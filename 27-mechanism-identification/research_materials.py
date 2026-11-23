"""Generate bounded mechanism conclusions from the complete saved cohort only."""
import csv
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from s27_common import HERE,ROOT,read,write,sha,now


LABEL={'H0':'HStem','LA':'No-GELU','LP':'Long-Patch'}
COLOR={'H0':'#176b8a','LA':'#b67632','LP':'#747b87'}


def main():
    result=read(HERE/'RESULTS.json');audit=read(HERE/'EXPERIMENT_INTEGRITY.json')
    assert result['new_fits']==6 and audit['status']=='PASS'
    evidence={r['arm']:r for r in result['summary']};primary=result['paired_comparisons'][0]
    assert (primary['a'],primary['b'])==('H0','LA')
    source=read(HERE/'SOURCE_RECOVERY.json')['rows'];cells=[]
    for f in (0,1):
        for seed in (201,202,203):
            pos=f*3+[201,202,203].index(seed)
            for arm in ['H0','LA','LP']:
                if arm=='LA':
                    m=read(HERE/'evidence'/f'LA_f{f}_s{seed}'/'metrics.json')
                    accuracy=m['source_validation']['summary']['acc_mean'];ce=m['best_source_ce'];epoch=m['best_epoch'];wall=m['wall_seconds']
                else:
                    r=next(r for r in source if r['id']==f'{arm}_f{f}_s{seed}')
                    accuracy=r['source_accuracy'];ce=r['source_ce']
                    m=next(r for r in read(HERE/'REFERENCE_METADATA.json')['rows'] if r['id']==f'{arm}_f{f}_s{seed}')
                    epoch=m['best_epoch'];wall=None
                dev=evidence[arm]['fit_accuracies_pct'][pos]
                cells.append(dict(arm=arm,fold=f,seed=seed,source_accuracy_pct=accuracy*100,source_ce=ce,
                    dev_accuracy_pct=dev,source_minus_dev_pp=accuracy*100-dev,best_epoch=epoch,new_fit_wall_seconds=wall))
    rows=[]
    for r in cells:
        if r['arm']!='H0':continue
        q=next(q for q in cells if q['arm']=='LA' and q['fold']==r['fold'] and q['seed']==r['seed'])
        rows.append(dict(fold=r['fold'],seed=r['seed'],source_delta_pp=r['source_accuracy_pct']-q['source_accuracy_pct'],
            dev_delta_pp=r['dev_accuracy_pct']-q['dev_accuracy_pct'],gap_delta_pp=r['source_minus_dev_pp']-q['source_minus_dev_pp'],
            source_ce_delta=r['source_ce']-q['source_ce']))
    means={a:dict(source_accuracy_pct=float(np.mean([r['source_accuracy_pct'] for r in cells if r['arm']==a])),
        source_ce=float(np.mean([r['source_ce'] for r in cells if r['arm']==a])),
        source_minus_dev_pp=float(np.mean([r['source_minus_dev_pp'] for r in cells if r['arm']==a]))) for a in LABEL}
    for name,data in [('source_dev.csv',cells),('source_dev_contrasts.csv',rows)]:
        with (HERE/name).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)
    write(HERE/'SOURCE_DEV_ANALYSIS.json',dict(time=now(),means=means,paired_fit_contrasts=rows,
        scope='Descriptive historicalDev gap; sourceval already used for checkpoint selection,40vs200packets/TX/RX. Not the independently trained4RX/1heldRX C Pilot. No gapbootstrap or causalRXsuppression claim.'))
    plots(result,cells,rows)
    delta=primary['mean_delta_pp'];lo,hi=primary['conditional_rx_ci95_pp'];positive=primary['primary_practical_rule_met']
    judgment=('三处 GELU 对当前 HStem 与固定训练流程的组合有一致正收益。' if positive else
        '未达到预先登记的 GELU 正收益一致性门槛；不能据此证明等效或断言激活无用。')
    inference='该干预同时改变非线性函数族、梯度路径、训练中 BN 轨迹和最终权重，不能把差值唯一归给推理非线性，更不能解释为 RF 硬件不变性。'
    ds=float(np.mean([r['source_delta_pp'] for r in rows]));dd=float(np.mean([r['dev_delta_pp'] for r in rows]));dg=float(np.mean([r['gap_delta_pp'] for r in rows]))
    # Rank research questions without launching or redesigning either frozen arm.
    next_step='C' if positive and abs(dd-ds)>=.5 else 'B'
    reason=('A 的 Source 与 Dev 变化并不同幅。优先用已冻结的四源 RX/一留出 RX Pilot 检查性能损失，才能进一步区分整体分类能力与条件性接收机迁移收益。两台留出 RX 仍只提供 Pilot 证据。' if next_step=='C' else
        '当前结果没有充分分离 Tokenizer 自身与后端的作用。已冻结的 2×2 后端对照直接检验收益是否依赖所选 MHA，保持同一 Token 接口和近似参数预算；不能推广到所有读出头。')
    write(HERE/'MECHANISM_DECISION.json',dict(time=now(),status='A_COMPLETE_STOP_AUTOMATIC_FITTING',new_fits=6,
        A_rule_met=positive,primary_delta_pp=delta,conditional_rx_ci95_pp=[lo,hi],source_delta_pp=ds,dev_delta_pp=dd,gap_delta_pp=dg,
        judgment=judgment,limitations=inference,next_highest_identification_value=next_step,next_reason=reason,
        next_rank_scope='Research-priority judgment from disclosedA outcomes; B/C designs and RXs unchanged. Neither launched.',
        B_fits=0,C_fits=0,IQ_only_fits=0,ask_strict_fits=0,remaining_authorized_automatic_fits=0))
    text=f'''# S27-A：GELU 干预的科研结果与机制边界

本轮以 S26 ad5443e 为基线，仅完成 No-GELU（LA）六次新训练，每次 200 轮。H0 与 LP 的十二个冻结模型复用，全部六个 LA checkpoint 封存后才统一评估。工程合成更新为 8 次，科学优化更新为 150,000 次。B/C 仅预注册，执行数均为零。

## 实际结果

|方法|参数|Dev Accuracy（%）|六组拟合 SD（pp）|Macro-F1（%）|最差 RX（%）|Source Accuracy（%）|Source CE|
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    for a,param in [('H0',133690),('LA',133690),('LP',138346)]:
        r=evidence[a];m=means[a]
        text+=f"|{LABEL[a]}|{param:,}|{r['accuracy_pct']:.3f}|{r['fit_sd_pp']:.3f}|{r['macro_f1_pct']:.3f}|{r['worst_rx_accuracy_pct']:.3f}|{m['source_accuracy_pct']:.3f}|{m['source_ce']:.6f}|\n"
    text+=f'''
主比较 H0−LA：**{delta:+.3f} pp**，条件性 RX 95% 区间 **[{lo:.3f}, {hi:.3f}]**；{primary['positive_fit_pairs']}/6 配对正向，两折变化为 {primary['fold_deltas_pp']} pp。一致性门槛为事前冻结的均值 ≥0.5 pp、区间下界 >0、至少 5/6 配对正向且两折均正向。

{judgment}{inference}

LA−LP：{result['paired_comparisons'][1]['mean_delta_pp']:+.3f} pp。该比较仍混杂分组、宽度、BN、参数化和计算量，不作等效或纯层级因果检验。

## 哪些解释受到约束

1. A 中输入、Token 接口、全部初始 state_dict、卷积、BN 数量、MHA、训练包、增强随机流、选模规则和参数量相同，唯一架构干预为三处 GELU→Identity。因此“参数变少、卷积支持变短或后端更换”不是 A 差值的解释。
2. H0 与 LA 的部分卷积/矩阵 MACs 均为 95.028M；移除 GELU 确实减少逐元素计算，不能声称完整 FLOPs 或延迟完全相同。
3. 固定共享前端输入和推理 BN 后，LA Tokenizer 是仿射映射。整个网络仍含前端非线性、LayerNorm、MHA softmax 等，训练 BN 也跨样本/位置统计，因此“LA 整网线性”是错误说法。
4. 本轮复用 S21 已有但未训练的 H3 定义；不是新增结构发明，也不追溯修改过去的零执行记录。

## Source 与历史 Dev：必须同时看

H0−LA 的 Source Accuracy 变化均值为 {ds:+.3f} pp，Dev 变化为 {dd:+.3f} pp；两者性能损失差 ΔG（H0−LA）为 {dg:+.3f} pp。每组完整值见 source_dev_contrasts.csv。

这些是已用来选 checkpoint 的 Source Validation 与有开发历史的 Dev；包预算为每 TX×RX 40 与 200。它们不能替代 C 的重新训练、四源 RX/一留出 RX 验证，也不能证明接收机变化被特别抑制。不能只报告较小 gap 而隐去源域拟合差异或准确率天花板。

## 尚未识别的机制

非线性表达、优化、BN 运行统计、卷积边界与 MHA 联合作用尚未分离。RX 条件散度和类间/类内比不是互信息或硬件不变性。Source-only 指本次训练权限，不表示这些接收机在科研历史中从未被观察。旧 Confirm/blind 和同步敏感性等负面记录保持原样；没有新的独立采集、安全攻击或未知 TX 拒识结果。

## 下一项最有辨识价值的实验：{next_step}（未启动）

{reason}

B/C 的结构、种子、包身份规则、初始化、正/负/不确定结果解释和预算分别冻结于 PLAN_B.json、PLAN_C.json；C 的单 RX 几何指标适用范围补充见 PLAN_C_MATHEMATICAL_SCOPE.json。不根据 A 的 Dev 成绩修改任一计划。B 最多拟合 12 次，C Pilot 最多拟合 12 次，都不在本轮自动执行；IQ-only、ASK Strict-IB 和额外搜索均未启动。

## 可复现性

科学初始登记提交 ff79743 先于本轮 Source 波形访问和训练；完整六模型 seal 提交先于新 Dev 推理。一次 Windows 进度文件原子替换的 WinError5 在已保存 epoch6（750 步）后中断了 LA_f0_s203；单独登记的 IO 重试包装保留全部冻结源代码，从 epoch7 恢复模型、优化器和全部 RNG，不重新执行已完成的更新。六个独立拟合共七次训练进程调用，科学更新总数仍是 150,000。原错误、快照 SHA 与恢复记录保留。

独立核验 90 个模型×RX 单元、180,000 条预测的标签/行身份、混淆矩阵、Accuracy、F1、保存 logits 的 CE、配对区间、源域选模、恢复流和历史模型哈希。全部种子和接收机保留，无结果驱动的修改或补跑。逐 RX、逐 TX、Source/Dev、完整训练曲线、成本及实际 stdout 均归档；大波形、checkpoint 与逐包预测留在本地。推理成本为单独同会话 GPU-resident 测量，不能与旧会话拼接；并行拟合的 wall time 之和不是独占 GPU 小时。
'''
    (HERE/'RESEARCH_REPORT.zh-CN.md').write_text(text,encoding='utf-8',newline='\n')
    write(HERE/'REPORT_PROVENANCE.json',dict(time=now(),status='GENERATED_FROM_FULL_RECOMPUTED_EVIDENCE',
        results_sha256=sha(HERE/'RESULTS.json'),audit_sha256=sha(HERE/'EXPERIMENT_INTEGRITY.json'),generator_sha256=sha(__file__)))
    print('Complete mechanism report and3 reproducible figures generated; no further fit.',flush=True)


def plots(result,cells,contrasts):
    folder=HERE/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    s={r['arm']:r for r in result['summary']};arms=['LP','LA','H0']
    fig,ax=plt.subplots(1,2,figsize=(8,3),gridspec_kw={'width_ratios':[.8,1.3]})
    for i in range(6):ax[0].plot(range(3),[s[a]['fit_accuracies_pct'][i] for a in arms],color='#b9bfc6',lw=.6)
    for j,a in enumerate(arms):
        ax[0].scatter([j]*6,s[a]['fit_accuracies_pct'],s=16,color=COLOR[a],zorder=3)
        ax[0].scatter(j,s[a]['accuracy_pct'],s=28,marker='D',color=COLOR[a],zorder=4)
    ax[0].set(xticks=range(3),xticklabels=[LABEL[a] for a in arms],ylabel='Dev accuracy (%)',title='All six paired fits')
    for comp,color in zip(result['paired_comparisons'][:2],[COLOR['H0'],COLOR['LA']]):
        names=list(comp['per_rx_delta_pp']);ax[1].plot(range(10),list(comp['per_rx_delta_pp'].values()),marker='o',ms=3,lw=.8,color=color,label=f"{LABEL[comp['a']]} - {LABEL[comp['b']]}")
    ax[1].axhline(0,lw=.6,color='#777777');ax[1].set(xticks=range(10),xticklabels=names,ylabel='Accuracy difference (pp)',title='Every RX: three-seed means')
    ax[1].tick_params(axis='x',labelrotation=35,labelsize=6)
    for label in ax[1].get_xticklabels():label.set_ha('right')
    ax[1].legend(frameon=False,fontsize=7);fig.tight_layout();savefig(fig,folder/'fig01_complete_comparison')
    fig,ax=plt.subplots(1,3,figsize=(9,2.8))
    for j,metric in enumerate(['source_accuracy_pct','dev_accuracy_pct']):
        for i in range(6):ax[j].plot(range(3),[[r[metric] for r in cells if r['arm']==a][i] for a in arms],color='#bcc2c8',lw=.6)
        for x,a in enumerate(arms):ax[j].scatter([x]*6,[r[metric] for r in cells if r['arm']==a],s=13,color=COLOR[a])
        ax[j].set(xticks=range(3),xticklabels=[LABEL[a] for a in arms],ylabel='Accuracy (%)',title='Source validation' if j==0 else 'Historical Dev')
    for key,color,name in [('source_delta_pp',COLOR['LP'],'Source delta'),('dev_delta_pp',COLOR['H0'],'Dev delta'),('gap_delta_pp',COLOR['LA'],'Gap delta')]:
        ax[2].plot(range(6),[r[key] for r in contrasts],marker='o',ms=3,lw=.8,color=color,label=name)
    ax[2].axhline(0,lw=.6,color='#777777');ax[2].set(xticks=range(6),xticklabels=[f"f{r['fold']}/{r['seed']}" for r in contrasts],ylabel='HStem - No-GELU (pp)',title='All paired contrasts')
    ax[2].tick_params(axis='x',labelrotation=40,labelsize=6);ax[2].legend(frameon=False,fontsize=6.5);fig.tight_layout();savefig(fig,folder/'fig02_source_dev')
    fig,ax=plt.subplots(1,2,figsize=(8,2.8))
    for f in (0,1):
        for seed in (201,202,203):
            name=f'LA_f{f}_s{seed}'
            with (HERE/'evidence'/name/'history.csv').open(encoding='utf-8',newline='') as stream:history=list(csv.DictReader(stream))
            ce=[float(r['source_val_ce']) for r in history];epochs=[int(r['epoch']) for r in history]
            ax[f].plot(epochs,ce,lw=.8,label=str(seed));m=read(HERE/'evidence'/name/'metrics.json')
            ax[f].scatter(m['best_epoch'],ce[m['best_epoch']-1],s=14)
        ax[f].set(yscale='log',xlabel='Epoch',ylabel='Clean source RX-equal CE',title=f'No-GELU fold {f}: all200 epochs');ax[f].legend(frameon=False,title='Seed',fontsize=7)
    fig.tight_layout();savefig(fig,folder/'fig03_complete_training')


def savefig(fig,path):
    for ext in ['pdf','png']:fig.savefig(path.with_suffix('.'+ext),dpi=230,bbox_inches='tight',pad_inches=.035)
    plt.close(fig)


if __name__=='__main__':main()
