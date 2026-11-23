"""Describe actual no-fit results and inactive plans, without filling missing cells."""
import csv
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pypdf import PdfReader
from s28_core import HERE,ROOT,read,write,sha,now

def main():
    audit=read(HERE/'AFFINE_AUDIT.json');precision=read(HERE/'PRECISION_DIAGNOSTIC.json');source=read(HERE/'SOURCE_RECOVERY.json')
    assert source['status']=='PASS' and len(audit['models'])==len(precision['models'])==6
    rows=[]
    for a,p in zip(audit['models'],precision['models']):
        assert a['id']==p['id']
        rows.append(dict(id=a['id'],fp64_token_max=a['source_FP64']['errors']['tokens']['max_abs'],fp64_logit_max=a['source_FP64']['errors']['logits']['max_abs'],
            naive_fp64_token_max=a['source_FP64']['errors']['naive']['max_abs'],native_fp32_token_max=a['source_FP32']['errors']['tokens']['max_abs'],
            native_fp32_logit_max=a['source_FP32']['errors']['logits']['max_abs'],native_changed=a['source_FP32']['changed_predictions'],
            native_contract=a['source_FP32']['errors']['tokens']['allclose'] and a['source_FP32']['errors']['logits']['allclose'],
            strict_probe_token_max=p['cuDNN_TF32']['False']['errors']['tokens']['max_abs'],strict_probe_logit_max=p['cuDNN_TF32']['False']['errors']['logits']['max_abs'],
            strict_probe_contract=p['cuDNN_TF32']['False']['errors']['tokens']['allclose'] and p['cuDNN_TF32']['False']['errors']['logits']['allclose'],
            strict_probe_changed=p['cuDNN_TF32']['False']['changed_predictions']))
    with (HERE/'affine_summary.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    source_means={arm:np.mean([r['source_accuracy'] for r in source['models'] if r['arm']==arm])*100 for arm in ['H0','LA','H2','LP']}
    b=read(ROOT/'27b-tokenizer-backend-interaction/RESULTS.json');a=read(ROOT/'27-mechanism-identification/RESULTS.json')
    old=read(ROOT/'21-hstem-mechanisms/summary.json')
    dev={'H0':next(q['accuracy_pct'] for q in b['summary'] if q['arm']=='H0'),'LP':next(q['accuracy_pct'] for q in b['summary'] if q['arm']=='LP'),
         'LA':next(q['accuracy_pct'] for q in a['summary'] if q['arm']=='LA'),'H2':next(q['acc'] for q in old['arms'] if q['arm']=='H2' and q['role']=='dev_val')}
    write(HERE/'FACTORIAL_READINESS.json',dict(time=now(),status='THREE_HISTORICAL_CELLS_VERIFIED_FOURTH_NOT_RUN',
        rows=[dict(arm=arm,source_accuracy_pct=source_means[arm],historical_dev_accuracy_pct=dev[arm],new_fits=0) for arm in ['H0','LA','H2','LP']],
        H2_LA=dict(planned_fits=6,completed_fits=0,source_accuracy=None,dev_accuracy=None),interaction=None,
        reason='The missing cell has no scientific fit; do not invent a factorial interaction or infer a unique optimization cause.'))
    summary=dict(time=now(),scientific_fits=0,synthetic_gate_updates=read(HERE/'GATE.json')['synthetic_optimizer_updates'],
        source_reference_models=24,source_predictions=48000,affine_models=6,native_source_packets=12000,FP64_source_probe_packets=300,
        FP64_all_pass=all(q['source_FP64']['errors'][k]['allclose'] for q in audit['models'] for k in ['tokens','logits']),
        FP64_max_token_error=max(q['fp64_token_max'] for q in rows),FP64_max_logit_error=max(q['fp64_logit_max'] for q in rows),
        native_contract_all_pass=all(q['native_contract'] for q in rows),native_max_token_error=max(q['native_fp32_token_max'] for q in rows),
        native_max_logit_error=max(q['native_fp32_logit_max'] for q in rows),native_changed_predictions=sum(q['native_changed'] for q in rows),
        strict_probe_all_pass=all(q['strict_probe_contract'] for q in rows),strict_probe_max_logit_error=max(q['strict_probe_logit_max'] for q in rows),
        strict_probe_changed_predictions=sum(q['strict_probe_changed'] for q in rows),A_completed=0,C_completed=0,S28B_completed=0,
        original_native_failure_preserved=True,scientific_train_authorization='PENDING_EXPLICIT_USER_REPLY',new_Dev_Confirm_blind_reads=0)
    write(HERE/'SUMMARY.json',summary)
    prior=read(ROOT/'27b-tokenizer-backend-interaction/EVIDENCE_MATRIX.json')['rows']
    prior += [dict(claim='Frozen No-GELU tokenizer is affine conditional on shared-stem output.',definition='Finite strided zero-padded operator with frozenBN; all32token positions.',
                   result=f"Six FP64Sourceprobes pass; maxlogiterror {summary['FP64_max_logit_error']:.3g}. NativeTF32 runtime fails1e-4 and changes1/12000 predictions; all retained.",
                   status='QUALIFIED_FUNCTION_REPRESENTATION_EVIDENCE',evidence=['28-affine-bn-gelu-audit/AFFINE_AUDIT.json','28-affine-bn-gelu-audit/PRECISION_DIAGNOSTIC.json'],
                   limit='The shared stem and MHA stay nonlinear; training BN is batch dependent. Equivalence does not prove an optimization cause.'),
              dict(claim='BN moderates the usefulness of tokenizer GELU.',definition='(H0-LA)-(H2-H2_LA)',result='Three historicalcells verified; fourthcell0/6fits; interactionNA.',
                   status='NOT_YET_TESTED',evidence=['28-affine-bn-gelu-audit/PLAN_A.json'],limit='Exploratory, post-S27B; BN also changes352affine parameters and optimization.'),
              dict(claim='HStem specifically reduces source-to-heldRX degradation.',definition='OriginalS27C fresh4RX fit; pairedheldaccuracy anddeltaG.',
                   result='Originalplan preserved byteforbyte;0/12fits.',status='NOT_YET_TESTED',evidence=['28-affine-bn-gelu-audit/PLAN_C.original.json'],limit='OnlytwoheldRX, not independentnewacquisition orpopulation evidence.'),
              dict(claim='Layer factorization uniquely causes a favorable training bias.',definition='Sameinitialfunction, differing parameterization requires future auditedcontrol.',
                   result='Motivation strengthened by affine representation, no direct-operator fits.',status='CANDIDATE_NOT_PROVEN',evidence=['28-affine-bn-gelu-audit/RESEARCH_REPORT.zh-CN.md'],
                   limit='Directunstructured operator changescapacity,initializationconstraints andoptimizerstate; no automatic S28B budget.')]
    write(HERE/'EVIDENCE_MATRIX.json',dict(time=now(),rows=prior,prior_S27B_rows_retained=True,old_experiments_not_renamed=True))
    plots(audit,precision)
    text=f'''# S28-0：仿射算子与 BN×GELU 机制审计

本轮实际完成零训练审计、预注册和工程 Gate。**新增科学拟合0次，S28-A为0/6，S27-C为0/12，S28-B未启动。** 用户附件明确要求新增训练另行授权。工程 Gate 共7次合成更新，含首次比较器设备错误前的3次和修复后的4次；均未使用RF数据。错误只涉及CPU/CUDA优化器标量的检查方式，模型、优化器及随机流恢复执行代码未改。

## 1. 核心发现与边界

六个冻结 No-GELU（LA/H3）模型均支持有限长度仿射物化：在相同Source均衡探针上，float64最大Token误差 **{summary['FP64_max_token_error']:.3e}**，最大logits误差 **{summary['FP64_max_logit_error']:.3e}**；预定atol=rtol=1e-9，全部通过，预测均不变。普通共享340点卷积在首尾Token不等价，误差被完整保留；加入边界权重与位置偏置后才得到完整算子。

**原生数值运行并未通过等价合同。** FP32张量、cuDNN TF32开启时，完整12000条Source预测的最大logits差为 **{summary['native_max_logit_error']:.6f}**，有 **{summary['native_changed_predictions']}/12000** 个argmax改变，六个模型均未达到原定atol=rtol=1e-4。没有放宽容差、删位置或替换失败结果。随后预先登记的同探针TF32开/关诊断中，关TF32全部通过，最大logits差 **{summary['strict_probe_max_logit_error']:.3e}**，预测改变 **{summary['strict_probe_changed_predictions']}/300**。后者是300包数值诊断，不是全12000包的新验收，也不改变历史训练/评价精度。

这与S23已有的另一类转换精度问题方向一致；S23旧实验没有重命名为本轮结果。转换仅保存到本地，不替换历史模型或成绩。数学函数表示可等价，与给定浮点内核运行逐数值等价，是不同命题。

## 2. 数学定义与全部边界

令 z 为共享CNN前端输出，形状32×8192。四个有限长度卷积算子为C1…C4；前三个推理BN为D_i h+d_i（running统计冻结），则Tokenizer为有限仿射映射：

`W_eff = C4 D3 C3 D2 C2 D1 C1`，偏置由逐层递推得到。C_i本身包含各层stride、padding和有限区间裁切，不能把中间零填充忽略。

核/步长/填充为 `(8,4,2)→(8,4,2)→(8,4,2)→(4,4,0)`；长度 `8192→2048→512→128→32`。合成支持340、stride256、padding42，第一中心127.5；加上共享CNN的11点支持，名义CNN路径支持为350。TrueIB与FFT使其不能解释成端到端物理RF局部支持。

`T(z)[:,t] = b_t + Σ_(c,q) K_t[:,c,q] z[c,256t-42+q]`，输入外按零处理；q=0…339。

- t=1…30 共用K；t=0和31使用显式K_0、K_31，并保留全部32个位置。
- 偏置b_t由原Tokenizer零输入得到；中间30位置相同，首尾可不同。
- 系数通过逐层局部伴随反传、逐层合法区间裁切构造，独立于分类标签；不构造2048×262144的巨大稠密矩阵。
- 该表示存储2091008个系数，作为冻结buffer没有可训练参数；不是参数压缩结论。

**仿射只相对于共享前端输出成立。** 两个共享BN、两个LeakyReLU、TrueIB/幅相/FFT前端以及MHA均保留。No-GELU在训练时的BN依赖batch，不是贯穿训练的固定仿射映射。

## 3. 六组完整误差

|模型|FP64 Token最大差|FP64 logits最大差|原生FP32 logits最大差（2000包）|原生argmax改变|严格FP32探针logits最大差（50包）|
|---|---:|---:|---:|---:|---:|
'''
    for q in rows:text+=f"|{q['id']}|{q['fp64_token_max']:.3e}|{q['fp64_logit_max']:.3e}|{q['native_fp32_logit_max']:.6f}|{q['native_changed']}|{q['strict_probe_logit_max']:.3e}|\n"
    text+='''
原生FP32全Source验证每模型2000包；FP64及后续精度诊断使用每RX×TX的首个合法验证包，每模型50包。固定抽样、全部种子、全部位置及失败元素数保留在JSON/CSV。最大相对误差在接近零的位置可能很大，因此同时报告绝对误差、RMS和原定逐元素allclose，不以单一相对误差作结论。

![同一300包探针的精度诊断](figures/precision.png)

![六模型边界处理对比](figures/boundaries.png)

## 4. BN×GELU 预注册：缺失单元未填写

|模型|Tokenizer中间BN|GELU|参数|历史Dev Accuracy（%）|本轮恢复Source Accuracy（%）|
|---|---|---|---:|---:|---:|
'''
    for arm,bn,gelu,param in [('H0','有','有',133690),('LA','有','无',133690),('H2','无','有',133338)]:
        text+=f'|{arm}|{bn}|{gelu}|{param}|{dev[arm]:.3f}|{source_means[arm]:.3f}|\n'
    text+='''|H2_LA（待训练）|无|无|133338|未运行|未运行|

主交互 `(H0−LA)−(H2−H2_LA)` 目前为 **NA**。仅新增H2_LA一种结构，共六次，复用其余18个历史模型；LP六个模型仅作次要对照。两个共享CNN的BN仍保留。GELU无参数，所以新模型与H2的完整初始state相同；所有存活参数与对应H0逐元素相同。移除BN会去掉352个仿射参数，不能把因子效应唯一归因于批统计。

PLAN_A固定原两折、201/202/203、Source160/40、200轮、batch64、AdamW、AWGN和最小Source RX等权CE选模。六模型全部封存后才统一Dev评价。交互按10唯一RX内先平均三个种子，再做10000次RX bootstrap，seed28028；采用双向探索性0.5pp、区间排零、5/6配对同向及两折同向规则。所有四格、反例和Source学习均报告，不做成绩驱动调参、替换种子或声称独立确认。

## 5. S27-C 保持独立且未激活

PLAN_C.original.json与原计划逐字节相同。fold0训练rtl_1/rtl_3/rtl_6/rtl_9，留出b210_1；fold1训练b200_mini_1/pluto_1/rtl_2/rtl_7，留出rtl_5。H0/LP各六个新模型，从S27C-H4x160配对种子重新初始化，不能复用五RX训练权重。每次6400训练/1600源验证、100更新/轮、总20000更新；仅四RX验证选模，12个模型全部封存后才评价留出2000包。

同时报告源域、留出准确率和 `ΔG=(Source−Held)_H0−(Source−Held)_LP`。沿用原规则：两个fold/RX均有held差≥0.5pp、ΔG≤−0.5pp，且至少5/6配对同时方向有利。两台留出RX不是六个独立RX，不做总体显著性/bootstrap。Held-only只有一台RX，跨RX散度按构造为零，Jc不能识别RX不变性，须标为不可识别；不能用epsilon比值制造不变性证据。

本轮数学审计用的是既有五RX模型的合法Source验证，包含这些历史RX；它不产生新C模型或新C留出评价，也不意味着这些接收机在研究历史中从未观察过。C的架构、角色和种子预检已通过，完整科学执行仍须单独激活。

## 6. 研究判断与贡献边界

**可继续研究的候选机制**是多阶段局部卷积的参数化与训练路径；当前仅建立了推理函数表示的数学对象，尚未证明其是性能差异的唯一原因。LA仍保留训练BN，且与LP的分组、自由度、初始化和算力不同；不能从LA−LP的旧2.083pp直接推出纯分层优化因果。

若将有效算子直接作为自由参数，普通32→64、340点稠密卷积仅核与共享偏置就有696384参数，而无BN四层卷积仅86256；若让边界系数独立学习，自由度进一步改变。相同起始函数也不保证参数空间、约束或AdamW状态可比。S28-B须另行审计匹配条件，本轮不设计搜索或拟合。

创新证据矩阵保留S27-B原有全部条目，新增“有精度限定的仿射表示”证据，并明确BN×GELU、RX专项收益、唯一优化归因仍未验证。MHA协同和单一Jw解释均未被重新包装为已证实结论；没有独立新采集或物理RX不变性证明。

## 7. 权限、复现与下一步

24个旧checkpoint的配置、200轮轨迹、最小Source CE选择及48000条原Source预测指标严格恢复；历史权重/config/history哈希保持不变。新增Dev、Confirm、blind数据读取和科学优化均为0。数学代码/容差和数据权限在首次Source读取前提交；精度诊断在看到原失败后单独登记并保留原失败。原始日志、工程更新计数、源数据角色记录和全部数值误差随报告归档。

执行顺序见 registration.json、PLAN_A.json、PLAN_C.original.json。当前launch.py按零预算拒绝训练。用户附件原文：“不要自动启动新实验。先完成零训练审计、预注册和工程 Gate；新增训练须经过明确授权。” 此要求来自本轮用户方案，不是额外推定的批准流程。

建议下一步授权范围明确为A六次，或A六次加原C十二次；两项分别回答BN×GELU交互和RX专项收益，不以一项结果决定是否挑选另一项设计。当前无A/C新成绩，不能提前给出交互或ΔG。
'''
    (HERE/'RESEARCH_REPORT.zh-CN.md').write_text(text,encoding='utf-8',newline='\n')
    write(HERE/'REPORT_PROVENANCE.json',dict(time=now(),generator_sha256=sha(__file__),inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in [HERE/'AFFINE_AUDIT.json',HERE/'PRECISION_DIAGNOSTIC.json',HERE/'SOURCE_RECOVERY.json',ROOT/'21-hstem-mechanisms/summary.json',ROOT/'27-mechanism-identification/RESULTS.json',ROOT/'27b-tokenizer-backend-interaction/RESULTS.json']}))
    pdf(rows,summary,dev,source_means)

def plots(audit,precision):
    folder=HERE/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7,2.7))
    for a,p in zip(audit['models'],precision['models']):
        values=[a['source_FP64']['errors']['tokens']['max_abs'],p['cuDNN_TF32']['True']['errors']['tokens']['max_abs'],p['cuDNN_TF32']['False']['errors']['tokens']['max_abs']]
        ax.plot(range(3),values,marker='o',ms=3,lw=.8,label=a['id'].replace('LA_',''))
    ax.set(yscale='log',xticks=range(3),xticklabels=['FP64','FP32 tensors / TF32 on','FP32 / TF32 off'],ylabel='Maximum absolute Token error',title='Matched fixed Source probes: every frozen LA model')
    ax.legend(ncol=3,frameon=False,fontsize=7);fig.tight_layout();save(fig,folder/'precision')
    fig,ax=plt.subplots(figsize=(7,2.7));names=[q['id'].replace('LA_','') for q in audit['models']];x=np.arange(6)
    ax.bar(x-.18,[q['source_FP64']['errors']['naive']['max_abs'] for q in audit['models']],width=.36,label='Shared kernel + constant bias',color='#ac6735')
    ax.bar(x+.18,[q['source_FP64']['errors']['tokens']['max_abs'] for q in audit['models']],width=.36,label='Boundary weights + position bias',color='#176b8a')
    ax.set(yscale='log',xticks=x,xticklabels=names,ylabel='Maximum absolute Token error',title='All 32 positions retained; FP64, balanced Source probes')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2,frameon=False,fontsize=7);fig.tight_layout();save(fig,folder/'boundaries')

def save(fig,path):
    for ext in ['pdf','png']:fig.savefig(path.with_suffix('.'+ext),dpi=220,bbox_inches='tight',pad_inches=.035)
    plt.close(fig)

def pdf(rows,summary,dev,src):
    body=r'''\documentclass[10pt]{article}
\usepackage[letterpaper,margin=.8in]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,hyperref,xurl}
\hypersetup{hidelinks}\graphicspath{{../figures/}}\setlength{\parskip}{4pt}
\title{S28-0: Finite Affine Operators and Factorial Readiness}
\author{Read-only mechanism audit}\date{23 November 2026}
\begin{document}\maketitle
\section{Outcome and Scope}
Six frozen No-GELU (LA/H3) tokenizers can be materialized as finite affine operators conditional on the shared CNN output. A single shared convolution is insufficient at the two boundary tokens. All six double-precision Source probes pass the predeclared $10^{-9}$ absolute/relative tolerance after explicit boundary handling.

The original runtime uses FP32 tensors with cuDNN TF32 enabled. It fails the unchanged $10^{-4}$ tolerance in all six models and changes one prediction across 12,000 Source packets. A separately preregistered, matched 300-packet TF32-off numerical diagnostic passes. It does not replace the full native-runtime failures or change the historical scientific recipe.

There are zero new scientific fits. The BN-by-GELU fourth cell remains untrained (0/6), and the unchanged source leave-one-RX pilot remains untrained (0/12). The user's attached plan explicitly requires a separate authorization for new fits. Seven synthetic engineering updates are disclosed, including three before a CPU/CUDA comparator error and four after its correction; no RF data enters that gate.
\section{Finite-Length Mathematical Object}
Let $z\in\mathbb R^{32\times8192}$ be the shared CNN output. Each $C_i$ is a finite strided convolution, including its intermediate zero padding and crop. Frozen evaluation BN is the affine map $D_i h+d_i$. Then
\[
T(z)=W_{\rm eff}z+b_{\rm eff},\qquad
W_{\rm eff}=C_4D_3C_3D_2C_2D_1C_1.
\]
The shared CNN contains two BN and two LeakyReLU operations; the RF views and MHA also remain nonlinear. Training BN is batch dependent. Only the tokenizer conditional on $z$ is a frozen affine map.

The convolution (kernel,stride,padding) sequence is $(8,4,2)$ three times, followed by $(4,4,0)$. Lengths are $8192\to2048\to512\to128\to32$. The effective kernel has length 340, stride 256 and padding 42. Its first center is 127.5. The shared CNN extends the nominal local path to 350; global FFT and TrueIB prevent an end-to-end physical locality interpretation.
\[
T(z)_{o,t}=b_{o,t}+\sum_{c=0}^{31}\sum_{q=0}^{339}
K_{t,o,c,q}\,z_{c,256t-42+q},
\]
where outside-input coordinates are zero. Positions $t=1,\ldots,30$ share $K$; positions 0 and 31 need $K_0,K_{31}$. The bias is computed by the original tokenizer's zero response. Local adjoints crop at every intermediate layer; they do not construct a huge dense matrix. All 32 positions remain included. The stored frozen representation has 2,091,008 coefficients, not a model-compression claim.
\clearpage
\section{All Six Frozen Models and Precision Failures}
\begin{center}\small\begin{tabular}{lrrrrr}\toprule
Fold/seed&FP64 token&FP64 logits&Native logits&Native flips&Strict logits\\\midrule
'''
    for q in rows:
        name=q['id'].replace('LA_','').replace('_',r'\_')
        body+=f"{name}&{q['fp64_token_max']:.2e}&{q['fp64_logit_max']:.2e}&{q['native_fp32_logit_max']:.5f}&{q['native_changed']}&{q['strict_probe_logit_max']:.2e}\\\\\n"
    body+=r'''\bottomrule\end{tabular}\end{center}
Errors are maximum absolute differences. Native uses all 2,000 Source-validation packets/model. FP64 and strict-FP32 probes use the first legal validation packet in every TX-by-RX cell: 50/model. Strict here means cuDNN and matmul TF32 disabled; the original native setting is retained in the primary record. Original and converted logits are compared within each setting. No tolerance is relaxed or failing model replaced.
\begin{figure}[htbp]\centering\includegraphics[width=.97\linewidth]{precision.pdf}
\caption{Matched 300-packet Source probes, all six models. These are numerical differences of frozen functions, not training-performance changes.}\end{figure}
\begin{figure}[htbp]\centering\includegraphics[width=.97\linewidth]{boundaries.pdf}
\caption{The shared-kernel baseline fails at the boundaries. Explicit boundary weights and position bias restore the double-precision finite operator.}\end{figure}
\clearpage
\section{BN by GELU: A Registered Missing Cell}
\begin{center}\small\begin{tabular}{lccrrr}\toprule
Cell&Mid BN&GELU&Params&Historical Dev&Source val.\\\midrule
'''
    for arm,bn,gelu,param in [('H0','yes','yes',133690),('LA','yes','no',133690),('H2','no','yes',133338)]:
        body+=f"{arm}&{bn}&{gelu}&{param:,}&{dev[arm]:.3f}&{src[arm]:.3f}\\\\\n"
    body+=r'''H2\_LA&no&no&133,338&not run&not run\\\bottomrule\end{tabular}\end{center}
\[
\Delta_{BN\times GELU}=(A_{H0}-A_{LA})-(A_{H2}-A_{H2\_LA}).
\]
The interaction is not estimated while its fourth cell is missing. Exactly six new H2\_LA fits are planned; eighteen original factorial models are reused and six LP models are secondary controls. Only the three intermediate BN and three GELU modules become Identity. H2 and H2\_LA have the same full initial state; all surviving values match H0 for each paired fold/seed. Two shared-stem BN modules remain. Removing middle BN also removes 352 affine parameters and changes batch statistics, running statistics and optimization jointly.

The exploratory plan follows the original two folds, seeds 201/202/203, first160/last40 packets/TX/RX, 200 epochs, batch64, AdamW, AWGN and minimum clean Source RX-equal CE. All six new models must be sealed before common Dev reporting. Ten unique RXs are resampled after averaging seeds within RX (10,000 repetitions, seed28028). The two-sided practical rule requires magnitude at least .5pp, conditional95\% interval excluding zero, at least5/6 paired effects and both fold means in the same direction. Failure to meet the rule establishes neither equivalence nor absence of a BN effect. The design follows historical Dev observation and is not independent confirmation.
\section{Independent Source-Held Pilot, Still Inactive}
The original S27-C JSON is preserved byte for byte. Fold0 trains rtl1/rtl3/rtl6/rtl9 and holds b210\_1; fold1 trains b200\_mini\_1/pluto\_1/rtl2/rtl7 and holds rtl\_5. Twelve fresh H0/LP models use the fixed S27C-H4x160 paired initialization, never the five-RX-trained weights. Each fit has 6,400 training packets, 1,600 Source-validation packets, 100 updates/epoch and 20,000 total updates. Only four-RX validation selects checkpoints; held evaluation waits for all twelve models to be sealed.

Report both absolute accuracies and $\Delta G=(A_{S}-A_{H})_{H0}-(A_{S}-A_{H})_{LP}$. Retain the original rule: held H0-minus-LP at least .5pp and $\Delta G\leq-.5$pp in both receiver/fold means, with at least5/6 jointly favorable fit pairs. Two held RXs do not become six independent receivers; no population-RX bootstrap or significance is justified. A held-only distribution has one RX: its across-RX scatter is zero by construction, so $J_c$ cannot identify receiver invariance. That diagnostic must be marked not identifiable, rather than interpreted using an epsilon ratio.
\clearpage
\section{What Is and Is Not Established}
The function representation motivates studying structured parameterization and training paths. It does not identify an optimization cause. LA retains batch-dependent training BN; LP differs in grouping, freedom, initialization and compute. Removing all middle BN/GELU is a distinct six-fit intervention, not a proof that the complete RF network is linear.

A free dense 32-to-64, length340 convolution already has 696,384 kernel/bias parameters, versus 86,256 for the four convolution factors without BN. Allowing independent boundary kernels changes capacity further. Equal starting predictions do not equalize parameter constraints or AdamW states. A future direct-operator optimization control needs a separate audited design; no S28-B search or fit is started.

The evidence matrix retains every previous S27-B entry and adds the qualified affine representation. BN-by-GELU interaction, RX-specific gap reduction and a unique optimization explanation remain untested. Neither MHA amplification nor a monotonic Source $J_w$ explanation is reinstated. No new independent acquisition, blind RX or physical receiver invariance is established.
\section{Integrity and Execution Boundary}
All 24 historical models strictly recover their original Source metrics over 48,000 predictions. Model, configuration and 200-epoch history hashes remain unchanged; minimum-Source-CE selection is verified. The mathematical contract and data permissions were committed before Source IO. After native precision failures were recorded, the supplemental matched precision diagnostic was separately frozen and committed before execution. It preserves the six native failures and the one changed prediction.

This audit reads historical five-Source validation, including RXs later designated held in C. It does not create a new C model or a C held evaluation, and never claims these physical receivers have no research exposure. There are no new Dev, Confirm or blind packets. Synthetic engineering updates are counted separately; the corrected optimizer comparator compares exact dtype/values across CPU/CUDA scalar placement without changing the recovery implementation.

The scientific fitting budget is zero. The training entry rejects execution until an explicit user authorization and committed activation. Architecture, initialization, mathematical composition, resources, roles and resume checks have passed; scientific A/C cohorts have not been executed. Raw waveforms and converted coefficient tensors remain local. Public code, plans, complete error tables, logs and audit records support reproduction. The Chinese report contains the detailed evidence boundaries and exact plan links.
\end{document}
'''
    for old,new in {'first160/last40':'first 160/last 40','batch64':'batch 64','seed28028':'seed 28028','conditional95':'conditional 95',
                    'least5/6':'least 5/6','Fold0':'Fold 0','fold1':'fold 1','length340':'length 340'}.items():body=body.replace(old,new)
    folder=HERE/'paper';folder.mkdir(exist_ok=True);tex=folder/'s28_audit.tex';tex.write_text(body,encoding='utf-8',newline='\n')
    temp=HERE/'tmp/pdfs/report';temp.mkdir(parents=True,exist_ok=True)
    r=subprocess.run(['D:/bianyiqi/texlive/2026/bin/windows/latexmk.exe','-pdf','-interaction=nonstopmode','-halt-on-error','-outdir='+temp.as_posix(),tex.name],cwd=folder,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (temp/'build_stdout.txt').write_bytes(r.stdout);assert r.returncode==0,'See build_stdout.txt'
    out=HERE/'output/pdf/s28_affine_audit.pdf';out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes((temp/'s28_audit.pdf').read_bytes())
    render=temp/'render';render.mkdir(exist_ok=True)
    subprocess.run(['D:/bianyiqi/texlive/2026/bin/windows/pdftoppm.exe','-png','-r','115',str(out),str(render/'page')],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    write(HERE/'DOCUMENT_BUILD.json',dict(time=now(),documents=[dict(path=str(out),sha256=sha(out),pages=len(PdfReader(out).pages),render=str(render),log=str(temp/'s28_audit.log'))]))
    print('Reports/figures generated from complete no-fit audit; PDF visual review pending.',flush=True)

if __name__=='__main__':main()
