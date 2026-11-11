"""Publish derived JSON diagnostics as a report/plots; no original arrays opened."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent


def read(name): return json.loads((HERE/name).read_text(encoding="utf-8"))
def number(v, digits=2): return f"{v:.{digits}f}"
def pct(v, digits=2): return number(100*v,digits)


def main():
    summary=read("SUMMARY.json"); registry=read("INPUT_REGISTRY.json")
    complete=read("ANALYSIS_COMPLETE.json"); qa=read("QA_REPORT.json")
    assert complete["status"]=="COMPLETE_NO_NEW_TRAINING" and qa["status"]=="PASS"
    models={(r["fold"],r["arm"],r["role"]):r for r in summary["models"]}
    groups={(f,a,r):read(f"groups/{a}_f{f}_{r}.json") for f in range(4) for a in ("B0","ASK") for r in ("fit_validation","held_source")}
    repeated=read("REPEATED_INPUT_FOLDS.json")
    pairs={(r["fold"],r["role"]):r for r in summary["paired_models"]}

    # Figures use already-archived derived summaries, never scientific IQ/logit files.
    colors={"B0":"#2878A3","ASK":"#D87828"}
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    x=np.arange(4)
    for arm in ("B0","ASK"):
        for role,style,label in [("fit_validation","--","fit RX validation"),("held_source","-","held source RX")]:
            axes[0,0].plot(x,[100*models[f,arm,role]["rx_equal_accuracy"] for f in x],style+"o",color=colors[arm],label=f"{arm}: {label}")
    axes[0,0].set(title="A. Seen-RX validation and held-RX accuracy",xlabel="Historical fold",ylabel="RX-equal accuracy (%)",xticks=x)
    axes[0,0].legend(frameon=False,fontsize=8)
    for j,role in enumerate(("fit_validation","held_source")):
        axes[0,1].bar(x+(j-.5)*.35,[100*models[f,"B0",role]["tx_rx_day_concentration"]["top_10pct_error_share"] for f in x],width=.35,color=["#72AFCB","#2878A3"][j],label=["Fit RX validation","Held source RX"][j])
    axes[0,1].set(title="B. HStem errors in the worst ~10% of cells",xlabel="Historical fold",ylabel="Share of errors (%)",xticks=x,ylim=(0,100))
    axes[0,1].legend(frameon=False,fontsize=8)
    rescued=[100*pairs[f,"held_source"]["packet"]["b0_wrong_ask_correct"]/pairs[f,"held_source"]["packet"]["n"] for f in x]
    broken=[100*pairs[f,"held_source"]["packet"]["b0_correct_ask_wrong"]/pairs[f,"held_source"]["packet"]["n"] for f in x]
    axes[1,0].bar(x,rescued,color=colors["B0"],label="B0 error corrected by ASK")
    axes[1,0].bar(x,-np.array(broken),color=colors["ASK"],label="B0 correct, ASK wrong")
    axes[1,0].plot(x,np.array(rescued)-broken,"ko",label="Net packet accuracy gain")
    axes[1,0].axhline(0,color="#555555",lw=.8)
    axes[1,0].set(title="C. Small net gain, many changed decisions",xlabel="Historical fold",ylabel="Share of packets / gain (pp)",xticks=x)
    axes[1,0].legend(frameon=False,fontsize=8)
    for arm in ("B0","ASK"):
        row=next(r for r in repeated["rows"] if r["rx"]=="3-19" and r["tx"]==4 and r["arm"]==arm)
        axes[1,1].plot(row["folds"],100*np.array(row["fold_errors"])/row["n"],"o-",color=colors[arm],label=arm)
    axes[1,1].set(title="D. RX3-19 / TX4: same 200 input rows",xlabel="Historical fold (fit set/seed also change)",ylabel="Error rate (%)",xticks=[1,2,3],ylim=(-1,36))
    axes[1,1].legend(frameon=False)
    for ax in axes.flat: ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Frozen source-only classification diagnostics — no new model fits",fontsize=14)
    fig.savefig(HERE/"DIAGNOSTIC_OVERVIEW.png",dpi=180);plt.close(fig)

    receivers=sorted({r for f in registry["folds"] for r in f["held_rx"]})
    fig,axes=plt.subplots(1,2,figsize=(13,7),constrained_layout=True)
    cmap=plt.get_cmap("magma").copy();cmap.set_bad("#D7D7D7")
    for ax,arm in zip(axes,("B0","ASK")):
        matrix=np.full((len(receivers),10),np.nan)
        for i,rx in enumerate(receivers):
            for t in range(10):
                available=[row for f in range(4) for row in groups[f,arm,"held_source"]["per_tx_rx"] if row["rx"]==rx and row["tx"]==t and row["n"]]
                if available: matrix[i,t]=100*np.mean([1-r["accuracy"] for r in available])
        heat=ax.imshow(np.ma.masked_invalid(matrix),cmap=cmap,vmin=0,vmax=100,aspect="auto")
        ax.set(xticks=range(10),yticks=range(len(receivers)),yticklabels=receivers,xlabel="TX class index (0-based)",ylabel="Unique held-source RX",title=arm+": error by TX x RX")
        for i in range(len(receivers)):
            for t in range(10):
                if np.isfinite(matrix[i,t]) and matrix[i,t]>=20:
                    ax.text(t,i,f"{matrix[i,t]:.0f}",ha="center",va="center",fontsize=8,color="white" if matrix[i,t]<65 else "black")
    fig.colorbar(heat,ax=axes,label="Error rate (%), mean over available frozen folds",shrink=.8)
    fig.suptitle("Each RX is one row; repeated folds are averaged, not independent receivers\nGrey = no observed TX support; same input cohort is reused across folds1–3",fontsize=12)
    fig.savefig(HERE/"TX_RX_RISK.png",dpi=180);plt.close(fig)

    def cell(f,arm,t,rx,day):
        return next(r for r in groups[f,arm,"held_source"]["per_tx_rx_day"] if r["tx"]==t and r["rx"]==rx and r["day"]==day)

    lines=[
        "# 冻结 HStem / ASKNet：分类功能失效诊断",
        "",
        "日期：2026-11-11。基于 `7ae38a7`；研究分支 `research/classification-failure-diagnostics`。诊断代码、数据白名单及描述性规则先封存于 **`0a968be`**，随后读取旧预测。",
        "",
        "状态：**DIAGNOSTIC_COMPLETE_MECHANISM_UNRESOLVED**。本轮完成只读功能诊断，没有训练新模型或执行模型 forward，没有打开新 IQ、checkpoint 或特征缓存，没有进入路线 I/II/III 的算法试验。",
        "",
        "## 研究结论",
        "",
        "HStem 的失败确实集中在特定 TX×RX×Day 组合，部分组合可由另一个冻结强模型大幅恢复；同时出现了随 RX/day 条件变化的错误统计和随训练实例变化的局部混淆。**当前不能把原因唯一归结为‘局部扰动下边界脆弱’，尚不足以启动路线 I 的鲁棒 Margin 损失训练。**",
        "",
        "下一步应把可信变换族验证与一个同预算的既有组风险训练对照分别设计。路线 II 的环境混合/多原型解释缺少 embedding 和留 TX/day 可预测性证据，继续保留为条件候选。本轮没有新命名算法，也没有修改 S17–S20 的历史 STOP。",
        "",
        "![诊断总览](DIAGNOSTIC_OVERVIEW.png)",
        "",
        "## G0：本轮实际使用的信息",
        "",
        "| fold | train 包 | fit-validation 包 | held-source 包 | fit 缺失 TX×RX×Day cell | 非空 train cell 支持 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for f in range(4):
        rows=groups[f,"B0","fit_validation"]["per_tx_rx_day"]
        nonzero=[r["training_support"] for r in rows if r["training_support"]]
        lines.append(f"| {f} | {registry['folds'][f]['train_packets']} | {models[f,'B0','fit_validation']['packet']['n']} | {models[f,'B0','held_source']['packet']['n']} | {sum(r['training_support']==0 for r in rows)} / 360 | {min(nonzero)}–{max(nonzero)} |")
    lines += [
        "",
        "每折为 24 个 source RX 中 18 个 fit、6 个 held，另外 8 个该折 target RX 禁止访问。只使用日期1–2。held 组共 **12 个唯一 RX、23,800 条按同 RX 输入血缘去重的记录**，四折合计 47,800 条 held 预测/模型；folds1–3 重复同 6 个 RX、同一批 12,000 输入记录，不能将它们当成新增接收机或新包。该去重不声称跨 RX 是物理不同的发射事件，也不为其建立包配对。",
        "",
        "读取的是 16 个封存预测文件和 20 个 source 标签/索引文件；独立 QA 再读 16 个登记预测文件。数据边界、每路径 SHA 和顺序依据见 [G0_PROTOCOL](G0_PROTOCOL.json)、[INPUT_REGISTRY](INPUT_REGISTRY.json)、[ACCESS_LOG](ACCESS_LOG.json) 与 [QA_ACCESS_LOG](QA_ACCESS_LOG.json)。旧 cache 没有独立包 ID；同模型输入的配对依据是确定性索引/顺序及匹配的源 cell 哈希，详见 [D4 审计](D4_FRONTEND_AUDIT.zh-CN.md)。不同 RX 包从未按数组下标配对。",
        "",
        "## D1：错误集中，但不能简单归因于少样本",
        "",
        "| HStem fold | fit 验证：最高约10%组的错误份额 | held：最高约10%组的错误份额 | held 这些组的包份额 | train 支持数 vs 验证错误 Spearman |",
        "|---|---:|---:|---:|---:|",
    ]
    for f in range(4):
        a=models[f,"B0","fit_validation"];b=models[f,"B0","held_source"]
        lines.append(f"| {f} | {pct(a['tx_rx_day_concentration']['top_10pct_error_share'])}% | {pct(b['tx_rx_day_concentration']['top_10pct_error_share'])}% | {pct(b['tx_rx_day_concentration']['top_10pct_packet_share'])}% | {number(a['support_and_stability']['training_support_error_spearman'],3)} |")
    lines += [
        "",
        "最高风险约10%的观察组承载远超样本份额的错误；held 非空 cell 都有 100 包，只有 fold0 缺失 2/120 cell。训练支持与验证错误的相关性接近零，不能直接把组风险差异解释为采样数量不足。此处只有训练标签支持和训练 RX 验证风险，**没有训练集逐包预测/梯度，所以没有测得实际训练组损失的因果机制**。",
        "",
        "每折 HStem 最差的一个观察 cell（探索性选择，均 n=100）：",
        "",
        "| fold | TX索引 | RX | Day | HStem 错误数 | ASK 错误数 | HStem错误被ASK纠正 | HStem正确被ASK破坏 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for f,t,rx,day in [(0,9,"8-14",2),(1,5,"2-1",2),(2,2,"2-1",2),(3,4,"2-19",1)]:
        a,b=cell(f,"B0",t,rx,day),cell(f,"ASK",t,rx,day)
        pair=next(r for r in read(f"pairs/f{f}_held_source.json")["per_tx_rx_day"] if r["tx"]==t and r["rx"]==rx and r["day"]==day)
        lines.append(f"| {f} | {t} | {rx} | {day} | {a['errors']} | {b['errors']} | {pair['b0_wrong_ask_correct']} | {pair['b0_correct_ask_wrong']} |")
    lines += [
        "",
        "这些结果说明部分 HStem 失败包可被另一个既有分类器判对，不能将 HStem 失败直接解释为观测不可辨识。它们不证明模型利用了纯 TX 硬件信息，也没有证明其余包在信息论上可分或不可分。ASK 仍会制造其他局部错误，不能用挑出的成功 cell 宣称整体优越。",
        "",
        "![唯一RX下的TX风险](TX_RX_RISK.png)",
        "",
        "## D2：观察到裕量变化，尚未观察扰动鲁棒性",
        "",
        "`margin = true logit − largest wrong logit`。同时保存最难负类计数、预测混淆、置信度、熵及 5/25/50/75/95% 分位数。原始 Margin 受得分尺度影响，不能直接比较两模型。normalized margin 除以该包 10 类 logits 的标准差，对公共平移/正尺度缩放不变；它依然不是物理边界距离或认证半径。",
        "",
        "| HStem fold | fit RX Accuracy % | held RX Accuracy % | fit / held RX等权 normalized margin | 共同TX/day分层 Margin差 held−fit | held 错误中 confidence≥0.9 的比例 |",
        "|---|---:|---:|---|---:|---:|",
    ]
    gaps={(r["fold"],r["arm"]):r for r in summary["role_comparisons"]}
    for f in range(4):
        a,b=models[f,"B0","fit_validation"],models[f,"B0","held_source"]
        lines.append(f"| {f} | {pct(a['rx_equal_accuracy'])} | {pct(b['rx_equal_accuracy'])} | {number(a['rx_equal_normalized_margin'],3)} / {number(b['rx_equal_normalized_margin'],3)} | {number(gaps[f,'B0']['standardized_gaps']['normalized_margin_mean'],3)} | {pct(b['packet']['high_confidence_fraction_among_errors'])}% |")
    lines += [
        "",
        "HStem 的共同 TX/day 分层 Margin 四折均下降；但错误并非全在零附近。fold1/TX5/RX2-1/day2 的 100 个错误中有 69 个原 confidence≥0.9，Margin 中位数 −6.9466。未做概率校准，confidence 不能当作真实正确概率。",
        "",
        "一个反例是 ASK：其分层 normalized margin 也四折下降，但 fold2/3 的 held Accuracy 高于 fit-validation，且 CE 更低。这说明平均 Margin 不能单独决定未知 RX 分类收益。干净 logits 更不能给出局部 `T_eta x` 的最坏响应；没有可信变换族时，不把这项诊断叫作已验证的‘边界脆弱’机制。fit-validation 曾参与选模，角色比较也不是同包或随机实验。",
        "",
        "## D3：净增益小，错误集合变化大",
        "",
        "| held fold | HStem错→ASK对 | HStem对→ASK错 | 两者都错 | 错误Jaccard | ASK−B0 RX等权 pp | 标签oracle上界 % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for f in range(4):
        row=pairs[f,"held_source"];p=row["packet"]
        lines.append(f"| {f} | {p['b0_wrong_ask_correct']} | {p['b0_correct_ask_wrong']} | {p['both_wrong']} | {number(p['error_jaccard'],3)} | {pct(row['rx_equal_delta_accuracy'],4)} | {pct(row['rx_equal_diagnostic_oracle_accuracy'])} |")
    lines += [
        "",
        "例如 fold2：ASK 纠正 686 个 HStem 错误，同时把 583 个 HStem 正确包变错，净增 103 个正确包。两模型只共同错 376 个包。oracle 使用真实标签选择正确模型，只是错误互补的诊断上界；本轮没有设计或验证集成器，也不将它作为可部署性能。原始包等权净差和 RX 等权差可能不同，表中明确使用后者。",
        "",
        "S20 已定位的最差配对 RX3-19（fold2，−8.05 pp）可进一步拆成：",
        "",
        "| TX索引 | Day | n | HStem错 | ASK错 | ASK最主要错误类别及计数 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for t in (1,4):
        for day in (1,2):
            a,b=cell(2,"B0",t,"3-19",day),cell(2,"ASK",t,"3-19",day)
            neg=int(np.argmax(b["wrong_predicted_counts"]))
            lines.append(f"| {t} | {day} | {a['n']} | {a['errors']} | {b['errors']} | TX{neg}：{b['wrong_predicted_counts'][neg]} |")
    lines += [
        "",
        "TX1 的主要增错在 day2；TX4 在 day1 的 52 个错误全预测成 TX5。这里 TX 索引从0开始，TX1/4 对应历史 tx_list 的 `11-1` / `17-11`。这些是探索性定位，不是频带、Koopman 或某项训练设置的因果归因。",
        "",
        "同一 RX3-19、同一批 200 包/TX，在 folds1–3 的错误数为：",
        "",
        "| TX | 模型 | fold1 | fold2 | fold3 | 三折均错的包数 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for t in (1,4):
        for arm in ("B0","ASK"):
            r=next(r for r in repeated["rows"] if r["rx"]=="3-19" and r["tx"]==t and r["arm"]==arm)
            lines.append(f"| {t} | {arm} | {r['fold_errors'][0]} | {r['fold_errors'][1]} | {r['fold_errors'][2]} | {r['all_folds_wrong']} |")
    lines += [
        "",
        "TX1 的 ASK 困难反复出现，TX4 的严重失败主要出现在 fold2；因此同一接收机内也至少存在不同的功能模式。相同输入不变而模型输出变化，说明不能把这些错误全部归为观测不可辨识。各折 fit RX、data seed/split 和训练轨迹共同改变；它不是纯 seed 消融，也不能用 folds1–3 当独立确认。",
        "",
        "## 不确定性、解释范围与下一步",
        "",
    ]
    ci=summary["receiver_uncertainty"]
    lo,hi=ci["conditional_rx_bootstrap_interval"]
    lines += [
        f"按同 RX 跨 fold 先平均，12 个唯一 RX 的 ASK−B0 均值为 **{pct(ci['mean_delta'],4)} pp**。本次预先固定 bootstrap seed21111、5000次的条件区间为 **[{pct(lo,4)}, {pct(hi,4)}] pp**，仍跨零。S20 原区间 [−0.7681,+2.6029] pp（seed20101）保持原样；有限 Monte Carlo 抽样差异不是新科学证据。区间不覆盖共享训练集、单 macro-seed、模型选择和反复探索的不确定性，不能宣布等价或协议无效。",
        "",
        "D4 确认当前是未均衡版本、每 TX×RX×Day 前100包及原前端路径；没有核验本地 EQ/non-EQ 逐包映射，也没有证明任何归一化是失效原因，见 [完整前端审计](D4_FRONTEND_AUDIT.zh-CN.md)。",
        "",
        "三个竞争解释的支持、反例、与 S17/S18/S19 的实质区别和最小证伪实验见 [HYPOTHESES.zh-CN.md](HYPOTHESES.zh-CN.md)。目前的具体取舍：",
        "",
        "1. 路线 I：暂缓算法训练预注册。先补一个可信、TX 判别安全且对应真实变化的受限变换族；干净 Margin 下降不能代替这一步。",
        "2. 路线 III：把同骨干、同预算的强 ERM / 一个既有 TX×RX 风险目标列为必要控制设计，尚未启动；高风险集中不自动证明 GroupDRO/CVaR 有效。",
        "3. 路线 II：缺少 feature 条件结构和跨 TX/day 预测证据，不实现多原型或边缘化头。",
        "",
        "新训练、IQ 或新输入版本访问均需下一份具体登记/预注册。本轮不重新运行相位 FIR、静态 RX 子空间、频带 Teacher，不用历史 held 上的定向修复宣称新泛化。现有证据也没有检验统一 WiSig/LoRa 机制。",
        "",
        "## 复核与归档",
        "",
        "7 项合成检查通过；独立 float64 程序重算了全部16份旧预测，核对 3840 个 TX×RX×Day 组、1920 个模型配对组，精确重现 192 个已归档 RX 混淆矩阵。见 [QA_REPORT](QA_REPORT.json)。cell 的 Wilson 区间假设包独立，只作描述，不能代替 RX 聚类不确定性；Macro-F1 保留全部10类，缺失 cell 的指标为 null。",
        "",
        "- [总体与分层统计](SUMMARY.json)、[完整组表目录](groups/)、[模型配对表目录](pairs/)、[同输入跨fold](REPEATED_INPUT_FOLDS.json)。",
        "- [决定](DECISION.json)、[执行范围](ANALYSIS_COMPLETE.json)、[封存清单](MANIFEST.json)。",
        "- `python -m unittest discover -s research/classification_failure_diagnostics -p 'test_*.py' -v`：仅合成检查。",
        "- `python research/classification_failure_diagnostics/verify.py`：只重读白名单旧预测做数值复核；会更新本分支的派生 QA 报告。",
        "- `python research/classification_failure_diagnostics/report.py`：只读取派生 JSON 重生成报告/图表，不打开科学数组。",
        "",
        "原始执行 `diagnose.py` 会拒绝覆盖已有完成记录；不得删除记录来追逐结果。跨机器复现需要 INPUT_REGISTRY 所列本地旧预测与标签文件及匹配 SHA。原始 IQ、权重、逐包预测、外部作者源码均未上传。参数、时延和显存没有重测；模型未改，成本只引用 [S20 原测量](../../20-asknet-source-comparison/COSTS_AND_STRUCTURE.json)。",
    ]
    (HERE/"README.zh-CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
    decision=dict(status="DIAGNOSTIC_COMPLETE_MECHANISM_UNRESOLVED",client_date="2026-11-11",
                  preread_registration_commit="0a968be00cfe8a144effe0e17cb992c23d2c498e",
                  route_I_algorithm_training_admitted=False,
                  route_I_reason="Clean-score diagnostics do not validate a TX-safe environment transform or perturbation-specific boundary failure.",
                  route_III="Design a same-budget established group-risk control; efficacy untested, no training started",
                  route_II="Conditional only; no embedding multimodality or transferable nuisance model established",
                  new_classifier_fits=0,new_iq_reads=0,target_crd_reads=0,dates_3_4_scientific_reads=0,
                  historical_STOP_modified=False,novel_algorithm_claimed=False,independent_confirmation_claimed=False,
                  hypotheses_mutually_exclusive=False)
    (HERE/"DECISION.json").write_text(json.dumps(decision,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(dict(status=decision["status"],plots=2,new_model_fits=0)))


if __name__=="__main__": main()
