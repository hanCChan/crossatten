"""Render only derived audit metadata; no original arrays, weights or models."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
def read(p): return json.loads((HERE/p).read_text(encoding="utf-8"))


def main():
    audit=read("AUDIT.json");runs=read("RUN_FACTORS.json");plan=read("CONTROL_PANEL_PLAN.json")
    pairs=[p for p in audit["overlap"] if p["folds"][0]>0]
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    x=np.arange(3);v=[100*p["membership_disagreement_fraction"] for p in pairs]
    axes[0].bar(x,v,color="#2878A3")
    for i,value in enumerate(v): axes[0].text(i,value+1,f"{value:.2f}%",ha="center")
    axes[0].set(xticks=x,xticklabels=[f"Fold{p['folds'][0]} vs {p['folds'][1]}" for p in pairs],ylabel="Common-row train/val membership changes (%)",title="A. Same split seed, different packet assignment",ylim=(0,45))
    for j,arm in enumerate(["B0","ASK"]):
        rows=[r for r in runs if r["arm"]==arm]
        values=[r["selected_updates"] for r in rows]
        axes[1].bar(np.arange(4)+(j-.5)*.35,values,width=.35,label=arm,color=["#2878A3","#D87828"][j])
    axes[1].set(xticks=range(4),xlabel="Historical fold",ylabel="Optimizer steps to selected checkpoint",title="B. Different selected training trajectories")
    axes[1].legend(frameon=False)
    for ax in axes: ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Retrospective source-only factor audit — no causal effect estimated",fontsize=13)
    fig.savefig(HERE/"TRAINING_FACTORS.png",dpi=180);plt.close(fig)
    lines=["# 训练实例差异审计与分阶段控制设计","","日期：2026-11-11。基于 `b3035c9`，分支 `research/training-instance-attribution`。",
           "","状态：**CONFOUNDS_VERIFIED_CAUSAL_ATTRIBUTION_UNRESOLVED**。完成8个历史训练实例、20个旧标签/索引文件及代码的只读审计。没有训练新模型、读取 IQ/权重/逐包预测或修改历史 STOP。",
           "","## 结论","","现有三折不能区分‘初始化不稳定’与‘源RX组成敏感’：每对 folds1–3 仅共享10/18台 fit RX，三折共同 fit RX 只有 `7-7、7-14`；实际 seed、初始状态摘要、共同包的训练/验证身份、LR轨迹和选中epoch也同时变化。它们解释了为什么历史比较无法做单因素归因，但**尚未证明哪一因素主导错误，不能给出因果贡献比例**。",
           "","下一步先设计固定数据的初始化×顺序控制，再单独改变合法 source RX 组成；标准组风险方法留待这些控制之后决定。新 Margin Loss、多原型、新 Attention 和 HStem 替换均不启动。",
           "","![训练差异](TRAINING_FACTORS.png)","","## 1. 同一宏 seed 并没有固定随机性","",
           "实际 seed 来自 `SHA256(S17-WiSig|ManyRx|fold|first|18101)` 的前8位，fold 被包含在哈希输入中。宏 seed 都是18101，但各折实际 seed 不同；同一架构的4个初始化摘要也各不相同。这里只核查已记录的摘要，不曾加载初始权重；旧路径未封存可加载的初始 state。跨架构不能用相同 seed 声称相同初始化。","",
           "| fold | 实际seed | HStem执行/选中epoch | ASK执行/选中epoch | HStem选中更新步 | ASK选中更新步 |","|---|---:|---|---|---:|---:|"]
    for f in range(4):
        b=next(r for r in runs if r["fold"]==f and r["arm"]=="B0");a=next(r for r in runs if r["fold"]==f and r["arm"]=="ASK")
        lines.append(f"| {f} | {a['actual_seed']} | {b['epochs_executed']} / {b['selected_epoch']} | {a['epochs_executed']} / {a['selected_epoch']} | {b['selected_updates']} | {a['selected_updates']} |")
    lines += ["","依据：[原 HStem 训练循环](../../17-rx-distortion/s17_train.py)、[stable_seed](../../lib/xa/train.py)、[S20 包装](../../20-asknet-source-comparison/s20_run.py)。全部配置、初始化和 checkpoint 摘要及训练轨迹见 [RUN_FACTORS.json](RUN_FACTORS.json)。本轮没有重新拟合、重放旧 CUDA 顺序或重算模型输出。",
              "","## 2. 同一 split_seed 也没有固定共同包的身份","","用冻结 RX/day cell SHA 和缓存内确定顺序重建源行身份，未按不同 RX 下标配对。每对 folds1–3 的 fit RX 交集都是10台，RX集合 Jaccard 为10/26；其共同输入行的分配为：","",
              "| folds | 共同输入行 | 两者train | 前train/后val | 前val/后train | 两者val | 身份变化比例 |","|---|---:|---:|---:|---:|---:|---:|"]
    for p in pairs:
        lines.append(f"| {p['folds'][0]} / {p['folds'][1]} | {p['common_pool_rows']} | {p['both_train']} | {p['a_train_b_val']} | {p['a_val_b_train']} | {p['both_val']} | {100*p['membership_disagreement_fraction']:.2f}% |")
    lines += ["","**fold0/2 的整数 train_ids 和 val_ids 向量甚至完全相同，但对应的源RX与包不同**：在19068条共同源行中，仍有1673条改变train/val身份。整数索引是当前池的位置，不是跨fold的包ID。既有 split_seed16001 固定了池位置随机化规则，却没有在池组成改变后固定同一物理记录的分配。",
              "","重建依赖被审计的缓存提取顺序与52个RX/day源cell摘要，未重读 IQ，也没有独立 immutable packet ID。结果是缓存血缘层面的对应，不证明跨 RX 的物理发射事件相同/不同。完整6对比较见 [AUDIT.json](AUDIT.json)。",
              "","## 3. 顺序、学习率和选模还有哪些差别","",
              "| 因素 | HStem+MHA | 官方路径ASK | 本轮能否消除歧义 |","|---|---|---|---|",
              "| 每轮顺序 | 独立CUDA generator，seed=实际seed；历史未记录order SHA | CPU DataLoader使用全局Torch RNG；每轮记录order SHA并在原代码检查全样本覆盖 | 不能验证两模型顺序相同，也未重放旧顺序 |",
              "| 模型内随机性 | 当前路径未启用Dropout；主要显式随机源为初始化和批顺序，BN受批序影响 | dropout默认0，但训练截止含随机Gumbel选择 | 不能把同一numeric seed当相同随机过程 |",
              "| 输入增强 | M0无family，实际没有增强 | 无额外输入增强 | cfg中的augmentation_probability=.5并不说明M0用了增强 |",
              "| LR日志时点 | scheduler.step后记录，通常是下一轮LR | train_epoch内记录，当轮使用LR | 已转换成每轮实际使用的LR再比较 |",
              "| 自适应依据 | RX等权验证CE，最多40轮 | 包等权验证CE，最多200轮 | 实际LR/停止/选中epoch不同，尚无因果干预 |",
              "| checkpoint替代规则 | 未保存全轨迹的包等权CE | 原轨迹上包CE与RX等权CE最优epoch四折恰好一致 | 不能把ASK差异归因于两种规则在已发生轨迹上选了不同checkpoint |",
              "","统一日志时点后，HStem的降LR起始训练epoch为 f0:10、f1:12/19、f2:13、f3:12；ASK为 f0:22/35、f1:23/36、f2:25/37、f3:30/41。HStem在fold1的选中epoch12使用1e-4，其余选中epoch使用1e-3；ASK四折选中epoch均使用1e-4。",
              "","实际LR与最佳epoch可以是初始化/源组成影响的中介，而非互相独立的原因。ASK与HStem还存在不同架构和训练配方，不能把它们称为只改变一个因素的相近模型对照。deterministic标志也不等于已有同设置重放证据，HStem使用过warn_only模式。静态路径和原文件摘要见 [CODE_REVIEW.json](CODE_REVIEW.json)。",
              "","## 4. 已落实到索引层面的下一步设计","",
              "[分阶段控制设计](CONTROL_DESIGN.zh-CN.md)给出两个相互独立的区块：固定fold2原数据的2×2初始化/批顺序设计加一次相同设置重放（草案5次拟合），然后是固定core、替换extension的两臂组成设计（草案4次拟合）。两者都未执行，也不是本轮训练预算。HStem实验只能研究HStem的敏感性，不能直接解释ASK的0/66/5案例；ASK若要归因需另行登记其模型内随机流。",
              "","组成面板只从fold2既有fit18中选，不能借用该折target8。两臂每次各15849个train包，共用8149个core包及同一3551个core-validation包；extension各7700包，并精确匹配TX/day数量与规范标签顺序。生成了可复现的索引SHA，见 [CONTROL_PANEL_PLAN.json](CONTROL_PANEL_PLAN.json)，没有读取对应IQ。",
              "","未来必须固定初始state、各随机流、规范样本键、选模策略、信息权限和区块预算；先封存所有snapshot再读source-held。RX3-19/TX4属于反复查看的探索案例，不能充当独立泛化确认。5/4次小规模对比也不能当成总体方差估计或功效论证。",
              "","## 5. 范围、工程记录与复核","",
              "本轮独立检查8份旧CSV训练日志、6对源池对应与面板索引，原CSV/模型/历史代码保持原样。5项合成检查覆盖索引dtype、池位置非身份、成员交叉表、LR时点和面板匹配。两次纯审计实现错误在生成因素结果前被修正并保留于 [ENGINEERING_ATTEMPT.json](ENGINEERING_ATTEMPT.json)：原数组摘要应保留int32/int64原dtype；源RX名称列表不能与编码fit_rx数组混用。没有重训或修改任何科学结果/阈值。",
              "","20个唯一标签/索引路径在失败审计尝试共读40次、成功审计读20次、独立QA失败/成功各读20次；没有打开IQ、模型权重或旧逐包预测。面板的train索引显式为int64、validation索引为int32，摘要按各自dtype计算。访问记录见 [ACCESS_SCOPE.json](ACCESS_SCOPE.json)、[ACCESS_LOG.json](ACCESS_LOG.json)、[QA_ACCESS_LOG.json](QA_ACCESS_LOG.json)，数值核验见 [QA_REPORT.json](QA_REPORT.json)。这是脚本/声明动作的审计，不是OS级监控。",
              "","本轮状态与下一步限制见 [DECISION.json](DECISION.json)，全部文件校验见 [MANIFEST.json](MANIFEST.json)。`audit.py`拒绝覆盖已有AUDIT；`verify.py`只重读登记标签和历史日志；`report.py`只读取派生JSON。跨机器复核需要登记的本地文件与相同SHA；原始数组、权重与CSV源码不重复上传。",
              "","当前仍不能认定源RX组成、初始化或优化哪一个主导失效。已完成的是把混杂具体化，并准备可区分因素的下一份设计；未启动标准组风险训练、物理扰动训练或新Loss。"]
    (HERE/"README.zh-CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
    decision=dict(status=audit["status"],new_training_admitted=False,new_model_fits=0,causal_ranking_identified=False,
                  next_step="Register fixed-data initialization/order control before separate source-composition control",
                  group_risk_training="Decide only after attribution controls",margin_loss="deferred",multimodal_head="deferred",
                  change_HStem=False,add_attention=False,historical_STOP_changed=False,
                  proposed_fit_counts_not_executed=dict(fixed_data_factorial_plus_replay=5,composition_pairs=4))
    (HERE/"DECISION.json").write_text(json.dumps(decision,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(dict(status=decision["status"],new_model_fits=0,plots=1)))


if __name__=="__main__": main()
