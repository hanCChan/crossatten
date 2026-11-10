# S18 公平协议与历史证据审计

审计日期：2026-11-10。仅核对仓库已有汇总、配置与公开主来源；不读取真实目标 IQ、不运行目标评分。S15/S16/S17/S17B 及其历史结果保持冻结。S18 的新结果只能来自合法源 RX、日期1–2；源域诊断不能代替真实 CRD。

## 冻结公平 CRD 参照

以下为 RX 等权指标，来自 [S10 最终对照](../10-tifs-validation/FINAL_CR_CRD_COMPARISONS.md)及 [机器可读汇总](../10-tifs-validation/final_comparisons.csv)。IB-RFF 实际由 S06 公平重训，S10 重汇总；ASKNet 为 S10 新重训。S07 用于 ManyRx，S08 用于 ManySig。不可将官方发布单次权重、源域 held-RX 结果与本表混用。

| 数据 | 方法 | Accuracy % | Macro-F1 % | 最差 RX % | Accuracy 的 RX 聚类95%区间 |
|---|---|---:|---:|---:|---|
| ManyRx | IB-RFF | 88.0702 | 85.8595 | 67.1781 | [85.726,90.233] |
| ManyRx | ASKNet | 88.7697 | 86.5035 | 65.2028 | [86.222,90.997] |
| ManyRx | S07_frequency_mild | 87.6506 | 85.4252 | 68.0071 | [84.814,90.274] |
| ManySig | IB-RFF | 91.0926 | 90.8187 | 76.7778 | [87.616,94.102] |
| ManySig | ASKNet | 93.4838 | 93.1502 | 82.9167 | [91.007,95.532] |
| ManySig | S08_hard | 93.0116 | 92.7970 | 84.6667 | [90.090,95.713] |

成对 RX 差值来自 [S10 接收机区间](../10-tifs-validation/receiver_cluster_intervals.csv)：

| 数据 | 差值方向 | 差值 pp | RX 聚类95%区间 pp |
|---|---|---:|---|
| ManyRx | S07 − IB-RFF | −0.4197 | [−1.8907,+1.0249] |
| ManyRx | S07 − ASKNet | −1.1191 | [−2.5269,+0.3268] |
| ManySig | S08 − IB-RFF | +1.9190 | [−1.4537,+5.0695] |
| ManySig | S08 − ASKNet | −0.4722 | [−1.9491,+0.8796] |

这些区间均跨0，不能仅用均值宣称稳定优劣。统计单位是 RX；区间条件于已有模型和种子。外部基线与自有方法的训练种子未配对，RX 配对不等于训练随机性已配对。

[S10 数据与泄漏规则](../10-tifs-validation/PROTOCOL_AND_LEAKAGE_AUDIT.md)统一 non-equalized、256点 IQ、每个 TX/RX/day 前100包、逐包功率归一化、官方 RX 固定顺序四折、日期1–2为源、日期3–4为 CRD。这里的公平首先指数据与评价协议、源域选型和冻结流程；**不同方法保留各自训练配方，不能称全部优化器、训练轮数和选模规则完全匹配**。

[ASKNet 重训说明](../10-tifs-validation/ASKNET_FAIR_REPRODUCTION.md)：固定官方 commit `f00846d68425315b68ae69b8c6b712e08559c2fe`，调用官方模型/训练/loader，24个 formal fits，seed2023–2025，Adam1e-3、wd0、batch128、最多200ep、Plateau(.1,patience5,min1e-6)、ES10。未通过目标结果调整默认值；未核验上游原始 checkpoint。IB-RFF 保留匹配 FAN 初始化加 HSIC 微调的两阶段成本。

## 历史结果与重复创新边界

| 阶段 | 冻结出口与证据 | S18 必须保留的解释 |
|---|---|---|
| S15 | [decision.json](../15-hstem-attribution/decision.json)：`HSTEM_ADVANTAGE_REMAINS_UNDER_MATCHED_PARAMETER_CONTROL` | **S15不是STOP**。HStem 相对参数匹配 Patch-Lite 保持优势；仍混合感受野、BN和归纳偏置，未唯一证明层级性的贡献。 |
| S16 | [STATUS.json](../16-rf-qwen3/STATUS.json)：`STOP_SOURCE_GATE`；28尾部fit | 共同 tokenizer 已冻结；不能把其 MHA 尾部当成端到端 B0，也不能把约90%的 held-source 当作目标 CRD。 |
| S17 | [decision_d0.json](../17-rx-distortion/decision_d0.json)：`STOP_NO_STABLE_ERROR_LINK` | 幅频/相位/IQ稳定fold数2/3/0，错误Spearman中位数.263/.256/.327。相位是统计线索，未识别纯 RX 硬件机制，也未自动准入增强。 |
| S17B | [decision_d1.json](../17b-phase-intervention/decision_d1.json)：`STOP_D1_GATE` | 4个端到端基线、72组冻结干预；没有训练新增增强模型。FIR保持TX性能但相位对齐6.4%–12.8%，未达15%及比同EVM噪声多.02要求。不能推断所有物理增强无效。 |
| S11 | [结果](../11-spectral-generalization/RESULTS.md)：`STOP_MECHANISM_GATE` | 已有 TX/RX/day centroid 方差比频谱先验。散度比及 Fisher-like 原理不是S18新发明；该STOP不是候选网络经完整训练后的失败。 |
| S12 | [结果](../12-spectral-validation/RESULTS.md)：`STOP_SOURCE_GATE` | F2 ManyRx仅+0.0750pp、2/4fold正向，最差单配对−6.95pp。不能将其改名或据ManySig正向补救联合门槛。 |

[S15 结果](../15-hstem-attribution/RESULTS.zh-CN.md)：HStem−参数匹配Patch-Lite，MHA dev_val/confirm为+2.687/+3.142pp，HSTU为+1.733/+2.806pp；条件 RX 区间正向，但未做多重比较校正，confirm也非新盲测。[S17/S17B复盘](../17b-phase-intervention/S17_S17B_REVIEW.zh-CN.md)明确区分输入统计线索、冻结干预与新增方法训练收益。

S11/S12 的重叠尤其重要：[S11新意说明](../11-spectral-generalization/NOVELTY_AND_TIFS_DECISION.md)已将固定 TX/RX/day centroid 方差比先验称为 Fisher-like between/within scatter，门控仅由合法70%训练分区估计。S18若有价值，需落在**已学习的 HStem 中间表征、候选方向子空间、冻结分类干预及 TX 判别性保护**，不能将散度比公式、按RX统计或有界抑制单独宣称原创。

CCAlign 的历史同协议参照见 [S01结果](../01-strong-baselines/RESULTS.md)和 [Shen协议](../docs/PROTOCOL.md)：rl1 Strict-CIS rank16、Adam3e-4、20ep、无AWGN；CE两fold为82.67/85.45%，ccalign为84.91/89.00%。其损失只用源batch的特征/TX/RX标签，不用目标标签；模型和训练配方不同，不是HStem配对CCAlign试验。

本次在当前仓库的报告、配置和代码中**未找到已完成GRL实验的数值证据**。GRL出现在 [原研究方案](../refs/rffi_shen_crossattn_research_plan.md)及 [文献记录](../docs/LITERATURE_2025_2026.md)，不能据此宣称已完成某个HStem配对GRL实验，也不能把后来添加GRL包装为新创新。未找到不等于证明仓库外从未运行过。

## True In-Band 与 Information Bottleneck

Shen TrueIB 是 **True In-Band**：8192点LoRa，Fs1MHz、BW125kHz等固定信号约束。IB-RFF 的IB是 **Information Bottleneck**。WiSig 当前256点、FFT实虚部/对数幅度/相邻频点相位差的五通道前端，不等价于Shen严格带内处理。来源见 [Shen协议](../docs/PROTOCOL.md)、[S13说明](../13-mixer-cfo-mechanisms/README.zh-CN.md)。

IB-RFF的SAF/FAF是可学习频率感知滤波，也不等价于固定带宽裁剪。不得由Shen结果推断WiSig边缘频谱只有噪声；若将来测试裁剪，应在同骨干/配方下单独注册对照，不能与新瓶颈同时改变。

## S18 B0复用与隔离规则

S17B四个端到端基线位于 `../17b-phase-intervention/runs/pilot__ManyRx__M0__f{0..3}__first__s18101/`；例如 [fold0配置](../17b-phase-intervention/runs/pilot__ManyRx__M0__f0__first__s18101/config.json)。训练规则见 [S17B注册](../17b-phase-intervention/registration.json)：

- seed18101，source split16001；40ep上限、batch128、FP32、82426参数；原五通道RF输入、HStem、普通MHA。
- Adam1e-3、wd0、Plateau(.1,patience5,min1e-6)、earlywait10；由源RX等权CE选择checkpoint。
- 官方每fold源RX24个，其中first6留作held-source，fit18个，仅日期1–2；fit池按split16001分70%训练、30%验证。
- 原基线未读取held-source。S18 D0可复用其冻结权重而不新增训练；模型SHA和数据划分应另行核验、记录在S18。
- P1若使用不同初始化、数据顺序、选模准则、冻结tokenizer或训练轮数，便不能将历史B0视为严格配对。新B1–B4必须保持共同RF/HStem/MHA与训练配方；额外损失/瓶颈是唯一预定变量。严格复用成立才可省去4个B0 fit。

方向估计、中心、标准化、候选秩和模型训练不得使用held-source或评价分区统计。源训练70%与隔离源验证30%的索引必须显式封存；验证统计可检查稳定性，不能回流估计方向或反复调强度追分。推理只输入IQ/特征，不能输入RX标签或TX真值。

`s17_data.load()`会经`cache()`写阶段缓存；新阶段应独立配置输出目录与访问日志，避免污染历史。不得调用会`pickle.load`全部RX/日期的历史通用重建入口。源cell合法性应在打开文件前检查；没有合法源缓存时，使用受限元数据索引/字节跨度读取，而非读取整份pickle。

首轮新方法最多ManyRx四fold×一seed；D0无稳定差异即STOP，不能自动训练B1–B4。若进入P1，应预先冻结联合门槛：相对匹配B0平均至少+1pp、至少3/4fold正向、最差配对RX不低于−2pp，并明确RX聚类区间条件；之后才允许独立种子、另一held-source组和ManySig确认。历史目标结果只作背景，S18全过程禁止真实目标CRD访问。

## HSIC对照和公开主来源

[IB-RFF官方README](https://github.com/BeechburgPieStar/IB-RFF)明确FAN、HSIC信息瓶颈、两阶段训练且不需RX标签。[官方训练代码](https://github.com/BeechburgPieStar/IB-RFF/blob/main/main.py)将隐藏表征和输入分别flatten，计算 `cls_loss + alpha*hsic_data`；该项是输入与表征的依赖代理，不能称直接测量或最小化 `I(Z;R)`。

[官方HSIC实现](https://github.com/BeechburgPieStar/IB-RFF/blob/main/utils/hsic.py)实际采用 `hsic_normalized_cca`：Gaussian sigma5、带宽按输入维数缩放、epsilon1e-5、正则化核矩阵逆、输出两个归一化算子乘积的迹。普通中心化HSIC与该nHSIC-CCA公式不同。B1若实现普通HSIC，应明确它是一般信息压缩对照；若迁移nHSIC-CCA，应注明公式、核和数值稳定设置。**两者均不能称完整IB-RFF复现**，因为骨干、输入和两阶段配方未匹配。

结构可核对 [FAN](https://github.com/BeechburgPieStar/IB-RFF/blob/main/backbones/FAN.py)及 [SAF/FAF](https://github.com/BeechburgPieStar/IB-RFF/blob/main/backbones/Adaptive_Filter_Module.py)：SAF→Patch→FAF→残差卷积bottleneck→mean→分类头。公开主来源于本次审计只读访问；后续若使用上游代码训练，应固定实际commit，而非依赖可变化的main链接。

[RIEI主论文v1](https://arxiv.org/abs/2411.03636v1)已经提出Separable Condition，并将发射机与接收机相关特征解耦，仅以发射机相关特征识别。故“有选择抑制RX信息”“TX/RX分离”这一宽泛思想已有先例。S18当前只能称待验证候选；新意仍需完整对照，源域统计与干预也不足以证明纯TX/RX硬件因素可辨识或完全分离。无稳定增益时保留负结果，不宣称超过IB-RFF、ASKNet或形成已验证的新算法。
