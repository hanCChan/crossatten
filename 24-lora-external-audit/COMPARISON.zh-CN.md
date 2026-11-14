# LoRa同协议比较：新增主表与历史证据分开

**本阶段没有训练，没有新增测试准确率。** 以下两张是正式实验待填表，不能用于宣称超过任何方法；逐(fold,seed)空结果入口为 [native_full.csv](native_full.csv) 与 [strict_ib_full.csv](strict_ib_full.csv)。

表A：Native Full Method。相同原包、TX/RX/源验证权限；保留各方法前处理、网络和合理训练配方。各法可获得的频带可能不同。H0自己的原生前处理即TrueIB。

| 方法 | fit状态 | RX等权Acc | RX等权Macro-F1 | 最差RX | 参数/实自由度 | 成本 |
|---|---|---|---|---|---|---|
| 冻结HStem+PlainMHA H0 | 历史6模型已核验，可复用；未重新评估 | 见历史表 | 见历史表 | 见历史表 | 133,690 | 未来与其他模型同机重测 |
| Shen-CIS+RA/DANN source-only | 未训练，跨框架核验/协议修复待冻结 | — | — | — | 13,509,263训练；13,442,954推断 | — |
| ASKNet-LoRa原默认 | 未训练，完整默认GPU资源门槛待验证 | — | — | — | 74,153,872元素；141,246,353实自由度 | — |
| IB-RFF Shen8192适配 | 未训练，完整适配配置待冻结 | — | — | — | 待冻结patch后确定 | — |

表B：Strict-IB Full Method。统一以原包的±62.5kHz投影为信息预算，再保留方法主体。全部为信息预算控制适配；尤其不能称官方原数据/任务复现。算法内生成的其他频率不引入额外原始带外信息。

| 方法 | fit状态 | RX等权Acc | RX等权Macro-F1 | 最差RX | 参数 | 资格 |
|---|---|---|---|---|---|---|
| H0 | 同表A复用同一6模型 | 见历史表 | 见历史表 | 见历史表 | 133,690 | 原模型就是TrueIB |
| P_IB→Shen-CIS+RA | 未训练 | — | — | — | 同原版 | 前处理已数值核验，训练规则待冻 |
| P_IB→ASKNet-LoRa | 未训练 | — | — | — | 同原版 | 未缩小候选，完整资源门槛待验证 |
| P_IB→IB-RFF适配 | 未训练 | — | — | — | 待定 | 8192两阶段配置待冻结 |

候选TIFS2022 triplet/15NN与TIFS2023 Transformer完成资格与输入审计后可另登记；Ma/gMLP缺代码时保持缺项。不以输出0或随机网络精度代替未完成结果。

## 已有结果，仅作历史不同配方参考

这些数值从现有报告只读摘录，没有重新加载评估波形、选新模型或重新训练。原始角色仍按历史文件命名；confirm已曝光，不能称新盲测。不同表中的SD定义不同，禁止合并成一个显著性检验。特别是S06的S_hstem使用HSTU，不能拿它的97.48代替当前PlainMHA H0。

| 历史方法 | dev_val RX Acc% | 已使用confirm Acc% | dev_val Macro-F1% | 参数 | 原始证据 |
|---|---:|---:|---:|---:|---|
| **H0：HStem+PlainMHA A1_mha** | **97.335** | **96.508** | **97.326** | **133,690** | S15汇总，S13模型，S23登记 |
| Shen-CIS non_adv，native | 93.12 | 95.20 | 92.89 | 13,442,954 | S01 cis_flat，非DANN |
| Shen-CIS non_adv，Strict-IB | 94.62 | 95.72 | 94.16 | 13,442,954 | S01 cis_ib_flat，非DANN |
| IB-RFF α=.004 IB unified | 94.94 | 92.21 | 94.90 | 310,481 | S06历史Shen适配 |
| IB-RFF α=.04 full official-recipe | 73.55 | 76.30 | 72.57 | 310,481 | S06历史8192适配；名称中的official指配方，并非官方LoRa完整复现 |
| FAN full official-recipe | 73.62 | 75.50 | 72.60 | 310,481 | S06历史8192适配 |
| WaveMLP scaled full unified | 81.02 | 78.67 | 79.50 | 594,442 | S06历史适配，非原论文LoRa结果 |

证据文件：[S15结果](../15-hstem-attribution/RESULTS.zh-CN.md)、[S01逐臂表](../01-strong-baselines/summary/confirm_tables.md)、[S06结论](../06-paper-validation/summary/paper/CONCLUSION.md)。它们与H0之间的差值不能当成S24新配对实验的结论。

## Tokenizer-only机制表

保留原S15/S21/S22/S23，不覆盖。S15中Full Patch+MHA/E1、Patch-Lite+MHA/C2、HStem+MHA/A1的dev_val分别为95.222/94.648/97.335%，参数571,434/134,266/133,690；这是统一五通道与后端的结构归因证据，不能冒充外部方法完整管线对照。

未来正式分析：每方法2折×3seed全报；主指标先逐RX算Acc/F1再等权。公开每RX、每seed、最差RX与配对差，不按包伪增样本量。成本应同机同精度、包括原生前处理；ASK另报复数自由度和QR成本，IB计两阶段训练总时间。训练中止保留日志与状态，不因目标分数重新选超参或收缩评价RX。
