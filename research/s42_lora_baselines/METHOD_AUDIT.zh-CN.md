# Shen 近期基线适配核验

| 方法 | 本地可核对材料 | P-H5×160 处置 |
|---|---|---|
| Ma Spec+Siamese | 原文 §IV/VI/VII、Fig. 5、Table II，已核对无外部预训练的主结果 | 首批六拟合。相同低源 RX 和包预算。论文缺失的谱图和通道细节已在运行前登记为实现选择。 |
| Triad-GAN | ICC 论文 §III–IV、作者博士论文 §4.4–4.5，均用相同十 TX、二十 SDR 的公开数据 | 确认条件 G/D、四残差块编码器、仅编码器与分类头推理、三损失系数 2.0/0.5/0.5。公开材料中未能核对尺寸、逐步参数更新与编码器对判别器损失的梯度走向、LoRa 谱图前端和优化预算。精确实现资格待补。 |
| ASKNet LoRa | 作者 Lora/run_CIS.py 与数据加载器已锁定，当前机器有独立 receiver-drift 文件的路径。历史 S24 权重、轨迹与完整 Native 指标可核对 | 官方 N210-1 Day 1 → RTL-6 D2–D4 与 P-H5×160 针对不同问题。官方默认 batch 128、200 轮上限、patience 20。现有 S24 完整 LoRa 版已有六拟合，保留优化散度和成本。 |
| Receiver-Agnostic FL | 发表于 IEEE TMLCN，包含本地对抗学习、联邦聚合和 client selection | 与集中学习的参与方和数据协议不同，用于相关工作。 |

[Ma 原文](https://arxiv.org/html/2512.12070v1) · [Triad-GAN 作者版](https://livrepository.liverpool.ac.uk/3196889/) · [Wang 博士论文](https://hdl.handle.net/20.500.14242/359791) · [ASKNet 官方 LoRa 指令](https://github.com/BeechburgPieStar/ASKNet-RFF#lora-cis--asknet-refinement) · [Receiver-Agnostic FL](https://ieeexplore.ieee.org/document/11373634/)

## 14→6 与当前主表

本地 OOB 前稿的 Table VI 报告 Shen-CIS **99.24%**、Shen-RA **99.14%**，按六台原 blind RX 的五种子均值汇总。原稿训练跨 14 台开发 RX，HStem 的冻结 P-H5×160 模型则从每折五台 RX 的每 TX/RX 160 包拟合。资料核对范围内，HStem 没有可验证的同 14→6 队列。当前稿件只将旧数值注明为历史任务证据。没有读取旧 blind、Confirm 或 receiver-drift 波形。

## ASKNet 的解释

S24 Native source-val 六拟合准确率为 90.65/40.05/77.35/90.55/91.25/11.50%。其 65.092% 留出均值同时计入源训练弱的轨迹，官方 LoRa 入口含 CIS+ASK。默认 8191 模态 QR 每次前向重新计算，同一机器记录的 batch 128 IQ 到 logits p50 为 2.260 s。优先顺序据此为 P-H5×160 的近期竞争对照，官方协议验证在单独档案中处理。

## Triad-GAN 原协议未见 RX 复算

作者图 5 有 20 台 RX 的柱状图，其中含五台训练 RX。使用[抽取程序](triad_fig5.py)从作者接受稿 PDF 的矢量柱高和 40/60/80/100% 刻度恢复数值，按图中的 RX 排序核对，移除原文 §IV-A 列出的五台源 RX。每个柱值以约 0.1 pp 精度重建。每组只在 15 台未见 RX 等权聚合。[逐 RX 近似值](evidence/TRIAD_FIG5.csv) [汇总及 PDF 哈希](evidence/TRIAD_FIG5.json)

| 训练 RX 方案 | 作者参照 | Triad-GAN | 平均差 |
|---|---:|---:|---:|
| 同构（RTL 1–5） | 70.03% | 84.43% | +14.40 pp |
| 异构（五种接收机） | 90.01% | 95.76% | +5.75 pp |

两种原论文的训练集合、剩余 15 台 RX 与当前 P-H5×160 两折不同。这一复算用于确认方法的对照价值。其同折同预算表现待获得可复现训练实现后测量。
