# 跨领域机制审计：先确定哪些变化可以消除

日期：2026-11-10。基线提交：`9815503880291f4b66c3566b07769f6a5fc8c9a6`（S20）。

状态：**REVIEW_COMPLETE_NOT_EMPIRICAL_GO**。本次完成文献、数学与既有证据审查，以及合成反例复现；没有执行新的数据实验或训练。A/B/C 仍是竞争假设，没有选出胜出算法，也没有撤销 S11–S20 的任何 STOP。

## 研究判断

下一步最值得解决的是：**在现有采集与推断条件下，能否找到一个跨 TX 可预测、与 TX 判别信息不过度重合的接收环境变化族？** 若不能，继续设计归一化、生成增强或等变层，仍会回到“模块先行”的循环。

跨领域方法提供了有用的假设，但原方案中的三个前提需要收紧：

1. **语音时间统计与单包频率平滑不是同一操作。** RASTA 处理特征随时间帧的轨迹；PCMN 学习对运行帧均值的频率相关权重。它们没有证明单包频谱的平滑部分主要来自 RX。[RASTA 原文](https://www.ee.columbia.edu/~dpwe/papers/HermM94-rasta.pdf)、[PCMN 作者说明](https://machinelearning.apple.com/research/parametric-cepstral-mean-normalization-for-robust-speech-recognition)。
2. **传播信道一般也依赖 TX。** 更合适的一阶模型是 `Y_t,r,d = X_t C_t,r,d G_r,d + N`。WiSig 是逐 TX 发送、多 RX 在同一采集时段接收；官方还说明 RX 不同步、单元支持不均衡。不能据此把不同 RX 数组中的相同下标视为同一包。[WiSig 官方采集说明](https://cores.ee.ucla.edu/downloads/datasets/wisig/)。
3. **不变性需要保护可分性。** 若允许的环境变换能把一种 TX 信号变成另一种 TX 信号，对这组变换完全不变的表示就会合并两者。另一方面，绝对硬件/信道分解存在尺度自由度，并不排除相对 TX 信息可辨识。详见[数学边界](IDENTIFIABILITY.zh-CN.md)。

更接近 RFFI 的语音类比是“说话人身份与录音会话因素分离”，而非只保留语言内容的 ASR。i-vector 等因子模型可帮助设计重复身份/环境的统计模型；它们本身也不保证得到纯身份变量。[Front-End Factor Analysis for Speaker Verification](https://sls.csail.mit.edu/publications/2011/Dehak_IEEE_May2011.pdf)。

## 与已有实验的连接

| 既有证据 | 本轮必须保留的含义 | 不支持的外推 |
|---|---|---|
| [S11](../../11-spectral-generalization/RESULTS.md)、[S12](../../12-spectral-validation/RESULTS.md)：源频谱统计与先验筛选 | 已做过“源 RX 统计指导频率处理”；散度受幅度缩放影响 | 换成倒谱或平滑算子就自动成为新机制 |
| [S17 D0 修正解释](../../17-rx-distortion/D0_INTERPRETATION.zh-CN.md)：幅频稳定 2/4、相位 3/4、IQ 0/4 | 存在有限且混杂的 RX 相关证据；跨 TX 共享幅频假设并未稳定通过 | 所有 RX 效应都不存在，或已经识别出 RX 硬件响应 |
| [S17B](../../17b-phase-intervention/INTERPRETATION.zh-CN.md)：平滑相位 FIR 保持分类，却未通过特异覆盖门槛 | TX 保持与真实变化覆盖必须分别证明 | 增强物理上合理，所以代表真实 RX 变化 |
| [S18](../../18-selective-rx/RESULTS.zh-CN.md)：选择性 RX 子空间干预未过门槛 | 几何方向和实际分类收益不同；软残差可逆不等于信息消除 | RX 探针下降足以证明纯 TX 表示 |
| [S19](../../19-tx-spectral-evidence/RESULTS.zh-CN.md)：证据辅助 B4−B0 −0.6046 pp、B4−B3 −0.6741 pp | TX 重要性标签未产生独立收益 | “重要频带”就是 RX 稳定频带 |
| [S20](../../20-asknet-source-comparison/RESULTS.zh-CN.md)：ASK−B0 +0.8773 pp、4/4 正向，最差配对 RX −8.05 pp | 整体模型/配方的平均改善与局部风险同时存在 | 优势由 Koopman、频带或某个算子单独导致；两模型等价；协议无效 |

S20 的 12 个唯一 RX 条件区间为 [−0.7681,+2.6029] pp；fold1–3 重复同 6 个 held-source RX，不能当成 24 个独立接收机。当前已查看的 held-source 结果只能用于探索；由这些结果产生的下一假设不能再把同一结果称为独立确认。真实 target、日期 3–4、ManySig/Shen 盲测没有在本次打开。

## 三个方向的优先级

| 假设 | 当前可取之处 | 首个必须解决的问题 | 当前决定 |
|---|---|---|---|
| A：受约束的接收环境归一化 | 可以把响应族、估计误差和 TX 损失联系起来 | 跨 TX 可预测的环境分量是否存在；是否与平滑 TX 谱形重合 | 优先做可辨识性与可迁移性论证，暂不实现归一化层 |
| B：接收环境生成增强 | 可直接测试“生成变化是否对应真实源环境变化” | 生成族的来源、覆盖及 TX 信息保护；与 S17B 的实质差异 | 条件保留；复用 S17B 反证要求，不能仅扩大随机 FIR |
| C：物理结构约束 CNN | 对已知作用群可以给出可检验的等变关系 | 哪个变换是标签安全的；变换的物理适用范围 | 只对明确变换讨论；复杂值卷积或学习滤波器本身不构成机制证据 |

公平对照应包含强 ERM，并按科学问题选择 GroupDRO、SWAD 或 Fishr；无需把三者全部堆入模型。GroupDRO 优化源组风险，不能自动保证任意新硬件风险；SWAD/Fishr 的原始证据也不等于已验证 WiSig。公平调参、选模和算力预算必须先固定。[GroupDRO](https://arxiv.org/abs/1911.08731)、[SWAD](https://proceedings.neurips.cc/paper_files/paper/2021/hash/bcb41ccdc4363c6848a1d760f26c28a0-Abstract.html)、[Fishr](https://proceedings.mlr.press/v162/rame22a.html)、[DomainBed](https://arxiv.org/abs/2007.01434)。

## 查新中的直接重叠

“跨接收机不变表示”“TX/RX 解耦”“风格增强”“信道独立谱”均已有直接相关研究，不能单独作为创新点：

- Shen 等已研究 receiver-agnostic 对抗训练及协作推断。[2022 预印本](https://arxiv.org/abs/2207.02999)。
- RIEI 是已有接收机无关特征解耦工作，需列入后续逐项比较。[原论文](https://arxiv.org/abs/2411.03636)。
- Pan 等 `2510.09405` 的 **v1（2025-10-10）摘要明确提出 style transfer + adversarial training**；**v2（2026-05-26）改为 DRIFT 的对抗对齐、RX 正则与特征分离**。不能混用版本、作者表或把 v1 的描述当成 v2 的实现细节。本轮未复现其结果。[v1](https://arxiv.org/abs/2510.09405v1)、[v2 方法正文](https://arxiv.org/html/2510.09405v2)。
- Ma 等研究谱表示、对比预训练、Siamese 分类及增强的信道/RX 联合鲁棒性；本轮只完成摘要级筛查，不能据此认定与候选方法完全相同。[2025 原论文](https://arxiv.org/abs/2512.12070v1)。

当前只做定向首轮筛查，**没有完成全面查新，也没有证明 A/B/C 中任何方向具有原创性或超过 ASKNet**。逐文献的假设、可借用部分、阅读深度和迁移缺口见[文献表](LITERATURE.zh-CN.md)。

## 可直接使用的下一步材料

- [最小证伪设计](HYPOTHESIS_TESTS.zh-CN.md)：先准备性审计，再分别检验 A/B/C；包含失败出口和需要预注册的项目。
- [数学边界](IDENTIFIABILITY.zh-CN.md)：共同响应、轨道相交、时间/频率归一化、源组风险的适用条件。
- [可复现合成检查](synthetic_checks.py)及[输出](SYNTHETIC_CHECKS.json)：验证 6 项数学/工程陈述。它们不读取 IQ、权重或历史预测，也不估计 WiSig 性能。
- [示意图](SYNTHETIC_COUNTEREXAMPLES.png)、[访问与范围记录](REVIEW_SCOPE.json)、[文件清单](MANIFEST.json)。

本轮给出的可执行研究入口是“定义并证伪一个受限制的接收环境变化族”，而非新的网络名称。只有在真实源变化对应性和功能性 TX 保护均有独立证据后，才进入算法训练与相对强基线的收益检验。

复现时在本目录运行 `python synthetic_checks.py`，随后运行 `python archive_review.py --verify`。本次使用 NumPy 1.26.4、Matplotlib 3.10.0；合成检查的 6 项断言全部通过。哈希审计还检查文档本地链接及引用的历史文件与 S20 基线一致。这些边界推导是研究约束与反例，不是新的可辨识性定理或论文贡献。
