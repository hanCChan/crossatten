# 原始文献、迁移条件与重叠项

核查日：2026-11-10。以下只引用原论文、会议页面、作者机构或作者代码。阅读深度分别列明；“方法核查”表示检查了相关方法段落，并非完整复现。下表的无线迁移判断属于本轮分析，不是原作者给出的 WiSig 结论。

## 跨领域候选

| 工作与原始来源 | 年份与核查深度 | 可借用的机制 | 迁移所缺的条件 / 本轮判断 |
|---|---|---|---|
| [RASTA Processing of Speech](https://www.ee.columbia.edu/~dpwe/papers/HermM94-rasta.pdf) | IEEE TSAP 1994；方法核查 | 对语音特征的时间轨迹滤波，以抑制慢变卷积环境等因素 | 需要任务信息与环境的时间尺度差异；不是单包频率平滑。静态 TX 谱形可能一起被消除 |
| [Parametric Cepstral Mean Normalization](https://machinelearning.apple.com/research/parametric-cepstral-mean-normalization-for-robust-speech-recognition) | ICASSP 2019；作者摘要 | 频率相关的运行帧均值权重；可联合训练 | 运行帧上下文和 ASR 任务不同于单包 TX 分类；完整实现细节仍需原文审计 |
| [FSDR](https://openaccess.thecvf.com/content/CVPR2021/html/Huang_FSDR_Frequency_Space_Domain_Randomization_for_Domain_Generalization_CVPR_2021_paper.html) | CVPR 2021；会议摘要，PDF 访问返回 403 | 依据学习得到的域变/域不变成分，在图像频率空间随机化 | 图像空间频率不能直接解释为 RF 载频偏移或无线谱形；本轮不宣称完成其频率选择实现复核 |
| [Domain Generalization with MixStyle](https://arxiv.org/abs/2104.02008)；[作者代码](https://github.com/KaiyangZhou/mixstyle-release/blob/master/reid/models/mixstyle.py) | ICLR 2021；论文摘要与代码核查 | 逐通道实例均值/标准差混合，模拟特征风格变化 | 必须验证无线特征的这些统计主要对应环境而非身份；具体实现差异见下文 |
| [Speaker Recognition from Raw Waveform with SincNet](https://arxiv.org/abs/1808.00158) | SLT 2018；原文摘要 | 用低/高截止参数约束 sinc 带通滤波器 | 是有效的结构先验候选，不提供 TX/RX 分离保证；不是 ASKNet 之外从未出现过的“可学习频带”概念 |
| [LEAF](https://arxiv.org/html/2101.08596v1) | ICLR 2021；方法核查 | Gabor 复滤波、模平方、平滑池化、可学习压缩/归一化 | 包络与降采样可能舍弃任务需要的相位/瞬态；可学习前端不等于已识别接收机因素 |
| [Group Equivariant CNNs](https://proceedings.mlr.press/v48/cohenc16.html) | ICML 2016；论文页面/摘要 | 把特定群作用编码到卷积与特征变换 | 必须定义无线输入上的实际群、边界条件及标签安全范围；图像旋转群不是 RX 响应模型 |
| [Invariant Scattering Convolution Networks](https://arxiv.org/abs/1203.1513) | TPAMI 2013，预印本 2012；原文摘要 | 小波、模运算与平均带来特定平移不变性及形变稳定性 | 稳定性依赖指定变换；可能损失相位身份信息。不要混淆 Mallat 2012 的另一篇 Group Invariant Scattering |
| [Deep Complex Networks](https://arxiv.org/abs/1705.09792) | ICLR 2018，预印本 2017；原文摘要及构件核查 | 复卷积、复归一化与初始化 | 复数结构本身不保证全局相位等变，更不保证任意 RX 变化不变 |
| [Distributionally Robust Neural Networks for Group Shifts](https://arxiv.org/abs/1911.08731) | ICLR 2020，预印本 2019；原文摘要 | 控制源组最坏风险，并强调正则化与泛化 | 源组覆盖、估计误差、类别支持须控制；用作训练对照，不作为新贡献 |
| [SWAD](https://proceedings.neurips.cc/paper_files/paper/2021/hash/bcb41ccdc4363c6848a1d760f26c28a0-Abstract.html) | NeurIPS 2021；会议摘要 | 密集、感知过拟合区间的权重平均 | 需要固定可用的验证/选模规则及同等预算；不能看 held RX 后调平均区间 |
| [Fishr](https://proceedings.mlr.press/v162/rame22a.html) | ICML 2022；会议摘要 | 跨域匹配损失梯度方差 | 是现成的训练控制；论文会议年是 2022，不能沿用预印本年份代替 |
| [In Search of Lost Domain Generalization / DomainBed](https://arxiv.org/abs/2007.01434) | ICLR 2021；原文摘要 | 统一实现、调参和模型选择下比较 DG 与 ERM | 强 ERM 是必要参照；该论文不证明所有 DG 在所有任务无效 |

## 更接近可辨识性的参照

| 原始来源 | 核查深度 | 对本任务的作用与边界 |
|---|---|---|
| [Front-End Factor Analysis for Speaker Verification](https://sls.csail.mit.edu/publications/2011/Dehak_IEEE_May2011.pdf)，2011 | 原文方法概念 | 说话人身份与会话变化的类比更接近 TX 身份任务；total variability 同时含身份和通道变化，后续仍需补偿，不能等同纯身份提取 |
| [Blind Deconvolution Using Convex Programming](https://arxiv.org/abs/1211.5608)，IEEE TIT 2014 / 预印本 2012 | 原文摘要 | 恢复必须依赖具体子空间与随机/非相干条件；不能凭“盲反卷积”名称认定无线硬件可分 |
| [Nonlinear ICA Using Auxiliary Variables and Generalized Contrastive Learning](https://proceedings.mlr.press/v89/hyvarinen19a.html)，AISTATS 2019 | 会议摘要 | 需要辅助变量带来充分条件变化等假设；标签不是自动满足定理的证据 |
| [Variational Autoencoders and Nonlinear ICA](https://proceedings.mlr.press/v108/khemakhem20a.html)，AISTATS 2020 | 会议摘要 | 条件生成模型的可辨识性有明确假设与等价类；为机制论证提供参照，不是即用即得的硬件解耦算法 |

## 直接相关 RFFI 工作

| 原始来源 | 版本、核查深度 | 重叠与后续比较要求 |
|---|---|---|
| [Towards Scalable and Channel-Robust RFFI for LoRa](https://arxiv.org/abs/2107.02867) | 2021 预印本；摘要与仓库既有 CIS 实现 | channel-independent 特征和增强已经存在；倒谱/谱比值需明确相对既有信道消除方法新增什么条件或能力 |
| [Towards Receiver-Agnostic and Collaborative RFFI](https://arxiv.org/abs/2207.02999) | 2022 预印本；原文摘要 | 接收机对抗不变表示、协作推断和微调均为既有路径；推断信息权限须与本任务一致 |
| [Domain Generalization for Cross-Receiver RFFI / RIEI](https://arxiv.org/abs/2411.03636v1) | 2024-11-06，摘要标明 accepted by IEEE IoT Journal；摘要核查 | 已提出 Separable Condition 和 TX/RX 特征解耦；后续理论贡献需逐条比较其充分条件，不能把“可辨识性”一词本身当成创新 |
| [Pan 等 v1](https://arxiv.org/abs/2510.09405v1)、[v2 / DRIFT](https://arxiv.org/html/2510.09405v2) | 2025-10-10 / 2026-05-26；v1 摘要、v2 方法与协议段落 | v1 摘要提出风格迁移和对抗训练。v2 的 TX/RX 分支、GRL、同 RX 中心正则和分离损失直接重叠不变/解耦思路。使用 WiSig ManySig、均衡后的信号和另一评价配方，不能与 S20 数值直接排名 |
| [Towards Channel-Robust and Receiver-Independent RFFI](https://arxiv.org/abs/2512.12070v1) | 2025-12-12；原文摘要 | 谱表示、对比预训练、Siamese 与增强联合处理信道/RX；完整方法、是否需要微调和实验权限尚未逐条复核 |

这份表不复述 IB-RFF/ASKNet 的全部方法；它们仍是本仓库已有强参照。ASKNet 的锁定版本、输入和官方配方审计见 [S20 UPSTREAM](../../20-asknet-source-comparison/UPSTREAM.json) 与[正式结果](../../20-asknet-source-comparison/RESULTS.zh-CN.md)。上述新发现文献没有在本次复现或引入到实验。

## 本地 MixStyle 不是原版的严格复现

作者 `reid/models/mixstyle.py` 的 Git blob 为 `7eb151bf06a181ed4bc6f54f562bffc49f35deaf`。其输入为 `B,C,H,W`，对空间轴 `[2,3]` 求每通道统计，detach 均值/标准差，默认以 `p=0.5` 应用。

本地 [mixstyle.py](../../lib/legacy_rfhstu/mixstyle.py) 在 token `B,T,C` 的轴 `(1,2)` 上求统计，得到每包一个全局均值/标准差；不 detach，也没有概率跳过开关。[DeviceClassifier](../../lib/legacy_rfhstu/models.py) 默认关闭该模块，S19/S20 B0 的 ExploreRF 路径不调用它。

本轮检查了默认调用路径和 configs/既有报告中的相关引用，未确认一个足以代表原版 MixStyle 的正式启用实验。因此只能说仓库有同名变体，不能写“原版 MixStyle 已在当前协议上失败”。本次保留代码原样，没有为了对照修改历史实现。

## 检索记录与覆盖限制

检索围绕用户列出的原始题名，并补充下列问题组：

| 问题组 | 实际使用的代表查询 | 筛选方法 |
|---|---|---|
| 语音归一化 | `RASTA processing speech 1994`；`parametric cepstral mean normalization 2019` | 优先作者原文与机构页面，区分时间/频率轴 |
| 结构先验 | `LEAF learnable frontend 2021`；`SincNet 2018`；`group equivariant CNN 2016`；`scattering 2013` | 区分原任务、群作用与可证明性质 |
| 鲁棒学习 | `GroupDRO SWAD Fishr DomainBed` | 校正会议年份，检查其声称范围 |
| 可辨识性 | `blind deconvolution identifiability`；`speaker factor analysis`；`nonlinear ICA auxiliary variables` | 只把有条件定理用于说明条件，不外推硬件已分离 |
| 近年 RFFI | `radio frequency fingerprinting receiver independent cepstral normalization 2024 2025 2026`；`radio frequency fingerprinting MixStyle domain randomization receiver` | 原论文核实重叠；搜索结果用于发现而非证明 |
| 近年 RFFI 补查 | `radio frequency fingerprinting receiver independent disentanglement 2025 2026`；`radio frequency fingerprinting equivariant complex neural networks phase invariance receiver` | 优先 arXiv、IEEE、作者页；排除与 RF 指纹身份无关的量子接收/定位结果 |

本记录是问题导向的筛查，不是逐数据库、逐检索式穷尽的系统综述。FSDR PDF 与部分 OpenReview 页面访问受限；未读取的全文或代码不能标记为方法复核。没有检索命中不等于不存在；没有查清同义词、专利、非公开工作与全部引用链。若候选机制成形，下一轮应针对其精确数学操作与 RIEI/DRIFT/CIS/ASKNet 逐项作差异表，再决定是否具备论文贡献。
