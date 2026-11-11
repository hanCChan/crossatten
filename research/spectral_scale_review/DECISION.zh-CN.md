# 研究决策：结束周期时域线，只保留一个频域尺度候选

**本轮完成只读证据复核、候选筛选及设计预注册。新增科学拟合0次，新增IQ、checkpoint和特征缓存读取均为0。** P2与固定窗口两个STOP原件保持不变，没有启动P0+P2融合或新的24次训练。

## 1. 当前P0/P2的现成互补性

按检查前登记的[12个主端点成员](REVIEW_SCOPE.json)，只解码原`SOURCE_PREDICTIONS.zip`中的P0/P2 selected预测；未读取epoch10/final40或历史ASK原始预测。逐行核对canonical key、pool_id、TX、RX、day及SHA。先在每个RX内平均三固定流，再对12台RX等权，独立复现：

| 已封存预测的量 | RX等权结果 |
|---|---:|
| P0 Accuracy | 85.0332% |
| P2 Accuracy | 28.9667% |
| P0错、P2对 | 1.9921% |
| P0对、P2错 | 58.0586% |
| 预测不一致 | 70.1570% |
| P0/P2至少一个argmax正确 | 87.0252% |
| P2恢复的P0错误比例 | 13.3099% |

完整逐RX/视图/流结果与等式检查见[EVIDENCE.json](EVIDENCE.json)。`P0 Accuracy + 纠错率 = oracle`与`P2−P0 = 纠错率−新增错误率`都在1e-12内一致。

87.0252%只约束逐包选择这两个**固定argmax标签**的理想选择器。它不约束重新训练的特征融合，甚至不约束所有固定logits融合：例如真类0，两个logits为(0.8,1,0)、(0.8,0,1)，两个argmax都错，但分数均值会选对0。本轮没有据此测试新的分数融合，也不把oracle当作F2的性能上限或设计目标。现成互补性有限，是不继续投入P0+P2侧路的决策依据，不能证明所有联合表征均无效。

## 2. 历史证据的范围与重复工作

| 已核对记录 | 支持的判断 | 本轮据此排除 |
|---|---|---|
| [S07](../../07-wifi-osu/RESULTS.md)：ManyRx单种子源初筛，frequency81.83%、time/frequency split71.75% | 简单时频双路有明确负面先例；不是本轮三流配对结果 | 再组合P0/P2做门控时频融合 |
| [S09](../../09-phase-invariance/RESULTS.md)：ManySig fixed first−null CRD +5.68pp，条件区间[2.41,9.12] | 同预算/配对下相对相位内容有贡献，不能转写成ManyRx收益或RX影响已消除 | 去斜率、二阶相位差分的改名重试 |
| S09源筛选：centered/curvature均未通过结转门 | 理想时延不变性不足以证明分类收益 | 继续搜索相位不变性强度 |
| [S19](../../19-tx-spectral-evidence/RESULTS.zh-CN.md)：静态band gain、power gate、TCSE及证据Loss已经比较 | 已研究频率选择与Teacher证据，B4未准入 | 新频带重要性Teacher、相同band gate或新辅助Loss |
| [现有多尺度代码](../../lib/legacy_rfhstu/multiscale.py) | 已有合并token后多encoder融合的通用实现；“多尺度”一词本身不是新贡献 | 把库内多尺度能力或普通Inception分支说成首创 |

S07/S09的正式CRD、S20旧source-held与当前C两视图source-CV分开呈现。没有用新目标评分筛结构，也没有把不同数据集、种子、角色和加权规则的百分比直接作配对差。

## 3. ASKNet真正提供的研究依据

只查阅官方固定提交`f00846d...`的[PatchNet.py](https://github.com/BeechburgPieStar/ASKNet-RFF/blob/f00846d68425315b68ae69b8c6b712e08559c2fe/WiSig/backbones/PatchNet.py)，缓存文件SHA与[S20上游清单](../../20-asknet-source-comparison/UPSTREAM.json)一致。本轮没有执行ASK代码或打开其权重。

| 维度 | HStem+MHA | 官方ASKNet路径 |
|---|---|---|
| 频谱操作 | 保留五通道FFT表示，局部频率卷积 | 复FFT、中心mask、127维复数unitary算子混合 |
| 截止/尺度 | 固定卷积tokenizer | 训练时全局selection_logits的Gumbel选择；推理用同一argmax |
| tokenizer坐标 | 32个频率token×64，名义卷积跨度32bins/hop8 | IFFT与能量归一化后，时间patch64/stride32，7token×128 |
| 后续映射 | 两层普通MHA、均值池化、分类 | token内通道MLP、均值池化、分类 |
| S20预算/选模 | 最多40epoch、RX等权CE | 最多200epoch、packet等权CE |
| S20参数 | 82,426实参数 | 100,240参数元素，116,369实标量自由度 |

ASK的cutoff是模型级参数，不是逐包或逐RX自适应选择。它在频谱处理后回到时域，因此旧整体对照并不支持“所有时域网络都失败”或“频率坐标单独导致优势”。不同的复数混合、截频、归一化、容量、预算、选模与shuffle仍混杂。原[S20结论](../../20-asknet-source-comparison/RESULTS.zh-CN.md)为INCONCLUSIVE_SOURCE_COMPARISON，不能只挑其可纠错样本归因为架构。

已归档的同输入[功能诊断](../classification_failure_diagnostics/README.zh-CN.md)显示，旧HStem/ASK source-held四折的RX等权标签oracle为89.7426%、95.9167%、96.8667%、97.7667%；四折分别有430/485/686/487个HStem错误被ASK纠正，同时新增341/368/583/375个错误。这里纠错数是包计数，oracle是RX等权，不能混为同一加权。fold1–3复用相同6台RX和输入；唯一RX平均净差+0.8324pp、条件区间跨零。没有把这些旧oracle与当前87.0252%作同协议排序。

**保留的启发是检查频域信息的组织与交互尺度。** 尚无证据把ASK的整体差异归因于多尺度，也没有证据说明某种平滑频谱主要属于RX。这只足以支持一个有普通卷积控制、允许失败的结构假设。

## 4. 候选取舍

| 想法 | 决定 | 原因 |
|---|---|---|
| P0+P2预测或时域特征融合 | 排除 | 当前固定预测互补较弱，S07已有时频双路负面先例 |
| 相位去斜率/二阶差分 | 排除 | S09源筛选STOP保持 |
| 新Teacher/band gain/频谱消除 | 排除 | 与S19或既有归一化假设重叠，不能把“平滑”等同RX |
| 复制ASK cutoff/Koopman | 不选 | 整体互补性不识别单个组件因果效应，会打开多个模块的搜索空间 |
| **冻结P0上的细/宽频率token残差F2** | **唯一设计候选** | 不替换已验证路径；可与严格同参数、同最大跨度的普通频率卷积直接比较 |

当前具体实现选择**冻结**原P0权重及BN、MHA和分类器，而非再一次同时改变主干。原stem之后提取细/宽两条频率邻域，形成32×64的残差token：`z=z0+alpha*G(h)`。alpha=0回到同一个冻结F0；只训练3,201个新增参数。F1普通卷积控制也训练3,201个参数，含骨干推断总参数都为85,627，最大侧路跨度都为50个频率bins，token中心相同。

F2的3/33两个kernel按原tokenizer几何事前固定，不根据错误RX、TX、频带热图或ASK cutoff选择。它不是动态SKNet，也没有显式预测RX或把某个尺度命名为TX/RX成分。能否提取增量判别信息仍未知，冻结主干也会限制能学到什么；本轮没有任何F2性能结果。

此设计与[Selective Kernel Networks](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Selective_Kernel_Networks_CVPR_2019_paper.html)、[InceptionTime](https://arxiv.org/abs/1909.04939)等已有多尺度网络直接相关，残差与零初始系数也非新概念。这里只冻结可证伪的工程候选，不认定原创性、物理机制或论文贡献。

## 5. 本轮交付与停止点

- [完整设计预注册](PREREGISTRATION.zh-CN.md)、[机器登记](REGISTRATION.json)和[未来作业表](PROPOSED_JOBS.json)。
- [原型](models.py)与[12项CPU合成验收](SYNTHETIC_ACCEPTANCE.json)：不含optimizer步骤或真实权重读取。
- [互补性复算](EVIDENCE.json)、[范围](REVIEW_SCOPE.json)、[访问记录](ACCESS_LOG.json)。
- [工程说明](ENGINEERING_NOTES.json)：冻结参数前后CPU可有约1e-7的舍入差异；保留冻结F0位级回退要求，并为未来封存P0设logit及argmax一致性验收。

未来方案是复用六个P0，只对F1/F2各作2视图×3流，共12次新增侧路拟合；冻结EMA E0作为既有强配方参照，既不新增EMA训练，也不把它叠到F2上。主比较F2−F1，同时要求对F0及E0的平均与局部风险联合门，不以弱基线加分称突破。

**执行仍关闭：本轮0次科学训练，没有新输入评估。** 完整受限训练器、CUDA验收、源模型/EMA可恢复性与执行附录尚待未来有明确训练授权的阶段完成。不重新打开P2、固定窗口、任何旧STOP或新RX确认；没有自动推进为训练。

设计复核[DESIGN_QA.json](DESIGN_QA.json)共241项通过：独立重算已派生RX均值和oracle等式、核对12个未来作业及严格参数计数、六个P0/六个EMA登记、原两个STOP的全部artifact摘要，以及71份代码/登记字节归档。该复核没有再次解码预测或读取IQ、权重。原始设计字节见[DESIGN_CODE_BYTES.zip](DESIGN_CODE_BYTES.zip)，不包含官方ASKNet实现或真实模型参数。
