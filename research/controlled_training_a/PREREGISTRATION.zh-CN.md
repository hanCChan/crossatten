# 阶段 A 正式预注册：固定数据的初始化 × 批顺序敏感性

登记日期：2026-11-11（Asia/Shanghai）。基于 `ebd7b20`。本文件及 `PRE_REGISTRATION.json`、实现和合成验收检查须在首次新拟合前提交；机器登记中的 SHA256 固定具体代码和输入。历史状态 `CONFOUNDS_VERIFIED_CAUSAL_ATTRIBUTION_UNRESOLVED` 保留，不改写旧审计、STOP 或模型。

## 研究问题和授权范围

在原 fold2 的同一 fit18、日期 1–2、同一 train/validation 分配和同一自适应训练策略下，选定的两组初始化及两种批顺序是否改变 HStem+普通 MHA 对同一输入的预测？这是有限水平敏感性实验，不能估计随机初始化总体方差、因素贡献百分比，不能解释 ASKNet 的 0/66/5 错误。

本阶段最多 5 个预定拟合尝试，固定顺序 A00、A00-replay、工程门槛、A10、A01、A11。前两次独立启动、独立构建初始化、独立训练；replay 计入预算，任何已开始训练的失败也计入成本。门槛失败即停下后续三次，不增加 seed 追逐历史坏例。工程修复须记录失败成本并先登记修订；不能静默重跑或追加拟合。阶段 B 的四次组成拟合不获本登记准入，也不随本脚本启动。所有训练完成后停止扩张这条归因线，不追加结构、Loss、种子扫描。

| 运行 | 初始化 seed | 批序 seed | 模型内随机流 seed |
|---|---:|---:|---:|
| A00 | 22101 | 22201 | 22301 |
| A00-replay | 22101 | 22201 | 22301 |
| A10 | 22102 | 22201 | 22301 |
| A01 | 22101 | 22202 | 22301 |
| A11 | 22102 | 22202 | 22301 |

## 输入权限与身份

`PRE_REGISTRATION.json` 的 `inputs` 登记唯一允许训练读取的 fold2 缓存绝对路径、文件 SHA256、manifest 和原索引；禁止调用可自动生成缓存或读取其他 fold 的历史 loader。源池 34668 包，原 train 24267 包，validation 10401 包，以登记后实际形状验收为准，不能重新划分。数据不足、哈希或角色不符直接失败。

规范样本键为 `RX / day / 历史源 cell SHA256 / cell 内确定性行号`；从同一池重建，保留原 train_ids / val_ids 的位置顺序，每轮置换作用于固定 train 序列。不能将不同 RX 的下标配对，也不把血缘键称为独立的物理发射事件 ID。输入按既有缓存和既有前端处理，不引入新预处理。

source-held6 固定为 `1-1、1-19、1-20、2-1、2-19、3-19`，仅在全部五个运行的 selected、epoch10、final 及完整逐轮证据封存后，统一一次探索性评价；重放工程门槛只用训练状态和 source-validation 输出。`held_inputs` 的文件哈希可在登记前用原 manifest / 原始字节校验，不能提前载入数组、推理或检查预测。无须重读历史预测或历史权重。

fold2 target8 `19-1、19-2、19-19、19-20、20-1、20-19、20-20、23-1` 及日期 3–4 不进入任何读取、训练、选模或评价。RX3-19/TX4 是反复观察的案例；全部 RX、TX、day 同时报告。未来算法必须在未参与方法选择的接收机上验证。

## 共同训练规则

使用原 `s17_models.build('ManyRx', initialization_seed)`，保持 HStem+两层普通 MHA、原分类头及 CE。Adam，lr=1e-3、weight_decay=0、batch=128、FP32、最多 40 epoch；不裁剪梯度，不引入增强或 Dropout。初始化单独保存可加载 state（包括 buffers）；加载后重置 CPU/CUDA/Python/NumPy 模型随机流为 22301。每轮顺序使用独立 CUDA generator，保存置换和规范键顺序摘要。

选模依据为 18 个训练 RX 等权 validation CE，严格 `<` 改善时保存 selected；ReduceLROnPlateau factor=.1、patience=5、min_lr=1e-6（其余原默认参数），连续 10 轮未改善停止。记录每轮实际使用 LR 和 scheduler 后的下一轮 LR。选中 epoch、更新步、LR 和早停可以是因素影响的中介；不声称固定了实际优化轨迹。

主要结果一律使用最终 selected checkpoint；epoch10 为固定辅助端点，二者完整同时呈现，禁止择优解释。final 用于工程复核。每轮保留模型/BN、optimizer、scheduler、CPU/CUDA/Python/NumPy RNG、顺序 generator 和选模状态，保存可加载 checkpoint 及规范语义摘要。记录时间、峰值显存、环境、参数数及全部失败尝试。

## 必须先通过的重放门槛

启用严格 deterministic algorithms（不使用 warn_only）、固定 cuDNN/CUBLAS 配置和同一软件硬件；本区块共同关闭 matmul/cuDNN TF32，并记录环境，这不是历史运行的逐位重建。A00 与 replay 必须在以下项目逐位一致：初始化参数和 BN buffers、train/validation 规范键、逐轮实际键序列和全样本覆盖、每轮模型及 BN、optimizer/scheduler/RNG/选模状态、所有数值训练记录（剔除时间）、selected/epoch10/final checkpoint 的张量语义内容和 validation 原始 logits。

checkpoint 压缩/序列化文件 SHA 用于各文件封存；跨运行比较以张量 dtype/shape/bytes 及递归非张量状态为准，避免将文件名等容器元信息误认成数值差异。失败须给最大绝对数值差、分类差异和缺失项目；不因 Accuracy 相近而通过。未通过禁止因素归因、剩余三次拟合和 B。通过后，同初始化水平的跨顺序运行必须初始 state 相同，同顺序水平的共有 epoch 必须规范键顺序相同；不满足同样停止解释。

## 预定分析

source-validation 与 source-held 分别分析；held 为探索性结果，validation 是已参与选模的数据。对各运行及两个端点报告 RX 等权 Accuracy、十类 Macro-F1（缺失类记零）、CE，以及每 RX、TX×RX、TX×RX×Day 的支持数、错误、真实类/最难负类与原始 logit margin 分布。保留逐包 raw logits 与规范键，报告 margin 的 1/5/25/50/75/95/99 百分位。

对 A00→A10、A00→A01、A01→A11、A10→A11、A00→A11 五对报告同输入预测分歧、纠正错误、新增错误、错误替换、配对最差组及支持数。报告包等权及 RX 等权分歧；总体 Accuracy 类似也不能掩盖局部分歧。replay 不作为第五个独立水平进入效应估计。

对预定标量指标 M：

- Δinit = ((M10 − M00) + (M11 − M01)) / 2。
- Δorder = ((M01 − M00) + (M11 − M10)) / 2。
- Δinteraction = M11 − M10 − M01 + M00。

不确定性使用同一组 RX 对所有运行共同重采样的配对 RX cluster bootstrap，10000 次、seed22401、percentile95% 区间；这只是固定已观察 RX 集合上的条件描述，不能推至独立 RX 总体或 seed 分布。held 仅 6 RX，区间可能很不稳定；同时给出所有逐 RX 差异，不以包数充当接收机样本量，不做事后单组显著性筛选。

事先约定用于后续研究取舍的工程量级：Accuracy/Macro-F1 配对绝对差 ≥1 个百分点，CE ≥0.05 nats；同输入预测分歧 ≥5% 为可观测的预测变化。它们是实用筛查线，不是功效证明或通用统计显著性定义。结合符号、配对区间及逐 RX 一致性解释；跨过阈值但区间宽或 RX 方向不一致则结论为不确定，不能从单 TX 坏例定义“大”。margin 无事后阈值。

## 独立 B 阶段与研究出口

B 保留为独立预注册：共同 Core6 的 8149 train 包、每臂 Extension6 的 7700 包，共 15849 train 包及共同 3551 validation 包。两组 extension 是特定设备集合的规则划分；设备编号是否对应硬件批次、位置或采集条件尚未由元信息核实，不能猜测。即使 B 有差异，首先只能解释为固定 Core、预算和随机流下替换这两个具体集合的效应。A 用 18 RX、B 用 12 RX，不能拿绝对性能相减来排名初始化和 RX 组成效应。

若稳定性差异明显，后续先评估 SWAD、EMA 等标准稳定化方法；若独立 B 的组成变化明显，优先源环境覆盖与受约束环境变化；二者都明显则先建立稳定性与覆盖的标准对照，再选一个机制；二者均不明显则保留未归因状态，回到信号处理、表征和 RX/TX 交互。B 未做时不为其填入“大/小”。本轮不实施这些后续方法。

HStem+Attention 保留为已有结构成果；S20 至当前工作为探索/归因依据，不计作新的算法创新。新的泛化机制与新接收机验证必须另行设计。本阶段结束后按预定规则报告，不因某个历史坏例未改善而追加实验。
