# S24 外部基线训练前审稿与执行方案（2026-11-16）

> 状态：REVIEWED_PROPOSAL_NOT_FROZEN；基于 exp/s24-lora-external-audit@0bb963c。本文是研究审阅建议，不修改既有 P-H5x160 协议、H0 权重或历史证据。本文件建立不代表已经训练、完成 GPU 资源验证或获得新的独立测试资格。

## A. 当前最重要的判断

- 外部主线三类方法有代表性：Shen TMC2024 CIS+RA/DANN；ASKNet **LoRa 原生完整**；IB-RFF **Shen8192 source-only适配**。不要以无法核验的近似版本冒充作者方法。
- 双主表保留：**Native Full**（同原始包、同源/开发RX角色，保留每法原生处理）与 **Strict-IB Full**（先对同一原始包投影±62.5kHz，其余完整网络保留）。Strict-IB 明确是修改过的信息预算对照，不是官方原生复现。
- HStem 本身 S15/S21/S22/S23 已有 Full Patch、Patch-Lite、HStem、Smooth-RoundTrip、Overlap-Patch、NoMidBN、AffineOnly、MHA/HSTU 对照；不重复 36 历史模型拟合，不把 BN/重叠/平滑统计联系宣称为已严格隔离的独立因果。
- 若审稿确有需要，可额外**预注册**“2级/3级/4级 HStem，同输出32×64、相同共享前端与PlainMHA、相近参数”的补充结构实验；深度同时改变核与BN等，不称其纯层数因果；首轮不启动。
- DANN 若用于解释“去RX机制”的优劣，建议新加入同CIS前端、同优化训练条件的 **non-adv CE** 配对控制（2 view×2 fold×3seed=12 fits）；如果仅用DANN作外部完整方法，CE控制可以放到补充表，但不能用历史非配对S01数值说明DANN增益。
- IB-RFF 的每个 stage1 α=0 checkpoint应**顺带作为 FAN CE 对照**，不增加拟合；stage2严格从对应fold/seed/view自己的stage1初始化，不偷用发布权重。
- 原草案 2视图×2折×3种子×(DANN1 + ASK1 + IB2)=48 fits、36下游最终checkpoint是**上限草案，不是承诺全部可执行**。若增加 non-adv CE，单独登记+12 fits；不可悄悄把总数仍写48。

## B. 阻断条件（所有项应在对应方法首次optimizer step前通过）

### B1 DANN：跨框架参数/行为等价

1. 固定作者 commit 8910103；根据已审计官方模型构建 TensorFlow/Keras 基准与 PyTorch 候选端口，映射卷积 HWIO→OIHW、Dense、channels-last Flatten、TF SAME padding、AvgPool、L2 normalization、GRL、TX/RX两头，并记录各层权重映射和hash。
2. 用已允许**源域**原始包生成固定无增强CIS，也可加合成输入；将同一组明确确定的参数从TF转移到Torch，在 eval 模式比较每层特征、TX logits/softmax、RX logits/softmax。两边cross-entropy/梯度分别核对（TF softmax输出对应Torch logits），包括GRL对共享encoder反号及对RX头不反号。事先注册适当的浮点绝对/相对误差容差，失败不能绕过。
3. 官方验证rx标签切片使用了前段索引，必须按同一包 train/val 行号同时切 data、tx_label、rx_label；记录源代码错误与source-only修复。
4. 提前固定 160/40 split、训练随机采样/有无放回、AWGN SNR分布、源验证是否加噪及固定噪声流、early stopping、保存best还是停止时权重、源验证指标组合（TX CE、TX+RX总损失等）和二头loss系数。不能先看dev再调。
5. 端口中的shen_cis已内部做RMS归一化；保持 source-only 入口对齐，不要重复增加不必要的信号预处理；如引入输入投影，核对投影/加噪次序。

### B2 ASKNet：默认完整 LoRa 算法必须先验证计算可行性

1. 使用官方 Lora ResCISNet 完整 k=[128,256,512,1024,2048,4096]、8191模态complex Koopman与每次forward QR，保留原CIS/ResNet、tau计划和训练/评估 hard-select差异。
2. 先在拟使用的真实GPU（优先高显存机器）做**正式训练以外**的逐级内存审计：模型构造、单样本完整forward/backward、batch1/小batch优化器状态与一次技术性step（仅沙盒随机/合法源包、不得视作科学fit）、峰值GPU内存、QR耗时、NaN/数值稳定性，最后根据记录冻结最大可用batch与gradient accumulation策略。
3. RTX4060 8GB上仅7模态梯度通过**不满足**原生完整模型验证；下界一块8191² complex64矩阵的权重+梯度+Adam双状态约2.147GB，另有QR工作空间/计算图，复杂度O(n³)。若不可行，标记原生ASKNet「RESOURCE_INFEASIBLE」；可以另登记资源受限变体，但名字/表格/参数/成本必须与原生完全分开，不能缩小k后称官方方法。
4. 不以调整本方法的batch、precision、频带或训练样本上限制造对HStem有利的比较；确认技术差异后冻结可比较的预算标签。同机重测H0，不能拿4060的H0与A100的ASK直接比较延迟。

### B3 IB-RFF：正式标为 Shen8192 adapted

1. 公开入口为WiSig 256点，缺官方LoRa完整config。适配应保持完整8192IQ，不截256、不用HStem五视图替代原FAN；显式冻结 SAF长度、patch、stride、位置编码尺寸、FAF、DSCNN、分类头与两阶段HSIC核。
2. 冻结stage1 CE和stage2 CE+α normalized HSIC-CCA的精确损失、初始化、数据管线、epoch、scheduler、选模规则及HSIC计算位置/成本。来源区分「作者WiSig默认」「S06历史Shen适配」「本轮自主新适配」。
3. α绝不能由当前Dev/Confirm成绩或历史看过Dev之后最优的结果暗中挑选。若要选择α，必须**预注册**源RX内部预算及其合法数据角色并另统计fits；若只固定一个α，记录其先验理由和可能的历史选择偏差。
4. stage1的α=0结果复用为内部FAN参照。stage2不得借用与本fold/source/view不一致的训练权重；在source-only微调和选择中严禁使用dev/confirm/blind标签。

## C. 精简的实施顺序（从易到难，避免一次开48次）

0. **只读再验证**：UPSTREAM 5仓29文件hash、PACKETS 20000个身份、两fold完全隔离、原协议及历史H0六权重hash、原CIS/ASK前处理、访问日志中confirm/blind=0；源码和任何训练配置变更均产生新的git commit和登记hash。
1. **方法级Gate**：优先完成DANN TF-Torch parity与正确RX切片；同时完成IB-RFF adapter和原始alpha来源核查；ASKNet完整默认资源审计单独进行。每方法有独立pass/fail，不能因一法失败取消其余可行基线。
2. **源域技术Pilot**：每个通过Gate的方法，先native fold0 seed201完成1个完整fit（IB必须stage1+stage2），只看source-train/source-val与技术指标，不加载held/dev，不以得分决定新超参。所有失败写清是代码/资源还是科学结果。
3. **冻结预注册**：输出 registration.json（允许拟合名单、视图、折、种子、固定参数、checkpoint selector、GPU与mixed precision、随机流、最大steps/epochs、early stopping、异常恢复、成本口径、STOP条件）。无正式pass时最大授权scientific_fits=0。
4. **正式配对矩阵**：仅对各自Gate通过且完成预注册的方法运行native和strict_ib：2fold×3seed。按method/view/fold/seed单独目录与独立初始化。过程中不访问Dev；结束后hash检查点+sealed manifest，一次性解锁Dev评估。Blind永久不打开，Confirm仅历史引用。
5. **报告**：独立两张主表、源域训练日志和逐RX/逐seed数据；按RX身份而不是包作为推断单位，给出paired差值与条件cluster bootstrap CI、Macro-F1、最差RX、参数及真实自由度、MAC/FFT/QR、同机同precision推断前处理p50/p95、batch吞吐、训练总GPU时长。RX在不同fold切换角色的依赖需披露，不把2×3当6个独立RX重复。
6. **结构消融选项**：仅在外部主表完成且出现明确论文审稿问题时，独立注册追加分层级数等价预算结构对照；不得启动新S24小模块搜索、抹去S21–S23的STOP结论。

## D. 验收和禁止事项

- DATA: 所有方法由同一PACKETS.csv确定训练身份；跨方法只可共享合法原始包与角色，不共享不合格权重/目标RX。
- SECURITY: no confirm/blind reads; Dev只在全部对应方法模型和配置封存之后读，Dev只报告不选模。
- PARITY: 模块级数值等价的详细误差与误差阈值必须事先注册；原版和修订版的算法身份分开。
- SCIENCE: 不把Native/Strict-IB两个不同信息预算统计混为同协议SOTA；历史Confirm不是新blind；ASK QR若失败应展示资源约束，不填假准确率；IB适配非官方LoRa。
- ABLATION: S15/S21–S23原六模型与机制否决保持原记录；独立层级因果证据不足时不要夸大。
- ARTIFACTS: 保留运行配置sha、代码版本、模型hash、包身份、失败日志、随机种子、逐RX预测、运行时峰值内存与运行时长。所有未训练/未运行维持NA而非0。

## E. 给本地 Codex/Claude 的执行授权范围

用户当前明确希望推进S24并开始训练，但当前仓库没有S24训练启动器，也未接入本地D盘与GPU。本文件**并未实际启动训练**。本地Agent可依以上方案先实现缺失端口/runner、执行无科学fit的Gate与技术Pilot，**只有在每个方法的Gate全部通过、registry冻结后才启动该方法正式fit**。若任何P0阻断无法解决，则停止该方法并报告，不得降级网络还冒充完整官方复现；可以推进其他已满足条件的方法。所有新增拟合以源域、源验证为限，未封存不得评价Dev。阶段上限与non-adv扩展应在最终冻结登记中准确计算。
