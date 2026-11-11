# 阶段 C：标准方法竞争的正式预注册

2026-11-11，Asia/Shanghai；基于 `140d2bc`。A/B 归因训练正式结束，不追加初始化/批序/组成实验、BN归因线或新 Loss。当前任务是冻结 C0/C1/C2 的方法、数据角色、有限预算、选型规则和确认边界；**本次不启动训练**。

状态：`C_SOURCE_METHOD_COMPETITION_PREREGISTERED_CONFIRMATION_GATE_CLOSED`。本登记不是算法有效性结果，也不是新机制准入。未来执行前必须把完整训练器、评价器、日志与成本测量代码登记为工程附录，保持本文方法和分析规则不变，通过工程验收并提交后才能开始科学拟合；不存在可绕过该要求的训练入口。

## 研究问题与三个竞争臂

问题是：在固定 RF 输入、HStem+普通 MHA、合法源数据及公平开发预算下，标准单轨迹稳定化或统一 RX 风险控制，能否减少未见于当前拟合的源 RX 上的平均与局部 TX 判别风险？不继续寻找 A/B 因素的重要性排名，不声称已经识别 RX 硬件机理。

| 臂 | 固定方法 | 要回答的问题 |
|---|---|---|
| C0 | 共同配方 ERM，原 RX 等权验证 CE 选 checkpoint | 建立有正则化候选、严格配对的标准基线 |
| C1 | C0 单轨迹 EMA，decay=.999，fit-only BN重估后按同一验证 CE 选模 | 标准参数平滑完整配方是否改善判别可靠性 |
| C2 | 既有 RX-group GroupDRO，dual step=.01 | 统一 RX 条件风险优化是否足够 |

三个臂不改变输入、骨干、Attention、类别数或推断接口；RX只供C2训练分组，推断只输入IQ。C2本轮明确按RX分组，**不同时加入TX×RX、类别对风险、CVaR、GRL或额外Margin目标**。类别对风险公式属于研究假设，不作为本轮新算法。GroupDRO是迁移到共同骨干/配方的既有目标，不是完整复现原论文的所有网络和优化器。

## 合法源域内选型，独立确认保持关闭

当前只使用 fold2 fit18、日期1–2的已登记源缓存。两组 source-CV 视图由 `SHA256('C-source-CV-v1|'+RX)` 排序的前6/随后6台分别作内层留出；其余12台作该视图训练RX。顺序规则在标签/IQ结果之前固定，不按历史错误挑选RX。具体列表、成员dtype/哈希、支持数见 [SOURCE_PLAN.json](SOURCE_PLAN.json)。

每个视图分三种角色，所有方法完全一致：

1. **fit_train**：原 train_ids 中属于该视图12台训练RX的行，保留原位置顺序；只这些行参与梯度和EMA BN重估。
2. **checkpoint_val**：原 val_ids 中属于同12台RX的行；仅按RX等权CE选checkpoint及驱动plateau，保持原包级train/val身份。
3. **source_selection**：该视图6台内层留出RX在源池中的全部日期1–2行。它们不进入该视图拟合、BN或逐epoch选模；所有候选checkpoint冻结后统一评分，用于本轮正则化/方法选型。

两组内层留出集合互不相交，共12台不同合法源RX；同一物理RX可在另一个视图作训练RX，这是源域内部交叉验证，**不是新的独立确认**。本轮不得把不同视图间的性能变化再当成新的RX组成归因实验。

历史 held6 `1-1、1-19、1-20、2-1、2-19、3-19` 本轮不打开；target8 `19-1、19-2、19-19、19-20、20-1、20-19、20-20、23-1`、日期3–4及ManySig/其他数据本轮均无访问权限。不为选型查看旧 held logits 或训练权重；A/B旧模型不能替代C0。

历史 [S10协议](../../10-tifs-validation/PROTOCOL_AND_LEAKAGE_AUDIT.md)及 [receiver汇总](../../10-tifs-validation/final_receiver_metrics.csv)已覆盖当前ManyRx的32台RX。不能把当前target8、另一个历史fold或未看过的新seed称为新接收机盲测。[CONFIRMATION_GATE.json](CONFIRMATION_GATE.json) 因而明确为关闭：当前未登记与方法选择隔离的新RX集合、兼容TX标签映射和最终样本量。

真正独立确认需单独登记新/可证明未参与方法选择的RX、采集条件、TX映射、输入哈希、统计方案和冻结模型；不得使用其标签、IQ或无标签统计调参/BN适配。当前预算不包括全fit18重拟合、历史held描述性复核或独立目标评分，均不得自动启动。

## 固定配方、公平搜索与预算

共同：ManyRx10类、原5通道前端、HStem+两层PlainMHA、原分类头、FP32、batch128、Adam lr1e-3；无增强/Dropout/梯度裁剪。每个视图每个候选严格训练 **40 epoch**，所有梯度族的实际更新数均为 `40*ceil(N_train/128)`。这次统一关闭early stopping以固定实际更新预算；保留原“源RX等权validation CE严格最小、同值选最早epoch”的checkpoint规则。不能将这一共同变化说成与历史A/B完整配方相同。

共同沿用A的确定性环境规则：deterministic algorithms=True、warn_only=False、CUBLAS_WORKSPACE_CONFIG=:4096:8、cuDNN deterministic=True/benchmark=False、matmul和cuDNN TF32=False、CPU threads=2。执行附录须核对本机PyTorch/CUDA/cuDNN/GPU及实际包文件摘要，禁止静默升级或只记录numeric seed而不保留初始参数、BN、优化器/调度器/RNG及逐轮实际键序列。

共同使用 ReduceLROnPlateau factor=.1、patience5、min_lr1e-6、其他PyTorch2.5原默认值。C0的raw验证CE驱动其共享轨迹（含C1）的LR；C2也以同一普通RX等权验证CE驱动自身LR，不用GroupDRO验证加权值。实际LR可因方法而异，记录为共同自适应策略下的中介，不声称固定了实际LR路径。

正则化网格只含 **Adam coupled weight_decay={0, .001}**，三个臂拥有完全相同两个候选，不按某个方法的历史held效果扩展。GroupDRO原研究指出正则化会影响最差组泛化，故不只用wd0作弱对照；此网格是有限标准竞争，不宣称已经找到最优ERM或最优GroupDRO。[原论文](https://arxiv.org/abs/1911.08731v2)

| 配对流 | 初始化 seed | 顺序 seed | 模型随机流 seed |
|---|---:|---:|---:|
| S0 | 24101 | 24201 | 24301 |
| S1 | 24102 | 24202 | 24301 |
| S2 | 24103 | 24203 | 24301 |

每个视图/流/weight_decay 只有两个梯度轨迹族：**ERM_EMA**一次ERM拟合同时保存C0和C1；**RX_GDRO**保存C2。C0/C1共享同一学生训练轨迹是预定的配对设计，不能把它们当两次独立seed。其余配对的初始化模型摘要、每轮相对样本顺序、标签和覆盖均须一致。

总预算：2视图×3流×2wd×2梯度族=**24个梯度拟合尝试**，最多960个梯度epoch；产生36个臂候选（含12个EMA影子模型），不是36次独立梯度拟合。另有最多480个无梯度BN校准整遍，及其验证/EMA计算成本，必须单列，不能把EMA当零成本。所有失败计入预算，不自动重试、补种子或追加候选。当前执行数为0。

## C1 EMA与BN的精确定义

使用已安装PyTorch2.5.0的 `AveragedModel`、`get_ema_multi_avg_fn(.999)`，`use_buffers=False`。每次optimizer.step后更新一次EMA；第一次更新复制第一个已更新学生参数，此后 `EMA←.999*EMA+.001*student`。从首个更新开始，不调warmup、起始epoch或decay，不平均不同初始化。官方说明：[SWA/EMA](https://docs.pytorch.org/docs/stable/optim.html#weight-averaging-swa-and-ema)。实际函数实现文件哈希一并登记，不依赖在线stable版本。

每个epoch评价C1之前，复制EMA模型并仅在该视图 **fit_train** 的同一规范行序、batch128、无shuffle/增强、drop_last=False上用官方 `update_bn` 重估BN；参数不更新，复位running stats，momentum=None，遵循该工具的逐batch累计语义（不把末批改成自定义样本加权估计）。随后eval模式计算与C0/C2完全相同的checkpoint_val CE。保存该参数+重估buffers整体作为候选，不能训练后才更改BN而仍沿用旧checkpoint排名。[update_bn官方接口](https://docs.pytorch.org/docs/stable/optim.html#torch.optim.swa_utils.update_bn)

重估只作用于复制模型，不改变学生、EMA影子状态或参数；保护并恢复CPU/CUDA/Python/NumPy随机状态，不推进训练order generator。部署用选中EMA+其已经冻结的fit-only BN，不在source_selection、历史held或新RX上重估。本轮C1检验的是“EMA+合法BN处理”完整标准配方，不能把所有收益归因为纯参数平滑或证明BN中介。

## C2既有RX GroupDRO的精确定义

每视图12个真实训练RX为组，组集合只由fit_train登记，所有组全局支持必须>0，否则该视图验收失败；缺失TX×RX cell不制造数据、不当作额外空组。q初值均匀，范围[0,1]、总和1。

对batch内存在的组计算逐包CE的组平均 `L_g`；batch缺失组设0、更新增量0，不能把缺失组作为真实零风险报告。更新后使用新q加权本batch组损失：`logq←logq+.01*detach(L)`、再logsumexp归一化，训练损失 `sum_g q_g L_g`。logq/q为float64用于稳定性，反传损失为FP32；q不求梯度。不对活跃组额外重归一化，不裁剪为事后权重上限，不做group-size adjustment、loss normalization、BTL/CVaR或group-balanced sampler。

与[官方参考](https://github.com/kohpangwei/group_DRO/blob/cbbc1c5b06844e46b87e264326b56056d2a437d1/loss.py)的乘法权重更新在精确算术下等价；step=.01依据其固定默认。原参考commit为`cbbc1c5b06844e46b87e264326b56056d2a437d1`。不调用其可能每轮访问test的完整训练器，只迁移目标到共同受限数据路径。记录q、组计数/损失、缺失batch、梯度/更新数及实际权重范围，遇非有限值停止并记失败，不在看输出后切换实现。

## 选型、统计与预定出口

主要endpoint始终是各臂按checkpoint_val CE选中的checkpoint；epoch10和final40仅作预定辅助，禁止择优。全部轨迹候选冻结后才统一读取source_selection结果。每个方法在两wd中按12个内层留出RX、三固定流平均的RX等权CE最小选择wd；精确同分选较小wd。不用Accuracy、最差组或历史held作额外搜索目标。

完整报告两wd所有臂/视图/流。**相同wd的配对C1−C0和C2−C0**用于受控方法比较；各方法独立选wd之后的源域差异属于同一开发集上的选择性结果，可能乐观，不能称独立方法验证或纯目标函数因果效应。三臂选择信息、网格与最大更新预算相同。

报告RX等权Accuracy/Macro-F1/CE，最差配对RX、全部TX×RX×Day支持与风险、最差组（同时给支持≥50版本）、真实类/竞争类/最难负类与margin、纠错/新增/均错改类和同输入分歧。TX字段使用零起始模型类索引，不擅自改成物理TX硬件编号。C2减少RX平均风险并不自动解决类条件失败；不只报告过线组。比较实例分歧不能代替风险改善。

配对RX cluster bootstrap 10000次、seed24501、95%percentile，同一RX的所有方法/三流一起重采样。两个视图留出RX不重复，平均先按RX/固定三流聚合；各视图和各流独立表格同时呈现。统计单位是已观察12个开发RX，不是包、epoch或36个模型；条件区间不是seed或RX总体保证。两个方法比较不据源域未校正区间宣称总体显著性。

开发晋级只表示“值得单独确认”：相对各自选wd的C0，平均Accuracy≥+1pp、配对RX条件区间下界>0、两视图均正、至少2/3流正、最差配对RX差≥−2pp、支持≥50的配对TX×RX平均风险增加不超过5pp。所有条件须共同满足，且报告CE/Macro-F1；未通过不认定方法完全无用，也不追加seed追门槛。两候选都通过则按source_selection CE较低者排优先，精确同分优先C1；C0和两个方法完整保留，不能删除负结果。

当前只比较标准方法，不新增类条件Loss。若标准RX风险控制仍保留危险类别条件错误，将其作为下一单一机制的候选动机；若EMA有效，只能先说明标准稳定化配方有效。没有可信`T_eta`、物理尺度、局部近似适用范围及独立新RX证据前，不启动Jacobian原型，不把A/B或源域开发收益当作第三项算法创新。

## 参数、延迟和成本

模型有效推断参数预期82426，推断无RX标签、无EMA双分支或GroupDRO权重。训练额外EMA、q和存储开销另计。未来按同GPU/软件环境，预定source训练输入、resident CUDA tensor、batch1/128，100次warmup、分别1000/200次测量，CUDA event同步、三轮旋转方法次序，报告p50/p95与吞吐；计入完整RF前端+HStem+MHA，不含磁盘I/O/主机拷贝，不冒称端到端系统延迟。

训练报告梯度fit、EMA更新、BN校准样本/遍数、训练/验证/校准/评分wall time、显存和总成本，失败成本保留。完整训练器须先对无科学数据的合成样本验收，再锁代码；任何变更方法/网格/选型/权限的修订须在C科学输出前登记，不能结果出来后悄悄更换。
