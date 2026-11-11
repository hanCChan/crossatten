# 阶段 C：标准方法竞争已预注册，尚未训练

2026-11-11，Asia/Shanghai。基于已完成的 A/B，状态为 **C_SOURCE_METHOD_COMPETITION_PREREGISTERED_CONFIRMATION_GATE_CLOSED**。

A/B归因线正式结束。本轮完成的是下一项方法竞争的可复核预注册、源域成员计划、标准方法核心实现及合成验收。科学拟合、IQ解码、历史权重/逐包预测加载和目标评分均为0；不追加BN归因或新Loss。

## 三个固定方法

| 臂 | 配方 | 关键边界 |
|---|---|---|
| C0 | 共同HStem+MHA ERM | 使用原RX等权validation CE选checkpoint，有与其他臂相同的正则化候选 |
| C1 | 同一ERM轨迹的EMA，decay=.999 | 每个候选用fit_train重估BN；只部署一个冻结平均模型 |
| C2 | 标准RX GroupDRO，dual step=.01 | 仅按RX分组；不同时加入TX×RX、CVaR或新的类别对Loss |

共同保持RF输入、HStem、Attention、分类头、数据角色及配对随机流。原“严格最小RX等权验证CE”选模规则保留；这次各臂统一训练40轮、关闭early stopping，以匹配实际更新预算。与A/B训练配方不完全相同，不能拿旧A/B模型直接替代C0。

正则化候选为Adam coupled weight_decay=0或.001，三臂相同。选择规则仅看合法源内部跨RX开发结果，不能用历史held6。C1的评价BN来自EMA复制模型的fit-only校准，参数、学生及影子状态不被重估改变；其效果只能解释为完整标准配方，不能当成BN中介证明。

方法细节、权重/缺失组处理、完整统计与晋级条件见[正式预注册](PREREGISTRATION.zh-CN.md)。[GroupDRO原研究](https://arxiv.org/abs/1911.08731v2)提示正则化的重要性，因此本协议包含共享的小网格，而非只用wd0给出弱组风险基线。它仍是有限候选竞争，不代表已找到任何方法的最优超参数。

即使C2不能解决类别条件失败，后续更细的TX×RX分组也首先属于既有组风险方法对照，不能仅凭更换分组就宣称新算法。

## 源域内部三种角色

本轮只登记fold2 fit18、日期1–2。按与性能无关的SHA256接收机排序形成两个内层视图，各12台RX训练、6台RX留出供源域选型；两视图的留出RX不重复。

| 视图 | fit_train包 | checkpoint_val包 | source_selection包 | 每轮更新 |
|---|---:|---:|---:|---:|
| V0 | 16246 | 6954 | 11468 | 127 |
| V1 | 16173 | 6995 | 11500 | 127 |

fit_train/checkpoint_val分别保留原train_ids/val_ids中的对应成员，不改变包的原分配。source_selection的RX完全不进入当前视图训练、BN或逐epoch选模；全候选checkpoint封存后才统一评分，用于方法和正则化选择。它们是合法源域内部交叉验证，不是新的外部接收机确认。

全部RX列表、原dtype索引SHA、TX/day及TX×RX×day支持（包括零支持）在 [SOURCE_PLAN.json](SOURCE_PLAN.json)。[ACCESS_SCOPE.json](ACCESS_SCOPE.json) 在本轮读取5个标签/索引数组之前写出；只做成员计划，没有解码IQ。未来输入的IQ文件仅作原始字节哈希核对，记录于机器登记。

## 有限预算和公平比较

2视图×3固定流×2个weight_decay候选×2梯度族，共 **24个梯度拟合尝试上限**、最多960个梯度epoch、121920次更新。每个ERM_EMA轨迹同时产出C0/C1；另一个RX_GDRO轨迹产出C2。因此有36个臂候选，其中12个EMA影子读出，不将它们算成独立seed或独立梯度拟合。

EMA额外最多480遍无梯度BN校准、7780560个校准样本处理及验证前向必须单列成本。失败也计预算，不静默重试、补seed或扩网格。[JOBS.json](JOBS.json) 固定24个作业及每个作业的成员视图、seed、配置、读出和5080次更新预算。

所有候选完整报告；同wd配对C1−C0/C2−C0是受控比较。每方法各自按source_selection CE选wd之后的结果包含开发选择偏差，只用于是否值得独立确认，不能称新泛化机制已成立。当前真实执行数为 **0**。

## 独立确认为什么没有登记为可执行

历史S10的 [final_receiver_metrics.csv](../../10-tifs-validation/final_receiver_metrics.csv) 已有ManyRx全部32个RX的报告；其[协议](../../10-tifs-validation/PROTOCOL_AND_LEAKAGE_AUDIT.md)也明确这些不是新盲测。此次只从历史汇总提取RX身份核对资格，没有重算或使用其Accuracy作方法选型。

因此[独立确认门槛](CONFIRMATION_GATE.json)保持关闭：没有把另一历史fold、target8、新seed或日期3–4登记为新接收机盲测。真正确认需另行登记未参与方法选择的新/可证明未用RX、兼容TX映射、采集条件、哈希和统计方案，并先冻结模型。本轮也不打开历史held6做描述性复核，不自动全fit18重拟合。

## 当前交付与执行边界

[PRE_REGISTRATION.json](PRE_REGISTRATION.json) 锁定科学协议、来源、成员/作业计划及方法核心代码；[REFERENCES.json](REFERENCES.json) 固定GroupDRO上游commit和本机PyTorch2.5.0实际SWA/EMA实现文件摘要。

新增12项纯合成CPU检查通过，覆盖EMA精确更新和BN buffer语义、fit-only角色拒绝、BN校准不改学生/影子并恢复随机流、GroupDRO概率/梯度/缺失组/非有限值、状态重放，以及源域三角色隔离和旧分配保持。没有构建或前向科学HStem模型。

完整科学训练/评价执行器尚未登记，当前没有C训练入口。执行前需完成实现与日志、配对/成本验收的工程附录并提交；不得借工程补充修改已冻结的网格、方法、选型规则或评价权限。此项边界明确区分“研究预注册完成”和“训练已经执行”。

源域晋级也不是独立最终收益。后续只在强标准对照揭示其作用边界后，才考虑一个新的类条件判别机制；可信物理变化族、Jacobian含义与独立新RX证据仍未建立。HStem+Attention保留已有贡献，类别对风险与鲁棒裕量原型不计作已验证创新。
