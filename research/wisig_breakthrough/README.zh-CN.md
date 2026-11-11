# 研究主线：WiSig CRD突破，保留LoRa高性能基础

本页是研究目标与已有工作出口的索引，不是新的训练阶段。A/B归因已结束，C是最后一次普通标准方法竞争；之后只允许在证据和独立协议支持下选择最多一个新机制，不能以新名称继续扩张普通因素实验。

## 要突破的历史参照

| 数据/协议 | 已有模型结果 | 强复现参照 | 定位 |
|---|---:|---:|---|
| Shen LoRa dev RX | HStem+RF-HSTU 97.48% | 同协议历史记录 | 保留已有基础 |
| Shen LoRa historical confirm RX | HStem+RF-HSTU 96.84% | 同协议历史记录 | 新方法须配对验证不退化 |
| WiSig ManyRx严格CRD | S07系列87.6506% | ASKNet88.7697% | 主要突破对象，90%为挑战目标，不是保证 |
| WiSig ManySig严格CRD | S08系列93.0116% | ASKNet93.4838% | 迁移验证对象 |

来源：[S06 LoRa正式汇总](../../06-paper-validation/RESULTS.md)、[WiSig正式CR/CRD对照](../../10-tifs-validation/FINAL_CR_CRD_COMPARISONS.md)、[机器汇总](../../10-tifs-validation/final_comparisons.csv)。不同数据/协议的百分比不能直接相减。LoRa这些数值是本项目历史实验，不是Shen原论文报告值。

普通MHA不能借用RF-HSTU最优数字：[S15](../../15-hstem-attribution/RESULTS.zh-CN.md) 的HStem+MHA为Dev97.335%、Confirm96.508%。后续LoRa迁移必须保持其合法信号前端及对应已验证配置，在配对协议下检验，而不是给换架构的新跑分贴“无退化”标签。当前没有新增LoRa训练或新方法无退化声明。

## 本轮三个有确定出口的工作包

1. **[C 标准方法竞争](../standard_method_competition_c/EXECUTION_ADDENDUM.zh-CN.md)**：原预注册24个梯度fit，C0/C1共享ERM轨迹，C2为RX GroupDRO。工程验收后按原方法/网格/seed/视图/选模和预算执行。只允许fit18内部source-CV，不允许把其Accuracy称为正式CRD。完成后无论晋级与否不追加种子、普通Loss或BN归因。
2. **[M0真实双视图](../wisig_raw_eq_m0/README.zh-CN.md)**：已在配对硬门结束，`STOP_NO_VERIFIABLE_LOCAL_PAIRING`。官方均衡失败筛包及独立compact截取不保留原packet身份，本地审阅材料没有补足；TX信息保留/相关性未评估，不假设安全，不训练双视图候选。这只针对当前证据，不是物理上不可能配对。
3. **[Full-WiSig确认资格](../wisig_full_confirmation_metadata/README.zh-CN.md)**：额外9 RX身份和官方聚合计数已核验。原10TX×D3–D4各100 raw包的预算仅`2-20`、`8-13`满足；历史未用于方法选择仍未知，已确认合格RX为0、访问关闭。没有读取候选IQ、预测或性能。

## 执行后的固定研究出口

- C1有可靠收益：将EMA完整标准配方作为强基线，不将EMA本身计为第三项创新。
- C2有可靠收益：将RX风险优化作为强基线，保留全部未保护的类别条件风险，不直接换分组冒充新方法。
- 两者都有效：按冻结规则选更可靠参照，不自动叠加。
- 两者都不显著：不扩普通Loss搜索，不把区间跨零写成证明无效。

新机制必须有可信信号变化、TX判别保护的可证伪依据，与普通Raw+EQ CE/一致性及最强标准基线公平比较，且最终有正式CRD和迁移证据。当前M0未过配对门，因而**没有获准训练的Raw/EQ新机制**；不因理论公式好看而强行平均/对齐，更不自动改名重试。

独立确认资格与性能选择分离。新RX若不能证明未参与方法选择，后续只能按另行协议报告回顾性benchmark和跨任务迁移，不得称全新RX盲测。当前工作不授权正式target8/日期3–4评分、Full新数据训练或新的LoRa/ManySig拟合。
