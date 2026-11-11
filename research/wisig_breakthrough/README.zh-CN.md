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

1. **[C 标准方法竞争已完成](../standard_method_competition_c/EXECUTION_RESULTS.zh-CN.md)**：24/24次梯度拟合成功，960个epoch、121,920次更新，没有失败重试或额外拟合。主端点C0/C1/C2的source-CV Accuracy为85.0332%/86.1811%/84.5601%。EMA平均+1.1479 pp，但最大受支持TX×RX错误率增加21 pp，未过局部风险门；GroupDRO平均−0.4731 pp，亦未晋级。最终为`NO_STANDARD_METHOD_DEVELOPMENT_PROMOTION`。全部网格、108端点、局部风险、推断延迟和成本已封存；不追加种子、普通Loss或BN归因。这些百分比不能称为正式CRD。
2. **[M0真实双视图](../wisig_raw_eq_m0/README.zh-CN.md)**：已在配对硬门结束，`STOP_NO_VERIFIABLE_LOCAL_PAIRING`。官方均衡失败筛包及独立compact截取不保留原packet身份，本地审阅材料没有补足；TX信息保留/相关性未评估，不假设安全，不训练双视图候选。这只针对当前证据，不是物理上不可能配对。
3. **[Full-WiSig确认资格](../wisig_full_confirmation_metadata/README.zh-CN.md)**：额外9 RX身份和官方聚合计数已核验。原10TX×D3–D4各100 raw包的预算仅`2-20`、`8-13`满足；历史未用于方法选择仍未知，已确认合格RX为0、访问关闭。没有读取候选IQ、预测或性能。

## 执行后的固定研究出口

- C1有可靠收益：将EMA完整标准配方作为强基线，不将EMA本身计为第三项创新。
- C2有可靠收益：将RX风险优化作为强基线，保留全部未保护的类别条件风险，不直接换分组冒充新方法。
- 两者都有效：按冻结规则选更可靠参照，不自动叠加。
- 两者都不显著：不扩普通Loss搜索，不把区间跨零写成证明无效。

新机制必须有可信信号变化、TX判别保护的可证伪依据，与普通Raw+EQ CE/一致性及最强标准基线公平比较，且最终有正式CRD和迁移证据。当前M0未过配对门，因而**没有获准训练的Raw/EQ新机制**；不因理论公式好看而强行平均/对齐，更不自动改名重试。

本轮已按冻结出口结束三个工作包。EMA的平均收益保留为后续独立预注册应纳入的强参照，但未获稳健晋级；标准方法没有被包装成第三项创新。当前可训练新机制候选为0。用户对额外RX的[历史资格回复](../wisig_full_confirmation_metadata/OWNER_QUALIFICATION_RESPONSE.json)为“无法确认，继续保持资格未定”，合格新RX仍为0、确认访问保持关闭。

独立确认资格与性能选择分离。新RX若不能证明未参与方法选择，后续只能按另行协议报告回顾性benchmark和跨任务迁移，不得称全新RX盲测。当前工作不授权正式target8/日期3–4评分、Full新数据训练或新的LoRa/ManySig拟合。
