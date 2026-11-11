# 阶段 C 执行完成：平均收益与局部风险必须同时报告

24 次预注册梯度拟合全部成功完成，没有失败重试、额外种子或新增配方。最终决策为 **`NO_STANDARD_METHOD_DEVELOPMENT_PROMOTION`**。EMA 有一致的平均收益，但未通过预注册的类别条件风险限制；RX GroupDRO 也未满足联合门槛。A/B 归因线和 C 标准方法竞争在此结束，不追加普通训练因素实验。

这是合法 fit18 内部、12 台 source-selection RX 与三套固定配对训练流的开发结果。**不是 ManyRx 正式 CRD、新接收机盲测、随机种子总体方差估计或接收硬件物理机制证据。** 历史 ManyRx CRD 87.6506%、ASKNet 88.7697% 与挑战目标 90% 仍按原协议保留；不能用本表百分比与它们直接相减。

## 主结果和预注册决策

各方法均按 selected checkpoint 的 source-selection RX 等权 CE 选中 weight decay=0.001。因此本次各自选参比较也恰好是同 wd 的受控比较；另一个 wd 的全部结果仍保留。checkpoint 由 fit RX 的 checkpoint-validation CE 选择，source-selection 在全部训练、读出和哈希封存并提交后才统一评分。

| selected 主端点 | RX 等权 Accuracy | RX 等权 Macro-F1 | RX 等权 CE | 相对 C0 / pp | 配对 RX 95% 条件区间 / pp |
|---|---:|---:|---:|---:|---:|
| C0 ERM | 85.0332% | 0.813311 | 0.516880 | — | — |
| C1 EMA=.999 + fit-only BN 重估 | 86.1811% | 0.826064 | 0.486013 | +1.1479 | [+0.2993, +1.9902] |
| C2 RX GroupDRO | 84.5601% | 0.810272 | 0.511247 | −0.4731 | [−1.5573, +0.6016] |

区间先在每台 RX 内平均三套固定流，再用同一组 10,000 次 RX 簇配对重采样（seed=24501）。区间只描述这些观察 RX 的条件不确定性；同一 source-selection 信息还用于两档 wd 选择，因此保留开发选择偏差，不作独立确认或总体显著性声明。C0/C1 共享学生梯度轨迹，但各自的选中 epoch 可以不同；比较的是完整配方和选模结果，不是纯粹的单因素目标函数效应。

| selected 相对 C0 | C1 | C2 |
|---|---:|---:|
| V0 平均 Accuracy 差 / pp | +1.0665 | −0.6736 |
| V1 平均 Accuracy 差 / pp | +1.2293 | −0.2725 |
| S0 平均 Accuracy 差 / pp | +1.0707 | −1.8259 |
| S1 平均 Accuracy 差 / pp | +1.1929 | +0.3343 |
| S2 平均 Accuracy 差 / pp | +1.1801 | +0.0724 |
| 同输入预测分歧率 | 8.4628% | 10.9335% |
| 同输入纠错占比 | 2.9518% | 2.9553% |
| 同输入新增错误占比 | 1.8039% | 3.4283% |
| 最差配对 RX Accuracy 差 / pp | −1.3205（24-6） | −4.2167（8-14） |
| 最大受支持 TX×RX 错误率增加 / pp | +21.0000 | +25.0000 |

纠错、新增错误和分歧率按相同 RX/流权重汇总，分母是全部配对输入。分歧还包含两边都错但预测类别不同的情形；全部类别变化、竞争类别和 margin 分布随完整指标保留。

| 冻结的联合晋级条件 | C1 | C2 |
|---|---|---|
| 平均 Accuracy 至少 +1 pp | 通过 | 未通过 |
| 配对 RX 区间下界 >0 | 通过 | 未通过 |
| 两视图均为正 | 通过 | 未通过 |
| 至少 2/3 训练流为正 | 通过（3/3） | 通过（2/3） |
| 最差配对 RX 不低于 −2 pp | 通过 | 未通过 |
| 每个受支持 TX×RX 平均错误率增加 ≤5 pp | **未通过** | **未通过** |

C1 的失败门是局部风险：RX `24-6`、TX 类索引 `8`，每流 200 包，三流平均错误率增加 21 pp。C2 的最大恶化为 RX `24-16`、TX 类索引 `5`，每流 200 包，增加 25 pp。TX 索引沿用冻结类别顺序，不与设备名或其他报告的编号约定混用。共有 117 个满足每流至少 50 包的 TX×RX 组；没有把缺失组填零，全部组均参与冻结规则检查。这些例子解释门槛为何失败，不用于事后选组或提出新的 Loss。

EMA 的平均收益不能被抹去，也不能越过局部风险门槛晋级；该配方应作为后续独立预注册时保留的强标准参照。GroupDRO 的区间跨零不证明所有组风险方法无效。本次没有将 EMA 和 GroupDRO 自动叠加，也没有改变 EMA、分组或更新预算。

## 全部端点、训练控制与成本

selected 始终是主端点；epoch10 和 final40 是辅助结果，不用于事后替换主结论。三方法沿用主端点选中的 wd，final40 Accuracy 分别为 86.2242%、86.2464%、85.6147%；epoch10 分别为 85.5628%、85.0185%、84.2376%。

- [完整比较表](FULL_COMPARISONS.zh-CN.md)：两个 wd、三个端点、两视图、三训练流及全部同 wd/选参比较。
- [108 个视图×训练流×方法端点](CANDIDATE_CELLS.md)及[逐候选标量指标](CANDIDATE_CELLS.json)：保留各读出的实际 snapshot epoch、每 RX 和 TX×RX 支持与风险。
- [机器可读汇总](SUMMARY.json)：全部网格、配对差异、RX 与 TX×RX 风险及晋级条件。
- [完整指标压缩档](ANALYSIS.json.gz)：全部 108 个候选端点，包括支持、混淆、类别对、margin 和错误变化；解压得到原始 ANALYSIS.json。[压缩及本地产物说明](ARTIFACT_ARCHIVE.json)记录校验值。
- [读出汇总](RESULTS.zh-CN.md)、[最终决策](DECISION.json)、[独立数值复核](EXECUTION_QA.json)、[训练控制复核](TRAINING_EXECUTION_QA.json)。
- [训练与部署成本](COSTS.zh-CN.md)、[机器成本记录](COSTS.json)、[36 个 selected 读出的逐轮延迟](LATENCY.json)。

24 次 fit 各 40 epoch、5,080 次更新，共 960 个梯度 epoch、121,920 次更新。C0/C1 共享的 12 次 ERM 轨迹产生两种读出，另有 12 次 GroupDRO；36 个方法读出不是 36 次拟合。EMA 共 60,960 次更新；480 次 fit_train-only BN 重估，共 7,780,560 个校准样本访问。122 项封存控制检查全部通过，108 个读出的 checkpoint-validation 推断重放均完全一致；每轮 EMA 校准均保留学生、shadow 和训练随机状态。

在任何科学训练前完成 34 项 CPU 合成测试及 3 条 GPU 合成轨迹验收；随后单独登记的 8 项元数据复核测试通过。实际训练和分析的独立复核均 PASS。合成验收与科学训练分开记账，未使用额外科学拟合进行调试。

执行器与工程附录提交于 `bf92856`，源域预检提交于 `e46394b`；全部拟合后，封存与训练 QA 提交并推送于 `007dc62`，之后才进行 source-selection 评分。见[执行工程附录](EXECUTION_ADDENDUM.zh-CN.md)、[训练封存](STAGE_C_SEAL.json)、[逐轮证据](RUN_EVIDENCE.json)、[评价哈希清单](EVALUATION_MANIFEST.json)和[数据访问记录](EXECUTION_ACCESS_LOG.jsonl)。旧预注册、旧 QA/MANIFEST 和原 README 保留原始状态；本页记录后续实际执行。

[发布一致性核验](CLOSURE_QA.json)确认 83 个冻结代码/证据文件未变、压缩档逐字节往返一致、108 个候选标量表与原分析相符、访问记录仅追加且路径均在源域白名单内。[最终发布清单](EXECUTION_MANIFEST.json)登记本轮 C、M0、确认资格及研究主线文档的文件哈希。

Git 文本换行规范化导致部分发布字节与执行字节的哈希不同。已补充[字节复核说明及原字节档](BYTE_PRESERVATION.zh-CN.md)，保留冻结文件的实际字节，并在最终清单中同时列出两种表示的哈希；训练、评价和决策均不变。

## 与 M0、独立确认及研究主线的衔接

[M0](../wisig_raw_eq_m0/README.zh-CN.md) 在首个配对硬门停止：`STOP_NO_VERIFIABLE_LOCAL_PAIRING`。官方处理链存在均衡失败筛包和独立 compact 截取，当前审阅材料不能建立原始包身份。TX 判别安全性和跨 RX 相关性尚未评估，不能把“未知”写成“均衡安全”或“均衡有害”；当前不训练 Raw/EQ 双视图候选，也不改名重试。

[Full WiSig 元数据核查](../wisig_full_confirmation_metadata/README.zh-CN.md) 找到额外 9 台 RX，其中 `2-20`、`8-13` 满足预设原 10 TX×日期 3–4 包数预算。但它们也出现在官方 SingleDay 设备集合，历史独立性无法确认。用户明确选择[继续保持资格未定](../wisig_full_confirmation_metadata/OWNER_QUALIFICATION_RESPONSE.json)：合格新 RX 数仍为 0，确认访问保持 CLOSED。没有读取这些候选的 IQ、预测或准确率。

三个工作包至此均有明确出口。C 的平均收益仍未保护全部类别条件风险；M0 缺少可验证配对，因而目前获准训练的新机制候选为 **0**，不为填满“最多一个”名额而发明新配方。研究目标继续是保留 LoRa 的已验证结构基础、在合法独立协议下超过 ManyRx ASKNet 并挑战 90% CRD，再验证 ManySig/LoRa 迁移。当前没有实现或宣称该突破，未对 historical held6、target8、日期 3–4 读取 IQ 或评分，也没有新增 ManySig/LoRa 训练。Full 资格工作仅使用公开聚合计数。[研究主线索引](../wisig_breakthrough/README.zh-CN.md)。
