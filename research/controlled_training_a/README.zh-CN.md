# 阶段 A：完成预注册、严格重放和有限水平敏感性实验

日期：2026-11-11，Asia/Shanghai。

状态：**STAGE_A_COMPLETE_PREDICTION_CHURN_VERIFIED_SELECTED_EFFECTS_UNCERTAIN**。已完成全部 5 个预定拟合尝试，没有追加训练；阶段 B 未启动。历史审计的 `CONFOUNDS_VERIFIED_CAUSAL_ATTRIBUTION_UNRESOLVED` 保留，未改写旧 STOP、模型或结论。

## 主要发现

在完全相同的 fold2 训练/验证数据和共同自适应训练策略下，改变选定初始化或批序，会改变同一批 source-held 输入的预测。selected checkpoint 的五组预定配对分歧率为 **7.425%–8.217%**，均超过预注册的 5% 预测变化筛查线。这是固定水平下可重现的预测敏感性证据，不是初始化/顺序总体方差贡献。

主要端点的 RX 等权 Accuracy：

| 运行 | 训练轮数 | 选中 epoch | source-validation | source-held6 |
|---|---:|---:|---:|---:|
| A00 | 14 | 4 | 92.4196% | 91.0083% |
| A00-replay | 14 | 4 | 与 A00 逐位相同 | 与 A00 逐位相同 |
| A10 | 13 | 3 | 92.2653% | 91.7417% |
| A01 | 14 | 4 | 92.3705% | 91.3667% |
| A11 | 14 | 4 | 92.5373% | 92.4333% |

source-held6 的 selected 平均因素对比（Accuracy，单位：百分点）：

| 对比 | 点估计 | 配对 RX 重采样 95% 区间 |
|---|---:|---|
| 初始化 | +0.9000 | [-0.6500, +2.6292] |
| 批顺序 | +0.5250 | [-0.6583, +1.6417] |
| 交互 | +0.3333 | [-1.3917, +2.3667] |

上述三个点估计均未达到 1 个百分点的预注册因素筛查线，但区间宽且跨零，不能据此认定初始化或批序影响“小”，也不能认定某因素占主导。主要端点的 Macro-F1/CE 平均因素对比也未达到各自实用筛查线。A00→A11 的整体配对 Accuracy 差为 +1.425 个百分点，说明平均因素对比与某一具体配对不能混为一谈。

A00→A11 的 12000 个共同 held 包中，986 包预测不同：450 包纠错、279 包新增错误、257 包在两次均错时改变了预测类。因此净 Accuracy 改善不能代替逐包翻转分析。全部 TX×RX×Day 支持、组错误、最差配对组、真实类/最难负类和 margin 分布均已保留。

epoch10 辅助端点的初始化 Accuracy 对比为 +1.7833 个百分点，区间 [+0.4417, +3.3042]；Macro-F1 为 +2.1779 个百分点，CE 为 −0.1241 nat。这提示该固定辅助时点有较强初始化敏感性，但不替换 selected 主结果，不将它转述为最终选模策略下已确定的大效应。所有端点、指标和配对完整呈现在[预定分析报告](RESULTS.zh-CN.md)及[完整分析 JSON 压缩归档](ANALYSIS.json.gz)；本地保留原始 `ANALYSIS.json`，归档为其无损 gzip 副本。

上述区间只描述这 6 个固定历史 held RX 的条件重采样，不是未见接收机总体或 seed 分布的置信保证。source-validation 已参与选模；source-held6 也是历史探索接收机，均不能充当新的独立泛化确认。

## 工程证据与访问边界

- 首次训练前的预注册提交：`9afc836`，见[正式文本](PREREGISTRATION.zh-CN.md)及[输入/代码登记](PRE_REGISTRATION.json)。独立复核及 19 项纯合成检查通过；未用科学 IQ 做工程试跑。
- A00/replay 的 23 项检查全部逐位相同，最大数值差 0：初始化、规范样本键和成员分配、每轮完整状态及 BN/optimizer/scheduler/RNG、selected/epoch10/final 和 validation 输出。见 [REPLAY_GATE.json](REPLAY_GATE.json)。
- 其余三次完成后，同初始化的模型状态、同顺序的共同 epoch 规范键序列等隔离检查均通过。五次运行与全部 snapshot 在 `be3414c` 提交中封存，之后才首次载入本轮 held 数组。见 [STAGE_A_SEAL.json](STAGE_A_SEAL.json)。
- 固定训练 24267 包、验证 10401 包；held 共 12000 包，6 RX、日期 1–2。target8 及日期 3–4 没有读取，没有旧预测或历史模型权重访问。
- 总计 69 epoch、13110 次参数更新；各 fit 内计时合计 135.50 秒，峰值已分配 CUDA 显存 0.190 GiB。计时包含拟合内保存/验收，不含进程启动、登记预检、统一 held 评价或分析。环境为 PyTorch 2.5.0+cu124、RTX 4060 Laptop、严格确定性 FP32，详细配置见 [RUN_EVIDENCE.json](RUN_EVIDENCE.json)。
- [ACCESS_LOG.jsonl](ACCESS_LOG.jsonl) 记录 65 次封存前登记 fit 文件的哈希/数组动作，以及封存后 12 个 held 源 cell 各一次校验和载入，共 89 条。登记前的原始字节哈希读取另载于预注册；这是脚本动作记录，不是操作系统级监控。

提交前曾发现 Windows 历史文件 CRLF checkout 与 Git LF 文本校验不兼容。修正仅影响“是否已提交”的文本比较，运行代码仍按原始字节 SHA256 锁定；增加回归检查后才形成正式预注册。旧登记草稿保留于 `DRAFT_REGISTRATION_BEFORE_LINE_ENDING_FIX.json`，其时科学数组载入、训练和模型前向均为零。

## 固定研究出口

本次已完成 A 的工程重放与有限水平预测敏感性检验，到此停止追加拟合。**尚未测量 RX 组成效应，不能比较初始化与组成的重要性，也不能因此将研究分流到某个已确定的算法方向。**

B 若执行，须独立正式登记那两个特定 extension 集合的替换对照。设备编号与采集/硬件属性的关联仍待元信息核实；B 采用 12 RX，不能与 A 的 18 RX 直接相减排名因素。HStem 的结果不能解释 ASKNet 的 Gumbel 随机过程或 0/66/5 具体错误。

HStem+Attention 保留为已有结构成果；本阶段属于归因依据，不是新算法创新。后续标准稳定化或跨环境机制应按研究出口另行选择，最终在未参与方法选择的接收机上验证。

## 文件与复核入口

运行级 CONFIG/HISTORY/COMPLETE 的可阅读副本见 [RUN_EVIDENCE.json](RUN_EVIDENCE.json)。每轮可加载训练状态、实际顺序、raw logits、逐包诊断位于本地忽略目录 `artifacts/`；文件摘要分别由封存和 [EVALUATION_MANIFEST.json](EVALUATION_MANIFEST.json) 固定，不上传 IQ 或模型二进制。

重新拟合、评价或分析命令会拒绝覆盖原尝试。合成验收可运行：

```powershell
D:\crossatten\.venv\Scripts\python.exe -m pytest research/controlled_training_a/test_controls.py tests/test_controlled_training_analysis.py -q
```

训练 CLI 顺序为 `runner.py train --run A00`、`A00-replay`、`replay-check`、其余三个运行、`seal`；提交 seal 后才允许 `evaluate`，随后运行 `analyze.py`。这仅解释已经执行的步骤，不授权重复这五次拟合。跨机器复核还需相同登记数据、代码字节及环境，Git 源码本身不包含本地科学数组。
