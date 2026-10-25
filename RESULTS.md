# crossatten 全部结果汇报

证据截止：**2026-10-25**。范围：Shen 阶段 00–03、WiSig 数据与发布权重验证（M2），以及阶段06论文验证的实施状态。
阶段 00–03 与 WiSig 发布权重参考保持冻结。2026-10-25 按用户授权新增[论文验证阶段06](06-paper-validation/README.md)，已完成预登记、69/69运行检查与成本实测，正式源域训练已启动，最终目标域结果尚待完成。本报告不把运行检查或实施规格中的预测当成结论。

## 1. 核心结论

1. **当前 Shen 最强已验证配置是 `s03_hstem`：dev_val 97.48±0.41%，confirm 96.84±0.58%，142,392 参数。** 相对预登记对照 `s03_preln`，dev_val 提升 2.00 pp，6/6 配对为正，判为 `improves`。
2. 从全频带原始表征转向**严格带内 + AWGN**是阶段 01 的决定性改善；阶段 02 再把训练周期延长到 200 epoch，达到 95.84 / 93.89。模型参数仍为 580,008。
3. **重复结构主假设尚未成功。** 阶段 03 的单位 RMS 残差通道和等参数符号堆叠分别降低 7.47 pp、5.20 pp；瞬时频率通道为 `unclear`。不能把分层 stem 的工程收益写成已经证实的 Chirp 结构创新。
4. **Shen 已完成 CIS + non_adv_net 的 PyTorch 重新实现和同协议训练，不是完整原版 TensorFlow/对抗训练流程复现。** rank-16 CIS 以 658,314 参数保留了接近原版的准确率，是有效的轻量对照。
5. **WiSig ManyRx / ManySig 的数量、数据转换和发布权重验证已通过。** IB-RFF CRD 为 89.1879 / 93.5694；WaveMLP CRD 为 88.6759 / 93.1250。这是上游权重评估，不是我们的公平重训或跨数据集方法成功。

主要证据：[S01 结果](01-strong-baselines/RESULTS.md)、[S02 确认表](02-single-factor-screen/summary/confirm_tables.md)、[S03 结果](03-chirp-structure/RESULTS.md)、[WiSig M2 结果](05-wisig/RESULTS.md)。

## 2. 口径与完成情况

Shen 使用冻结协议 **P-H5x160**：10 个 TX，每折 5 台源 RX，160 包/TX 用于训练、40 包/TX 用于源验证；5 台 dev_val RX、4 台两个 fold 共用的 confirm RX。
每台评估 RX 为 2,000 包。6 台 blind RX 未读取。

- 探索：fold 0/1 × seed 101/102，共 4 次训练/臂；确认：fold 0/1 × seed 201/202/203，共 6 次训练/臂。
- 单次训练只按源验证 CE 选择 checkpoint；阶段 03 的结转判定只用确认阶段的 dev_val。
- 主指标为 RX 等权准确率。下表的均值±SD 来自 6 个 `(fold, seed)` fit，**SD 包含 fold 差异，不是纯 seed 波动**。seed 和包都不能视为额外独立接收机。
- confirm RX 已多次查看，只作报告，不再是独立泛化证据；历史 14→6 协议没有在本轮重新验证。
- `improves` 需要平均增益至少 1 pp、确认时至少 5/6 正配对，并满足最差 RX 门槛；`harms` 和 `unclear` 按预登记规则原样报告。

| 阶段 | 做了什么 | 已完成规模 | 状态与完整证据 |
|---|---|---|---|
| 00 | 物理参数、频段、CIS 坐标、CFO/SNR、符号均值/残差审计 | 10 台非 confirm RX，20,000 包 | [审计结果](00-signal-audit/RESULTS.md) |
| 01 | 训练配方、Shen-CIS/GAP、输入频带、时频分离 | 19 臂；76 探索 + 114 确认 fit | 全部完成；[完整结果](01-strong-baselines/RESULTS.md) |
| 02 | 强基线上的 Chirp/OOB/结构/训练单因素及 rank-16 CIS | 13 臂；52 探索 + 78 确认 fit | 全部完成；[完整确认表](02-single-factor-screen/summary/confirm_tables.md) |
| 03 | 基线重跑、Pre-LN、加深、残差、符号堆叠、瞬时频率、分层 stem | 8 smoke + 32 探索 + 48 确认 fit | 88/88 核验通过；[完整结果](03-chirp-structure/RESULTS.md) |
| 04 | 原 Shen 外部方法实施规格 | 历史目录未训练 | FAN/IB-RFF、WaveMLP 已转入阶段06；Ma 不在本次注册矩阵内 |
| 05/M2 | 两份 WiSig 数据转换、源域审计、发布权重验证 | 两数据集、四折、三个方法系列 | 已完成；[M2 结果](05-wisig/RESULTS.md) |
| 05/M4 | 原 WiSig 公平重训安排 | 历史目录未训练 | 本方法、FAN/IB-RFF、WaveMLP 已转入阶段06执行 |
| 06 | WiSig W0/W1/W2 与 CNN 控制、两个外部强基线；Shen 同协议比较与机制消融 | 69/69 smoke 完成；正式源域矩阵进行中 | [预登记](06-paper-validation/README.md)、[实时验证报告](06-paper-validation/RESULTS.md)；完成冻结提交后才允许最终评估 |

阶段 01–03 共保存 **400 个正式训练记录**，包含对照重跑；这些不是 400 个独立泛化实验。逐臂、逐 seed、逐 RX、F1、训练曲线、预测与失败/跳过状态均保存在对应 `runs/`、`summary/`。
协议与来源见 [PROTOCOL](docs/PROTOCOL.md)、[PROVENANCE](docs/PROVENANCE.md)。

## 3. 阶段 00：发现了什么

阶段 00 是信号审计，不报告一个端到端模型的 dev_val / confirm 分数。

- 核验了 Fs=1 MHz、BW=125 kHz、SF7、8 个 1,024 点 chirp 的边界对齐。
- 相干均值保留 99.6–99.7% 能量，但符号残差具有 U 形能量分布和首符号瞬态；相邻残差相关系数 +0.49～+0.67，不符合独立白噪声的约 −0.14。
- IB 残差频谱跨 RX 线性探针达到 52.5%，支持“残差包含可迁移信息”的描述性观察，不保证显式残差网络会改善准确率。
- CFO/SNR 元数据单 RX 内能区分类别，跨 RX 则约为随机水平，未作为模型输入。
- Shen-CIS 原裁剪覆盖约 −203.1～+195.3 kHz，并不严格带内。
- “远带 >200 kHz 噪底是主要捷径”后来被阶段 01 否定；带外有类别信息也不等于原始带外侧路能带来增益。

全部频段、探针及后续更正见 [00 RESULTS](00-signal-audit/RESULTS.md)。

## 4. 阶段 01–02：强基线与负面结果

### 主要模型

准确率单位 %；除注明外均为确认阶段 6 fit 的 RX 等权均值±SD。

| 阶段 / 模型 | 参数 | dev_val | confirm | 解释 |
|---|---:|---:|---:|---|
| 01 全频带 CrossAttn + AWGN | 625,896 | 50.46±4.56 | 56.75±4.40 | 原始全频带迁移差 |
| 01 TrueIB + AWGN，100 epoch | 580,008 | 94.74±1.33 | 92.11±0.85 | 严格带内强基线 |
| 01 Shen CIS Flatten | 13,442,954 | 93.12±1.62 | 95.20±0.87 | 原裁剪，PyTorch 重实现 |
| 01 严格 IB CIS Flatten | 13,442,954 | 94.62±3.03 | 95.72±0.12 | 带内 CIS 对照 |
| 01 CIS GAP | 270,218 | 58.67±8.96 | 54.84±9.60 | 轻量替代失败 |
| 02 TrueIB + AWGN，200 epoch | 580,008 | 95.84±0.69 | 93.89±0.48 | 唯一确认有效的基线改进 |
| 02 rank-16 CIS | 658,314 | 92.36±2.08 | 94.75±1.23 | 相对 Flatten 仅 −0.76 / −0.46 pp，判为 unclear |
| 03 hstem | 142,392 | **97.48±0.41** | **96.84±0.58** | 当前最强已完成配置 |

不同模型之间的这张总表是系统级参照，不能代替单因素预登记比较或阶段 04 的竞争方法重跑。

### 单因素结论

阶段 01：全频带模型的源验证→未见 RX 差距为 25–39 pp，严格 IB 只有约 0–3 pp。仅去除 >200 kHz 没有可测改善，必须把所有视图由严格 IB 波形重新生成。AWGN 有助于训练稳定；GAP 不能替代 CIS Flatten。

阶段 02 以 100-epoch TrueIB 为对照：

| 因素 | dev_val / confirm (%) | dev_val Δ (pp) | 确认 dev_val 判定 |
|---|---|---:|---|
| 200 epoch | 95.84 / 93.89 | +1.11 | improves |
| 均值 chirp | 81.09 / 80.40 | −13.64 | harms |
| 梳状 FFT | 88.99 / 86.48 | −5.74 | harms；探索时曾为 improves |
| dechirp | 90.98 / 88.66 | −3.76 | unclear |
| 原始全 OOB 侧路 | 83.44 / 83.63 | −11.30 | harms |
| 原始近带 OOB 侧路 | 84.11 / 83.95 | −10.63 | harms |
| 原始左近带 OOB 侧路 | 84.25 / 83.43 | −10.48 | harms |
| CIS 谱商 OOB 侧路 | 94.11 / 91.96 | −0.62 | unclear |
| depth4，无 Pre-LN | 67.08±30.42 / 65.59±29.08 | −27.65 | unclear；显著失稳但不足 5/6 负配对 |
| 严格 IB 时频分离 | 93.05 / 90.42 | −1.68 | harms |
| AWGN U[10,50) | 94.72 / 92.14 | −0.01 | unclear |

全部臂及逐 RX 结果：[01 探索](01-strong-baselines/summary/explore_tables.md)、[01 确认](01-strong-baselines/summary/confirm_tables.md)、[02 探索](02-single-factor-screen/summary/explore_tables.md)、[02 确认](02-single-factor-screen/summary/confirm_tables.md)。

## 5. 阶段 03：第一轮全部结果

统一采用严格 IB、AWGN U[0,50)、200 epoch。所有 8 臂均进入确认，未按探索分数淘汰。
`symstack` 的干净对照是 `nofft`；其余对照见下表。预登记文件与判定规则未按结果修改。

| 实验臂 | 参数 | dev_val (%) | confirm (%) | dev_val Δ / 对照 (pp) | 判定 |
|---|---:|---:|---:|---|---|
| base | 580,008 | 95.84±0.69 | 93.89±0.48 | 0.00 / s02_ep200 | 确定性对照 |
| preln | 580,136 | 95.48±0.91 | 93.79±0.59 | −0.36 / base | unclear |
| preln_d4 | 622,374 | 95.58±0.79 | 93.73±0.75 | +0.10 / preln | unclear |
| resid | 580,584 | 88.01±1.68 | 87.21±1.70 | −7.47 / preln | harms，6/6 负 |
| nofft | 579,912 | 95.48±0.95 | 93.65±0.39 | 约 0.00 / preln | unclear |
| symstack | 579,912 | 90.27±0.75 | 89.02±2.03 | −5.20 / nofft | harms，6/6 负 |
| ifreq | 580,360 | 95.50±1.23 | 93.94±0.91 | +0.02 / preln | unclear |
| **hstem** | **142,392** | **97.48±0.41** | **96.84±0.58** | **+2.00 / preln** | **improves，6/6 正** |

`preln` 在探索时改善 +1.04 pp，确认时为 −0.36 pp，必须以确认结论为准。
所有 6 个确认 baseline 的权重及 dev_val/confirm 预测均与阶段 02 ep200 **逐元素完全相同**，4 个探索 baseline 也通过确定性核对。
见 [冻结决策](03-chirp-structure/decision_confirm.json)、[确定性检查](03-chirp-structure/summary/confirm_determinism.json)、[完整表](03-chirp-structure/summary/confirm_tables.md)。

### hstem 收益与代价

- 相对原始 `base` 的事后系统比较：dev_val **+1.64 pp**、confirm **+2.95 pp**，两者均为 6/6 正配对；相对预登记 `preln` 的 confirm 为 +3.06 pp。
- 相对 `preln` 参数减少 **75.46%**，但计数 FLOPs 从 139.2 增至 **190.6 MFLOPs/包（+36.93%）**。高分辨率卷积使小参数不等于低算力。
- A100 batch512 吞吐从 41,234 降至 **36,101 包/s**，不能宣称更快；hstem GPU batch1 p50/p95 为 **2.53/2.99 ms**，CPU 4 线程为 **6.08/7.57 ms**。
- 最差 `(fold, RX)` seed 均值：dev_val **95.57%**，confirm **90.22%**。Macro-F1 为 97.48 / 96.78。
- 确认运行/评估实测最大分配显存 **3.36 GB**，含 batch500 的 checkpoint 评估，超过规格估计的 3.1 GB；无 OOM、无漏跑。

成本测量包含从设备上的原始 IQ 到 logits 的前处理；FLOPs 只完整统计卷积/矩阵运算，FFT 和逐元素操作不完全计入。基准在正式训练前无竞争 GPU 任务时运行，见 [bench](03-chirp-structure/summary/bench.md)。

### 机制诊断与边界

使用 resid/symstack 全部 6 个确认 checkpoint，共 **432 条逐 RX/干预记录**，未重新拟合或据此选择下一轮结构。

| dev_val 干预 | 原准确率 (%) | 干预后 (%) | Δ (pp) |
|---|---:|---:|---:|
| resid 残差置零 | 88.008 | 43.568 | −44.440 |
| resid 包间残差循环移位 | 88.008 | 38.613 | −49.395 |
| symstack 仅保留均值 chirp | 90.273 | 34.637 | −55.637 |
| symstack 逆序符号 | 90.273 | 29.882 | −60.392 |
| symstack 错位符号 | 90.273 | 12.845 | −77.428 |
| symstack 包间残差循环移位 | 90.273 | 27.825 | −62.448 |

这些下降说明模型依赖相应表征及包内耦合，**不等于这种表征提高了泛化准确率，也不是身份信息的纯因果证明**。
注册的 batch 循环移位仅使 **0.80%** 包的残差 donor 换了 TX，其余仍为同 TX；干预同时改变包配对并造成分布偏移。未干预 batch64 与保存的 batch500 评估准确率差为 0.0000 pp。
完整 dev_val/confirm 数据与 donor 诊断见 [mechanisms](03-chirp-structure/summary/mechanisms.md)。

源验证集上的事后尺度检查：原 base token RMS 为 106.5–176.6、sigmoid 饱和比例 99.94–100%；Pre-LN 将 RMS 控制到约 1，但仍有 73–95% 饱和，depth4 更高。**不能说 Pre-LN 消除了打分饱和，或已经证明历史崩溃的因果机制。**
见 [token_scale](03-chirp-structure/summary/token_scale.md)。

## 6. Shen 复现到什么程度

**已验证：** `shen_cis` 与原 numpy/scipy 逻辑逐元素最大误差小于 1e-5（实际约 2e-7）；网络匹配 Keras glorot、same 非对称填充和 channels-last 展平顺序，参数为 **13,442,954**。
实现、测试、训练与同协议结果均在仓库内。

**未复现的范围：** 没有完整运行原版 TensorFlow/对抗训练流程，没有声称严格复现论文原协议的数字。这里按 epoch 打乱不放回遍历而不是每 batch 随机抽样，源验证 AWGN 固定而不是每 epoch 重生成，并使用 P-H5x160 的源验证划分。
详情见 [PROVENANCE](docs/PROVENANCE.md)、[CIS 测试](tests/test_core.py)、[实现](lib/xa/models.py)。

## 7. WiSig：数据已齐，M2 已通过

ManyRx 复用 `gai` 中已有 Kaggle archive；ManySig 从 UCLA 官方页面发布的 Google Drive 下载，校验值记录在 manifest。
两个官方 loader 与我们 cache 的 float32 输入及标签在每个 RX/day **完全相同，最大差为 0.0**。

| 数据集 | 官方训练池数量（各 fold） | 数据门槛 |
|---|---|---|
| ManyRx | 46,468 / 46,468 / 46,668 / 47,200 | 与官方日志逐一相等 |
| ManySig | 10,800 / 10,800 / 10,800 / 10,800 | 与官方日志逐一相等 |

源 RX、第 1–2 天实测 Fs=**25 MHz**、STF 周期 **20 点**，严格 IB cutoff 冻结为 **7.67578125 MHz**。
256 点 crop 不含完整 LTF/CP 匹配尾部，**尚不能完成完整 LTF 对齐验收**，后续不能照搬 LoRa 的长度或声称完整 WiFi 重复结构已核实。

以下为上游发布权重四折均值，官方包等权口径；**FAN 的 CR 与另两者的 CRD 不是同一测试任务**。

| 方法 / 协议 | ManyRx (%) | ManySig (%) | 验证情况 |
|---|---:|---:|---|
| FAN / CR（第 1–2 天测试） | 89.6840 | 93.0625 | 对应官方 stage1 CR 日志 |
| IB-RFF / CRD（第 3–4 天测试） | 89.1879 | 93.5694 | 对应官方 stage2 CRD 日志 |
| WaveMLP / CRD | 88.6759 | 93.1250 | 四折均值与 README 四舍五入值相同 |

FAN/IB-RFF 全部 16 个 fold 验证通过，最大日志差 0.0109 pp。历史 FAN “不匹配”是 CRD 对 CR 日志的脚本口径错误，已更正，不是发布模型失败。
旧 pickle 的 GELU 兼容问题通过重建**未改动的官方 FAN**并严格加载完全相同 state 解决。
WaveMLP 未发现许可证，代码仅本地运行、未复制入仓库。

新增阶段06已开始官方/统一配方的源域训练及内部源RX选模；公平seed2023/2024/2025最终重训必须等待源域决策冻结提交。当前尚无本方法WiSig目标域成绩。
因此不能宣称本方法超过 89.19 / 93.57。证据、每折指标、上游 SHA、来源限制见 [05 RESULTS](05-wisig/RESULTS.md) 和 [文献报告](docs/LITERATURE_2025_2026.md)。

## 8. 验证、发布与下一步

测试覆盖 legacy core、models2 及比特回归、models3、S03 干预/状态恢复/损坏检测、WiSig 转换与 CR/CRD 区分。
S03 的 88 个保存 fit 核验了完整预算、源 CE 最低 checkpoint、协议 hash、参数、逐 RX 分母/F1/汇总、有限权重和冻结结果表 hash，见 [verification.json](03-chirp-structure/summary/verification.json)。

**2026-10-25 发布范围：** 用户最新请求“整理汇报所有结果并推送到github”授权发布，覆盖旧规格中的“不要 push”，不改变预登记或训练授权。
本次提交代码、实施规格、文献报告、全部已有指标/曲线/预测/日志和诊断，更新 `main` 与 `exp/s00-s01`。
00–02 的历史文件、冻结协议、legacy 与 guard 均保持不变。
数据集、cache、token、第三方仓库和 `model.pt` 不进入 Git。现有 Release `snapshot-2026-10-25` 只覆盖此前阶段 00–02 的 checkpoint；**新增 S03 checkpoint 保留本地，未上传到该 Release**。

**新的用户授权：** 2026-10-25进入论文竞争力验证，停止Shen新Loss/复杂结构搜索，WiSig优先，Shen近期强基线与CNN机制消融并行。原有历史结果不覆盖，既有confirm RX不重新称为盲测。
当前可成立的结论仍是“更强、更少参数的工程基线 + 重复结构负面结果与机制诊断”，不是已完成SOTA/新方法贡献。
阶段06完整注册矩阵、源域选择和最终结果另见[验证报告](06-paper-validation/RESULTS.md)；注册矩阵完成后停止并报告，不自动追加实验。
