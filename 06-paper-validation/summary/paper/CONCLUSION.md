# 论文竞争力验证汇报

证据时间：2026-10-25T21:28:26+08:00。预登记：`bc30820`；WiSig 源域冻结：`4babf3a`；Shen 源域冻结：`88be16b`。
状态：**全部完成**；已审计 588 个正式 fit。

## 结论

**收缩为 LoRa 特定证据路线；本轮不支持跨 LoRa/WiFi 通用方法主张。**
这是按注册实验得出的证据路线，不是新颖性审查、全领域 SOTA 或投稿就绪判定。
停止新增 Loss、复杂特征和架构搜索；本轮完成后不自动追加实验。

## WiSig 主结果

下表为四折 RX × 三 seed 的 RX 等权均值及 fit SD，CRD 为主任务。SD 包含 fold 和 seed 差异；seed/fold/包不是新增独立接收机。
W0/W1/W2 分别报告，没有在测试域选出胜者。公平重训不是发布权重结果，也不是加载 LoRa 权重迁移。

| 数据集 | 模型 | fit | CR (%) | CRD (%) | CRD 包准确率 | CRD RX Macro-F1 | 参数 |
|---|---|---:|---:|---:|---:|---:|---:|
| ManyRx | W0 | 12 | 85.76 ± 5.77 | 83.19 ± 4.82 | 83.19 | 80.86 | 72232 |
| ManyRx | W1 | 12 | 86.74 ± 6.41 | 83.56 ± 6.38 | 83.58 | 81.14 | 91128 |
| ManyRx | W2 | 12 | 85.51 ± 6.67 | 80.55 ± 5.69 | 80.54 | 77.44 | 91128 |
| ManyRx | FAN_source_selected | 12 | 90.19 ± 3.05 | 87.56 ± 3.55 | 87.55 | 85.26 | 151761 |
| ManyRx | IB-RFF | 12 | 90.22 ± 2.96 | 88.07 ± 2.85 | 88.08 | 85.86 | 151761 |
| ManyRx | WaveMLP | 12 | 90.83 ± 3.17 | 88.06 ± 2.27 | 88.05 | 85.62 | 84490/85258 |
| ManySig | W0 | 12 | 72.97 ± 9.72 | 69.70 ± 11.17 | 69.70 | 67.21 | 71972 |
| ManySig | W1 | 12 | 75.90 ± 8.63 | 72.33 ± 8.22 | 72.33 | 69.99 | 90868 |
| ManySig | W2 | 12 | 70.00 ± 10.15 | 66.50 ± 10.28 | 66.50 | 64.25 | 90868 |
| ManySig | FAN_source_selected | 12 | 89.22 ± 6.67 | 88.27 ± 5.24 | 88.27 | 87.76 | 146605 |
| ManySig | IB-RFF | 12 | 91.17 ± 5.96 | 91.09 ± 5.02 | 91.09 | 90.82 | 146605 |
| ManySig | WaveMLP | 12 | 92.73 ± 2.74 | 91.04 ± 3.57 | 91.04 | 90.41 | 83974/84742 |

W0 为普通 Stem；W1 为分层 Stem；W2 为 W1 加冻结截止点。全部固定 32×64 Token 和两层 HSTU。各折 RF 臂共用源域选出的训练配方。
竞争方法按各自内部源 RX 选配置；IB 包含匹配 FAN 初始化与 HSIC 微调两阶段。

IB 与其实际 FAN 初始化的收益见 `ib_initialization_pairs.csv`。配对是描述性补充，不改变冻结决策或增加新的选择标准。

## Shen 同协议

P-H5x160：10 TX、每折 5 台源 RX、每 TX/RX 160 训练包和 40 源验证包。每台评估 RX 为 2,000 包。确认训练为两 fold × seed201/202/203。
checkpoint 和方法配置仅按源验证 CE；dev_val 只报告。四台 confirm RX 已反复查看，不是新盲测；六台 blind RX 未读取。
当前训练器验证 batch 为 128，S03 历史结果保持冻结，不要求训练比特一致。

| 模型/冻结配置 | fit | dev_val (%) | 已使用 confirm (%) | dev_val RX Macro-F1 | 参数 |
|---|---:|---:|---:|---:|---:|
| FAN_full_official | 6 | 73.62 ± 3.12 | 75.50 ± 4.52 | 72.60 | 310481 |
| FAN_full_unified | 6 | 82.79 ± 2.74 | 82.85 ± 4.34 | 81.39 | 310481 |
| FAN_ib_unified | 6 | 89.51 ± 3.60 | 86.53 ± 3.18 | 89.22 | 310481 |
| IB-RFF_a0.004_ib_unified | 6 | 94.94 ± 1.27 | 92.21 ± 1.34 | 94.90 | 310481 |
| IB-RFF_a0.04_full_official | 6 | 73.55 ± 3.84 | 76.30 ± 4.57 | 72.57 | 310481 |
| S_base | 6 | 95.50 ± 0.90 | 93.78 ± 0.58 | 95.47 | 580136 |
| S_cnnflops | 6 | 86.45 ± 0.76 | 83.78 ± 2.46 | 86.21 | 655358 |
| S_cnnonly | 6 | 94.38 ± 0.99 | 92.99 ± 1.03 | 94.26 | 100154 |
| S_cnnparam | 6 | 86.44 ± 1.13 | 83.64 ± 0.83 | 86.31 | 138898 |
| S_hstem | 6 | 97.48 ± 0.42 | 96.84 ± 0.57 | 97.47 | 142392 |
| WaveMLP_scaled_full_official | 6 | 56.68 ± 23.56 | 55.70 ± 23.16 | 54.92 | 594442 |
| WaveMLP_scaled_full_unified | 6 | 81.02 ± 3.86 | 78.67 ± 4.88 | 79.50 | 594442 |

FAN/IB-RFF 与 WaveMLP 分别保留官方及统一配方，所列 full/ib、α 和输入适配由源 CE 冻结。没有根据目标成绩重选。额外 FAN 输入臂用于 IB 初始化，不可冒充另一独立方法。
WaveMLP 的 8192 点适配不等于复现其原论文 LoRa 数字。完整 CIS、严格 IB CIS、rank-16 CIS 与 S00–S03 结果见仓库总报告，未覆盖。

## 机制消融

| 数据集 | 任务 | 配对 | fit | Δ pp | 正配对 | 最差 RX Δ pp | 注册判定 |
|---|---|---|---:|---:|---:|---:|---:|
| ManyRx | CRD | W1-C0 | 12 | +3.98 | 10/12 | -4.33 | unclear |
| ManyRx | CRD | W1-CF | 12 | +8.59 | 12/12 | -5.83 | unclear |
| ManyRx | CRD | W1-CP | 12 | +10.34 | 12/12 | -4.03 | unclear |
| ManyRx | CRD | W1-W0 | 12 | +0.37 | 7/12 | -5.95 | unclear |
| ManyRx | CRD | W2-W1 | 12 | -3.02 | 3/12 | -11.78 | unclear |
| ManySig | CRD | W1-C0 | 12 | +3.37 | 8/12 | -5.94 | unclear |
| ManySig | CRD | W1-CF | 12 | +8.37 | 11/12 | -4.19 | unclear |
| ManySig | CRD | W1-CP | 12 | +11.05 | 12/12 | -3.75 | unclear |
| ManySig | CRD | W1-W0 | 12 | +2.63 | 8/12 | -7.42 | unclear |
| ManySig | CRD | W2-W1 | 12 | -5.83 | 2/12 | -32.94 | harms |
| Shen | confirm | S_hstem-S_base | 6 | +3.07 | 6/6 | +1.20 | improves |
| Shen | dev_val | S_hstem-S_base | 6 | +1.98 | 6/6 | +1.22 | improves |
| Shen | confirm | S_hstem-S_cnnflops | 6 | +13.07 | 6/6 | +7.65 | improves |
| Shen | dev_val | S_hstem-S_cnnflops | 6 | +11.03 | 6/6 | +7.37 | improves |
| Shen | confirm | S_hstem-S_cnnonly | 6 | +3.85 | 6/6 | +0.88 | improves |
| Shen | dev_val | S_hstem-S_cnnonly | 6 | +3.10 | 6/6 | +0.47 | improves |
| Shen | confirm | S_hstem-S_cnnparam | 6 | +13.20 | 6/6 | +8.43 | improves |
| Shen | dev_val | S_hstem-S_cnnparam | 6 | +11.04 | 6/6 | +8.25 | improves |

CNN-only 去 HSTU、保留分层 Stem 和分类头；同参数/同 FLOPs CNN 是两个分别按结构成本匹配的普通 Stem 控制，误差均小于 5%。它们不是准确率搜索得到的最佳 CNN，也不能证明所有普通 CNN 均不可替代。均值提升但未满足正配对或最差 RX 门槛仍为 unclear，不解释成确定无效。

## 计算成本

训练前在空闲 A100 上测量，FP32、设备上 IQ 到 logits。包含前处理，不含主机传输；FFT 和逐元素计算未完整计入 FLOPs。

| 结构 | 参数 | MFLOPs/包 | GPU 单包 p50/p95 ms | batch512 包/s | CPU4t p50 ms |
|---|---:|---:|---:|---:|---:|
| S_base | 580136 | 139.2 | 2.164/2.237 | 42602 | 4.34 |
| S_hstem | 142392 | 190.6 | 2.533/3.806 | 35867 | 5.02 |
| S_cnnonly | 100154 | 187.4 | 1.838/1.927 | 37895 | 4.88 |
| S_cnnparam | 138898 | 18.5 | 1.889/3.19 | 97639 | 2.6 |
| S_cnnflops | 655358 | 188.1 | 1.55/2.058 | 31735 | 6.73 |
| W0_ManyRx_official | 72232 | 7.7 | 2.015/2.836 | 242215 | 1.59 |
| W1_ManyRx_official | 91128 | 10.8 | 2.295/3.059 | 200982 | 1.87 |
| W2_ManyRx_official | 91128 | 10.8 | 2.494/3.446 | 185488 | 2.02 |

不能因 hstem 参数少便宣称推理更快。完整竞争方法成本、逐 RX、seed/fold 波动、F1 和两阶段训练时间保存在 CSV 与 `../report.json`。

## 负面证据

- WiSig W1-W0 的 CRD 稳定改善需满足原注册门槛，不能只看均值。
- W2 截止点来自早期 fold0 源域 PSD，相关 RX 在其他折可能为目标域。因此 W2 不是每折严格源域独立的预处理证据；W0/W1 不使用该截止点。
- 所有低准确率、早停、负配对以及失败/中断记录保留，不改配置重试以挑好结果。
- FAN CUDA median 有上游非确定性警告，固定 seed 不保证比特一致。
- 未完成 Ma 2025 等其他方法；不能以两个强基线对照直接宣称全领域 SOTA。

文件审计问题：0；保留失败/中断条目：10。
指标来自原始 float32 logits。float16 存档可能合并最高分；权重/预测审计核验非平分结果、标签分母及平分兼容的混淆计数，不覆盖原指标。

## 证据索引

- [完整验证表](../../RESULTS.md)、[原始审计汇总](../report.json)
- [主指标](main_metrics.csv)、[逐 RX 指标](receiver_metrics.csv)
- [注册消融](registered_ablations.csv)、[成本](measured_costs.csv)
- [精度与计算量图](accuracy_cost.png)、[导出来源 hash](evidence.json)
