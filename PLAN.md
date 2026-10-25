# 研究路线：Shen 跨接收机 CrossAttn v2（第二篇论文）

**目标：** 在严格 0-shot 跨 RX 闭集识别中，通过改进信号前处理、RF 表征、时频融合和训练机制，尽可能提升效果。参数量暂不设上限，但每个实验臂都要记录参数、FLOPs、吞吐和端到端延迟。

**原则：**
- 先探索（只看 dev_val），再确认（新 seed，加上 confirm RX）。
- 每次只改一个因素，组合必须能归因到各个因素。
- 所有尝试都要记录，失败的也写进结果。

## 四个技术方向与阶段对应

| 阶段 | 方向 | 内容 | 状态 |
|---|---|---|---|
| [00-signal-audit](00-signal-audit/) | 数据与信号 | 物理核验、梳状谱剖面、各频段信息探针、SNR/CFO 捷径、CIS 坐标、残差 | ✅ 完成，见 [RESULTS](00-signal-audit/RESULTS.md) |
| [01-strong-baselines](01-strong-baselines/) | A + C + D 的起点 | Q1 训练配方阶梯，Q2 Shen-CIS 及其 GAP 版本，Q3 频带，Q4 时频分离 token | ✅ 完成：严格 IB + AWGN 的 CrossAttn 达到 94.7/92.1（CIS 为 93.1/95.2），见 [RESULTS](01-strong-baselines/RESULTS.md) |
| [02-single-factor-screen](02-single-factor-screen/) | A/B/C/D 单因素 | 以 TrueIB+AWGN 为基线：chirp 结构（均值 chirp/梳状/dechirp）、OOB 增量（独立的原始 OOB 路径或 CIS 谱商 OOB 路径）、结构（深度、时频分离）、训练（epoch 数、AWGN 范围），外加 rank-16 CIS 轻量对照 | 进行中 |
| 03-combination | 组合 | 只组合在阶段 02 确认中判为 improves 的因素，并保留"组合减去一项"的对照；时域 IB 与 CIS 互补融合作为候选（两者最差的 RX 不同） | 待阶段 02 |
| 04-fusion-architecture | C：融合结构 | 独立时域/频域 encoder；比较 concat、gate、cross-attn；mixer 深度/宽度；分层 stem（替换 52 万参数的 patch 卷积） | 待阶段 03 |
| 05-training | D：训练机制 | RX 均衡采样、ccalign（只用源数据）、增强强度；必须先有某个具体缺陷的证据，才引入 GRL/SupCon/KD | 待阶段 04 |
| 06-confirmation | 确认 | 冻结的组合加全部单因素对照，多 seed 确认；14→6 历史协议的兼容性复核（标注为复用）；效率 Pareto 与蒸馏 | 最后 |

各阶段的顺序可以根据证据调整，但任何调整都要在本文件中写明理由。

## 决策门槛（每阶段在运行前冻结）

- **探索：** 4 对配对 fit（2 fold × 2 seed）。判为 "improves" 需同时满足：Δ 均值 ≥ +1 pp、至少 3/4 对为正、最差 RX 的 Δ ≥ −2 pp。
- **确认：** 6 对配对 fit。判定条件相同，但要求至少 5/6 对为正，并且在 confirm RX 上单独报告。
- **组合：** 只有在确认阶段判为 improves 的因素才能进入组合；组合本身也要重新确认，并且保留"组合减去某一项"的对照。

## 预期的论文贡献形态（待实验验证，现在不预设）

如果下面几条被证实：
1. 带外的身份信息集中在左侧近带，远带主要是 SNR/噪底捷径；
2. 时频分离的编码能稳定改善未知 RX 识别；
3. 均值 chirp 和残差/瞬态之间存在互补；
4. 消除接收机静态响应的差分表征能让带外信息变稳；

那么可以总结出一种"结构化谱证据提取与融合"方法。否则就如实报告负面结果，退回到强基线加机制诊断的写法。

## 可复用的已有资产

- 论文 1 的 CrossAttn 代码：从 gai@eb4ce2c 复制到 `lib/legacy_rfhstu/`，未做修改。
- Shen 的 CIS 与网络：用 PyTorch 移植（CC BY-NC-SA 4.0），已与 numpy 参考实现核对，误差小于 1e-5。
- rl1 的折与包：逐字节相同，因此 rl1 的 CE/ccalign 结果可以作为同协议参照（Strict CIS，rank16 头，Adam 3e-4，20 epoch，无 AWGN）：CE 82.67/85.45，ccalign 84.91/89.00（fold0/fold1 的 outer 准确率）。
