# S42 Shen 强基线补充结果

**Spec+Siamese 在与 HStem 完全相同的 P-H5×160 Native 包与 RX 划分上达到 95.89%。** 它是目前外部重实现中点估计最高的一项，高于原 Native Shen-DANN 的 94.21%。HStem 为 97.33%，两者配对区间跨零。HStem 参数约为其 0.96%。相对表现随 RX 改变。

## Native 主结果

| 模型 | RX 等权准确率 ± fit SD | Macro-F1 | 最差 RX |
|---|---:|---:|---|
| HStem | 97.333% ± 0.551 pp | 97.325% | Pluto-1，93.683% |
| Ma Spec+Siamese（论文重实现） | 95.890% ± 1.152 pp | 95.690% | RTL-2，91.517% |
| Shen-DANN | 94.208% ± 1.028 pp | 94.047% | Pluto-1，88.417% |
| Shen-CE | 93.535% ± 1.011 pp | 93.347% | Pluto-1，85.817% |

| 配对准确率差 | 均值 | 条件 RX 95% 区间 | 正向折×种子 |
|---|---:|---:|---:|
| HStem − Spec+Siamese | +1.443 pp | [−0.565, +3.353] pp | 5/6 |
| Spec+Siamese − Shen-DANN | +1.682 pp | [−0.087, +3.900] pp | 4/6 |
| Spec+Siamese − Shen-CE | +2.355 pp | [+0.282, +5.043] pp | 6/6 |

先在每个 RX 内平均三种子，再重采样十台历史留出接收机。HStem 两折平均差为 +0.59/+2.30 pp，且在七台 RTL 上均领先。Spec+Siamese 在 B200-mini-1、Pluto-1、B210-1 上领先 HStem，分别为 +2.45、+4.48、+1.02 pp。最大绝对差在 RTL-2，HStem 高 6.02 pp。这个分组是事后描述，硬件原因仍待隔离。Strict-IB 未新增 Ma 拟合。

## 训练、核验与成本

六个正式拟合均按源验证 TX CE 选择权重，源验证准确率 99.05%–99.45%。五条轨迹达到 160 轮上限，一条在 156 轮早停，合计 239,000 次更新、约 4,105 秒拟合时间。没有外部预训练、补种子或用留出 RX 选模。所有结果基于相同两折、三个种子和冻结包行。独立复算验证 60,000 条 Ma 预测及全部 30 个种子×RX 切片。S24 三方法的对照预测逐文件哈希及逐包标签/行号一致。原 S24 HStem、CE、DANN 保存为 CPU FP32 预测，Ma 在同一精度策略的 CUDA FP32 运行，TF32 关闭。记录显示训练来源访问先于报告读出。Confirm 和旧 blind 仍无读取。

Ma 重实现有 13,935,882 个参数。单独的 RTX 4060 Laptop 测时中，从 GPU 端 IQ 到 logits，B1 p50/p95 为 1.100/2.316 ms，B128 为 29.267/29.543 ms，B128 p50 对应 4,374 包/秒。S24 原计时来自另一次会话，论文分开报告。Ma 的网络规模和主要学习目标依原文，STFT 数值、padding、TDL 离散化是先登记的实现细节。[论文](https://arxiv.org/html/2512.12070v1) [协议与实现选择](PROTOCOL.zh-CN.md)

## 论文处置

Native 表增加 Spec+Siamese，HStem 正文以匹配结构对照和参数效率为主。与最强外部重实现报告 1.44 pp 点估计和跨零区间，并保留不利 RX。ASKNet 的弱源拟合与极端运行成本均按原记录报告。Triad-GAN 另有作者原协议的 15 台未见 RX 估计，异构方案约 95.76%，但同一 P-H5×160 的实现仍缺关键更新细节。历史 14→6 的 99.24% 单独注明旧协议。[其他强基线核验](METHOD_AUDIT.zh-CN.md)

[封存拟合](evidence/FIT_SEAL.json) · [机器汇总](evidence/RESULTS.json) · [逐 RX 指标](evidence/RX_METRICS.csv) · [访问顺序](evidence/ACCESS.jsonl)
