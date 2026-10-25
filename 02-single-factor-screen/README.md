# 02 单因素筛选：在严格 IB 强基线上逐项检验四个方向

## 1. 起点与问题

阶段 01 确认了新的强基线 **s02_base = CrossAttn-TrueIB + long cosine + AWGN U[0,50) dB**：dev_val 94.74%，confirm 92.11%，58 万参数。它的两个已知弱点：
- 训练还没收敛，最佳 epoch 在 87–97/100；
- 在 b200_1 上只有 80%，而训练源中没有 B200 硬件。

本阶段只做一件事：**每个实验臂只比基线多一个因素**，按用户提出的四个方向逐项检验，测出每个因素的独立效应。不做组合；组合留到阶段 03，而且只组合在本阶段确认中独立有效的因素。

## 2. 因素、理由与事先写下的预测

预测在任何训练开始之前写下，用于事后判断哪些理解是对的。

| 方向 | 实验臂 | 改动（唯一因素） | 为什么测它 | 事先预测 |
|---|---|---|---|---|
| B chirp 结构 | `s02_meanchirp` | 先对 8 个符号做相干平均（等价于只保留 DFT 的梳状频点），再做 IB 限带 | 用户方向 B。理论上能把独立噪声降低 9 dB；但阶段 00 显示残差不是白噪声（相邻残差相关 +0.5 到 +0.67），IB 残差的跨 RX 探针有 52.5% | **harms**：说明残差/瞬态中有被模型使用的身份信息 |
| | `s02_combfft` | 只有 FFT 视图保留梳状频点，IQ 和幅相视图不变 | 检验 FFT 视图中非周期部分（噪声+符号间变化）是否有贡献 | unclear |
| | `s02_dechirp` | IB 限带后乘以理想 SF7 upchirp 的共轭 | dechirp 后每个符号变成近似单音，硬件失真（相位非线性、频率相关的幅度）以偏离单音的形式出现，可能对 CNN 更友好 | 无强先验 |
| A 带外增量 | `s02_rawoob_full` | 主路保持严格 IB；论文 1 的 OOB 路径（ratio 视图，取自全频带包）作为 cross-attn 的 K/V | 关键问题：在严格 IB 强基线之上，OOB 通过独立路径能否带来增量？阶段 01 显示原始 OOB 放在主路会破坏迁移 | unclear 或 harms |
| | `s02_rawoob_near` | 同上，K/V 只保留 62.5–200 kHz | 去掉远带噪声 | unclear |
| | `s02_rawoob_nearL` | 同上，只保留左侧近带（−100 到 −62.5 kHz） | 阶段 00：这是带外里可迁移信息最多的区域（跨 RX 探针 49%） | 如果 OOB 有增量，最可能出现在这里 |
| | `s02_cisoob` | K/V 改为 CIS 谱商中 \|f\|>62.5 kHz 的 35 行 | 谱商可以抵消接收机的静态频响（用户方向 A）；但阶段 01 中 cis_ib ≈ cis_flat，CIS 里的带外行没有显示出增益 | unclear |
| C 结构 | `s02_depth4` | mixer 从 2 层增加到 4 层 | 用户关心"层数太少"；参数从 58.0 万增加到 62.2 万 | unclear 或小幅提升 |
| | `s02_tfsplit` | 在严格 IB 输入上把时间流和频率流分开 | 阶段 01 的 Q4 只在全频带下测过 | unclear |
| D 训练 | `s02_ep200` | 训练 200 epoch（warmup + cosine） | 最佳 epoch 都在 100 轮的末尾 | 小幅提升 |
| | `s02_awgn10` | AWGN 的 SNR 范围从 U[0,50) 改为 U[10,50) | AWGN 是阶段 01 的第二关键因素；0–10 dB 的训练包可能噪声过大，近乎无法学习 | 无强先验 |
| 参照 | `s02_cis_rank16` | Shen-CIS 换成 rank-16 头（658K 参数，保留时频位置），官方配方 + AWGN | 公平的轻量 CIS 对照：阶段 01 中 GAP 版本崩溃，是因为丢掉了位置信息，而不是因为参数少 | 相对 13.4M 的 flatten 版本没有实质损失 |

实现见 [lib/xa/models2.py](../lib/xa/models2.py)。各实验臂的参数量：
- s02_base 和 B 方向的三个臂：580,008，与基线相同。
- depth4：622,246。
- rawoob 系列：625,896，即论文 1 的 OOB 路径。
- cisoob：619,816。
- tfsplit：583,294。
- cis_rank16：658,314。

单元测试（[tests/test_models2.py](../tests/test_models2.py)）验证了以下几点：
- s02_base 与阶段 01 的 TrueIB 在相同初始化下输出逐元素相同。
- 均值 chirp 等价于梳状频点投影。
- dechirp 把理想 chirp 变成 DC 单音。
- CIS 的 OOB 行数为 35。

## 3. 预登记对比（Δ = B − A，[02-3-aggregate.py](02-3-aggregate.py) 中的 `PAIRS`）

- 11 个单因素对比：每个 A/B/C/D 臂减去 s02_base。
- 效率对比：s02_cis_rank16 − cis_flat，其中 cis_flat 用阶段 01 中 (fold, seed) 相同、数据流相同的训练结果。
- 参照行（不给判定）：
  - s02_base − 阶段 01 的 xa_trueib_cos_awgn：确定性检查，预期 Δ 恰好为 0。
  - s02_base − s02_cis_rank16：系统级对比。
- 判定阈值与阶段 01 相同：
  - 探索：4 对，improves 要求 Δ 均值 ≥ +1 pp、至少 3/4 对为正、最差 RX 的 Δ ≥ −2 pp；harms 要求 Δ 均值 ≤ −1 pp 且至少 3/4 对为负。
  - 确认：6 对，同号对数要求至少 5/6。

## 4. 决策与结转规则（运行前冻结）

1. **探索**（seed 101/102，只看 dev_val）：只记录判定和 run 表的 SHA，写入 `decision_explore.json`。**不做筛选，全部实验臂进入确认。**
2. **确认**（seed 201–203，dev_val + confirm）：
   - 结转到阶段 03 的因素 = 在确认 seed 上 **dev_val 判定为 improves** 的因素。
   - confirm RX 只用于报告，**不参与任何结转决策**（从本阶段起明确这一点，避免 confirm RX 被反复用于选择）。
3. **OOB 主张：** 四个 A 方向的臂中，如果没有任何一个在确认中判为 improves，就写明"在本协议下，没有证据表明在严格 IB 强基线之上 OOB 带来增量"，并把论文的"可靠 OOB"主张降级。
4. **轻量 CIS：** 如果 rank16 − flatten 的均值 ≥ −1 pp 且最差 RX 的 Δ ≥ −2 pp，就判定"CIS 不需要 1344 万参数"。此后，CrossAttn-TrueIB 与 CIS 的效率对比必须以 rank-16 CIS 为准。

## 5. 运行

```bash
cd /workspace/crossatten && export PYTHONPATH=lib PY=/data/miniforge/envs/env-3.12/bin/python
$PY 02-single-factor-screen/02-1-make-jobs.py smoke   && $PY -m xa.launch --jobs 02-single-factor-screen/runs/jobs_smoke.jsonl --per-gpu 4
$PY 02-single-factor-screen/02-1-make-jobs.py explore && $PY -m xa.launch --jobs 02-single-factor-screen/runs/jobs_explore.jsonl --per-gpu 5
$PY 02-single-factor-screen/02-3-aggregate.py explore --freeze
$PY 02-single-factor-screen/02-1-make-jobs.py confirm && $PY -m xa.launch --jobs 02-single-factor-screen/runs/jobs_confirm.jsonl --per-gpu 5
$PY 02-single-factor-screen/02-3-aggregate.py confirm --freeze
```
