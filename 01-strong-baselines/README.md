# 01 强基线：四个基础问题

本阶段不提出新方法，目的是在同一个困难协议（[P-H5x160](../docs/PROTOCOL.md)）下，先把四个基础事实测准，
再决定后续往哪个方向（相干 chirp、频域差分、新的融合结构）走。

## 1. 要回答的问题

| 编号 | 问题 | 为什么先问它 |
|---|---|---|
| Q1 | 原 CrossAttn-OOB 经过充分训练、调好学习率调度、加入 AWGN 之后，跨 RX 准确率能提高多少？ | 论文 1 的 82.32% 只训练了 4 个 epoch，没有调度，也没有增强，而 Shen-CIS 训练了 74 个 epoch 并加了 AWGN。不先补齐训练配方，就无法判断差距来自结构还是来自训练 |
| Q2 | Shen-CIS 在这个困难协议下是多少？把 Flatten-Dense 换成 GAP 后还强吗？ | Shen-CIS 的 1344 万参数里 98% 在一层 Flatten→Dense(512) 上。如果换成 27 万参数的 GAP 版本性能不降，"参数少"就不能作为本文的贡献 |
| Q3 | 近带 OOB（62.5–200 kHz）比完整 OOB 好吗？ | 阶段 00 显示 \|f\|>200 kHz 几乎测不到周期信号，这部分带外可能主要携带噪底和 SNR 信息。200 kHz 只是开发阶段的候选边界，不是已经证实的最优截止 |
| Q4 | 时间/频率混合 token 与分开的时域、频域编码相比，表现有何不同？ | 原 stem 把 FFT 频点当作一个与 IQ 时间样本并列的卷积通道，第 i 个 token 同时包含"第 i 段时间"和"第 i 段频率"，两者在物理上不对齐 |

## 2. 实验臂（arm）（[01-arms.json](01-arms.json)，在任何训练开始前登记）

**CrossAttn 训练配方阶梯（Q1）**：每一级只比上一级多加一个因素。
- `xa_p1fixed`：完全照搬论文 1 的配方（AdamW 1e-3，wd 5e-4，batch 64，训练 4 个 epoch，取最后一轮权重）。
- `xa_const`：训练轮数增加到 100，按源 RX 验证 CE 选 checkpoint。
- `xa_cos`：再加 5 epoch warmup + cosine 学习率调度。
- `xa_cos_awgn`：再加 Shen 的 AWGN 增强，SNR 服从 U[0,50) dB。

**频带（Q3）**：全部使用 long_cos 配方，且有 / 无 AWGN 两个版本都跑，用来检验"AWGN 是否会掩盖远带噪声捷径"这一交互作用。

| 臂 | 主路输入 | OOB 路 |
|---|---|---|
| `xa_cos[_awgn]` | 全频带 | 全部 OOB |
| `xa_oobnear_*` | 全频带 | 只保留 62.5<\|f\|≤200 kHz |
| `xa_lp200_*` | 整包先限带到 \|f\|≤200 kHz，再重新生成所有视图 | 因此自动只剩近带 |
| `xa_nooob_*` | 全频带 | 去掉显式 OOB 路（注意：这不是严格 IB，因为 FFT/IQ 里仍含 OOB） |
| `xa_trueib_*` | 论文 1 的 CrossAttn-TrueIB：输入限带到 \|f\|≤62.5 kHz | 无 OOB 路 |

**token 设计（Q4）**：`tf_cos[_awgn]`。
- 时域 stem 读 IQ 和幅相，频域 stem 读 FFT，两路各自形成一段 token，并且各有独立的位置编码。
- OOB 的 key/value 使用频域位置编码。
- mixer 的相对位置偏置只在同一段 token 内部按相对位置计算，跨段只学一个标量。
- 每路 stem 宽 16（原来是共享的 32），总参数量与原模型相差不到 1%。

**Shen-CIS（Q2）**：配方是 SGD 1e-3（动量 0.9），ReduceLROnPlateau(0.2, 10)，EarlyStopping(20)，最多 500 个 epoch，取停止时的权重。AWGN 版本照 Shen 的做法，验证集也加固定一次实现的噪声。
- `cis_flat`：官方结构的移植版，13,442,954 参数。
- `cis_gap`：GAP 版本，270,218 参数。
- `cis_*_noawgn`：不加 AWGN。
- `cis_ib_flat`：严格 IB 的 CIS。

参数量已由单元测试核对：
- CrossAttn-OOB 625,896，CrossAttn-TrueIB 580,008，与论文 1 完全一致。
- Shen-CIS 13,442,954。论文 1 表 III 写的是 13,510,424，多出的 67,470 正好是 RA 版本的 RX 判别头（Dense 512→128→14）。这是论文 1 的一处小笔误。

## 3. 预先登记的单因素对比（Δ = B − A，[01-3-aggregate.py](01-3-aggregate.py) 中的 `PAIRS`）

| 问题 | 对比 |
|---|---|
| Q1 | const−p1fixed，cos−const，cos_awgn−cos |
| Q2 | gap−flat（有 / 无 AWGN 各一次），awgn−noawgn（两种头各一次），ib_flat−flat；外加系统级对比 cis_flat−xa_cos_awgn（输入、模型、配方都不同，所以只列数字，不下结论） |
| Q3 | 下面每项都在有 / 无 AWGN 两种条件下各比一次：oobnear−full，lp200−full，nooob−full，trueib−nooob，trueib−full |
| Q4 | tf−mixed（有 / 无 AWGN 各一次） |

**探索阶段的判定**：2 fold × 2 seed = 4 对配对 fit。
- **improves**：Δ 均值 ≥ +1.0 pp，且至少 3/4 对为正，且最差 RX 的 Δ ≥ −2 pp。
- **harms**：Δ 均值 ≤ −1.0 pp，且至少 3/4 对为负。
- 其余一律记为 **unclear**。

**确认阶段**：2 fold × 3 个新 seed = 6 对，阈值相同，但要求至少 5/6 对同号，并在 dev_val 和 confirm 两组 RX 上分别报告。

## 4. 预先登记的结转规则（只看探索阶段 dev_val 的结果；`decide()`）

- **配方**：从 {const, cos, cos_awgn} 中取 dev_val 均值最高者。如果几者相差不到 1 pp，选组件更少的那个（const 最简单，cos_awgn 最复杂）。
- **频带**：在选定配方下比较 {full, oobnear, lp200, nooob, trueib}。只有某个候选超过 full 至少 1 pp 才替换，否则保持原设计（没有证据就不改）。
- **token**：只有 tf 相对 mixed 在选定配方下判为 improves，才采用 tfsplit。
- **CIS 头**：如果 gap−flat 的均值 ≥ −1.0 pp 且最差 RX 的 Δ ≥ −2 pp，就判为"GAP 没有实质损失"。
- 结转结果写入 `decision_explore.json`，文件内记录探索阶段 run 表的 SHA256。这个文件生成之后才能运行确认任务（guard 也会检查）。
- 确认阶段跑**全部**登记的实验臂（seed 201–203，评估 dev_val 和 confirm 两组 RX），因此确认集上不存在挑选偏差。

## 5. 运行方式

```bash
cd /workspace/crossatten && export PYTHONPATH=lib PY=/data/miniforge/envs/env-3.12/bin/python
$PY 01-strong-baselines/01-1-make-jobs.py smoke   && $PY -m xa.launch --jobs 01-strong-baselines/runs/jobs_smoke.jsonl
$PY 01-strong-baselines/01-1-make-jobs.py explore && $PY -m xa.launch --jobs 01-strong-baselines/runs/jobs_explore.jsonl --per-gpu 4
$PY 01-strong-baselines/01-3-aggregate.py explore --freeze        # 冻结 decision_explore.json
$PY 01-strong-baselines/01-1-make-jobs.py confirm && $PY -m xa.launch --jobs 01-strong-baselines/runs/jobs_confirm.jsonl --per-gpu 4
$PY 01-strong-baselines/01-3-aggregate.py confirm
$PY 01-strong-baselines/01-4-bench.py
```

结果写在 [RESULTS.md](RESULTS.md)；自动生成的表格在 `summary/`。
