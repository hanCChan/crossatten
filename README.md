# crossatten：Shen LoRa 跨接收机 CrossAttn v2（第二篇论文工作区）

本目录用于在**严格 0-shot 跨接收机**场景下，系统地改进论文 1 的 CrossAttn-OOB。研究路线见 [PLAN.md](PLAN.md)，协议见 [docs/PROTOCOL.md](docs/PROTOCOL.md)。

**全部结果：[RESULTS.md](RESULTS.md)（2026-10-25）。** [阶段06论文验证](06-paper-validation/README.md)已完成 **588/588 正式 fit**，另有69次运行检查。
分层 Stem + HSTU 在 Shen 重训为 **97.48±0.42% dev_val / 96.84±0.57% 已使用 confirm，142,392 参数**，
相对 CNN-only dev_val 提升3.10 pp，6/6正配对。S00–S03历史不覆盖，confirm不是新盲测。
WiSig W1的 ManyRx / ManySig CRD为 **83.56 / 72.33%**，落后公平 IB-RFF **88.07 / 91.09%**
与 WaveMLP **88.06 / 91.04%**；W1−W0未通过注册稳定改善门槛。
**当前收缩为 LoRa 特定证据路线，不支持跨 LoRa/WiFi 通用方法或投稿就绪结论。**
最终逐RX、F1、seed/fold波动及成本见[中文结论](06-paper-validation/summary/paper/CONCLUSION.md)和[完整审计](06-paper-validation/RESULTS.md)。

## 目录

```
configs/protocol.json      冻结的协议 P-H5x160（各 RX 角色、包的抽取规则、seed、禁止事项）
docs/PROTOCOL.md           协议说明与披露
PLAN.md                    各阶段路线、决策门槛
refs/                      论文 1 PDF、GPT 方案（原件，只读）
data/raw -> gai/data/external/shen_receiver_agnostic   原始 HDF5（软链接，未复制 30 GB）
data/cache/<rx>/           每台 dev RX 抽取的 2000 包（I/Q float32 [2000,2,8192]，共 1.8 GB，不纳入 git）
lib/xa/                    实验库：guard（防泄漏）、data、frontends、models、train、evaluate、launch
lib/legacy_rfhstu/         论文 1 的模型代码（gai@eb4ce2c 原样复制，SHA 记录见 docs/PROVENANCE.md）
tests/test_core.py         单元测试：参数量、CIS 移植、限带、AWGN、guard、跨进程 seed
RESULTS.md                 全部结果、参数/成本、负面证据、复现范围与停止点
00-signal-audit/           阶段 00：数据与信号审计（README=方法与理由，RESULTS=结果与解读）
01-strong-baselines/       阶段 01：四个基础问题（arms.json 预登记、runs/、summary/、RESULTS）
02-single-factor-screen/  阶段 02：13 臂单因素探索与确认
03-chirp-structure/       阶段 03：8 臂第一轮及确认、成本和机制诊断
05-wisig/                 WiSig M2：数据准备、源域审计、发布权重验证
06-paper-validation/      WiSig 公平重训、Shen 强基线、CNN 机制消融及成本验证
logs/                      数据访问日志（data_access.jsonl）、运行日志
```

## 阶段索引

| 阶段 | 问题 | 方法说明 | 结果 |
|---|---|---|---|
| 00 | 数据是什么？哪些频段有可迁移的身份信息？ | [README](00-signal-audit/README.md) | [RESULTS](00-signal-audit/RESULTS.md) |
| 01 | 训练配方 / Shen-CIS 与 GAP / 近带 OOB / 时频分离 | [README](01-strong-baselines/README.md) | [RESULTS](01-strong-baselines/RESULTS.md) |
| 02 | 严格 IB 强基线上的单因素筛选 | [README](02-single-factor-screen/README.md) | [确认指标](02-single-factor-screen/summary/confirm_tables.md) |
| 03 | Pre-LN / 残差 / 符号堆叠 / ifreq / 分层 stem | [README](03-chirp-structure/README.md) | [RESULTS](03-chirp-structure/RESULTS.md) |
| 04 | 原 Shen 强方法规格；FAN/IB/Wave转入06，Ma未训练 | [实施规格](docs/S03_S05_IMPLEMENTATION_SPEC.md) | [06同协议结果](06-paper-validation/RESULTS.md) |
| 05/M2 | WiSig 数据与发布权重验证；公平重训转入06 | [README](05-wisig/README.md) | [RESULTS](05-wisig/RESULTS.md) |
| 06 | WiSig 公平多 seed、Shen 强基线与结构消融，已完成 | [预登记](06-paper-validation/README.md) | [中文结论](06-paper-validation/summary/paper/CONCLUSION.md) |

## GitHub 快照与模型权重

仓库包含代码、协议、预登记、实验配置、逐次指标、训练曲线、预测结果和运行日志。
原始数据软链接 `data/raw`、`data/cache/`、`data/wisig/`、第三方仓库、token 和 Python 缓存不上传。

此前阶段 00–02 的 `model.pt` 保持原相对路径，单独打包到同仓库的 GitHub Release
`snapshot-2026-10-25`，不放入 Git 历史。附件分块的校验值见
`artifacts/SHA256SUMS`，逐个模型的校验值见 `artifacts/checkpoints.sha256`。
在仓库根目录、下载附件所在目录执行：

```bash
sha256sum -c artifacts/SHA256SUMS
cat crossatten-checkpoints-2026-10-25.tar.zst.part-* | tar --zstd -xf -
sha256sum -c artifacts/checkpoints.sha256
```

阶段03与阶段06的737个本地权重（包括阶段06 smoke）另存独立 Release
`paper-validation-2026-10-25`，不覆盖旧附件，也不进入 Git 历史。
新归档校验、范围及复原步骤见[归档说明](artifacts/PAPER_VALIDATION.md)；
逐块及逐权重清单为 `artifacts/PAPER_VALIDATION_SHA256SUMS` /
`artifacts/paper-validation-checkpoints.sha256`。

2026-10-25 本次更新保留阶段03及WiSig M2，新增完成的阶段06全部指标、预测、曲线、配置与10次中断尝试。
用户授权执行阶段06注册矩阵并发布，覆盖旧的“不要 push”约束；不修改历史协议、阶段00–03或实验判定规则。
全部588个正式fit使用同一训练代码hash，源域选择先冻结提交再评估目标RX。
注册实验完成后停止，不自动新增Loss、特征、架构或数据集实验，也不重新开放 Shen blind RX。

## 命名与规则

- 阶段目录 `NN-主题/`，脚本 `NN-k-动作.py`，按 k 的顺序执行。每个阶段都有 README（方法与理由）和 RESULTS（结果与讲解）。
- 运行任何训练之前，先把实验臂列表和判定规则提交到 git，作为预登记。实验在 `exp/*` 分支上进行。
- 环境：`/data/miniforge/envs/env-3.12/bin/python`（torch 2.5.0+cu124），机器为 4×A100-40GB。

```bash
cd /workspace/crossatten && export PYTHONPATH=lib PY=/data/miniforge/envs/env-3.12/bin/python
$PY tests/test_core.py
```
