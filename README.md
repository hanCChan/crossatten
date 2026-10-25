# crossatten：Shen LoRa 跨接收机 CrossAttn v2（第二篇论文工作区）

本目录用于在**严格 0-shot 跨接收机**场景下，系统地改进论文 1 的 CrossAttn-OOB。研究路线见 [PLAN.md](PLAN.md)，协议见 [docs/PROTOCOL.md](docs/PROTOCOL.md)。

**全部结果：[RESULTS.md](RESULTS.md)（2026-10-25）。** 阶段 03 第一轮已完成：
分层 stem 为 **97.48±0.41% dev_val / 96.84±0.58% confirm，142,392 参数**。
残差通道、符号堆叠均确认有害，重复结构方法贡献尚未成立。WiSig 两份数据和发布权重验证通过。
2026-10-25 已按新授权进入[阶段06论文验证](06-paper-validation/README.md)：
WiSig 176/176 源域选型与 Shen 100/100 源域比较均已冻结提交，公平多 seed 最终验证正在运行。
实时结果与完整性审计见[阶段06结果](06-paper-validation/RESULTS.md)，最终结论以完整注册矩阵为准。

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
| 04 | Shen 同协议强方法复现（未训练） | [实施规格](docs/S03_S05_IMPLEMENTATION_SPEC.md) | 尚无结果 |
| 05/M2 | WiSig 数据与发布权重验证（未重训） | [README](05-wisig/README.md) | [RESULTS](05-wisig/RESULTS.md) |
| 06 | WiSig 公平多 seed、Shen 强基线与结构消融 | [预登记](06-paper-validation/README.md) | [实时结果](06-paper-validation/RESULTS.md) |

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

新增的阶段 03 `model.pt` 保留本地，未上传到上述 Release，也不进入 Git 历史。

2026-10-25 本次更新发布阶段 03 已完成的 32 次探索、48 次确认、机制诊断和总报告，
以及 WiSig M2 的准备/验证代码与结果。发布整理本身未重新训练、未修改冻结协议或实验判定规则，
也未改动阶段 00–02 的历史文件。
用户最新发布请求覆盖实施规格中旧的“不要 push”约束，但不授权自动启动阶段 04 或 WiSig 重训。
随后用户明确授权阶段06的注册验证矩阵；该授权不扩展为新 Loss 或模型搜索，也不重新开放 Shen blind RX。

## 命名与规则

- 阶段目录 `NN-主题/`，脚本 `NN-k-动作.py`，按 k 的顺序执行。每个阶段都有 README（方法与理由）和 RESULTS（结果与讲解）。
- 运行任何训练之前，先把实验臂列表和判定规则提交到 git，作为预登记。实验在 `exp/*` 分支上进行。
- 环境：`/data/miniforge/envs/env-3.12/bin/python`（torch 2.5.0+cu124），机器为 4×A100-40GB。

```bash
cd /workspace/crossatten && export PYTHONPATH=lib PY=/data/miniforge/envs/env-3.12/bin/python
$PY tests/test_core.py
```
