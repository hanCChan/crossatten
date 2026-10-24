# crossatten：Shen LoRa 跨接收机 CrossAttn v2（第二篇论文工作区）

本目录用于在**严格 0-shot 跨接收机**场景下，系统地改进论文 1 的 CrossAttn-OOB。研究路线见 [PLAN.md](PLAN.md)，协议见 [docs/PROTOCOL.md](docs/PROTOCOL.md)。

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
00-signal-audit/           阶段 00：数据与信号审计（README=方法与理由，RESULTS=结果与解读）
01-strong-baselines/       阶段 01：四个基础问题（arms.json 预登记、runs/、summary/、RESULTS）
logs/                      数据访问日志（data_access.jsonl）、运行日志
```

## 阶段索引

| 阶段 | 问题 | 方法说明 | 结果 |
|---|---|---|---|
| 00 | 数据是什么？哪些频段有可迁移的身份信息？ | [README](00-signal-audit/README.md) | [RESULTS](00-signal-audit/RESULTS.md) |
| 01 | 训练配方 / Shen-CIS 与 GAP / 近带 OOB / 时频分离 | [README](01-strong-baselines/README.md) | [RESULTS](01-strong-baselines/RESULTS.md) |

## 命名与规则

- 阶段目录 `NN-主题/`，脚本 `NN-k-动作.py`，按 k 的顺序执行。每个阶段都有 README（方法与理由）和 RESULTS（结果与讲解）。
- 运行任何训练之前，先把实验臂列表和判定规则提交到 git，作为预登记。实验在 `exp/*` 分支上进行。
- 环境：`/data/miniforge/envs/env-3.12/bin/python`（torch 2.5.0+cu124），机器为 4×A100-40GB。

```bash
cd /workspace/crossatten && export PYTHONPATH=lib PY=/data/miniforge/envs/env-3.12/bin/python
$PY tests/test_core.py
```
