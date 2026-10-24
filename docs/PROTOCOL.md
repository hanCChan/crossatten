# 协议 P-H5x160（第二篇论文主协议，2026-10-25 冻结）

机器可读版本：[configs/protocol.json](../configs/protocol.json)。所有运行都会把它的 SHA256 写入各自的 `config.json`。

## 任务

闭集识别 10 个 TX；目标 RX 完全未见（0-shot），训练和模型选择都不使用目标 RX 的任何样本或标签。

## 数据与包

- 数据集：Shen et al. TMC2024 的 20-SDR LoRa 数据（CC BY-NC-SA 4.0）。采样参数为 Fs=1 MHz、BW=125 kHz、SF7，每包 8192 点，经阶段 00 实测核验。
- 每台 dev RX 的 `_train.h5` 中，每个 TX 抽 200 包，抽取规则与 rl1 的 g1_cache_v1 相同（seed 20261012），逐字节一致。
- 源 RX 上，每个 TX 的前 160 包（按行号升序）用于训练，后 40 包作为源验证集（source_val）。
- 评估 RX 使用全部 200 包/TX，即每台 RX 2000 包。

## 接收机角色

| 角色 | fold 0 | fold 1 | 用途 |
|---|---|---|---|
| source（5 台） | b210_1, rtl_1, rtl_3, rtl_6, rtl_9 | b200_mini_1, pluto_1, rtl_2, rtl_5, rtl_7 | 训练（160 包/TX）；源验证集（40 包/TX）只用于选 checkpoint、学习率调度和早停 |
| dev_val（5 台） | b200_mini_1, pluto_1, rtl_2, rtl_5, rtl_7 | b210_1, rtl_1, rtl_3, rtl_6, rtl_9 | 探索阶段在各实验臂之间比较、挑选候选；任何单次训练内部都不使用 |
| confirm（4 台，两个 fold 共用） | b200_1, n210_1, rtl_4, rtl_8 | 同左 | 本阶段的决策文件冻结后才允许读取；不参与本阶段的任何选择 |
| blind（6 台，不读取） | b200_2, b200_mini_2, b210_2, n210_2, n210_3, pluto_2 | — | 论文 1 已用作盲测（使用历史属于复用）；本协议不打开这些文件 |

- 源 RX 组与 rl1 R01/P2 的 formal 源组相同：fold0 对应 rl1 的 CV0，fold1 对应 CV1。dev_val 就是 rl1 的 outer RX，所以 rl1 的 CE/ccalign 结果可以作为同协议的参照点（模型和训练配方不同）。经核查，ccalign 只用源 batch 内的特征、标签和 RX 编号，不用任何目标标签（`p2_losses.py::ccalign`），因此它是严格 0-shot 方法，可以和本协议的结果直接比较。
- 4 台 confirm RX 是 gai 里登记的 development_receivers，rl1 从未使用过。论文 1 用过全部 14 台 dev RX，其中 b200_1 是论文 1 的一个开发 fold。

## 披露

2026-10-24 的代码审阅（在本协议冻结之前）在 b200_1 和 n210_1 上计算过信号层统计：梳状谱频段比、与 SNR 字段的相关，用了不超过 400 包。同一次审阅还对 14 台 RX 的 CFO 字段做过汇总。这些 RX 没有被用来训练或评估过任何模型。冻结之后，只有 stage=confirm 的任务可以读取它们，并且必须带有已存在的决策文件（由 `lib/xa/guard.py` 强制检查，每次访问都记录在 `logs/data_access.jsonl`）。
需要说明的是，阶段 00 提出的 200 kHz 候选边界，在 10 台非确认 RX 上同样成立，因此它不依赖这两台 RX。

## 训练和选择规则

- 只有 source_val 能影响 checkpoint 选择、学习率调度和早停，具体规则因配方而异（`lib/xa/train.py::RECIPES`）。
- 两个实验臂如果 (fold, seed) 相同，它们的数据顺序和 AWGN 噪声实现也相同（data_seed = sha256('P-H5x160|fold|seed')），因此可以直接配对比较。
- 探索阶段用 seed {101, 102}，确认阶段用 seed {201, 202, 203}。
- 统计单位是接收机。seed 只表示训练重复，包也不是独立样本单位。

## 指标

- 主指标：各 RX 等权平均的包级准确率。
- 其他指标：RX 等权的 macro-F1、最差 RX、逐 RX 准确率、seed 标准差、按 (fold, seed) 配对的差值（Δ）。
- 禁止只报告最好的 seed、最好的 RX 或最好的实验臂。

## 禁止事项

- 打开盲测 RX 的文件。
- 用 dev_val 或 confirm RX 来选 checkpoint、早停或调度学习率。
- 在同一阶段内用 confirm RX 做任何选择。
- 把 CFO/SNR 字段作为模型输入（本轮只用于诊断）。
