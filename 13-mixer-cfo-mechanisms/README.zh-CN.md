# S13：LoRa Mixer 对照与 WiSig CFO 机制诊断

工作目录：`D:\crossatten\stage_1\13-mixer-cfo-mechanisms`。
分支：`exp/s13-mixer-cfo`，继承 S12 的 `f3c9729`。
用户原始方案原样保存在 [PROPOSAL_ORIGINAL.txt](PROPOSAL_ORIGINAL.txt)。

## 执行内容和顺序

| 步骤 | 任务 | 预算 | 允许的数据 |
|---|---|---:|---|
| 工程检查 | 参数/FLOPs/4060 实测延迟；10 个两轮冒烟 fit | 10 | 只用源域 |
| P0 | STF/CFO 增量误差、层级 TX/RX 探针、归一化位移比 rho、BN/权重敏感性 | 8 个新诊断 fit，最多40轮 | 各折 WiSig 源 RX，第1–2天 |
| P1 | 8 种 Mixer，2折×3种子，共48 fit | 每 fit 200轮 | Shen P-H5x160；只按源验证 CE 选模 |
| P2 | P0 估计器门槛通过后登记并运行 CFO 候选 | 最多40个 pilot fit | 同 P0；模拟留出源 RX |
| P2 独立确认 | pilot 联合门槛通过后登记；另一个留出组、新种子、新基线 | 最多64 fit | 同 P0 |

P1 的所有源域 checkpoint 完成后，先提交 SHA 清单，再统一评估 dev_val 和历史已用 confirm。始终禁止 Shen blind。
S13 没有 WiSig 目标 RX/第3–4天评估器；真实 CRD 要在源域确认成功后另行冻结最终矩阵。
根据方案第七节，暂时不启动 P3；浅层 UDA/TTA 要独立协议，不能混入严格零样本 DG。

## P1 为什么需要这些对照

所有模型：`[B,2,8192]` IQ → RMS/TrueIB → 原有五视图 → 同一 HStem → `[B,32,64]` token → 交互模块 → 同一池化/10类分类头。
HStem、位置参数、分类头、数据顺序、AWGN 实现及噪声随机流按同一 fold/seed 配对。
AdamW，lr=1e-3，wd=5e-4，5轮 warmup+200轮 cosine，batch64，AWGN 0–50dB。
每轮计算源 RX 等权验证 CE，选其最低点；所有臂按相同128评估 batch 运行。

| 实验臂 | 变化 | 总参数 |
|---|---|---:|
| A0_cnn | 无 Mixer | 100154 |
| A1_mha | 两层普通四头 Softmax MHA | 133690 |
| A1_mha_matched | 内部投影80维，仍输出64维 | 141978 |
| A2_transformer | 两层标准 Encoder，FFN256 | 200122 |
| A2_transformer_matched | FFN32 参数匹配对照 | 142330 |
| A3_rfhstu | 历史 Sigmoid RF-HSTU | 142392 |
| A4_softmax_gate | 同 QKVU、门控、位置偏置，只换 Softmax | 142392 |
| A5_silu_gate | 同上，只换 SiLU 打分 | 142392 |

匹配 Transformer 的 FFN32 是瓶颈配置，不能替代自然 FFN256 对照。
A5 只隔离打分函数，不称为原始 HSTU 完整复现。
主要回答“普通注意力能否替代 RF-HSTU”，不预设 RF-HSTU 获胜。
原始 RF-HSTU 沿用第一篇实现；S13 没有新的训练损失或训练机制，不能把 CE/预热/余弦称为原创。

## P0/P2 的设计边界

WiSig 输入为非均衡前导码 `[B,2,256]`，25MHz；使用历史 S09 的频谱一阶相位5通道和 HStem/RF-HSTU，输出 ManyRx 的10类或 ManySig的6类 logits。
每个 TX/RX/day 按历史协议最多前100包，源日期1–2；内部70/30分区，用 fit-RX 训练/选模，留出的源RX仅评估。
ManyRx 每折留6个源RX，ManySig留2个；pilot用源列表前部，确认用尾部，保持不相交。
这是新训练的机制诊断，不声称复现了缺失的旧 checkpoint。

STF 假设区间0:160，延迟20点；输出 CFO 和归一化相关置信度。
合成注入 ±5/10kHz，并加30/20/10dB噪声。估计器准入要求每个数据集/折都满足：
置信度≥0.7的包不少于70%；20dB条件下增量误差中位数≤500Hz、P95≤2000Hz。
这些门槛在读取实际诊断数据前登记。合成增量准确不等于天然 CFO 绝对准确，也不能分离 TX/RX 本振。

P2 条件登记时冻结源 RX/day CFO 中位数相对总体中位数的绝对偏差95%分位，裁剪到500–10000Hz，作为均匀随机增强半径。
它与 S07 固定 ±2000Hz 有明确区别；不把随机 CFO 增强本身称为新发明。
B0原始、B1源统计 CFO 随机化、B2校正单路、B2原始+校正双路、B2双原始支路五组都报告。
校正使用逐包 CFO，低置信度回退原始输入，无目标批量统计。
双路与双原始控制参数和初始 state 完全一致。晋级双路还必须胜过双原始控制。

联合门槛：ManyRx平均至少+1pp，至少3/4折正向；两个数据集最差RX平均变化都≥−2pp；ManySig平均≥0，至少3/4折≥−1pp。
失败就 STOP，不追加 seed、不读取目标数据补救；通过再用16011/16012与另一个留出组独立配对确认。

## 在本机查看与运行

当前使用已安装的 `D:\miniconda\envs\crossatten4060\python.exe`，不改动 `.venv`。
配置生成后要先提交代码/registration/jobs，程序会检查提交和代码 SHA。

```powershell
Set-Location D:\crossatten\stage_1
$env:PYTHONUTF8 = '1'
& 'D:\miniconda\envs\crossatten4060\python.exe' '13-mixer-cfo-mechanisms\status.py'
Get-Content '13-mixer-cfo-mechanisms\execution.log' -Tail 20
```

后台任务已启动时不要再启动一个。工作流通过文件锁串行调度，一个 fit 对应一个独立 Python 进程。
手动前台运行/从完整任务边界续跑：

```powershell
& 'D:\miniconda\envs\crossatten4060\python.exe' -X utf8 -u '13-mixer-cfo-mechanisms\execute.py'
```

计划暂停：新建 `13-mixer-cfo-mechanisms\STOP_REQUESTED` 空文件，会在当前 fit 完成后停止调度。
续跑前只删除这个标记文件。中途失败的 fit 保留全部记录并阻止自动重试，需要先定位原因、记录技术修订，不能删除失败记录挑好结果。
阶段文件夹的 `execution.pid` 保存后台主进程 PID；实际 epoch 在对应 `runs/*/history.csv` 与 `live.json`。

## 结果与证据

- `RESULTS.md` / `progress.json`：已完成数量；`status.py`：当前 fit/epoch。
- `P0_MECHANISM_REPORT.md`、`P0_LAYER_PROBES.png`、`layer_probes.csv`：机制证据；未完成前不存在。
- `P0_INTERPRETATION.zh-CN.md`：P0结论边界与探针收敛限制；`P0_CFO_SUPPLEMENT.md`为事后描述性分析，不参与训练或门槛决策。
- `P1_MIXER_RESULTS.md` 与逐 fit/逐 RX CSV：48个模型完成并冻结后统一产生。
- `costs.csv`：本机 FP32 成本，FLOPs 计数不完整覆盖 FFT/逐元素计算，不能声称全算子精确成本。
- `decision_*.json`：每阶段 STOP/GO；`registration*.json` / `jobs*.json`：事前配置。
- `runs/`：checkpoint、逐轮记录、预测与失败现场；`cache/`、`access.jsonl`：新阶段独立缓存与数据访问记录。

接收机是统计单位；不把包或训练 seed 当成独立接收机。接收机聚类 bootstrap 区间条件于已训练种子，未做多重比较校正。
confirm 是历史已用接收机，不称为全新盲测。
本地自动提交仅包含本阶段文件，身份为 `Codex <codex@local.invalid>`；不会推送 GitHub。
未完成/技术失败不能解释为科学负结果。旧 S00–S12 的 STOP、代码与报告保持原样。
