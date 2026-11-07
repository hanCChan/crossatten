# S13 启动审计

2026-11-08 执行 `git fetch --all`。远端最新仍为 S12 `f3c9729`，新分支从该提交派生。
以下为仓库证据审计；用户附文中的外部论文数值和创新性判断不作为本阶段实验结论。

| 阶段 | 查阅重点 | S13 如何处理 |
|---|---|---|
| S06 | `06-paper-validation/RESULTS.md`、`decision_shen.json`、`configs/protocol.json` | 冻结 Shen P-H5x160、TrueIB/HStem、200轮训练、源CE选模；新配对基线不直接减历史均值 |
| S07 | `07-wifi-osu/RESULTS.md`、`lib/xa/explore_models.py` | 已有相位/时移/CFO/信道增强，CFO固定±2kHz；P2只考察新登记的源分布半径及受控双路 |
| S08 | `08-wisig-training/RESULTS.md`、`training.py` | 沿用频谱表征训练逻辑；本阶段明确重新按RX等权源验证CE选模，不能称旧权重精确重放 |
| S09 | `09-phase-invariance/S09_FINAL_RESULTS.md`、`MECHANISM_PROBES.md`、`phase_features.py` | 冻结一阶相位；±10kHz敏感性支持机制排查，不推断天然错误由CFO导致 |
| S10 | `10-tifs-validation/RESULTS.md`、`decision_final.json` | 保留 STOP；不继续软可靠性候选，不读目标补救 |
| S11 | `11-spectral-generalization/RESULTS.md`、`decision_bn.json` | 保留频谱/BN筛选结果；P0仅做新诊断，BN只用fit训练分区重校准 |
| S12 | `12-spectral-validation/RESULTS.md`、`decision_pilot.json` | 保留 STOP_SOURCE_GATE；新门槛属于S13，不回改S12 |

结构来源：`lib/legacy_rfhstu/models.py` 为第一篇遗留 RF-HSTU；`lib/xa/models3.py` 的分层 Stem 和前端作为固定底座。
审计发现 RF-HSTU 使用 Sigmoid 打分、QKVU门控、相对位置偏置，无独立FFN；因此补普通MHA/Transformer，以及同门控下换打分函数的对照。
历史代码存在 Attention、CFO、BN 相关组件，不以字符串出现就认定已完成本次完全配对实验；S13的新矩阵和输出单独归档。

原始 Shen 数据选行沿用200包/TX，其中源训练160/验证40；核对原始文件SHA后建立新缓存。
WiSig官方pickle包含全部RX和日期，直接 `pickle.load` 会一次载入禁用IQ；本阶段实现受限NumPy元数据索引，只在合法源cell检查通过后读取对应字节跨度。
索引过程只解析形状、TX/RX/date列表和数组跨度，不 materialize 目标IQ；已用合成pickle测试验证。
原有缓存/日志不会被写入，所有数据访问日志在S13下。

Windows单卡环境已用 `mingling/check_env.py` 检查CUDA与两个模型前后向；新阶段另外有10项单元检查及真实源数据冒烟矩阵。
没有恢复历史 checkpoint，所以P0标为fresh source diagnostic fits。
实验前提交代码、配置及SHA；源选模后、目标评估前再次提交模型SHA清单。
不自动重试失败 fit，不筛选 seed；P3 暂不启动。
