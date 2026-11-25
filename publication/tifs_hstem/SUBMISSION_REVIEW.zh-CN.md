# HStem TIFS 投稿审阅

稿件以高效跨接收机 LoRa 识别为主线。正文先给出 HStem 的分层构造和参数效率，再用同支持对照、BN×GELU 与冻结表征结果解释性能。接收机来源在协议中说明，同步和独立采集边界集中在 Discussion。

| 项目 | 审阅结论 |
|---|---|
| 核心创新 | 同输入、32×64 Token、最终名义卷积支持、中心及 MHA 下对照 HStem 与 Long-Patch，增益 2.835 pp。Long-Patch 多 3.48% 参数。流程图按信号、特征和运算逐步展开，结构图单列两条 Tokenizer 路径，并用相同坐标展示 350 点支持、256 点步长及 94 点重叠。 |
| 结构归因 | S26 同支持对照为主，S22 的参数化和归一化对照、S28 的 BN×GELU 正向交互与未达门槛的 held-source pilot 一并保留。 |
| 外部协议 | 正文主要对照为 Spec+Siamese、Shen-DANN 和 Shen-CE，保留 Native 与 Strict-IB 的统一协议和配对统计。文献表标出各自 RX 划分和训练预算；Triad-GAN 的约 95.8% 为原文 Fig. 5(b) 中 15 个未见 RX 的估算均值。14→6 的两个结果引用关联 OOB 稿。 |
| 统计与运行成本 | 新增 Accuracy–Parameters 和 Accuracy–Throughput 主图。配对条件区间统一进入结果表，保留 B1/B128 p50/p95、吞吐、峰值 PyTorch CUDA 分配和参数。S24 和 S26 的独立计时保留为历史补充。 |
| 复现质量核验 | S44 的一次 ASKNet 原协议复训在 RTL-6 Day 2–4 达到 95.48%，作者日志为 84.52%。同时核对了 Gul 优化预算及 IB-RFF 的公开 LoRa 配置和权重可用性。[S44 核验](../../research/s44_baseline_verification/RESULTS.zh-CN.md) |
| WiSig | 正文保留天然输入、受控频偏和时移三个对照及条件区间。两套历史源域视图的详情与 CFO 增强结果收在补充材料。 |
| 新 LoRa 采集 | 2026-11-25 复核工作区仍缺少带采集来源的新数据。给出 HStem、Spec+Siamese、Shen-DANN 的冻结评价及 Long-Patch 结构对照方案。[独立确认方案](INDEPENDENT_LORA_VALIDATION.zh-CN.md) |
| OOB 论文贡献边界 | 两篇工作共享语料、TrueIB、五视图和前端。OOB 用冻结路径干预诊断频带依赖，当前稿用同一普通 MHA 后端和配对训练测量 Tokenizer 构造的效果。[关联工作说明](PRIOR_WORK_DISCLOSURE.md) |

## 投稿材料

- 英文 [LaTeX 主稿](main.tex)、[补充材料](supplement.tex)、[投稿信草稿](COVER_LETTER.md)、[复现说明](REPRODUCTION.md)和[关联工作说明](PRIOR_WORK_DISCLOSURE.md)。构建脚本生成并逐页渲染四份 PDF。
- [数字来源](EVIDENCE_SOURCES.json)以 Git 提交号、路径和 SHA-256 绑定表格和主结果图。用 Python 运行 [build_tables.py](build_tables.py) 的 check 模式核对生成表。[build_figures.py](build_figures.py)从相同输入重建效率与消融图。
- 使用 [build_pdfs.py](build_pdfs.py) 和 TeX Live 构建。期刊初投主稿上限为 13 页双栏，参考文献计入主稿，补材另算。[IEEE 作者指南](https://signalprocessingsociety.org/publications-resources/information-authors) [TIFS 可复现性说明](https://signalprocessingsociety.org/publications-resources/ieee-transactions-information-forensics-and-security)

S44 是既有外部对照的质量核验。HStem、既有权重及 P-H5×160 历史评价沿用冻结版本。

## 需要作者确认

- 作者名单、排序、单位、通讯方式、ORCID、基金和利益声明。当前 PDF 留作无署名审阅版本。
- 关联 OOB 论文的实际投稿或出版状态、正式引用与可提供的版本。核实后同步更新参考文献与披露材料。
- 最终数据及代码可用性声明，包含大体量数据、权重、逐包预测和受限数据的实际访问方式。
- 独立 LoRa 采集结果若可获得，须先完成来源核验和一次冻结评价，再决定如何加入正文。当前版本保留已对齐、历史留出 RX 的研究范围。
