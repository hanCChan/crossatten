# HStem TIFS 投稿审阅

稿件以高效跨接收机 LoRa 识别为主线。正文先给出 HStem 的分层构造和参数效率，再用同支持对照、BN×GELU 与冻结表征结果解释性能。接收机来源在协议中说明，同步和独立采集边界集中在 Discussion。

| 项目 | 审阅结论 |
|---|---|
| 核心创新 | 同输入、32×64 Token、最终名义卷积支持、中心及 MHA 下对照 HStem 与 Long-Patch，增益 2.835 pp。Long-Patch 多 3.48% 参数。主图同时展示完整架构和四阶段与直接投影的对照。 |
| 结构归因 | S26 同支持对照为主，S22 的参数化和归一化对照、S28 的 BN×GELU 正向交互与未达门槛的 held-source pilot 一并保留。 |
| 外部协议 | Native 共享原包权限并保留原实现的处理方式。S42 新增同协议 Spec+Siamese 六拟合，95.890%，与 HStem 的配对差区间跨零。Strict-IB 约束带内信息，保留 ASKNet 延期和 Ma 的空栏以及逐 RX 差异。 |
| 统计与运行成本 | 新增 Accuracy–Parameters 和 Accuracy–Throughput 主图。配对条件区间统一进入结果表，保留 B1/B128 p50/p95、吞吐、峰值 PyTorch CUDA 分配和参数。S24 和 S26 的独立计时保留为历史补充。 |
| WiSig | 正文只保留 S29 三个预定端点及 S31 的背景句，标注历史源域两视图。 |
| 新 LoRa 采集 | 2026-11-25 复核工作区仍缺少带采集来源的新数据。给出 HStem、Spec+Siamese、Shen-DANN 的冻结评价及 Long-Patch 结构对照方案。[独立确认方案](INDEPENDENT_LORA_VALIDATION.zh-CN.md) |
| OOB 论文贡献边界 | 两篇工作共享语料、TrueIB、五视图和前端。OOB 用冻结路径干预诊断频带依赖，当前稿用同一普通 MHA 后端和配对训练测量 Tokenizer 构造的效果。[关联工作说明](PRIOR_WORK_DISCLOSURE.md) |

## 投稿材料

- 英文 [LaTeX 主稿](main.tex)、[补充材料](supplement.tex)、[投稿信草稿](COVER_LETTER.md)、[复现说明](REPRODUCTION.md)和[关联工作说明](PRIOR_WORK_DISCLOSURE.md)。构建脚本生成并逐页渲染四份 PDF。
- [数字来源](EVIDENCE_SOURCES.json)以 Git 提交号、路径和 SHA-256 绑定表格和主结果图。用 Python 运行 [build_tables.py](build_tables.py) 的 check 模式核对生成表。[build_figures.py](build_figures.py)从相同输入重建效率与消融图。
- 使用 [build_pdfs.py](build_pdfs.py) 和 TeX Live 构建。期刊初投主稿上限为 13 页双栏，参考文献计入主稿，补材另算。[IEEE 作者指南](https://signalprocessingsociety.org/publications-resources/information-authors) [TIFS 可复现性说明](https://signalprocessingsociety.org/publications-resources/ieee-transactions-information-forensics-and-security)

## 需要作者确认

- 作者名单、排序、单位、通讯方式、ORCID、基金和利益声明。当前 PDF 留作无署名审阅版本。
- 关联 OOB 论文的实际投稿或出版状态、正式引用与可提供的版本。核实后同步更新参考文献与披露材料。
- 最终数据及代码可用性声明，包含大体量数据、权重、逐包预测和受限数据的实际访问方式。
- 独立 LoRa 采集结果若可获得，须先完成来源核验和一次冻结评价，再决定如何加入正文。当前版本保留已对齐、历史留出 RX 的研究范围。
