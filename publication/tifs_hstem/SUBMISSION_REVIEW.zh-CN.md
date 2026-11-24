# HStem TIFS 投稿审阅

稿件以已对齐 Shen LoRa 窗口上的闭集跨接收机识别为主任务。正文把构造 RF Token 的系统效应作为主张，并给出捕获和统计范围。

| 项目 | 审阅结论 |
|---|---|
| 核心创新 | 同输入、32×64 Token、最终名义卷积支持、中心及 MHA 下对照 HStem 与 Long-Patch。Long-Patch 多 3.48% 参数。相关工作分别区分卷积嵌入、分层注意力、分段对数频谱和 LoRa 跨 RX 学习。 |
| 结构归因 | S26 同支持对照为主，S22 的参数化和归一化对照、S28 的 BN×GELU 正向交互与未达门槛的 held-source pilot 一并保留。 |
| 外部协议 | Native 共享原包权限并保留原实现的处理方式。Strict-IB 约束带内信息，记录 HStem 参考成绩的复用、ASKNet 的整组延期、IB-RFF 的适配及逐 RX 异质性。 |
| 统计与运行成本 | 正文主表和补充表使用 RX 等权指标、六个折×种子的 SD 和接收机条件区间。S24 和 S26 分别给出测量会话、B1/B128 p50/p95、B128 吞吐与参数。计数的 MAC 范围注明。 |
| WiSig | 正文只保留 S29 三个预定端点及 S31 的背景句，标注历史源域两视图。 |
| 新 LoRa 采集 | 待获得可核验的新数据。旧 Confirm、旧盲测和已有 Location 变体均按历史语境处理。见[独立确认方案](INDEPENDENT_LORA_VALIDATION.zh-CN.md)。 |

## 投稿材料

- 主稿为 10 页，补充材料为 5 页。全部页面已渲染并逐页检查。编译日志无未解析的引文、交叉引用或越界警告。S24 的 11 个已完成方法×协议单元、S26 的 30 条逐 RX 记录、S22 的 6 个消融组和 S25 的 24 个最终层读数已从证据表独立复算。
- 英文 [LaTeX 主稿](main.tex)、[补充材料](supplement.tex)、[投稿信草稿](COVER_LETTER.md)和[关联工作说明](PRIOR_WORK_DISCLOSURE.md)。
- [数字来源](EVIDENCE_SOURCES.json)以 Git 提交号、路径和 SHA-256 绑定 S24、S26、S28、S29、S31。已有 S22/S25 表和图沿用 S26 封存稿，新增结构图为 LaTeX 源。用 Python 运行 [build_tables.py](build_tables.py) 的 check 模式可复核七张生成表，Git 对象须包含已发布的相应实验分支。
- 本地构建可用 TeX Live 的 latexmk。分别编译 main.tex 和 supplement.tex，或使用 [build_pdfs.py](build_pdfs.py)。两份 PDF 单独上传。期刊初投主稿上限为 13 页双栏，参考文献计入主稿，补材另算。[IEEE 作者指南](https://signalprocessingsociety.org/publications-resources/information-authors) [TIFS 可复现性说明](https://signalprocessingsociety.org/publications-resources/ieee-transactions-information-forensics-and-security)

## 需要作者确认

- 作者名单、排序、单位、通讯方式、ORCID、基金和利益声明。当前 PDF 留作无署名审阅版本。
- 关联 OOB 论文的实际投稿或出版状态、正式引用与可提供的版本。核实后同步更新参考文献与披露材料。
- 最终数据及代码可用性声明，包含大体量数据、权重、逐包预测和受限数据的实际访问方式。
- 独立 LoRa 采集结果若可获得，须先完成来源核验和一次冻结评价，再决定如何加入正文。当前版本保留已对齐、历史留出 RX 的研究范围。

WiSig 后续的冻结 A1 实验仍按独立研究计划执行。真实方向与反向、随机方向使用同一触发集和边界算子，另用原始捕获的真实邻接样点区分截窗与填充效应。正式目标 RX 和新日期继续封存。
