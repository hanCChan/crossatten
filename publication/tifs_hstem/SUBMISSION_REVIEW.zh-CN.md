# HStem TIFS 投稿审阅清单

| 审阅项目 | 当前交付 |
|---|---|
| 主稿 | 12 页英文 IEEEtran 双栏稿，参考文献计入页数。八张主图提供 PDF 和 SVG，LaTeX 编译无未解析的引用和越界警告。 |
| 补充材料 | 6 页英文稿。保留额外基线全组均值和复现细节、原文协议对照、因子消融、分层几何、逐 TX 与同步诊断，以及全部六组源域 PCA。 |
| 关键证据 | 同支持 GPU 中 HStem 97.335%、Long-Patch 94.500%，六组配对均领先，均值 2.835 pp，条件 95% 区间 [2.253, 3.530] pp。Native HStem 为 97.333% 准确率、97.325% Macro-F1、93.683% 最差 RX。 |
| 参数与计时 | HStem 133,690 参数，Long-Patch 138,346。相对 Native Spec+Siamese 缩减 99.04%。同 GPU 会话 B128 8,451 packets/s，为 Spec+Siamese 的 1.96 倍。B1 p50/p95 及峰值显存保留。复算了原始 B128 计时样本的中位数和吞吐。 |
| 额外对照 | Native/Strict-IB 分开。参数化、支持扩张、BN×GELU 和 WiSig 结果保留对应条件区间。WiSig 两套视图的样本数和相同内容的时间位移对照写入正文与补充材料。 |
| 冻结边界 | 本次只读模型实现、论文和冻结汇总。模型训练 0 次，封存目标波形访问 0 次。未改模型、权重、协议和实验记录。 |
| 关联工作 | TrueIB、五视图、共享 CNN 和 token 接口明确归属于关联 OOB 工作。贡献定位为渐进式 token 构造、同支持对照和效率。见 [重叠披露](PRIOR_WORK_DISCLOSURE.md)。 |
| 文献与插图 | 六篇全文完成下载、文本核阅与渲染。Wang 的 IB-RFF 按官方出版及作者材料核对元数据，全文标为未取得。见 [文献调研](TIFS_REFERENCE_REVIEW.zh-CN.md)、[图稿审阅](FIGURE_REVIEW.zh-CN.md)。 |

## 投稿文件

- [主稿](output/pdf/hstem_tifs_manuscript.pdf)、[补充材料](output/pdf/hstem_tifs_supplement.pdf)、[投稿信](output/pdf/hstem_tifs_cover_letter.pdf)、[复现说明](output/pdf/hstem_tifs_reproduction.pdf)。
- [主稿源码](main.tex)、[补充材料源码](supplement.tex)、[证据清单](EVIDENCE_SOURCES.json)、[绘图规范](FIGURE_STYLE.md)、[新旧页对照](output/pdf/hstem_tifs_page_comparison.pdf)。
- 首投 Regular Paper 上限 13 页，通常无需编辑批准的补材篇幅为 6 页。参见 [IEEE SPS 作者指南](https://signalprocessingsociety.org/publications-resources/information-authors) 与 [TIFS 期刊页面](https://signalprocessingsociety.org/publications-resources/ieee-transactions-information-forensics-and-security)。

## 提交前由作者确认

- 作者顺序、单位与通讯信息，ORCID、基金与利益声明。当前稿为匿名审阅版。以四作者及两条脚注测试版面，仍为 12 页。填充 [作者信息模板](author_metadata.example.tex) 后复核篇幅与投稿信签名。
- 关联 OOB 工作的出版或投稿状态、正式引用及可供编辑审阅的版本。
- 正式数据与代码可用性声明，特别是权重、逐包预测和波形的获得方式。
