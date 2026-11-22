# S25：TIFS 冻结证据整合与工作稿

基于 S24 `dde0ef9`，本阶段完成只读机制分析和论文材料。**新增训练 0 次，未启动 ASK Strict-IB 或 Long-Patch；没有新的独立采集或盲测。**

- [英文工作稿 PDF](output/pdf/hstem_tifs_draft.pdf)、[LaTeX 源码](paper/main.tex)、[参考文献](paper/references.bib)
- [中文 Introduction / Contributions、限制与十项审稿问答](RESEARCH_NOTES.zh-CN.txt)
- [主张—定义—实验—结果—局限矩阵](EVIDENCE_MATRIX.json)、[GO / STOP](GO_STOP.json)
- [论文 1 重合审计](OOB_OVERLAP.json)、[独立数据状态](INDEPENDENT_DATA_STATUS.json)、[文献复现审计](LITERATURE_AUDIT.json)
- [完整四臂机制结果](MECHANISM_EXTENSION.json)、[逐 RX 结果与优势贡献](PER_RX_ANALYSIS.json)、[Long-Patch 结构审计](LONG_PATCH_AUDIT.json)

H0/P0/S1/H1 各六个冻结模型，使用每折合法 Source RX 的原有留出包，共 48,000 次特征预测。分析固定 Tokenizer、Pre-MHA、最终分类前三个接口及逐包 L2 归一化。中间层反例、弱种子和不占优接收机均保留。Source CFO/SNR 只作诊断，不作为输入；Dev 只读取已保存结果，不读取新波形或目标元数据。

最终源域特征的类间/类内比支持判别性改善；部分中间层指标不改善，不能宣称逐层去 RX、不变性或物理机制已被唯一证明。S24 Strict-IB 对 CE 的总体优势约 82% 来自 Pluto-1。没有新采集，独立泛化仍未验证。

`registration.json` 在首次新提取前提交，`analysis_extension_001.json` 在 S1/H1 与源域元数据分析前追加提交。历史成绩和源域选模信息已经暴露，本分析属于透明的探索性诊断，不是独立确认性预注册实验。

## 复现

在仓库根目录运行：

```powershell
./25-tifs-integration/build_manuscript.ps1 -RenderPages
```

该命令只使用已提交图表重新编译 PDF，无需波形、checkpoint 或逐样本特征。TeX Live 2026 提供 `latexmk`、IEEEtran 和 Poppler。默认本地路径写在脚本中，迁移环境时需修改工具路径。

完整科研复现先恢复注册中列出的相同 SHA256 checkpoint、源域缓存与原始源域元数据到注册路径，再运行 `source_geometry.py` 和 `mechanism_extension.py`。它们执行数据角色检查并追加访问日志，不应在原始证据目录重复运行以覆盖封存结果。建议复制到独立目录复现。图表重建使用 `paper_assets.py`，需要本地忽略的 `features/*.npz`；单独编译 PDF 不需要这些文件。

公开内容包含聚合 CSV、注册与哈希、代码、七张主图、六组完整配对 PCA 图、正式英文表格和工作稿。波形、权重、逐包特征不上传。历史 S15–S24 文件保持原样。最终完整性审计见 `FINAL_AUDIT.json`。

Long-Patch 审计扫描固定 504 个无数据结构候选，暴露参数匹配与秩/BN 的冲突。结论为停止自动训练；如后续确需一个六次训练的控制臂，须另行明确研究问题、冻结设计与权限。
