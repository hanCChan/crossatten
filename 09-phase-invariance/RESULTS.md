# 阶段09：机制、严格消融与冲刺结论

基线 `c0aceeb`；独立分支 `exp/s09-phase-invariance-cpu-probe`。已完成156次正式训练及5次GPU冒烟测试。历史00-08保持不变。

## 严格同预算消融

ManySig，四折×三个新种子。共享权重逐项相同、分包和训练shuffle相同。固定策略沿用S08源域冻结轮数[10,6,4,13]及[EMA,raw,EMA,raw]；packet为Official原始包验证CE早停。参数为三通道90420、五通道90868。

| 表征 | 训练策略 | CRD RX准确率 | Macro-F1 | 种子SD pp | 最差RX |
|---|---|---:|---:|---:|---:|
| first | fixed | 92.36% | 91.90% | 2.36 | 79.75% |
| first | packet | 92.90% | 92.48% | 0.53 | 83.72% |
| frequency | fixed | 86.69% | 85.67% | 2.36 | 62.11% |
| null | fixed | 86.69% | 85.67% | 2.36 | 62.11% |
| null | packet | 88.51% | 87.56% | 0.49 | 69.25% |

**真实相位相对零相位的条件增益：+5.68pp，接收机聚类95%区间[+2.41, +9.12]，改善10/12个RX。**

这支持相位内容本身在固定训练策略下有正向贡献，不只是新增448个权重。区间条件于这三个已训练种子，未做多重比较校正，不能升级为普遍因果结论。

## 分离训练策略

| 对比 | CRD差值 pp | 条件RX95%区间 pp |
|---|---:|---|
| first:fixed-null:fixed | +5.68 | [+2.41, +9.12] |
| first:fixed-frequency:fixed | +5.68 | [+2.41, +9.12] |
| null:fixed-frequency:fixed | +0.00 | [+0.00, +0.00] |
| first:packet-null:packet | +4.39 | [+1.71, +7.53] |
| first:fixed-first:packet | -0.53 | [-2.30, +1.16] |
| null:fixed-null:packet | -1.82 | [-3.57, -0.06] |

2×2交互项：+1.29pp，[-0.82, +3.48]。固定策略相对packet同时改变选轮/停止机制与部分折的EMA，仍不能单独量化EMA。

## 新时延机制筛选

Both datasets，四折×两种源留RX分组×两个种子×三种相位表征，共96次40 epoch训练。去斜率与二阶差分用同一个五通道网络，候选冻结前没有读取目标RX成绩。

| 数据集 | 候选:规则 | 支持折数 | 源域平均增益 pp | 结转 |
|---|---|---:|---:|---|
| ManyRx | centered:raw_held | 1/4 | +0.23 | 未通过 |
| ManyRx | centered:ema_held | 1/4 | +0.36 | 未通过 |
| ManyRx | curvature:raw_held | 0/4 | -0.70 | 未通过 |
| ManyRx | curvature:ema_held | 0/4 | -0.45 | 未通过 |
| ManySig | centered:raw_held | 1/4 | -0.68 | 未通过 |
| ManySig | centered:ema_held | 0/4 | -1.28 | 未通过 |
| ManySig | curvature:raw_held | 1/4 | -0.24 | 未通过 |
| ManySig | curvature:ema_held | 0/4 | -0.08 | 未通过 |

**没有新时延表征通过预登记门槛，没有追加新候选目标集训练/测试。**不能因数学不变性正确而忽略TX识别损失或源RX退化。

## 强基线与SOTA边界

| 数据集 | 本轮模型 | 公平基线 | 本轮CRD | 基线CRD | 差值 pp | 条件RX95%区间 pp |
|---|---|---|---:|---:|---:|---|
| ManySig | first:fixed | IB-RFF | 92.36% | 91.09% | +1.27 | [-2.24, +4.56] |
| ManySig | first:fixed | WaveMLP | 92.36% | 91.04% | +1.32 | [-1.56, +3.91] |
| ManySig | first:fixed | FAN_source_selected | 92.36% | 88.27% | +4.09 | [+0.01, +7.95] |
| ManySig | first:packet | IB-RFF | 92.90% | 91.09% | +1.80 | [-1.78, +5.06] |
| ManySig | first:packet | WaveMLP | 92.90% | 91.04% | +1.86 | [+0.03, +3.77] |
| ManySig | first:packet | FAN_source_selected | 92.90% | 88.27% | +4.62 | [+0.25, +8.85] |

S05官方发布权重历史参考：ManySig IB-RFF93.57/WaveMLP93.13；ManyRx89.19/88.68，不与公平多种子重训混用。新核实官方ASKNet方法，尚未完成同协议重训，因此不宣称全局SOTA。确切的Liverpool2026高阶谱差分文献仍未核实，不能虚构引文或宣称二阶差分首创。

两个已登记训练策略分别报告，不能在看到目标分数后择优作为新候选。S08原93.01%与本轮固定策略92.36%使用不同种子且初始化配对方式不同，不是同种子回退比较。普通早停92.90%也属于预登记消融，不是新源域结转赢家。

## 机制与论文主张

- 相位是已有Re/Im FFT信息的确定变换，属于显式归纳偏置，不增加原始信息量。
- 容量控制：12个固定策略fit分别在CR/CRD评测，零相位与三通道共24组预测的最大logit差为0；448个零输入权重没有带来额外收益。
- 新附加通道对理想循环平移更稳定；真实IQ补零截窗、CFO和频选信道并不普遍不变。保留的Re/Im使整个网络也不具有严格时移不变性。
- 源域特征Fisher比、逐TX召回及模型扰动响应为机制诊断，不是独立验证，不证明接收机时延是剩余误差主因。
- 机制选择跨源折聚合，而各折RX角色互换，不等同于完全嵌套逐折选参；源held CE用于选模和评价也有选择乐观偏差。目标RX曾被查看，不是新盲测。
- 全部负面结果、失败尝试、配置、训练曲线、预测及本地权重哈希保留。没有目标成绩驱动的追加实验，没有新Loss或大网络。
- 计算成本在GPU空闲后测量；参数较少不等于推理更快。

完整表格：`S09_FINAL_RESULTS.md`、`SOURCE_RESULTS.md`、`MECHANISM_PROBES.md`、`DIAGNOSTICS.md`、`S09_COSTS.md`、`LITERATURE_CHECK.md`。本轮执行到此停止；不自动合并、推送或扩展实验。
