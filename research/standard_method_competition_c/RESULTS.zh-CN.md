# 阶段 C：标准方法源域竞争结果

Source-CV development/selection findings on 12 observed source-selection RX and three fixed paired streams; not independent confirmation, an unseen-RX population estimate, seed-population uncertainty, or a pure objective causal effect.

主端点为 selected；epoch10/final40 为全部展示的辅助结果。24 次梯度拟合产生 36 个读出候选，C0/C1 共享学生轨迹。

| 端点 | 方法 | weight decay（按主端点 CE 选择） | RX 等权 Accuracy | Macro-F1 | CE |
|---|---|---:|---:|---:|---:|
| selected | C0 | 0.001 | 85.0332% | 0.813311 | 0.516880 |
| selected | C1 | 0.001 | 86.1811% | 0.826064 | 0.486013 |
| selected | C2 | 0.001 | 84.5601% | 0.810272 | 0.511247 |
| epoch10 | C0 | 0.001 | 85.5628% | 0.819518 | 0.609614 |
| epoch10 | C1 | 0.001 | 85.0185% | 0.813966 | 0.543945 |
| epoch10 | C2 | 0.001 | 84.2376% | 0.804743 | 0.608118 |
| final40 | C0 | 0.001 | 86.2242% | 0.826106 | 0.624765 |
| final40 | C1 | 0.001 | 86.2464% | 0.826326 | 0.615771 |
| final40 | C2 | 0.001 | 85.6147% | 0.820902 | 0.613333 |

全部 wd、视图、流、TX×RX×Day 支持与风险、类别对计数、混淆、裕量及同输入错误变化见完整 ANALYSIS.json。

| 比较（selected，各自选 wd） | Accuracy 差 / pp | RX 条件区间 / pp | 分歧率 | 开发晋级 |
|---|---:|---:|---:|---|
| C1−C0 | +1.1479 | [+0.2993, +1.9902] | 8.4628% | False |
| C2−C0 | -0.4731 | [-1.5573, +0.6016] | 10.9335% | False |

开发决策：`NO_STANDARD_METHOD_DEVELOPMENT_PROMOTION`；候选优先序：[]。

该结果用于源域方法筛选，不能作为独立 CRD 确认。未通过不证明方法完全无用；不追加种子或普通因素实验。训练成本、EMA/BN 开销与延迟见单独成本记录。
