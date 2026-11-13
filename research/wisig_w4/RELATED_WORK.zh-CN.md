# W4 查新与可归因边界

核验日期2026-11-13。以下按作者/出版方一手来源核对，区分“有依据值得检验”和“已经具有原创性”。

| 一手来源 | 与本轮直接相关的已有思想 | 本轮仍需证明或不能声称的内容 |
|---|---|---|
| [GAN-RXA，作者全文](https://arxiv.org/html/2303.14312v2) | 先借助辅助TX/RX校准特征提取器，再为新TX建立分类器的两阶段接收机无关框架已存在。 | A1只是普通辅助监督预训练参照，不是首次使用更多TX；本轮没有重现其生成增强或对其作同协议排名。 |
| [IB-RFF，官方仓库](https://github.com/BeechburgPieStar/IB-RFF)；[IEEE SPS出版方介绍](https://signalprocessingsociety.org/publications-resources/ieee-transactions-information-forensics-and-security/2026/09/information) | 信息瓶颈与频率感知骨干用于减少接收机冗余，且指出盲目对齐可能损害判别性。 | 本轮继续使用已冻结HStem+MHA EMA，未复现FAN/IB；源侧试验不能宣传超越IB-RFF公开跨RX/跨日结果。 |
| [ASKNet，官方仓库](https://github.com/BeechburgPieStar/ASKNet-RFF) | 自适应频带及谱Koopman信号精炼是其方法内容；S20已按固定代码commit做过同数据模型/配方比较。 | E只重用S20原预测，不加载S10已见留出RX的模型；不同模型和配方的互补不等于识别出某个组件的因果机制。 |
| [IJCAI2025：Indirect Alignment and Relationship Preservation](https://ijcai.org/proceedings/2025/229) | 通过类别/域之间的关系一致性进行间接对齐，以避免直接特征对齐造成结构损伤。 | “保持跨域类别关系”本身不能叫原创。A2是简洁的类中心关系MSE探针，不声称完整实现该论文。 |
| [ICML2026：Geometric Rate–Distortion Invariance](https://proceedings.mlr.press/v306/liu26cc.html) | 研究跨域类条件表示的几何对齐与判别几何复杂度控制，采用Grassmann子空间与率失真框架。 | 本轮点中心距离MSE既没有该理论，也不是其Grassmann算法复现；共同动机不能当成新的物理结论。 |
| [CVPR2019：Relational Knowledge Distillation](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html) | 更早已有距离/角度的关系监督思想；归一化成对距离不是新数学原语。 | 本轮对象是不同源RX的TX类中心，而非教师/学生，但应用对象变化本身不足以构成论文级算法创新。 |

## 本轮可以回答的归因问题

1. **A1−A0**：辅助TX预训练这个完整训练方案是否有源域分类收益？由于额外辅助数据与额外优化步同时变化，该差不能单独归因于“更多TX身份监督”。本预算没有额外的纯计算匹配臂，不补画因果结论。
2. **A2−A1**：相同辅助数据、顺序、初始模型、更新步数、EMA与下游策略下，预定关系约束是否有额外收益？只有这一配对差才直接检验关系项。
3. **E的融合差**：两套现成模型的固定组合能纠正多少错误？即使平均Accuracy提高，也必须披露最差RX、模型双份推理成本和受影响类别。普通集成本身不自动构成原创机制。
4. **24TX诊断**：预训练表示是否对未参与W4拟合的TX保留有用类别结构？这仍使用有标签支持原型，RX也是预训练见过的源RX，不能替代未知RX上的原10TX分类端点。

## 无论结果正负都不改变的边界

本轮是反复开发过的一个内部源RX视图、三种配对seed；6个接收机上的条件区间不是外部独立确认。接收机、位置、信道和采集过程依旧混杂，没有识别出独立的发射机/接收机硬件响应。未知真实目标RX和D3/D4保持关闭，LoRa成果只读，既有STOP不改。若分类门失败，不依靠关系诊断、训练曲线或某个较好seed改判；若通过，也只支持这个源侧假设，不能直接宣布新硬件机制、世界SOTA或期刊级新算法成立。
