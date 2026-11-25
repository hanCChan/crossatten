# TIFS 全文与插图调研

页码按下载版本的 PDF 页序计，含机构封面和作者简介。2026-11-25 下载、提取正文和渲染全文，并从论文版面检查图表的字号层级、占栏、配色和面板组合。

## Ma 等，TIFS 2025

[arXiv v1 全文，14 页](https://arxiv.org/pdf/2512.12070v1) · [出版记录](https://doi.org/10.1109/TIFS.2025.3630316)。

- pp. 1–3 由 IoT 鉴别引出信道、接收机和标注数据三项挑战，随后给出训练环节分工。§II（p. 3）先写基带 chirp，再用串联 TX 失真、信道、RX 失真和噪声表达问题。§§III–V 将系统、Siamese 微调、对比预训练分开，§VI 先说明数据集，§§VII–VIII 再报告各自的验证。
- Fig. 1（p. 3）用同类轴的四面板配对波形和时频图。Fig. 3（p. 4）用浅色区分预训练、微调、推理，Fig. 4（p. 5）单独解释配对，Fig. 5（p. 6）在单栏紧凑层序图标注容量。
- Table I（p. 8）对齐数据集与研究问题，Figs. 9–13（pp. 10–11）将逐 RX 结果放入双栏图，其余场景用紧凑彩色柱阵列。Tables II、IV 单列计算和时间。本文借鉴先交代场景与模型口径，再给逐 RX 数据的组织，图中再加入形状编码。

## Shen 等，TIFS 2022

[arXiv 全文，12 页](https://arxiv.org/pdf/2107.02867) · [出版记录，14 页](https://doi.org/10.1109/TIFS.2022.3152404)。指定 Liverpool 下载未完整通过本地 PDF 校验，分析采用完整预印本。

- pp. 1–2 从设备接入需求落到可扩展性与信道鲁棒性。§II 给系统，§§III–IV（pp. 3–4）处理 chirp、卷积信道、同步与 CFO、STFT 和相邻帧商，§V（pp. 4–6）才说明 ResNet 和 triplet loss，§§VI–VII 说明注册、识别和验证。
- Fig. 2（p. 2）按角色分区，Fig. 6（p. 5）用单栏纵向层图标核宽和维度。Figs. 3–5（pp. 4–5）使用相同轴尺度的蓝色波形和时频图。Table IV（p. 8）把新设备、信道、Doppler、极化列为具体试验。
- 本稿先定义闭集任务和采样条件，再分出固定信号变换与可学习模块。概念信号和实测图分别标识。

## Shen 等，TMC 2024

[arXiv 全文，14 页](https://arxiv.org/pdf/2207.02999) · [出版记录，17 页](https://doi.org/10.1109/TMC.2023.3340039)。下列图号按预印本。

- §§I–II（pp. 1–3）从 TX 鉴别转入 RX 失真和训练、推断分布变化。§III（pp. 3–6）依次给采集、增广、表示、梯度反转与融合。§IV 映射到 LoRa；§§V–VI 区分受控条件和办公室。
- Fig. 2（p. 4）用双栏多面板表达训练、融合和微调，Fig. 3（p. 5）分开 TX/RX 训练头，Fig. 6（p. 7）用紧凑层图给 CNN 尺寸。Table II 和 Figs. 8–11（pp. 9–11）分别评估漂移、设备更换和预算，逐 RX 图共享轴尺度并使用斜标签。
- 本文借鉴训练和推理角色分区，给折的互换关系和完整 RX 名称。闭集实验采用每包、单 RX 推断。

## Shen 等，TIFS 2024，联邦对比学习

[作者主页 IEEE 全文，12 页](https://www.eng.auburn.edu/~szm0001/papers/tifs24.pdf) · [出版记录](https://doi.org/10.1109/TIFS.2024.3469820)。

- §§I–II（pp. 1–3）从标注不足和数据共享困难推导朴素联邦设置的缺陷。§III（pp. 3–6）定义本地优化和聚合，§IV（pp. 6–8）将协议映射到 LoRa 前导、CIS 和对比损失，§V 给训练轮次和标记预算。
- Fig. 2（p. 4）用双栏多流程图表达客户端、参数和数据去向。Figs. 4–5（p. 7）以并列单栏图分开信号表示及 CNN，层图有紧凑的核宽和池化标签。Figs. 7–10（pp. 9–10）使用按条件分组、同刻度曲线，Table I 单列客户端设备。
- 本文借鉴固定变换与学习模块的区隔，以及分别记录训练资源和推理资源。

## Wang，TIFS 2026，IB-RFF

[IEEE SPS 作者说明](https://signalprocessingsociety.org/publications-resources/ieee-transactions-information-forensics-and-security/2026/09/information) · [作者代码与书目](https://github.com/BeechburgPieStar/IB-RFF) · [DOI](https://doi.org/10.1109/TIFS.2026.3723081)。

未取得可验证的公开全文。已核对 Yu Wang、vol. 21、pp. 8008–8021，以及作者对 FAN、信息瓶颈损失和测试场景的说明。章节、公式、图号和实际版面未审阅。本文在定义中区分训练目标中的信息瓶颈和固定 125 kHz 投影 TrueIB。

## Kong 和 Chen，DeepCRF，TIFS 2025

[arXiv 全文，15 页](https://arxiv.org/pdf/2411.06925) · [出版记录](https://doi.org/10.1109/TIFS.2024.3515796)。

- §§I–II 从已有 LoS micro-CSI 过渡到复杂信道下的估计。§III（pp. 3–5）组合理想通道、硬件微结构和噪声，并定义分类。§IV 先说明模型驱动增广，再说明 CNN；§§V–VI 分开采集与评估；结尾讨论分布外条件。
- Fig. 3（p. 5）画完整两阶段训练，Fig. 4（p. 6）上下排列单栏 CNN 和局部模块放大图，颜色编码运算类别，字号统一。Fig. 7（p. 10）用共享尺度的双栏六面板组织 SNR × 信道。Tables IV–V、Fig. 9（p. 12）分开输入和增广；Fig. 10（p. 13）辅助检查特征。
- 本文采用完整架构与 MHA 放大图，以及分开的消融面板。PCA 单独注明坐标和量化解释范围。

## Li 等，PUF-Assisted RFF，TIFS 2024

[Edinburgh 接收稿，16 页含封面](https://www.pure.ed.ac.uk/ws/portalfiles/portal/432817152/PUF_Assited_RFF_in_TIFS.pdf) · [出版记录，15 页](https://doi.org/10.1109/TIFS.2024.3389570)。

- §§I–II（pp. 2–4）从同型 PA 难区分推导 RFF 增强。§III（pp. 5–7）以 AM/AM、AM/PM、负载反射及模型拟合解释信号，§IV 写 PUF 控制，§V（pp. 9–10）才定义累积分布表示和分类网络。
- Fig. 5（p. 6）分开硬件连接和照片，Fig. 8（p. 9）按 IQ、特征、网络分段。Figs. 15–18（pp. 13–14）用同坐标轴、字阶与色标汇报有线和 OTA，Table V 集中列超参数。混淆矩阵组合的高信息密度来自已观测矩阵。
- 本稿的对应做法是先定义算子，再按问题组织同轴对照。机制图使用冻结实验统计。
