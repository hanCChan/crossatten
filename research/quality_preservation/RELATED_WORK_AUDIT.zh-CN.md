# 参考方案文献核验与本轮可主张范围

核验日期：2026-11-13。以下以出版方、论文和作者仓库为依据，未用第三方摘要推断实验细节。本轮源域成绩不能与不同WiSig子集或自建LoRa数据的百分比直接比较。

|参考工作|已核实信息|对本轮定位的约束|
|---|---|---|
|[Information Bottleneck-Driven Cross-Receiver RF Fingerprinting for Physical-Layer Security，作者仓库](https://github.com/BeechburgPieStar/IB-RFF)|作者明确说明FAN、HSIC代理的信息瓶颈以及预训练/微调两阶段方案，使用WiSig和LoRa；仓库列出TIFS 2026与DOI 10.1109/TIFS.2026.3723081。|FAN加正则或通用域不变学习不能当本项目全新贡献；本轮保留EMA为强参照，不新增IB因素搜索。|
|[Robust Radio Frequency Fingerprint Identification via Multi-Scale Complex-Valued Learning and Balanced Domain Alignment](https://ieeexplore.ieee.org/abstract/document/11684799)|IEEE索引摘要确认原信号/EMD两支路、多尺度复值网络、BMMD类条件对齐。其WiSig选定12台RX组成6个域，区别于本项目ManyRx协议；出版日期2026-09-09，DOI 10.1109/TCCN.2026.3732050。|普通多尺度、类条件对齐、低信噪比和遮蔽已被研究；不能靠模块命名或跨协议跑分主张领先。|
|[RFF-TTA: Physical Information-Aware Prototype for Temporally Varying RF Fingerprinting Online Test-Time-Adaptation](https://ojs.aaai.org/index.php/AAAI/article/view/37034)|出版方索引确认题名及2026记录；本次直接访问页面/PDF遭访问控制，未重新阅读全文。|可以确认同题工作存在；本轮不将其全文机制、数据角色或无标签类中心校准的新颖性写成已逐项核验的事实。|
|[A lightweight radio frequency fingerprint identification method for complex channels and cross-hardware scenarios](https://www.sciencedirect.com/science/article/abs/pii/S1874490726001473)|出版方摘要与章节索引列出TQCD时序商、DRCI距离倒数置信度融合与知识蒸馏轻量结构，实验为自建20个LoRa TX及异构SDR。页面不同段落的性能数字不一致，本轮不引用其百分比作为比较。|“质量/置信度加权”和“轻量跨硬件”本身不足以保证原创；其任务与本轮WiSig源域机制竞争有区别。|

## 本轮物理假设的来源

[WiSig原论文](https://arxiv.org/html/2112.15363)描述25Msps捕获、能量检测及256点预处理；[L-STF官方技术说明](https://www.mathworks.com/help/wlan/ref/wlanlstf.html)给出20MHz下8μs、10个短周期。因此20点是有来源的重复候选，不是从分类结果搜索得出。能量检测的包相对坐标不保证完整前导码或统一绝对边界。

## 创新结论的限定

W1证明的是在同TX/RX/Day条件下，重复一致性与识别错误存在稳定关联；不是“低质量造成错误”的唯一因果解释。W2必须进一步超过同容量普通门控与打乱代理负对照，才支持物理代理对本实现有额外价值。即使通过，也需要证明与上述工作在机制和任务上的实质差异，并补足独立外部验证，不能直接升级为已证实的TIFS/IoT-J创新。
