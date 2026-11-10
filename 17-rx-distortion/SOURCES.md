# S17 技术来源与适用边界

已核对的主来源：

1. [WiSig官方数据说明](https://cores.ee.ucla.edu/downloads/datasets/wisig/)：原始/均衡两版256点前导；跨RX采集没有包级时间同步；部分子集不平衡，不同子集也可能含相同信号。本轮沿用已冻结的非均衡缓存和RX/day分区，不把跨RX样本当作同一信道、同一包的配对观测，ManySig只是协议内确认而非全新独立采集数据。
2. [Zhang等，Domain Generalization for Cross-Receiver Radio Frequency Fingerprint Identification，arXiv:2411.03636](https://arxiv.org/abs/2411.03636)：RIEI以表征解耦学习发射机相关特征。说明跨RX解耦已有先例；本轮没有复现其网络，也不宣称首次处理RX失真。
3. [Shi等，Receiver distortion compensation for communication emitter identification using GAN and RLS，Physical Communication 77，103162，2026](https://www.sciencedirect.com/science/article/abs/pii/S1874490726001710)，DOI 10.1016/j.phycom.2026.103162：出版社摘要描述GAN失真补偿与RLS残余IQ失衡修正。当前可获取摘要，出版社全文访问受限；没有据此推断其训练数据访问协议与本轮源域泛化协议等价，也未声称方法细节已完整复现。

复数线性滤波、广义线性IQ模型 `y=alpha*z+beta*conj(z)`、EVM匹配噪声和标签置乱均是通用技术。本轮新增的是一个待验证的实验组合：从训练RX的TX/day条件化差异构造有界扰动，依次经稳定性、错误关联、TX保持代理与同EVM负对照筛选。该组合的有效性及新颖性需由结果和后续文献审查支持。

幅频描述子为16段中心化log谱；相位描述子为FFT相位二阶差分的圆统计；IQ描述子为中心化复信号的伪协方差/功率。它们都不是接收机硬件参数的唯一估计量。已知前导、发射机IQ失衡、传播路径、CFO及窗口对齐仍可能影响这些统计。

HStem、first-order相位特征、MHA、数据选择、源CE选模和WiSig预处理直接沿用冻结仓库S09/S13。S15/S16仅作为已收尾证据，不改变原STOP决定。
