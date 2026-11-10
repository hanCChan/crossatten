# S18：HStem 接收机选择性表征审计

本阶段继承 `04c6f08`，只在 `18-selective-rx/` 写入。S15 的正向归因、S16/S17/S17B 的历史 STOP 及 S11/S12 的负结果全部冻结。禁止打开真实目标 RX、日期3–4、完整原始 WiSig pickle；D0 不读留出源 RX。

公平基线、历史重叠和文献边界见 [PROTOCOL_AUDIT.zh-CN.md](PROTOCOL_AUDIT.zh-CN.md)。本实验是探索性、预先固定分析规则的研究，不保证原创性或有效性。

## D0 预先固定的实现

复用 S17B 四个端到端 HStem+普通 MHA，ManyRx、seed18101、split16001，82,426参数。每fold官方24个源RX中first6为留出源、余18个fit，日期1–2；原70/30包划分不变。源验证30%已参与历史checkpoint选择和S17B诊断，不能称从未触碰的独立确认集。D0无新增优化器步骤；没有使用S16冻结tokenizer尾部作为B0。

训练70%中每TX/RX/day cell固定seed最多取48包估计方向，不按结果重抽样。评价使用隔离30%的全部包，索引无交集；cell散度始终等权。模型看过训练估计集，因此这里研究的是既有表征；“数据隔离”指方向估计与评价隔离，不表示冻结模型未见过估计集。

四个固定观测点：第一卷积激活 `frequency.stem.2`（32通道）；中间tokenize激活 `frequency.tokenize.2`（48通道）；HStem输出 `frequency`（32×64 Tokens）；Mixer末尾 `encoder.final_norm`（32×64）。空间/token位置均值池化保留通道，研究对象限于通道子空间，不覆盖所有2048维Token位置交互。

令 $\mu_{t,r,d}$ 为cell均值，使用population covariance：

$$S_R=E_{t,d}\operatorname{Cov}_r(\mu_{t,r,d}),\qquad S_T=E_d\operatorname{Cov}_t(E_r\mu_{t,r,d}).$$

所有cell必须存在，缺失不填补、不改变RX集合。中心 $\mu$ 为训练cell均值的等权平均。解 $S_Rv=\lambda(S_T+\epsilon I)v$，$\epsilon=\max(10^{-3}\operatorname{tr}(S_R+S_T)/C,10^{-12})$；取最大3个广义特征向量做Euclidean QR得到 $U$。QR保持子空间，单列不必仍是广义特征向量。

四层均报告训练/验证RX及TX散度占比、TX/day条件化RX centroid描述性probe（随机率1/18），以及包二分、日期1/2、TX0–4/5–9、训练/源验证的子空间平均平方主角余弦。评价集拟合的方向只用来检查稳定性，不能回流到干预方向、中心、rank或alpha。RX probe用真实TX/day作诊断去中心，不能被当成无需标签的分类推理。

唯一主要干预位置在HStem输出Tokens，其他层只作描述。使用源训练估计的 $U,\mu$，在所有token位置施加：

$$\widetilde z=z-\alpha U\operatorname{diag}(g)U^\top(z-\mu),\quad \alpha=0.5.$$

D0固定 $g=\mathbf1$；对照为同rank、同alpha的高TX方向和3个预固定seed随机正交子空间（18200+10fold+draw）。只有这一强度，不搜索rank/剂量/层。模块前向不读RX标签或TX真值。冻结分类器检查RX等权Accuracy、Macro-F1、CE、真实类logit margin、最差RX及最差TX；TX真值仅用于离线评分。RX probe在变换后的训练描述子上重估centroid，不能把欧氏距离缩短直接解释成RX信息消失。

**机制限制：** 在 $\alpha<1,0\le g\le1$ 时该线性残差映射仍可逆，不能从方差降低推导互信息 $I(Z;R)$ 严格降低。这里的“瓶颈”是有限模型中的软抑制与归纳偏置；它也不识别纯RX硬件，与信道、同步等因素混合。散度比、特征选择和有界门控本身均是已有数学工具。

## 固定 STOP/GO

机器可执行阈值以 [registration.json](registration.json) 和 [s18_decide.py](s18_decide.py) 为准。以下是操作性筛选标准，不是理论保证：

- Tokens子空间包二分及训练/验证重合度≥0.50，跨日及跨TX半数≥0.35，验证RX散度占比≥0.15。
- 选择性压缩比随机平均多减少RX散度≥0.05；高TX对照比选择性多减少TX散度≥0.05。
- 随机平均CE损伤减去选择性CE损伤≥0.005，高TX对照差≥0.01；避免只有几何统计差异而无分类差异。
- 每fold Accuracy/F1最多降1pp，CE最多增加0.05，平均margin最多降原margin标准差的0.10，最差配对RX最多降2pp、TX最多降5pp。
- 至少3/4 folds同时满足稳定性与特异性，全部fold保持TX判别指标；选择性相对随机Accuracy差的RX聚类95%区间下界>0，才允许进入P1。

以独立RX身份为统计单位，同一RX跨fold先平均再做5000次bootstrap。该区间仍条件于重叠训练集和单一seed，不覆盖训练随机性；不能把包、日期或重复fold中的同一RX算成独立样本。

未通过即 `STOP_D0_MECHANISM_GATE`，不训练新模块，不追加rank/alpha搜索，不扩展ManySig或独立seed。不能用结果反复修改本次门槛。

## 有证据才执行的 P1 定义

| 臂 | 定义 | 可归因问题 |
|---|---|---|
| B0 | 原HStem+普通MHA | 配对基线 |
| B1 | 同骨干+普通中心化RBF HSIC(X,Z)，明确核/权重，单独前瞻注册 | 信息压缩移植对照；不是IB-RFF复现 |
| B2 | 训练源方向、固定g=1、alpha=.5 | 静态方向信息 |
| B3 | 同方向，g=sigmoid(theta)，3个门控参数，alpha=.5 | 学习选择是否必要 |
| B4 | 固定seed随机rank3方向，同B3参数和初始化 | 新模块/容量控制 |

首轮ManyRx4fold×1seed，B0若能严格复用同初始化、batch顺序、数据、优化器与checkpoint规则则16新增fit，否则连B0共20fit；不以冻结尾部替代端到端配对。基线Adam1e-3/wd0/batch128/FP32/最多40ep，Plateau(.1,5,min1e-6)，earlywait10，源RX等权CE选checkpoint。

P1训练实现与完整超参数须在任何新fit前另行封存，尤其HSIC核及权重、门控初值、U的估计时点（预训练或warm-start）及所有臂对称性。不能把预训练权重或新训练配方仅给候选臂后归因于门控。若从S17B checkpoint共同warm-start，则B0也必须同预算续训，原B0不可直接复用。D0通过只准入受控研究，不自动证明P1的具体训练定义已经充分确定。

P1全部checkpoint封存后一次性读held-source RX；主要候选B3相对B0平均≥+1pp、至少3/4 folds正向、最差配对RX≥−2pp、RX聚类差区间下界>0，再扩展ManySig与独立seed。ManySig平均不下降且至少3/4 folds≥−1pp；重复观察held-source后不得追分。GRL/SupCon/CCAlign/频谱裁剪不叠加；源留一RX风险配方作为以后单独消融。

每阶段报告Accuracy/F1/最差RX/最差TX/margin、参数量、FLOPs定义及CUDA实测延迟。D0成本为原模型与静态干预，未训练B1/B3/B4的成本不能伪装成实测。真实目标CRD本任务始终禁止读取。

## 复现入口

环境：`D:/miniconda/envs/crossatten4060/python.exe`；工程测试仅用合成输入：`python -m unittest discover -s 18-selective-rx -p "test_*.py" -v`。先 `python 18-selective-rx/s18_register.py` 并本地commit注册及代码，后 `python 18-selective-rx/s18_run.py`，最后 `python 18-selective-rx/s18_decide.py`。正式执行逐fold只允许一次，意外中断保留attempt，不自动重试科学结果。
