# S19：统一RF表征与TX频谱证据学习

本阶段继承S18 `933d4a6`，独立分支 `exp/s19-tx-spectral-evidence`，只写`19-tx-spectral-evidence/`。历史S00–S18的有效证据、负结果和STOP不修改，见[historical_matrix.md](historical_matrix.md)；S18完整本地证据在[S18_VERIFIED.json](S18_VERIFIED.json)核验。

研究问题是：源训练TX margin的频段反事实影响能否为轻量频谱调制提供优于普通门控的监督。TCSE是工作名，没有原创性或有效性保证。FAN/ASKNet、频谱遮蔽与类条件监督均有先例；单纯增加门控不构成新意。

## 基线与数据隔离

首轮仅ManyRx4fold×1seed，官方每fold24个源RX，first6作为held-source，余18个fit；日期仅1–2。split16001，fit训练70%/源验证30%；10TX、256点、non-equalized、first100/TX/RX/day、逐包功率归一化。真实目标RX、日期3–4、全量原始pickle一律禁止读取。

B0严格复用S17B seed18101的4个端到端HStem+普通MHA模型（82,426参数）。统一训练种子为其data_seed：`stable_seed('S17-WiSig','ManyRx',fold,'first',18101)`。新增B1–B4均从匹配的随机主干初值训练，不用teacher权重warm-start。Adam1e-3、wd0、batch128、FP32、最多40epochs，Plateau(.1,patience5,min1e-6)、earlywait10；源RX等权CE选checkpoint与调LR，辅助证据损失不参与选模。

B0是固定源训练teacher，也为其他臂共享参照；其历史训练成本另报。teacher已用源验证选checkpoint，方向估计样本也曾被teacher训练，因此证据监督不是独立于teacher选择的因果真值。新B4只额外使用teacher在训练包上的输出，不增加held或目标数据。

所有新臂的数据、batch排列生成器、优化器和预算相同，主干state初始化SHA必须等于B0原配置，额外模块不消耗主干RNG。B3/B4同结构、同初值、同batch顺序，仅B4增加证据损失。16新增fit+4严格复用B0，共20个对照；不追加结构、band、剂量或lambda搜索。

S19开始前历史阶段已看过部分同名源留出RX与公开目标汇总，因此本轮held-source是本次候选训练/选模隔离的评估，不能称历史上从未触碰的独立接收机。全部20个checkpoint及元数据先封存并本地commit，之后一次性读取held-source。推理仅IQ；TX/RX标签只用于训练监督及离线统计。

## 统一输入约定与物理尺度

WiSig Fs25MHz，FFT默认未缩放forward+fftshift；保留原双层RMS调用及eps约定。五通道为复谱Re/RMS、Im/RMS、log1p幅度zscore、相邻复谱product Re/Im，phase amplitude有效阈值0.05，左pad1。所有调制先作用于同一个复数谱，再重新构造全部通道和相位有效mask。

新增前端通过Parseval相对RMS校正等效于IQ复谱滤波后的原归一化，避免生产推理IFFT/FFT来回；identity与历史五通道/logits逐位相同。±20%约束指归一化前实值gain在[.8,1.2]，不能宣称归一化后的每bin或RF通道仍有同样界。不把单特征通道缩放说成物理滤波。

LoRa仅做统一方案核对，未通过ManyRx门槛前不读Shen数据或训练。其合法基线为Fs1MHz/BW125kHz/SF7、8192点、TrueIB |f|≤62.5kHz、legacy5+HStem+MHA（133,690参数），不同于WiSig五通道。未来若准入，同一证据机制只能在TrueIB保留的复谱范围调制，再重建原legacy5，不复制WiSig采样率或±62.5kHz裁剪到WiSig。S03 HSTU的97.48/96.84不是MHA基线结果。

## 固定证据估计

全WiSig合法256bins分为K=8个连续子带；每band32bins，边界两侧各2bins raised-sine过渡，Nyquist周期边界同样过渡，8个mask非负且和为1。没有从历史结果再选低频区。

每fold训练70%按TX/RX/day固定seed19001+fold，每个非空cell最多取16包；空cell不插补、不删除整RX。训练估计与验证/held索引严格隔离。teacher冻结；对第j带施加 `S_j'=S*(1−0.5*w_j)`，IFFT回IQ走同一个teacher，记录正确类margin `m_y=logit_y−max_{k!=y}logit_k` 与 `d_j=max(m_y(clean)−m_y(masked),0)`。

先每TX/RX/day cell求d均值，再对可观察date等权平均，最后对每TX所有有支持源RX取线性插值Q0.25。得到q[T,8]；`bar_q=(q+eps)/(sum(q)+8eps)`，eps取`1e−6*max(mean(q),1e−6)`，全零为uniform。缺失支持、每TX的RX数和日期数完整记录；不同TX可有不同观测支持，不能称完整硬件分离。

每包每band另有两个逐包EVM匹配负对照：全谱实幅缩放(1−EVM)和全谱相位旋转2asin(EVM/2)。前者检测归一化/总能量伪迹，后者检测非频段特异的相位敏感性。记录相对保留能量、EVM、全部反事实logits、residual circular impulse在±4sample外的能量比例；遮蔽是圆周滤波，可能引入ringing。控制只描述，不根据其结果重新选择mask或扣减q。

## 固定实验臂

| 臂 | 调制器 | 新增参数 | 损失 |
|---|---|---:|---|
| B0 | 原前端 | 0 | CE；历史严格复用 |
| B1 | 输入无关的8band可学习实幅滤波 | 8 | CE |
| B2 | 8band相对log-power→MLP8/16/8（末层无bias） | 272 | CE |
| B3 | 8power+8邻频phase Re+8Im→MLP24/8/8 | 272 | CE |
| B4 | 与B3完全相同 | 272 | CE+0.1×evidence CE |

B1是一般频谱选择的机制控制，不是完整FAN/ASKNet复现；官方公平重训数字单独保留。B2/B3参数严格匹配，输入描述子与隐藏宽度不同，比较属于结构对照；B4−B3才隔离新增监督。静态/动态模块最终logits零初值，保留原谱残差。中心化band logits后`gain=1+0.2*tanh(logits−mean(logits))@masks`。

证据概率为softmax(logits)，B4辅助项为`−E sum_j bar_q[y,j] log_softmax(logits)_j`。标签y只在训练损失出现，forward只输入IQ。lambda=.1固定；不搜索loss权重、K或alpha。B0/teacher历史训练成本和额外证据25次teacher forward/包成本另计，不能只比较推理参数。

## 预先固定的联合门槛

B4唯一主要候选，不在看到held结果后改推B1/B2/B3。B4相对B0需平均≥+1pp、至少3/4fold正向、最差配对RX≥−2pp、描述性RX聚类95%区间下界>0。B4相对B3、B1、B2各需平均>0、至少3/4fold正向、同样区间下界>0；否则不能将收益归功于证据监督或声称优于一般滤波。

每个RX跨fold重复出现先平均，再按唯一RX身份5000次bootstrap，seed19191；包、日期、TX和重复fold不是独立接收机。CI条件于共享模型和单seed，不覆盖训练随机性。fold平均与RX身份平均各报，不混用它们的加权定义。

未全部通过即 `STOP_S19_SOURCE_GATE`，不继续gate/lambda/band搜索、不扩展ManySig/独立seed/Shen、不读真实目标CRD。即使B4−B0正向，B4−B3未通过也只能把结构作用当线索，不能写成证据监督有效。

只有通过，才另行封存确认阶段：固定source-last分组、至少3个独立seed；之后ManySig相同算法/主要超参数，再Shen TrueIB（blind仍需最终冻结及获准）。真实CRD始终不能参与调参，最终目标评估需要单独冻结的协议；本阶段没有自动访问真实目标的代码入口。

## 证据和复现

`registration.json`/代码/历史矩阵先commit，再`python s19_execute.py run`。失败attempt保留，不自动重试科学训练。所有输入仅只读；原始IQ、checkpoint和逐包logits本地保存，Git保存checkpoint哈希、配置、曲线、日志、paired metrics、CI、访问审计及失败记录。独立QA从保留logits重算CE/margin/F1/最差RX、从teacher logits和分组重算q，并核验访问封存顺序。
