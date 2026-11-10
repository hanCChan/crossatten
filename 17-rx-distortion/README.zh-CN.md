# S17：接收机相关失真机制审计与有界扰动Pilot

本阶段落实用户给出的S15/S16复盘路线。独立分支`exp/s17-rx-distortion`，工作区`D:\crossatten\worktrees\s17-rx-distortion`；继承S15 `db78148`和S16 `29a3d67`的冻结历史。S15/S16正式收尾，不再补HSTU/Qwen3训练。

## 目标与数据边界

目标是检查源RX相关的稳定分布变化是否与模型错误有关，并筛选可覆盖此变化、尽量保留TX判别能力的输入扰动。诊断不等于识别纯接收机硬件传递函数。

只读取既有S13缓存：ManyRx（必要时ManySig）的合法源RX、日期1–2。每fold源RX内first组留6个ManyRx/2个ManySig为留出源RX，其余为fit RX；Pilot与D0固定split16001划分70%训练/30%验证，以匹配冻结S16参考；确认阶段split等于独立seed。所有扰动统计只从训练70%估计，不读取留出源RX估计范围，不使用真实目标RX/date3–4，不访问Shen blind。

WiSig数据输入`[B,2,256]`，固定原RMS/FFT/first-order特征、HStem输出`[B,32,64]`、2层64维普通MHA、10/6类CE输出。完整模型ManyRx为82,426参数，端到端FP32训练。无新可训练网络结构、TTA或补偿器拟合。

## D0：一次有明确出口的机制诊断

D0先复用已冻结S16 R0_mha的4个源验证混淆矩阵作为错误关联参考，split16001完全对应。RX错误使用TX宏平均错误率，日期1–2合并，无法单独控制逐日验证错误；这是当前关联分析的额外限制。S16参考使用冻结RF前端，因此D0关联也条件于该表示与配方，不推广为所有端到端模型的机制。

若D0支持候选且工程检查通过，再训练4个ManyRx新端到端MHA基线（fold0–3，初始化seed18001，split16001，first组）用于D1与P1；随后复用为M0，不重复训练。训练协议固定Adam1e-3、batch128、最多40轮、plateau factor.1/patience5/min1e-6、连续10轮源CE未改善早停，最小源RX等权CE保存checkpoint。该顺序在读取D0数据前确定，不依据诊断结果调整阈值。

幅频：16段中心化log幅谱；相位：FFT相位二阶差分圆均值的cos/sin，降低常相位与线性相位影响；IQ：中心化复伪协方差/功率的实虚部。按TX/RX/Day构建训练cell均值（至少5包），先减去每个TX/day跨RX均值。以cell等权控制TX/day分布及包数差异，不用非配对频谱逐包比值。

单fold稳定条件全部满足：部分RX解释量eta²≥0.1、跨日RX profile相关≥0.4、偶/奇TX子集profile相关≥0.4、留出TX的RX最近中心识别率≥2×机会水平、199次TX/day块内RX标签置乱p≤0.01。至少3/4fold稳定，且RX级“profile偏离程度—源验证错误率”Spearman中位数≥0.2、至少3/4fold为正，才支持相应候选。阈值是探索出口，不代表已证明硬件因果机制；置乱p和条件区间无全流程多重比较保证。

相位统计完整审计，但由于256点非配对包的可识别性限制，本轮相位只进入合成干预，不自动转成训练补偿方法。自动训练最多幅频、IQ两类。若均无稳定错误联系，直接STOP，不强行设计增强方法。

## D1：冻结模型的合成干预与TX保持代理

只在fit RX验证30%上检查，不打开留出源RX。固定强度0.25/0.5/1、每种两个确定随机draw，不追加搜索。

幅频：抽取两个源RX相关log谱profile之差，目标逐频增益截断±3dB，平滑插值并实现为17tap复FIR。FFT长度512，满足线性卷积长度，移除已知8点群延迟，不使用256点循环卷积。额外约束冲激响应相对单位冲激的L1距离≤0.25，从而全频段幅度位于[0.75,1.25]，避免滤波零点抹除信息。

IQ：用TX/day条件化IQ profile差提供扰动方向与规模代理，`beta=delta_kappa/2`仅为小失衡、近proper输入下的近似启发，强制`|beta|≤0.05`。实际变换为`(z+beta*conj(z))/sqrt(1+|beta|²)`；不声称从该前导唯一估计真实RX beta。

相位probe：低阶平滑相位基，系数≤0.15rad×强度，使用同样17tap及L1有界FIR；仅做干预，不自动训练。

负对照包括：全局2倍增益、全局1rad相位（已知前端不变性检查），以及每包EVM严格匹配的proper复高斯噪声。全局不变性控制若任一Accuracy变化超过0.1pp则停止。强度需在每个fold的双draw均值上满足Accuracy/F1下降≤1pp、最差TX准确率下降≤5pp，且平均EVM≥0.002；从固定候选中取最大安全强度。

这些保持条件仅是冻结分类器上的TX信息代理，不能证明真实发射机信息完全不变。所有源域稳定性、错误关联及强度选择均在方法训练前提交SHA和决定。

## P1：公平的小模型训练Pilot

每个D0/D1通过的family有两臂：W为上述源统计有界扰动；N为相同逐包EVM复高斯噪声。每包训练概率0.5，两臂相同物理扰动随机流，噪声使用独立流。M0、W、N的初始化、训练数据顺序、优化器、轮数和源CE选模均一致；只改输入扰动。新M0共4次，再加最多16次方法及控制，即Pilot总计最多20次。D0无候选时新增训练数为0。

所有方法checkpoint先统一冻结，才读取first组留出源RX并计算RX等权Accuracy/Macro-F1、最差RX、源到留出差距、成本和配对条件区间。

候选W必须相对M0及同family的N均满足：ManyRx平均≥+1pp，≥3/4fold为正，最差配对RX≥−2pp，5000次RX聚类bootstrap的95%下界>0。重复跨fold出现的RX按整簇保留出现次数，控制臂不得晋升新候选。无候选通过则STOP。

## P2：通过后才执行独立源域确认

confirm1(seed18011)、confirm2(seed18012)，依次执行，改用last源RX留出组，同时ManyRx/ManySig四fold。每轮所有M0/W/N都重新端到端训练；扰动profile从该轮合法训练70%重新估计，但family、强度、物理上限和配方沿用Pilot，不能重新选择。每轮最多40 fit，条件确认总计最多80 fit。

两轮均须分别满足ManyRx准入条件，并在ManySig上均值≥0、≥3/4fold不低于−1pp、最差配对RX≥−2pp；任一轮失败立即STOP，不用后续种子救结果。ManySig与ManyRx可能共享部分采集信号，不宣称它是完全独立新数据集。通过后只标记可另行预注册最终真实目标评估，不自动打开目标。

## 工程与产物

复用`D:\miniconda\envs\crossatten4060\python.exe`及既有torch2.5.0+cu124/numpy/scipy/sklearn环境，不新增大模型依赖。当前Windows CodeIntegrity 3077/3033以签名策略拦截旧PyTorch的asmjit.dll（WinError4551），该DLL与安装RECORD相符。没有修改系统安全策略或替换DLL。环境证据见ENVIRONMENT_BLOCK.json。

NumPy独立测试先于D0真实数据，覆盖描述子不变性、TX/day去混杂、无RX效应负对照、RX/day角色、独立split和接收机聚类统计。涉及PyTorch的FIR线性边界/增益上限、IQ/幅频零强度、EVM匹配及模型不变性测试仍待运行，不能报告为通过。D0无需PyTorch可继续；如D0准入，则状态保持BLOCKED_ENVIRONMENT，直到环境恢复且完整工程测试通过，才允许D1/训练。

执行入口`s17_execute.py`，状态`STATUS.json`，逐模型`runs/*/history.csv/live.json`；D0输出`D0_fold*.json`、`D0_RESULTS.zh-CN.md`；D1输出干预JSON和固定决定；后续输出各阶段CSV、决定、最终`RESULTS.zh-CN.md`。没有科学训练自动重试，异常保留现场并STOP_TECHNICAL_REVIEW。总正式训练上限100，门槛早停时只运行已触发部分。

来源与已有工作的关系见[SOURCES.md](SOURCES.md)。当前方案是机制约束研究设计，不预先宣称新颖性或泛化收益已成立。
