# S18 D0S：在特征结果产生前修正缺失cell处理

原D0注册为 `8262eac`。首个fold在训练索引选择阶段触发 `Insufficient estimation cell support`，没有加载模型权重、提取已学习特征、计算方向或运行干预。失败attempt及原注册全部保留；原阶段状态为 `STOP_PRE_FEATURE_SUPPORT`。原实现对完整网格的假设不适合WiSig缺失支持，本文件记录单独的前瞻修正，不覆盖该失败，也不对已产生结果反复调参。

只读取源标签、RX/date与训练/验证索引的支持审计见 [SUPPORT_AUDIT.json](SUPPORT_AUDIT.json)。四fold各360个TX/RX/day理论cell，有13、15、13、8个空cell，训练和验证的空缺模式相同；非空训练cell均至少4包。若只保留完整RX，18个fit RX会分别缩为9、8、9、14个，因此本修正保留所有18个RX参与估计和分类评分，不以删除整RX或插补均值解决缺失。

修正只涉及描述子统计与条件probe的缺失支持处理。原checkpoint、seed、split、训练每cell最多48包、层位置、rank3、alpha=.5、随机方向seed、冻结分类器、分类评价、所有数值门槛与STOP规则均不变。新结果写入 `d0_support_matched/`；新代码/本说明/支持审计在任何已学习特征提取前另行注册并commit为 [registration_support.json](registration_support.json)。原注册代码逐字节保持不变。

对每个TX/date，令 $R_{t,d}$ 为源训练数据中有观测的RX集合。缺失cell不插补：

$$S_R=E_{t,d}\operatorname{Cov}_{r\in R_{t,d}}(\mu_{t,r,d}).$$

为防止TX均值因RX支持集合不同而混杂，TX散度使用成对TX的共同RX：

$$\delta_{t,t',d}=E_{r\in R_{t,d}\cap R_{t',d}}[\mu_{t,r,d}-\mu_{t',r,d}],\quad
S_T=\frac1{2DT^2}\sum_{d,t,t'}\delta_{t,t',d}\delta_{t,t',d}^{\top}.$$

各TX/date组及TX对等权，RX在各自共同支持内等权。每个TX/date至少2个观测RX，每个不同TX对/date至少2个共同RX，否则停止；不忽略无共同支持的TX对。该矩阵为PSD，完整矩形时严格还原原注册 $E_d\operatorname{Cov}_t(E_r\mu)$。缺失时它是共同支持的**成对TX散度代理**，不能说与原完整网格统计量完全同义，也不能外推到未观测cell。中心是已观测训练cell等权均值。

方向与中心仍完全来自源训练样本；评价只使用相同训练支持mask，训练/验证mask不一致即停止。probe的TX/day均值使用已观测RX等权，每RX prototype及准确率对其已观测TX/day cell等权；不把空cell当零准确率。TX分类Macro-F1沿用历史声明的10类规则，缺失类不因本修正被移除。

跨包二分、跨日、跨TX稳定性仍使用同一估计公式；不同day/TX半数的支持变化可能使重合度下降，这是该统计代理在真实支持条件下的限制，不能放宽门槛。对所有包的分类评价仍是18个fit RX等权，原冻结基线必须复现。

该修正由数据支持错误触发，未查看任何候选表现、held-source或目标结果。D0S结果仍是探索性机制筛选；不得将支持修正称为新的算法创新，也不得把几何统计变化解读为删除RX信息。若D0S未通过，停止当前固定候选，不启动训练。
