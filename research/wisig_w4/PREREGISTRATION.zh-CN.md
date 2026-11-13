# W4：先检验分类收益，再判断机制与创新（历史基线598ba17）

用户新方案明确授权W4-E、W4-P工程/数据验收后执行第一轮，新增神经拟合上限15。新分支research/wisig-w4-decision-pretrain；所有输出限定research/wisig_w4。既有W1/W2、W3-rank8、P2、F2、M1的STOP不改；LoRa文件只读。没有真实目标RX、日期3/4、Shen或历史外层held6原始IQ访问。W4-C不与W3-C混淆，本轮不是重开W3。

## E：旧预测上的互补性

只读取用户明确指定的S20四fold HStem+MHA/ASKNet历史预测及其训练角色/来源元数据。这里允许复核既有held-source预测和标签，不读取这些RX的原始IQ、不重新推理、不加载S10已见held-source的权重。该窄范围历史预测读取不能作为P阶段访问外层held6的许可。

先冻结输入SHA、fit/held/target身份及同源同序的生成路径，再读数值。saved预测没有显式packet-ID字段时，必须披露：依赖封存的相同source-cell SHA、同一确定性拼接加载器、相同label/RX/day顺序与occurrence重建键；绝不称它们原本包含独立采集包ID。来源或顺序无法成立则拒绝配对。

每fold、RX、TX和TX×RX报告A错B对、A对B错、共同错、两个单模型及Oracle。主Oracle余量为四fold等权平均的“RX等权Oracle Accuracy−该fold较强单模型RX等权Accuracy”，是有标签上界，不能部署；同时报告全局最强固定单模型和逐RX最强单模型参照。RX跨fold重复，另报唯一RX聚合及集中度，不把24个fold×RX当24独立RX。

主余量不足3pp即停止集成。达到3pp只执行一个事先固定的标量logit融合规则：每fold在同一fit18的已保存source-validation预测上最小化RX等权CE，T=1，alpha∈[0,1]，用凸目标导数二分64次；不网格搜索、不温度调参、不路由、不用held标签选alpha。与等权logit融合及两个单模型共同报告，不把Oracle或最好held组合当部署结果。是否允许做融合只由3pp预算门触发，alpha公式已经固定。

## P：72辅助TX普通监督与关系保持预训练

唯一源视图固定原C V0（fit12、内部source-selection6）。W3辅助六RX均须属于V0 fit12。不得把六RX直接跨fold复用；本轮没有第二视图训练。96TX按SHA256('W4-TX-v1|TX')排序，前72训练、后24仅封存后诊断；目标ManyRx/ManySig并集11TX始终排除。全部raw、D1/D2、32包/单元，沿用W3精确offset/row白名单，不假设跨RX相同行是同次空口发射。

72TX每RX/date前24包训练、后8包作辅助checkpoint验证：20,736训练包、6,912验证包。24TX的9,216包在所有正式模型封存前不解码、不做checkpoint/超参选择。它们曾参与W3观察，只是相对本次拟合未见TX，不能称历史全新盲测。

三臂：A0原随机初始化10TX EMA；A1辅助72类CE预训练→10TX；A2相同预训练+关系保持→10TX。A0优先严格复用原C V0 S0/S1/S2 wd=.001 selected EMA；必须核对全部角色、初始模型和验证logits精确重放，否则在查看本轮held结果前登记不可复用原因，最多新增3个相同配方A0；绝不按分数决定复用。

辅助骨干仍为82,426参数的10类HStem+普通MHA，换成72类头后86,456参数。A1/A2同seed初始state、同训练包、同顺序、同优化器与步数。每batch8TX×6RX×4包=192，类中心每TX/RX有4包支持；均匀覆盖72TX、6RX和两源日期，每epoch108步、40epoch共4,320步。每TX/RX的48个训练包每epoch恰好访问一次，不把重复采样伪装成新增样本。

对每RX的8类中心计算28个非对角平方距离，以本RX距离均值（+1e-8）规范化，再对15对RX的距离向量MSE等权平均得Lrel。不对齐不同RX的绝对类中心，也不建立跨RX空口配对。CE+lambda*Lrel只在辅助预训练使用，下游无新增loss。

lambda按每seed未更新初始模型、前四个固定fit训练batch的骨干梯度范数均值确定：lambda=0.1*||gradCE||/||gradRel||。只使用fit训练数据；使用复制模型及隔离RNG，原始初始权重/缓冲区不变，优化步为0。任何非有限/零梯度或退化支持拒绝验收；不调lambda或改损失补救。完整梯度与lambda在首个正式更新前封存。合成检验须证明关系损失非零梯度、共享旋转/平移/缩放一致性和塌缩可检测。真实每epoch记录类间结构，完全塌缩（类间均距/中心均平方<1e-6）则停止该科学尝试，保留失败，不换配方重试。

辅助训练Adam lr=.001、wd=.001，Plateau(.1,5,min1e-6)，40epoch无早停。scheduler看原student的辅助验证RX等权CE；checkpoint取校准后的EMA(.999)验证RX等权CE最小、严格更小、最早并列。BN校准仅aux-fit训练输入，保持student/shadow/RNG不变，A1/A2完全相同。

迁移取所选辅助EMA骨干与缓冲区，丢弃72类头；恢复与该seed A0相同的随机10类头。重置optimizer、scheduler、EMA，所有参数微调。直接调用原C冻结fit_loop：V0同样16,246 train/6,954 checkpoint_val、batch128、40epoch/5,080步、相同seed/order/stochastic流、wd=.001，精确EMA规则及选模；A0复用时验证逐比特等价。原内部source-selection11,468包在全部新模型封存提交后才评分。

新增辅助拟合6、下游6，预期12；A0若无法合法复用最多再3，总上限15。辅助新增25,920梯度步，下游30,480，预期总56,400步；每个attempt一次性守卫，无自动科学重试。工程合成优化与真实初始梯度探针单独计数，不能混入正式拟合，也不得用于选择超参。

## 主要分类门槛（均需满足）

全部A1/A2与A0身份、代码、权重、原checkpoint-val预测SHA先封存并commit，再看内部held-source分数。主要量为每seed六RX等权Accuracy、再三seed等权。A2−A1至少+1pp，3/3seed均正，最差RX配对差（先平均seed）≥−2pp；A2总体不弱于A0，且对A0最差RX差≥−2pp。逐seed最坏RX差也完整报告，不以它替换预定门槛。

报告packet Accuracy、RX等权Accuracy、Macro-F1、每TX×RX支持/错误、最差RX与条件RX聚类区间（10,000次，seed42101；同RX先平均三个seed）。只有6个反复开发过的内部RX，区间不作为独立泛化或显著性保证；门槛是探索性实用标准。

## 封存后的辅助诊断与成本

24未拟合TX只用作机制诊断：每对RX用一源日期全部32包均值构造有标签24类原型，另一源日期的另一RX作query，30有向对×两日期方向；报告A1/A2（预训练EMA）准确率、margin、规范化类距一致性和类间散布。此诊断使用支持标签，RX均是辅助训练见过的六源RX，不是目标10类Accuracy、未知RX零样本或独立24TX盲测，也不参与主门槛选模。

报告新增辅助数据、预训练/微调时间和步数、BN扫描成本、参数与batch1/128推理延迟。下游A0/A1/A2架构完全相同；不能因为数据多而称同数据SOTA。辅助TX、类别关系保持、跨RX表示学习都有先例，需与GAN-RXA、IB-RFF、ASKNet、IJCAI2025与ICML2026相关研究区别。普通关系MSE不是原创物理机制。

任一主门槛失败就停止本轮，不改划分、包数、lambda、rank、结构或添加loss；不自动进入第二视图/第二阶段。无论胜负保留全部指标、反例、权重/预测SHA和独立复核，并提交GitHub。
