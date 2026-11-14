# W5-A：固定A1在V1的跨源视图复现

2026-11-14。用户授权参考W4后续方案执行独立思考；本轮只执行W5-A。历史基线cd597a5，独立分支research/wisig-w5-a1-confirmation。W4-A2的STOP、W1/W2/W3/P2/F2/M1结论保持不变；不访问真实目标RX、D3/D4、外层历史held6原始IQ或Shen，不改LoRa文件，不启动W5-B。

## 资格与固定数据预算

使用原C阶段已存在的V1源视图，fit12与source-selection6身份不变。V1的七台ManyTx合法fit RX中，计数元数据表明只有不含8-8的六台组合能完整保留W4原72训练TX、每TX×RX×日期32个raw包。因此选14-7、18-19、18-2、7-14、7-7、8-7；这是保留完整既有TX/包数的唯一合法组合，不按Accuracy选RX。其余包含8-8的六种组合中TX20-1支持不足，不能偷偷替换它。

72TX身份和顺序完全沿用W4，排除ManyRx/ManySig目标TX并集11台。每单元前24训练、后8辅助checkpoint验证，D1/D2，共20,736训练包与6,912验证包。其余24TX本轮不读。新六RX全部在V1 fit12，且与V1 source-selection6无交集。官方ManyTx单体文件先进行只读头部索引，seek跳过数值载荷，记录offset/shape/row白名单并证明索引读取与数值区间交集为0，之后才允许解码选定的864单元。

V1下游保持原16,173 train、6,995 checkpoint-val、11,500 source-selection包及原顺序身份。三seed与原C V1 S0/S1/S2 wd=.001匹配。原V0预训练权重见过V1留出RX13-7/8-14，禁止直接迁移或续训它们；所有新辅助模型重新随机初始化。

## 固定A1实现与拟合预算

直接复制W4 p_models.py和p_fit.py中的fit_aux函数及p_data.py数据处理。对函数AST逐项核验，保持W4 A1训练算法、72类头、初始化、sampler、CE+0×关系监控、优化器、LR、EMA、BN校准、40epoch和checkpoint规则完全相同。lambda固定0，没有新的关系训练、梯度标定、Loss或调参。共享函数入口不允许启动W4的12拟合流程。

辅助每batch8TX×6RX×4包=192、每epoch108步、40epoch4,320步。下游丢弃72类头，恢复与A0同seed的随机10类头，迁移辅助selected EMA骨干及BN，重置optimizer/scheduler/EMA，使用原C不变fit_loop：Adam1e-3/wd=.001、Plateau(.1,5,min1e-6)、40epoch、batch128、EMA .999、RX等权CE最早最小选模，5,080步。

A0只在初始化、全部角色/索引、原验证logits逐比特核验通过后复用。预计3次辅助预训练+3次下游=6次新增科学拟合、28,200更新。A0不合格则先报告原因，不自动补训练。工程使用合成数据，单独计数；正式attempt不自动重试、不加seed或改变配方。

## 预注册主端点与三门

六个新模型和三份复用A0的身份/权重/训练记录先封存并commit，再读取V1内部source-selection IQ与预测。评测代码及以下判定必须在该读出前封存：

1. 三seed平均的RX等权Accuracy：A1−A0≥+1pp；
2. 3/3配对seed的RX等权Accuracy差均>0；
3. 每RX先平均三seed，最差配对Accuracy差≥−2pp。

完整报告每seed、每RX、每TX×RX支持/错误、Macro-F1、最差逐seed×RX风险和RX条件bootstrap区间（10,000次，seed43101）。区间不是额外晋级门，也不是新独立接收机总体显著性证明。Macro-F1全10标签、零分母0，缺失支持单列。

任一门失败则STOP_W5A_A1_NOT_REPLICATED，不改变RX角色、TX名单、数据量、lambda、训练长度或选模。通过则仅W5A_PASS_CROSS_VIEW_SOURCE_REPLICATION，W5-B不自动获授权。无真实目标CRD或SOTA主张。

## 独立性和归因

V1是相对本次训练严格隔离的新源视图复现，但这些内部RX曾在原C/W2中被评估，不能称历史全新盲测。与V0相同72TX/包数/更新预算，仅合法辅助接收机和下游视图变化。A1相对A0仍同时增加辅助数据和计算；本轮不能分离TX多样性、RX多样性与额外优化的因果贡献。W5-B的A0-Extra及RX-Control必须另行确定预算/资格，本轮不运行。

交付完整资格表、数据白名单、预注册、固定实现来源、权重/预测SHA、全部结果、独立复算和GitHub提交。原始IQ和优化器中间缓存不公开，所选模型与可移植统计复算输入归档。
