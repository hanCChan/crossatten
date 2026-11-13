# W2：质量条件化相对普通门控的直接竞争

W1通过之后、任何新深度训练之前冻结本实现。W1源域留出包已用于假设发现，W2是在已使用开发集上的有界比较，不能称为新的独立确认。未开放target8、historical held6、Day3–4或Full新RX。

## 模型、配方与预算

- B0复用C阶段weight_decay=.001、V0/V1×S0/S1/S2的已封存C1 selected EMA。原recipe、样本键与共同参数初始化须通过核验；否则停止登记执行，不默默借用不匹配B0。82,426参数。
- B1/B2/N0沿用原频域五通道HStem→32×64 tokens→两层普通MHA→分类器，所有参数端到端训练。三臂均在tokens后新增68→8→64的MLP门控，1,128新增参数、83,554总参数，残差乘数1+0.5tanh(g)，初始为1。门控接收64维token本身和4维上下文。没有时域替换、删包、对齐、补偿、额外Loss或新数据。
- B1上下文是当前token的sigmoid均值、tanh RMS、tanh最大绝对值、tanh标准差；B2是同一包[20,140)的平均/最小块相关、相位一致性、块相关标准差，均无标签。B2将包级质量广播到频域token，再与token内容联合学习通道权重；不声称时间块与FFT频点一一对应。
- N0同B2，但在每个完整数据角色内按预定Sattolo循环打乱上下文，保证无自配对，训练、checkpoint_val、BN校准和source_selection均使用对应固定映射；与批大小/次序无关。原始IQ和标签不打乱。
- 18条新轨迹：3臂×2视图×3配对流。每条40epoch、batch128、Adam lr=.001 / weight_decay=.001、ReduceLROnPlateau factor=.1/patience5/min_lr1e-6、EMA=.999。每epoch只用fit_train重校准EMA副本BN，保持学生/影子/RNG不变。调度器看学生checkpoint_val RX等权CE，EMA主端点按EMA checkpoint_val RX等权CE选择。此为原C代码完整复用，保留学生selected、epoch10和final40辅助端点，不按源留RX选择epoch。
- 原初始化24101+s、样本顺序24201+s、模型随机流24301；上下文打乱种子29101+s（fit_train）、29201+s（checkpoint_val）、29301+s（source_selection）。质量代理不回传梯度，前端/主干和分类器均回传。零输出门控首步内部层任务梯度为0是初始化性质，不属于冻结主干。

全部18次训练完成、状态/曲线/权重哈希提交封存后才一次性评价source_selection；工程失败留痕，不按性能追加种子/候选。原W1相关性不等同W2分类收益。

## 固定评估

主端点全包、clean、EMA selected的RX等权Accuracy。B2−B1和B2−B0均需≥1pp，RX聚类95%条件区间下界>0，两视图同向，至少2/3配对流改善，最差RX均值差≥−2pp。物理特异性还需B2−N0>0且RX条件95%下界>0。附加局部风险门：每流≥50包的TX×RX中，三流均值错误率增加不得超过10pp（相对B0/B1/N0均报告）；此门事前新增用于防止平均收益掩盖严重类别退化。

报告每RX、TX×RX、视图、随机流、Day、W1相同条件四分位分组，Macro-F1、CE、纠错/新增错误、参数与训练成本。没有可信RX型号映射则明确未知，不根据ID推断B210/N210。

名义额外AWGN条件固定20/10/0dB，噪声功率按每包当前观测IQ总功率定义、seed29401与29501分别用于V0/V1，所有臂/随机流用相同带噪IQ；这不是接收物理SNR。质量代理必须从带噪IQ重新计算，N0重新使用相同预定乱序。B0只做冻结推断，不增加训练。不用噪声结果更改clean主判定，不选最好SNR汇报。高一致性子集不替代总体Accuracy。

满足全部门才值得进一步验证；否则本实现STOP，保留普通门控和真实质量两者的正/负结果，不调参或增加GRL、SupCon、KD。
