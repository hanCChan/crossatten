# M1-A：共轭镜像观测量的物理与数值可行性

由 `81ec476` 新建独立分支。只做来源、物理模型、合成验证和登记后的有界源训练包审计。深度网络、线性分类探针、embedding 和模型权重访问均为零；M1-B/C 未授权。

先提交[预注册](PREREGISTRATION.zh-CN.md)、代码和输入白名单，再执行合成测试；工程通过后才能解码登记的 common-fit_train 包。源有效性或物理特异性门未过即 STOP，不调整窗口、阈值、控制或估计器。

交付入口：[结果](RESULTS.zh-CN.md)、[模型](PHYSICAL_MODEL.md)、[可辨识性](IDENTIFIABILITY.md)、[来源](SOURCES.md)、[注册](REGISTRATION.json)。所有系数均为参考规范下的有效观测描述子，不是分离出的真实 TX I/Q 参数。
