# WiSig频域信息组织：研究决策与关闭执行的预注册

本轮完成P0/P2封存预测复核、S07/S09/S19及ASKNet静态证据审查，只保留一个频域尺度候选。**新增科学拟合0次，IQ/checkpoint/特征缓存读取0次，optimizer步骤0次。**

- [研究决策与完整证据范围](DECISION.zh-CN.md)
- [未来受控实验预注册](PREREGISTRATION.zh-CN.md) / [机器登记](REGISTRATION.json)
- [封存预测互补性复算](EVIDENCE.json) / [有限读取范围](REVIEW_SCOPE.json)
- [结构原型](models.py) / [CPU合成验收](SYNTHETIC_ACCEPTANCE.json)
- [241项设计复核](DESIGN_QA.json) / [可复核代码字节](DESIGN_CODE_BYTES.zip)

唯一候选F2在完全冻结的P0频域stem上增加细/宽尺度残差token，不改变原五通道前端、32-token主路径和分类器。F1是同参数、同最大跨度的普通频域卷积控制。两者都只训练新增3,201参数，总推断参数85,627。

未来只登记12次侧路拟合预算，复用六个P0，并使用冻结EMA作强配方参照。当前不提供科学训练入口，不因本预注册自动获得执行权限。P2及固定窗口STOP保持不变。
