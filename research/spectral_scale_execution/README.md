# F2 频域尺度残差实验：已完成，按预注册 STOP

本轮 12/12 次科学拟合完成，480 epoch / 60,960 updates，无科学失败或重试。先完成真实源域预检，再训练、提交全部封存，最后统一评价合法 source_selection。未读取目标数据或新增正式 CRD 评分。

selected 主端点的 source-CV RX 等权 Accuracy：F0 **85.0332%**、F1 **85.0929%**、F2 **85.1169%**、EMA E0 **86.1811%**。F2 相对 F1 仅 +0.0240 pp，相对 E0 −1.0642 pp，未通过原联合门。独立复算 8,350 项通过。

- [完整中文结果、图表与成本](RESULTS.zh-CN.md)
- [独立复算](INDEPENDENT_QA.json) · [训练封存](TRAINING_SEAL.json) · [主端点判定](DECISION.json)
- [全部 48 份预测](SOURCE_PREDICTIONS.zip) · [完整科学 alpha/梯度/顺序轨迹](SCIENTIFIC_TRACES.zip)
- [用户正式授权原文](USER_AUTHORIZATION.zh-CN.md) · [封存材料清单](SCIENCE_ARTIFACT_MANIFEST.json)

原 [设计预注册](../spectral_scale_review/PREREGISTRATION.zh-CN.md)和 [9fccd65 工程附录](EXECUTION_ADDENDUM.zh-CN.md)保持原字节。`README.zh-CN.md` 与 `AUTHORIZATION_STATUS.json` 是授权前的冻结工程快照；当前完成状态见 [SCIENCE_STATUS.json](SCIENCE_STATUS.json)。

本候选结题，不自动调 kernel、alpha、Loss、解冻骨干或启动下一候选。逐 epoch 全状态及模型权重留在本地忽略目录的完整执行包，公开归档不含原始 IQ 或模型权重。
