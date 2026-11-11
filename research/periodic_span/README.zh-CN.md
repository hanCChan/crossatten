# 不依赖固定起点的周期跨度 HStem

本阶段按用户新建议独立推进，不重开固定窗口方案。

- [训练前冻结的完整协议](PREREGISTRATION.zh-CN.md)
- [模型](models.py)、[训练/封存/评价器](engine.py)
- `REGISTRATION.json`、`JOBS.json`、`SOURCE_PLAN.json`冻结24次源域竞争。
- 运行后通过`TRAINING_SEAL.json`及`RUN_EVIDENCE.json`核对初态和逐轮配对，再提交封存，才可评分source_selection。

## 已完成：候选停止

[完整结果与图表](RESULTS.zh-CN.md)：24/24次科学拟合、960epoch、121,920更新完成，无科学失败重试；独立复核5,258项通过。

| 臂 | source-CV RX等权Accuracy |
|---|---:|
| P0：原频域HStem+MHA | 85.0332% |
| P1：普通41抽头时域分支 | 28.7539% |
| P2：时域周期跨度20 | 28.9667% |
| N19：固定邻近跨度19 | 29.0587% |

P2−P1为+0.2127pp，RX条件95%区间[-0.6356,+1.0479]pp，最大受支持TX×RX错误率增加12.3333pp；P2−N19为−0.0920pp，未过特异性条件。最终`STOP_NO_PERIODIC_SPAN_PROMOTION`，不追加训练。

本轮三个时域臂替换了原频域前端，训练RX内验证约93.2%而留出RX约29%；没有检验保留强频域路径再融合时域周期侧路，不能把相对P0的巨大跌幅单独归因于周期跨度。固定窗口方案STOP保持不变。所有数字属于源域开发竞争，不是正式CRD或新RX盲测。
