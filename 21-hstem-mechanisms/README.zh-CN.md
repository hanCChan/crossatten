# S21：HStem 的局部覆盖与中间 BN 机制验证

本轮执行用户提供的“先证明有效机制，再形成创新”方案。范围限定为 Shen LoRa，新增 H1/H2 各 6 次拟合，两折 × seed201/202/203，共 12 次；不追加新 Loss、WiSig 训练或 kernel 搜索。

| 实验臂 | 模型 | 新拟合 | 参数 | CNN局部感受野 | 步长 | 相邻覆盖重叠 | BN总数 |
|---|---|---:|---:|---:|---:|---:|---:|
| H0 | S13/S15 原 HStem+MHA | 0 | 133690 | 350 | 256 | 94 | 5 |
| P0 | S15 参数匹配 Patch-Lite+MHA | 0 | 134266 | 266 | 256 | 10 | 2 |
| H1 | 固定插值 Overlap-Patch+MHA | 6 | 134266 | 350 | 256 | 94 | 2 |
| H2 | HStem-NoMidBN+MHA | 6 | 133338 | 350 | 256 | 94 | 2 |
| H3 | HStem-NoGELU，条件候选 | 0 | 133690 | 350 | 256 | 94 | 5 |

所有 tokenizer 输出为 `[B,32,64]`，第一 token 中心为 127.5。感受野仅为 CNN 坐标上的理论支持；全局 FFT 和 TrueIB 使其不能解释为原始 IQ 的端到端物理感受野。

H1 的分组数 8、低秩通道数 80 严格继承 S15。将原始 `[80,4,256]` 可学习卷积核 reshape 后按 `linear, align_corners=False` 插值到 340，再乘 `sqrt(256/340)`；stride256、padding42。核参数与原始偏置、后接 1×1 卷积的初始化逐元素保持一致。该幅值是预先固定的尺度规则，不声称线性插值后每个实际核的能量严格守恒。插值还改变频响和平滑性，所以不作“仅改变重叠”的纯因果解释。

H2 仅把 tokenizer 中三个 BN 换成 Identity，保留前端两个 BN、所有卷积、GELU、位置编码与 MHA。去除 352 个 BN 仿射参数，参数减幅 0.2633%；不通过增加占位参数人为掩盖此差异。H3 已登记但本轮执行器不允许运行。

训练直接复用冻结 S13 的 Shen200 优化循环：FP32、batch64、200 epoch、原 TrueIB、AWGN U[0,50)、AdamW、warmup/cosine，配对数据顺序和 AWGN 的 CUDA generator 种子不变。checkpoint 只按源 RX 等权验证 CE 取最小值，目标数据不进入选模。当前机器与基线同为 torch2.5.0+cu124、RTX4060 Laptop。

先检查全部 H0/P0 权重与归档 SHA、严格加载 state，并复算源验证 CE/指标。12 个新 checkpoint 全部完成后统一 SHA 封存并 git commit，随后才读取既有开发/历史 confirm RX。基线目标 float32 logits 必须逐元素完全复现，失配即停止配对报告。未接触六个 blind RX。

主指标是 RX 等权 dev Accuracy；恢复率使用总体均值之比 `(H1-P0)/(H0-P0)`，分母不达 0.25 pp 则未定义。恢复至少50%且两折 H1-P0 均为正，才记为值得后续验证的局部上下文信号。H2-H0 均值不高于−1pp、至少5/6负向且两折负向，才记为描述性 BN 重要信号。这些规则冻结于训练前，不是因果证明或统计等价检验。

交付包含 Accuracy/F1、逐 RX 与 TX×RX 错误变化、真类 logit margin、64维最终均值特征的类中心距离结构、参数/MACs/同机延迟。margin 包含误分类样本；距离矩阵先按各自45个非对角距离的均值归一化，再作跨 RX 相关和 RMSE。辅助指标不用于挑 checkpoint/模型。条件区间按 RX 聚类，种子重复不充当独立 RX。confirm 只作历史描述，不是新盲测。

运行入口（在本 worktree 根目录）：

```powershell
& D:/crossatten/.venv/Scripts/python.exe -X utf8 21-hstem-mechanisms/test_s21.py
& D:/crossatten/.venv/Scripts/python.exe -X utf8 21-hstem-mechanisms/s21_register.py
# 将代码、配置、architecture_audit 与 registration 提交到当前实验分支后：
& D:/crossatten/.venv/Scripts/python.exe -X utf8 -u 21-hstem-mechanisms/s21_execute.py
```

`STATUS.json` 记录进度；`runs/*/live.json` 记录每轮源指标。失败保留现场，不自动重训。`STOP_REQUESTED` 在下一 fit 边界停止。完成后生成 `RESULTS.zh-CN.md`、`decision.json`、`FINAL_AUDIT.json` 与 `evidence/`，并停止在12次预算。论文方法/实验初稿见 `MANUSCRIPT_DRAFT.zh-CN.md`，S21 完成后填入实际结果。
