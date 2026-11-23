# S27-B：Tokenizer × 后端交互

以 S27-A `636da47` 为基线，执行用户当前明确授权的 B：12 次新拟合。历史 `PLAN_B.json` 的零自动预算记录保持原样；本目录独立激活 B，不启动 C、IQ-only、ASK Strict-IB 或架构搜索。

冻结方案原件见 [PLAN_B.original.json](PLAN_B.original.json)，与 S27 原文件 SHA 相同。四组为 HStem+MHA、Long-Patch+MHA（各复用六个冻结模型），以及 HStem+TokenMLP、Long-Patch+TokenMLP（各新增六次训练）。参数分别为 133690、138346、133562、138218。新后端直接调用 S27 已定义的两层逐 Token 残差 MLP，宽度 128，保留输入投影、位置项、pre-LN、final-LN、mean pooling 和分类器。

全部新拟合沿用两折、种子 201/202/203、Source160/40、200 轮、batch64、AdamW、warmup/cosine、0–50dB AWGN 与源 RX 等权 CE 选模。同 fold/seed 共享数据顺序和噪声流，存活模块匹配对应 MHA 初始值，两种新模型的 MLP 初始参数一致。12 个新模型全部封存并提交后才统一 Dev 推理，不按任一新成绩修改后端或补种子。

主指标为 `(HStem_MHA−LP_MHA)−(HStem_MLP−LP_MLP)`。RX 内先平均三个种子，十个唯一 RX 做 10000 次 bootstrap，seed27028。沿用原门槛：均值 ≥0.5pp、条件区间下界 >0、至少5/6 配对正向、两折均正向。负结果和未达门槛都完整报告；未显著不等于等效，交互也不等于普遍注意力必要性。

统一评价共 24 个模型、120 个模型×RX 单元、240000 条 Dev 预测；次要分析覆盖每 RX、逐 TX、Source/Dev 和最差 RX。几何仅使用原 Source Validation 的全部固定样本，共 24 模型×2000包，固定 Tokenizer 均值、后端前均值、最终分类输入三个64维特征，统一逐包 L2 与 S25 分离/散度定义，全部种子和层保留。零或非有限嵌入标记未定义，不更换归一化。不同已训练模型之间的几何变化不是因果信息流或互信息。

同会话报告四组真实 GPU 延迟和部分 MACs。MLP 改变了非线性形式与算力；参数接近不代表功能类完全匹配。Source Validation 已参与选模，历史 Dev RX 也不构成新的独立采集。

工程 Gate 使用合成数据，检查形状/参数/MAC、初始化、有限梯度、两种新结构的完整 optimizer/RNG 恢复、非法角色拒绝、batch64/128 资源、源特征 hook 与原 forward 的逐位一致性及四格交互算术。8 次合成优化更新与 12 个科学拟合分开记录。吸收 S27 的 Windows 修复：JSON 使用进程独立临时文件，原子替换有界重试，保留完整 epoch 恢复；观察器使用共享删除读取。

恢复入口：

```powershell
D:/crossatten/.venv/Scripts/python.exe -u 27b-tokenizer-backend-interaction/resume.py
```

实时进度在 `LIVE_STATUS.json` 与 `runs/*/live.json`。波形、checkpoint、恢复快照、逐包 logits/特征留在本地；代码、配置、完整实际训练日志、聚合结果、科研报告和独立审计提交 GitHub。B 完成后停止新增拟合，判断协同证据及 C 的后续必要性，暂不重写正式 TIFS 全文。
