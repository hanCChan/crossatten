# S26：Long-Patch 六模型开发性控制与方法论证

<!-- completion -->
**本阶段已完成：6/6 新训练、18 模型统一评估、180000 条 Dev 预测独立核验。**

H0 97.335%，Long-Patch 94.500%，P0 94.648%；H0−LP +2.835 pp。全部六模型和十个 RX 保留，条件性解释见 [RESULTS](RESULTS.json) 与 [GO/STOP](GO_STOP.json)。

[修订英文稿](output/pdf/hstem_tifs_revised.pdf) · [补充材料](output/pdf/hstem_tifs_supplement.pdf) · [中英文贡献与十项问答](RESEARCH_NOTES.zh-CN.txt) · [科研完整性核验](EXPERIMENT_INTEGRITY.json) · [最终交付审计](FINAL_AUDIT.json)

仅重建文档：运行 `paper_revision.py`、`research_notes.py`、`build_documents.py`，不需要波形或权重；只用已提交聚合证据和图表。不要重新启动已完成训练或重复 Dev 评价。
<!-- /completion -->

基线为 S25 `657bdaa`。用户于 2026-11-22 要求执行附件中的后续方案，本阶段新增且仅新增一个 Long-Patch 控制臂、两折三种子，共六次训练。S25 的零训练登记保留原样，不追溯修改。

固定结构：组数 8、宽度 64、核长 340、步长 256、padding 42，随后 64→64 点投影；138,346 参数，比 HStem 多 3.48%；相同 32×64 Token、350 局部支持及普通 MHA 后端。内部宽度、BN、非线性深度与 MACs 仍不同，不能作为纯层级因果消融。

训练沿用原 Source 160/40、两折、种子 201/202/203、200 轮、batch 64、AdamW、warmup/cosine、0–50 dB AWGN 与源域 RX 等权 CE 选模。单个训练采用自己的相同数据 seed、排列与噪声发生器，保留共享前端和后端初始化。新增原子化完整 epoch 恢复，不重置种子或择优补跑。

结构、梯度、batch 64/128 显存、非法数据角色拒绝以及 optimizer/RNG 恢复后的逐位一致性先通过无数据 Gate。代码、参数、统计口径和六个作业在首次本阶段源域访问与训练前提交冻结。

全部六个 checkpoint/config/history 封存并提交后，才统一评价 LP、H0、P0 的 Dev 结果。禁止新 Confirm/blind 读取；不补 ASK Strict-IB；不获取或构造未经授权的独立采集。结果无论好坏均保留全部模型、接收机和种子，不改超参数。

主要比较为 H0−LP，次要比较为 LP−P0。十个 RX 内先平均三种子，再固定 10,000 次 RX bootstrap；只有条件性解释。未显著优于不能解释成等效，Long-Patch 表现接近时应收窄层级收益的主张。

当前恢复入口：`D:/crossatten/.venv/Scripts/python.exe -u 26-long-patch-control/resume.py`。入口检测已运行进程及已完成结果，避免重复启动同一批训练/评估。原 `s26_execute.py` 保留训练与封存入口；原评估文件末尾有一个多余括号，纯语法修复及全部 SHA 见 [TECHNICAL_FIX_001.json](TECHNICAL_FIX_001.json)。原登记文件不改字节，修复模块不改模型、训练或数据权限。本次后台修复入口等待原六组训练完成后接手评估，不重复拟合。

逐模型进度在 `runs/LP_f*/live.json`，总体在 `LIVE_STATUS.json`；GitHub 进度是提交时快照，本地日志为实时状态。大波形、checkpoint、恢复状态与逐包预测留在本地，不上传。

同时复核 Ma 2025 作者代码与数据预算条件，扩充方法推导，并围绕核心贡献重写稿件。不会用简化网络冒充其原生复现，也不会把数学结构描述称为新的理论保证。
