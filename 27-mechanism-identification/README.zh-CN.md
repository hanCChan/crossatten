# S27：HStem 机制识别

以 S26 `ad5443e` 为基线。本轮于 2026-11-23 仅执行 A：HStem-LinearActivation（LA）六次训练；B/C 完成独立预注册，自动训练预算为零。历史 S15–S26 文件和结果保持不变。

A 复用 S21 已定义但未训练的 H3：仅将 Tokenizer 的 `2/5/8` 三处 GELU 换为 Identity，卷积、BN、初始化、TrueIB 五视图、位置处理和 MHA 不变。H0/LA 均为 133,690 参数、95.028M 部分 MACs、350 点名义卷积路径支持及 32×64 Token。保留训练 BN，不能把 LA 整网称为线性系统。

原协议：两折 × 种子 201/202/203，5 个 Source RX/折，每 TX×RX 固定 160 训练/40 验证，200 轮、batch 64、AdamW、0–50 dB AWGN、源 RX 等权 CE 选模。每个配对种子的完整初始 state_dict 相同，排列和增强使用独立固定发生器。只允许完整 epoch 的模型、优化器、全部 RNG 状态恢复，不换种子或择优重启。

所有六个新模型、配置、历史记录封存并提交后，统一评估 H0/LA/LP 共 18 个模型、90 个模型×RX 单元、180,000 条 Dev 预测；不得新增 Confirm、旧 blind 访问、目标适应或超参数搜索。保留全部弱种子、RX 和负面结果。历史开发 RX 不构成独立采集。

主比较 H0−LA；次比较 LA−LP、H0−LP。先在每 RX 内平均三个种子，再以 10 个唯一 RX 为单位做固定 10,000 次 bootstrap（seed 27027）。解释门槛事前冻结；区间条件于现有模型和历史协议，不覆盖新采集、架构选择或新种子不确定性。接近、未显著下降都不等于证明等效。

独立计划见 [B：Tokenizer×后端](PLAN_B.json)、[C：Source 内留一 RX Pilot](PLAN_C.json)。B 的两层逐 Token 残差 MLP 保留同一 Token 接口，每层比 MHA 少 64 参数，总参数差 128；它也改变非线性形式，交互效应只能针对该后端解释。C 使用接收机名称的固定 SHA 排序，每折选一台留出 RX（不参考数据或成绩），只有两台测试 RX，不做接收机总体显著性声明。B/C 均不得在本轮自动运行。

工程检查：所有当前科学脚本 AST/导入、形状、参数/MAC、六组完整初始化、禁用数据角色、有限梯度、batch64/128 资源及优化器/排列/噪声/RNG 恢复逐位一致性。首次 Gate 的 JSON 变量复用错误及修复在 [ENGINEERING_CORRECTION_001.json](ENGINEERING_CORRECTION_001.json) 保留；总计 8 次合成更新，零 RF 数据训练，不混入六次科学拟合。

冻结登记见 [registration.json](registration.json) 与 [GATE.json](GATE.json)。恢复入口：

```powershell
D:/crossatten/.venv/Scripts/python.exe -u 27-mechanism-identification/resume.py
```

入口检测运行进程与已完成评估，避免重复队列。实时状态为 `LIVE_STATUS.json` 和 `runs/LA_f*/live.json`；GitHub 为提交时快照。波形、权重、恢复状态和逐包 logits 留在本地，公开配置、实际训练日志、逐 RX/TX 汇总、统计和审计。六次 A 完成后输出科研报告，说明哪些解释被约束、哪些仍混杂，并提出后续辨识价值最高的一项实验；本轮不定稿论文、不自动启动 B/C/IQ-only/ASK Strict-IB。
