# S28-0：仿射物化与 BN×GELU 预注册

基于 S27-B `48faa0e`。本阶段完成零训练数学/工程审计；新增科学拟合 **0**，S28-A **0/6**，S27-C **0/12**。本轮用户附件明确要求新增训练另行授权，故执行入口按零预算拒绝启动。

六个冻结 No-GELU Tokenizer 在 float64 下通过有限仿射等价检查；普通共享卷积在首尾 Token 不等价，须显式保留边界权重与位置偏置。**原 FP32 张量、cuDNN TF32 开启的完整12000包检查均未达到1e-4容差，且1个预测改变**，失败完整保留。单独登记的关闭TF32同探针诊断通过，不替换原失败或历史成绩。

- [中文科研报告](RESEARCH_REPORT.zh-CN.md) · [英文数学审计 PDF](output/pdf/s28_affine_audit.pdf)
- [六模型全部数值误差](AFFINE_AUDIT.json) · [完整误差表](affine_summary.csv) · [精度诊断](PRECISION_DIAGNOSTIC.json)
- [BN×GELU 六次预注册](PLAN_A.json) · [原 S27-C 十二次计划](PLAN_C.original.json) · [C 资格与边界](C_READINESS.json)
- [零训练登记](registration.json) · [合成 Gate](GATE.json) · [24 模型 Source 恢复](SOURCE_RECOVERY.json) · [创新证据矩阵](EVIDENCE_MATRIX.json)
- [最终审计](FINAL_AUDIT.json) · [用户原始方案](REQUEST.original.zh-CN.txt)

首次 Source 读取前已提交并推送登记、代码、容差与 Gate；原生精度失败后又单独登记精度补充诊断。仅使用既有合法 Source 验证数据，没有新增 Dev、Confirm、blind 或 C 留出评价。既有模型、配置和结果只读保留。

`report.py` 从已保存的聚合记录重建图表与报告。`s28_audit.py` 和 `s28_precision.py` 是真实的只读核验脚本，不能把重跑核验当作新实验；无需为审阅报告重新运行。原始波形、转换系数与工程快照保留在本地忽略目录。科学代码哈希依据本机精确字节，仓库字节与本地字节均在最终审计检查；请保留换行及原件。

S28-A 的 H2_LA 只移除三个中间 BN 与三个 GELU，133338 参数，32×64 Token，名义 CNN 支持350。两折三种子、其余模块初始值、输入预算和原训练配方均已冻结。缺失单元尚无成绩，BN×GELU 交互当前为 NA。

S27-C 原 JSON 逐字节保留，b210_1/rtl_5 留出不变；四RX新训练、两台留出RX的 Pilot 与 A 是两个独立问题。C不复用五RX权重，不把两台RX当六个独立接收机，也不证明独立新采集泛化。后续需明确授权A六次或A六次加C十二次，再冻结执行激活与适配器。
