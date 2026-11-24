# S28-A + 原 S27-C：18 次明确授权训练

<!-- completion -->
**已完成 A6+C12，共18次200轮训练，独立核验通过，自动训练停止。**

A主交互 +1.350 pp，双向规则满足：True；C两RX共同Pilot规则满足：False。

[中文机制报告](RESEARCH_REPORT.zh-CN.md) · [英文PDF](output/pdf/s28_ac_mechanism_results.pdf) · [完整结果](RESULTS.json) · [独立核验](EXPERIMENT_INTEGRITY.json) · [证据矩阵](EVIDENCE_MATRIX.json) · [最终审计](FINAL_AUDIT.json)

Rebuild materials and PDF from saved aggregates; do not rerun scientific fits or target evaluation.
<!-- /completion -->

基于 S28-0 `de7d010`，按用户2026年11月24日授权激活。A六次H2_LA先训练，C十二次H0/LP随后重新初始化训练。原两份计划逐字节保存，没有根据新成绩更改。A与C全部封存后才统一Dev/Held报告，完成后不增加训练。

执行入口：`D:/crossatten/.venv/Scripts/python.exe -u 28ac-mechanism-training/resume.py`。

每个模型在`runs/`中保存完整轮快照：当前及最优权重、优化器、所有随机状态、训练轨迹。`PAUSE_REQUESTED`使所有活动模型在完整轮次保存后退出；继续须依据用户指令清除该标记，恢复同一拟合。协调器及遗留训练进程检查防止重复执行。

`observe.py`只读报告真实轮次与估时；`registration.json`和`ACTIVATION.json`在首次RF读取前提交、推送。新18模型的原始波形/权重/逐包预测留本地；完整配置、轨迹、日志、聚合结果与机制报告随进度归档到GitHub。

沿用原生FP32张量、matmul TF32关闭/cuDNN TF32开启。S28-0 数值失败与精度限定不改，转换算子不替代原科学模型。C只有b210_1、rtl_5两台固定留出RX；仅四RX源验证选模，不加载五RX旧权重，不进行总体RX显著性宣称。
