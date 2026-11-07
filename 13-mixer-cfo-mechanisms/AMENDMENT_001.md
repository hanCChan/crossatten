# 技术修订001：ManySig六类指标

预注册后、正式 fit 和 CFO 诊断开始前，检查发现复用 `xa.evaluate.rx_metrics` 会把 ManySig 的 Macro-F1 按10类平均，而模型只有6类输出。
这会错误压低 ManySig 的 Macro-F1；准确率、训练 CE、梯度和准入门槛不受影响。

通过 `STOP_REQUESTED` 在第9个冒烟 fit 完成后暂停，未中断任何 fit。
已完成的8个 Shen 冒烟与1个 ManyRx 冒烟都是10类，结果保持原样；ManySig 和正式训练尚未开始。
暂停时的主流程错误记录归档至 `execution_events/001_metric_fix/`；该 RuntimeError 是主动停止调度，不是模型训练失败。

新增本阶段 `metrics.py`，按 logits 声明的类别数计算 confusion、Macro-F1，与 sklearn 验证一致；原有库文件不变。
同时记录每个新 fit 的当前 Git SHA 和技术修订 SHA；CFO 无可靠包时报告NA，避免格式化空值导致技术中断。
补全P2所有实验臂（包括corrected-only）的逐fit/逐RX导出及汇总表，P0探针记录求解迭代数。这些仅补充报告，不影响训练或门槛。
代码验证通过后，将本次完整代码 SHA 清单与已完成冒烟 checkpoint SHA 放进 `registration_amendment_001.json`，提交后再续跑。
原 `registration.json` 和原始提交保留，科学配置、种子、预算和门槛均不变。
