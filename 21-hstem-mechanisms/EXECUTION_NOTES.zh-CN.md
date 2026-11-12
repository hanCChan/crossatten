# S21 执行记录

2026-11-12，预注册提交 `d907ea2`，历史源域恢复提交 `7e6ee3d`。12个基线模型严格加载成功，源验证CE差值均为0，源验证全部指标逐项相同。首个H1正式训练已启动；没有读取本轮目标数据。

H1 的 PyTorch2.5 CUDA `upsample_linear1d_backward` 提示没有确定性实现。当前原S13训练器使用 `torch.use_deterministic_algorithms(True, warn_only=True)`，该提示未使执行失败。本轮保留预注册的原生线性插值、数据/噪声顺序与初始化，不在观察训练后替换插值算子、不丢弃本次拟合或另加种子。

因此须区分：数据顺序/AWGN与保留参数初始化按fold/seed严格配对，历史基线源预测严格恢复；H1新增插值反向可能有低位数值非确定性，不能承诺相同seed的新H1训练逐比特复现。该限制不等同于目标泄漏，也不改变源CE选模规则。目标评估仍要求历史float32 logits逐元素严格恢复，否则停止配对报告。首次提示保存在首个H1的stdout.log，最终证据附此记录。

当前实测每个200epoch新拟合约需十几分钟，12次串行训练预计为数小时。进度与完成状态以 `STATUS.json` 和各fit的 `live.json` 为准；未完成前不报告H1/H2目标准确率或机制结论。

为使长任务独立于前台工具会话持续运行，执行器在首个完整200epoch拟合后按既有fit边界停止规则交接给隐藏后台。交接程序位于 `D:/crossatten/.execution-s21/run_background.py`，只在首个fit完整、权重SHA相符且退出原因为预定边界停止时继续余下11次，不重试任何不完整fit。该预定退出记录保存在 `PLANNED_HANDOFF_EVENT.json`，不删除原始事件；进程交接不改变模型、配方、训练顺序、checkpoint、评价或12次预算。交接与后台完成状态见 `BACKGROUND_HANDOFF.json`。

2026-11-12 15:14?12?????????????????????H0?????????????15:28??????????????????????15:37??24???????????????????????EVALUATION_DIRECTORY_FAILURE/RECOVERY.json??????????registration_amendment_003.json?????0??
