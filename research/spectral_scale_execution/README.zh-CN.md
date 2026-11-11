# 频域尺度残差执行工程

当前状态：`ENGINEERING_ACCEPTED_SCIENTIFIC_EXECUTION_CLOSED`。

已完成十二个源 checkpoint 的身份恢复与 CUDA 合成前向、8 项 CPU 测试、两臂各 40 epoch 的 CUDA 合成训练循环和 589 项派生证据复核。科学拟合与源 IQ 读取均为 0。真实 checkpoint_val 预测一致性需在另行明确执行授权后检验。

详见[工程附录](EXECUTION_ADDENDUM.zh-CN.md)和[执行注册](EXECUTION_REGISTRATION.json)。原[设计预注册](../spectral_scale_review/PREREGISTRATION.zh-CN.md)、模型和全部门槛保持不变。

复核已公开的工程证据（只解码合成轨迹/状态，不解码源数组）：

```powershell
python research/spectral_scale_execution/verify.py --public
python research/spectral_scale_execution/executor.py status
```

复核全状态需要本机忽略目录内的原始合成 artifacts；公开 zip 提供完整轻量轨迹，代码 zip 保留实际运行字节。`preflight`、`train-all`、`evaluate` 和 `analyze.py` 需要匹配且已提交的明确用户授权；本轮没有生成授权文件。
