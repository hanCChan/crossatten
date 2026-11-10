# 复核与复现

最终入口为 [RESULTS.zh-CN.md](RESULTS.zh-CN.md)，决定为 `d0_support_matched/decision_d0.json`。原 `s18_run.py` 的完整网格设计因数据支持停止，保持原样用于审计；实际完成的研究是单独注册的D0S。

环境为 `D:/miniconda/envs/crossatten4060/python.exe`，PyTorch2.5.0+cu124、NumPy1.26.4、SciPy1.15.1，GPU RTX4060 Laptop。原源缓存和模型保留在 `D:/crossatten/worktrees/s17-rx-distortion/17b-phase-intervention/`，脚本只允许登记的ManyRx fit18/date1–2路径。Git不包含源IQ、模型权重或逐包特征。

在worktree根目录运行合成工程检查：

```powershell
& 'D:/miniconda/envs/crossatten4060/python.exe' -m unittest discover -s 18-selective-rx -p 'test_*.py' -v
```

复核已有结果，不做模型forward或读IQ：

```powershell
& 'D:/miniconda/envs/crossatten4060/python.exe' 18-selective-rx/s18_verify.py
```

该脚本会更新派生的`FINAL_ACCESS_AUDIT.json`时间及文件状态；正式归档保留的是本次封存版本。重生成报告及图表会更新派生文件时间和MANIFEST：

```powershell
& 'D:/miniconda/envs/crossatten4060/python.exe' 18-selective-rx/s18_reports.py
```

科学执行入口为`s18_support_run.py`，分析代码与门槛由原注册和支持修正注册共同固定。已有完整结果会被保留，已有未完成attempt会阻止自动重试；不要删除结果或attempt来追分。独立复现应保留原归档，在另一个明确隔离的复现目录中配置同样的登记输入和代码，并将其标为重复计算，不能覆盖本轮记录。

SHA引用包括历史checkpoint、源缓存manifest、估计/验证索引、池化feature缓存、原始结果、代码和协议。独立QA从本地feature缓存重算散度、方向与稳定性，从混淆矩阵重算分类指标；因无逐包logits缓存，CE/margin仅复核聚合与差值/门槛一致性。当前源域STOP已经封存，不执行训练、held-source或真实目标评估。
