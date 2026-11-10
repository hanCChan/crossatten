# S20 复核与复现入口

报告为[RESULTS.zh-CN.md](RESULTS.zh-CN.md)，原协议/代码冻结于[registration.json](registration.json)，本轮工程交接见[ENGINEERING_HANDOFF.json](ENGINEERING_HANDOFF.json)。所有命令从S20 worktree根目录执行，环境为`D:/miniconda/envs/crossatten4060/python.exe`。

合成单测：

```powershell
& 'D:/miniconda/envs/crossatten4060/python.exe' -m unittest discover -s 20-asknet-source-comparison -p 'test_*.py' -v
```

复核已有结果，不运行模型forward、不打开IQ，仅检查登记元数据、权重字节SHA和逐包logits：

```powershell
& 'D:/miniconda/envs/crossatten4060/python.exe' 20-asknet-source-comparison/s20_finalize.py audit
```

该命令会更新派生`FINAL_ACCESS_AUDIT.json`的时间和代码SHA；正式归档以本次MANIFEST封存版本为准。重生成报告/图表也会更新派生文件与MANIFEST：

```powershell
& 'D:/miniconda/envs/crossatten4060/python.exe' 20-asknet-source-comparison/s20_finalize.py report
```

原执行过程：先prepare/合成单测/CUDA检查，注册并commit；四次训练均成功。原`s20_run.py run`在seal函数拼装B0元数据时重复传入fold，尚未读held-source。保留原注册代码，使用单独提交的`s20_seal_finish.py`修复这一序列化交接、封存全部8个模型引用，随后调用原注册的held评价和decision函数。**没有重复任何训练，也没有更改门槛、模型或选模结果。**

独立重复计算应在新隔离目录保留本轮归档并标明是计算复现，不覆盖原结果；训练入口可分开用`s20_run.py fit`，封存/评价入口用`s20_seal_finish.py`。不要删除attempt以反复挑选结果。若只是对已有完整归档运行原`run`入口，完成结果会保留；没有未完成的fit会被自动重训。直接再次运行seal-finish或benchmark会被已有记录阻止。

作者源代码不在Git中转载。固定来源和3个Git blob SHA见[UPSTREAM.json](UPSTREAM.json)，外部只读目录为`D:/crossatten/.reference-s20/ASKNet-RFF-f00846d/`。原始S19 fit缓存/B0权重及已保存logits路径见[REUSE_AUDIT.json](REUSE_AUDIT.json)；新的ASK权重/source与held logits保留本地`20-asknet-source-comparison/runs/ASK__f*/`，由metrics/seal/manifest中的SHA引用。

成本和静态结构由`s20_finalize.py cost`在封存后计算，仅使用登记权重与随机IQ，不读取科研包。该入口没有改变或缓存ASK的原QR操作；成本已有记录时不自动重测。它与逐包分类的科学访问日志分开保存checkpoint-only访问清单。

原始IQ、模型权重、逐包预测和作者源码不进Git。代码、协议、官方来源SHA、复用拒绝理由、配置、曲线/顺序SHA、日志/k trace、checkpoint引用、指标/CI、封存、工程事件和最终审计均可复核。报告中的错误定位及K矩阵统计只作描述，不作为新方法选择；没有目标CRD、更多seed或新模块实验。
