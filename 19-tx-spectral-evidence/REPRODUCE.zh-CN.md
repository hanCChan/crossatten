# S19 复核入口

最终报告入口为[RESULTS.zh-CN.md](RESULTS.zh-CN.md)，历史矩阵为[historical_matrix.md](historical_matrix.md)。实际实验规格以[registration.json](registration.json)和被其SHA封存的代码为准。所有命令从worktree根目录执行，环境为`D:/miniconda/envs/crossatten4060/python.exe`。

合成工程检查：

```powershell
& 'D:/miniconda/envs/crossatten4060/python.exe' -m unittest discover -s 19-tx-spectral-evidence -p 'test_*.py' -v
```

已有科研结果的独立QA只读取登记元数据、checkpoint字节哈希和本地保存的logits，不做模型forward，也不打开IQ：

```powershell
& 'D:/miniconda/envs/crossatten4060/python.exe' 19-tx-spectral-evidence/s19_verify.py
```

该操作会更新派生`FINAL_ACCESS_AUDIT.json`的核验时间；正式归档的JSON和MANIFEST保存的是本次封存版本。重生成报告/图表也会更新派生文件时间与MANIFEST：

```powershell
& 'D:/miniconda/envs/crossatten4060/python.exe' 19-tx-spectral-evidence/s19_reports.py
```

正式执行入口为`s19_execute.py run`。既有完成结果会保留；已有未完成attempt会阻止自动重试科学训练。原始流程先估计四fold训练证据，再训练全部16新增模型，封存20个checkpoint及配置/指标并本地commit，之后一次性读取held-source，最后按既定门槛生成决定。没有目标RX/date3–4数据入口。

不要删除attempt/结果来重复选参。独立复现应在新隔离目录中保留本次原归档，配置相同登记输入与环境并记录为重复计算；改变频带数、alpha、loss权重或数据分组属于新的研究，不是本次复现。

本地输入引用：

- 历史fit源缓存和B0权重：`D:/crossatten/worktrees/s17-rx-distortion/17b-phase-intervention/`。
- 合法held-source cell缓存：`D:/crossatten/stage_1/13-mixer-cfo-mechanisms/cache/wisig/ManyRx/`，脚本固定每fold源名单并只读日期1–2。
- teacher反事实全部logits/labels/估计索引：`19-tx-spectral-evidence/cache/evidence_f*.npz`。
- 每模型源验证及held-source logits/labels：`runs/ManyRx__B*__f*__first__s18101/*predictions.npz`。
- 新训练权重：各run的`model.pt`；B0使用登记的历史只读权重。

原始IQ、权重和逐包logits不进Git；各JSON包含其SHA。Git保留所有代码、预注册、配置、逐epoch顺序SHA与曲线、日志、配对指标、CI、seal和访问审计。`live.json`是训练过程最后一轮快照，完成状态以`attempt.json`、`metrics.json`和`seal_pilot.json`为准。

成本测量用`s19_costs.py`，只在全部模型封存后加载fold0权重并输入随机IQ；`COSTS.json`单独记录五次checkpoint访问，不写主科研数据访问日志。benchmark不选择模型，不新增训练。完整推理算子路径与未覆盖算术项在该JSON列出，部分FLOPs不能作为完整总FLOPs。已有成本记录会阻止自动重复测量；保留正式记录。

注册后的报告、QA与成本脚本属于派生工具，不改变主分析、teacher证据、训练、门槛或决定；其最终SHA在MANIFEST及各输出中封存。
