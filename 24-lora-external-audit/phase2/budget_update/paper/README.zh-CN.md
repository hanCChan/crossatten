# 统一比较图

`plot_comparison.py`只读取本轮已生成的`RESULTS.json`和模型级metrics，不读取波形、不重算或选择统计假设，不参与训练。运行后输出Accuracy和配对差值两组PNG/SVG图：

```powershell
& D:/crossatten/.venv/Scripts/python.exe 24-lora-external-audit/phase2/budget_update/paper/plot_comparison.py
& D:/crossatten/.venv/Scripts/python.exe 24-lora-external-audit/phase2/budget_update/paper/verify_predictions.py
& D:/crossatten/.venv/Scripts/python.exe 24-lora-external-audit/phase2/budget_update/paper/verify_final_predictions.py
```

Accuracy图中的空心点是六个fold/seed的各自RX等权准确率，菱形是完整均值；它们不是六台独立设备。差值图采用RESULTS中全部预登记Accuracy配对和RX条件区间，不作多重比较校正。Native ASK未齐时不显示单个seed；Strict-IB ASK为延期。

图是生成时报告的快照。后续Native ASK完成并更新RESULTS后，须重新运行此脚本更新图片。稿件解释与边界见[../EVIDENCE_SCOPE.zh-CN.md](../EVIDENCE_SCOPE.zh-CN.md)，逐包预测核验见生成后的`../PREDICTION_INTEGRITY.json`。

`verify_predictions.py`仅核验baseline48与H0共54个模型的已保存logits：270个RX单元、540000条预测、所有方法/视图/seed的标签和packet rows一致性，以及独立NumPy重算的混淆矩阵、Accuracy、Macro-F1、交叉熵和报告均值。需本地忽略目录的预测NPZ，不上传波形或预测二进制。此核验不访问源波形、Dev波形、confirm或blind，也不改变统计分析。

2026-11-22最终图已包含全部ASK Native六模型。`verify_final_predictions.py`读取同一批已保存预测和封存元数据，覆盖54个新模型加6个H0、300个RX单元、600000条预测，独立重算全部指标、均值、fit SD及26个配对指标区间，同时核查权重/config/metrics哈希、延期任务未启动和成本结果完整性。最终产物为`../FINAL_INTEGRITY.json`，原baseline核验记录保留原样。
