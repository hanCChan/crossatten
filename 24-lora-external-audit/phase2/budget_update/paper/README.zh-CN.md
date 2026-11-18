# 统一比较图

`plot_comparison.py`只读取本轮已生成的`RESULTS.json`和模型级metrics，不读取波形、不重算或选择统计假设，不参与训练。运行后输出Accuracy和配对差值两组PNG/SVG图：

```powershell
& D:/crossatten/.venv/Scripts/python.exe 24-lora-external-audit/phase2/budget_update/paper/plot_comparison.py
```

Accuracy图中的空心点是六个fold/seed的各自RX等权准确率，菱形是完整均值；它们不是六台独立设备。差值图采用RESULTS中全部预登记Accuracy配对和RX条件区间，不作多重比较校正。Native ASK未齐时不显示单个seed；Strict-IB ASK为延期。

图是生成时报告的快照。后续Native ASK完成并更新RESULTS后，须重新运行此脚本更新图片。稿件解释与边界见[../EVIDENCE_SCOPE.zh-CN.md](../EVIDENCE_SCOPE.zh-CN.md)，逐包预测核验见生成后的`../PREDICTION_INTEGRITY.json`。
