# 阶段08结果

更新时间：2026-10-26T01:56:36+08:00。基线 `74f5943`，分支 `exp/s08-wisig-training`。

已完成正式训练 **48** 次，预算上限160次。历史阶段00-07保持只读。

**仍在执行注册实验，暂无完整正式CRD结论。**

## 证据

- `DIAGNOSTICS.md`：零GPU源域审计。
- `S08_SOURCE_RESULTS.md`、`source_results.csv`：所有源域训练/checkpoint结果。
- `registration.json`、`decision_after_*.json`、`decision_final.json`：预算与源域决策。
- `S08_WIFI_FINAL_RESULTS.md`、`final_*metrics.csv`：仅在完整确认后生成。
- `runs/`：配置、曲线、预测与失败记录；新权重仅本地保存。
- `S08_COSTS.md`：停止训练后实测成本；参数量不等于速度。

不自动合并、推送或发布权重；旧测试RX已被查看，本轮不是全新盲测。
