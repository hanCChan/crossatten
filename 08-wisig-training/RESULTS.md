# 阶段08结果

更新时间：2026-10-26T02:14:58+08:00。基线 `74f5943`，分支 `exp/s08-wisig-training`。

已完成正式训练 **136** 次，预算上限160次。历史阶段00-07保持只读。

## 正式确认

| 数据集 | 算法 | CRD RX准确率 | Macro-F1 | 种子SD pp | 最差RX |
|---|---|---:|---:|---:|---:|
| ManySig | candidate | 93.01% | 92.80% | 1.32 | 84.67% |
| ManySig | control | 87.56% | 86.67% | 1.94 | 70.50% |

## 对照结论

| 数据集 | 对照 | CRD差值 pp | 接收机聚类95%区间 pp | 改善RX |
|---|---|---:|---|---:|
| ManySig | S08-paired-control | +5.45 | [+2.09, +8.81] | 11/12 |
| ManySig | IB-RFF | +1.92 | [-1.33, +5.04] | 8/12 |
| ManySig | WaveMLP | +1.97 | [-0.19, +4.03] | 8/12 |
| ManySig | S07-source-training | +6.40 | [+2.32, +11.74] | 10/12 |

**ManySig 的公平重训均值超过两个强基线，但接收机级区间跨0，尚不能宣称稳定领先或SOTA。** 官方发布权重的ManySig参考为IB-RFF93.57%、WaveMLP93.13%，本轮93.01%仍低于这两项历史参考。

**ManyRx只通过2/4源域折，未开展新的正式CRD评测。** S07的87.65%继续作为历史结果，不能把它与S08拼成新方法的完整双数据集成绩。

本轮首先充分验证训练策略，包括16次完整200 epoch对照；温和AWGN原checkpoint不如Official，延长训练没有普遍收益。条件触发的频谱相邻相位特征与源域选预算/EMA联合策略取得了ManySig提升。最终各折无AWGN，特征与轮数由源域冻结；总收益尚不能拆分归因。

正式测试已完成并停止，不按目标成绩追加Loss、特征或模型搜索。

## 证据

- `DIAGNOSTICS.md`：零GPU源域审计。
- `S08_SOURCE_RESULTS.md`、`source_results.csv`：所有源域训练/checkpoint结果。
- `registration.json`、`decision_after_*.json`、`decision_final.json`：预算与源域决策。
- `S08_WIFI_FINAL_RESULTS.md`、`final_*metrics.csv`：仅在完整确认后生成。
- `runs/`：配置、曲线、预测与失败记录；新权重仅本地保存。
- `S08_COSTS.md`：停止训练后实测成本；参数量不等于速度。

- `S08_EXECUTION.md`：运行数、环境及耗时口径。
- `acceptance.json`：配置、权重哈希、预测、数据角色与历史保留验收。

不自动合并、推送或发布权重；旧测试RX已被查看，本轮不是全新盲测。
