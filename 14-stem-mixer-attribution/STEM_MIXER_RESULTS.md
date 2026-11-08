# Stem × Mixer配对消融

Patch+MHA、Patch+RF-HSTU为本阶段新增12 fit；HStem两格复用S13相同fold/seed的12个模型。所有非tokenizer初始权重逐项配对。
交互差值=(HStem+HSTU−Patch+HSTU)−(HStem+MHA−Patch+MHA)，先在同一RX/fold/seed内作差，再按RX聚类估计区间。

| 角色 | 实验臂 | RX准确率 % | Macro-F1 % | 最差RX均值 % | seed SD pp |
|---|---|---:|---:|---:|---:|
|dev_val|E1_patch_mha|95.222|95.186|89.317|0.415|
|dev_val|E2_patch_hstu|95.888|95.843|91.200|0.549|
|dev_val|A1_mha|97.335|97.326|93.667|0.308|
|dev_val|A3_rfhstu|97.460|97.457|95.500|0.269|
|confirm|E1_patch_mha|93.685|93.587|84.925|0.756|
|confirm|E2_patch_hstu|94.317|94.201|85.500|0.215|
|confirm|A1_mha|96.508|96.464|90.867|0.134|
|confirm|A3_rfhstu|96.448|96.382|90.775|0.641|

| 角色 | 指标 | 效应 | 均值 pp | 条件95%区间 pp | 正向配对 | 最差RX平均Δ pp |
|---|---|---|---:|---|---:|---:|
|dev_val|acc|hstem_effect_mha|+2.113|[+1.588,+2.748]|6/6|+1.267|
|dev_val|acc|hstem_effect_hstu|+1.572|[+0.952,+2.347]|6/6|+0.250|
|dev_val|acc|interaction|-0.542|[-0.743,-0.323]|1/6|-1.033|
|dev_val|acc|stem_main|+1.843|[+1.278,+2.545]|6/6|+0.758|
|dev_val|acc|mixer_effect_patch|+0.667|[+0.357,+0.998]|5/6|-0.033|
|dev_val|acc|mixer_effect_hstem|+0.125|[-0.160,+0.550]|3/6|-0.300|
|dev_val|acc|mixer_main|+0.396|[+0.120,+0.776]|6/6|-0.167|
|dev_val|macro_f1|hstem_effect_mha|+2.140|[+1.588,+2.815]|6/6|+1.265|
|dev_val|macro_f1|hstem_effect_hstu|+1.614|[+0.954,+2.459]|6/6|+0.252|
|dev_val|macro_f1|interaction|-0.526|[-0.738,-0.296]|1/6|-1.033|
|dev_val|macro_f1|stem_main|+1.877|[+1.278,+2.638]|6/6|+0.759|
|dev_val|macro_f1|mixer_effect_patch|+0.657|[+0.353,+0.978]|5/6|-0.039|
|dev_val|macro_f1|mixer_effect_hstem|+0.131|[-0.160,+0.566]|4/6|-0.300|
|dev_val|macro_f1|mixer_main|+0.394|[+0.119,+0.772]|6/6|-0.169|
|confirm|acc|hstem_effect_mha|+2.823|[+1.654,+4.898]|6/6|+1.542|
|confirm|acc|hstem_effect_hstu|+2.131|[+0.900,+4.204]|6/6|+0.808|
|confirm|acc|interaction|-0.692|[-0.867,-0.571]|0/6|-0.958|
|confirm|acc|stem_main|+2.477|[+1.277,+4.528]|6/6|+1.267|
|confirm|acc|mixer_effect_patch|+0.631|[+0.579,+0.683]|5/6|+0.575|
|confirm|acc|mixer_effect_hstem|-0.060|[-0.179,+0.050]|2/6|-0.250|
|confirm|acc|mixer_main|+0.285|[+0.235,+0.335]|4/6|+0.229|
|confirm|macro_f1|hstem_effect_mha|+2.877|[+1.657,+5.054]|6/6|+1.541|
|confirm|macro_f1|hstem_effect_hstu|+2.181|[+0.903,+4.346]|6/6|+0.816|
|confirm|macro_f1|interaction|-0.696|[-0.865,-0.572]|0/6|-0.956|
|confirm|macro_f1|stem_main|+2.529|[+1.280,+4.678]|6/6|+1.265|
|confirm|macro_f1|mixer_effect_patch|+0.614|[+0.547,+0.680]|5/6|+0.510|
|confirm|macro_f1|mixer_effect_hstem|-0.082|[-0.214,+0.050]|2/6|-0.255|
|confirm|macro_f1|mixer_main|+0.266|[+0.196,+0.336]|4/6|+0.168|

交互结论：当前条件区间不支持确定的正向互补性，不能只凭两个均值宣布协同。
参数、FLOPs、延迟见costs.csv。Patch与HStem参数不同；本12-fit矩阵未提供Stem参数匹配控制，不能把容量、感受野和层级结构效应完全分离。
confirm为历史已用接收机；不用于本阶段选模或选择最优结构。LoRa与WiSig前端不同，不据此宣称统一最优无线模型。
