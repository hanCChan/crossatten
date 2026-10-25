# 论文验证阶段：预登记

日期：2026-10-25。起点：`ee7bb10`。用户授权 WiSig 优先、Shen 两个近期强基线及结构消融并行。
本阶段不修改 00–03 的结果、冻结协议、legacy 或 guard，不新增候选 Loss、Chirp 分支或测试域调参。

## 实验与预算

- WiSig：ManyRx / ManySig，官方四折名单、非均衡输入、每 cell 前 100 包。
- W0 普通 stem、W1 分层 stem、W2 W1+7.67578125 MHz 源域审计 cutoff，均为 32 个 64 维 token、Pre-LN、两层 HSTU、相同五视图。W1 三个 k4/s2/p1 卷积依次 32→48→64→64，输出 32 token。
- C0 为 W1 去 HSTU，仍保留投影、位置、归一化、均值池化及分类器；CP/CF 为普通 k7/k5 stem + patch tokenizer + 相同无 HSTU 分类头，仅宽度以参数/FLOPs 匹配 W1。两种匹配分别报告，不能声称同一模型同时匹配两者。
- 所有 RF 臂候选两套配方：official Adam 1e-3/b128/Plateau factor0.1 patience5 min_lr1e-6/early-stop10/max200；unified AdamW 1e-3/wd5e-4/b64/warmup5+cos200/AWGN[0,50)。都按源验证 CE 最小 checkpoint。
- FAN / IB-RFF 官方结构，IB 从同 fold/seed/配方的 FAN 初始化，候选 α={0.004,0.04}×两配方，共四种；σ=5、epsilon=1e-5。只迁移官方 HSIC 的设备处理，不改变公式。FAN 单独保留。
- WaveMLP 两个固定 J/K 候选：ManyRx (5,32)/(4,16)，ManySig (4,16)/(5,32)，各两配方，共四种。官方源码仅本地导入；不复制无许可证源码。
- Shen：FAN/IB-RFF full/ib×两配方；WaveMLP 再比较 absolute/scaled 输入适配。8192 点 FAN patch512；WaveMLP absolute patch64/s32/J5/K32，scaled patch2048/s1024/J10/K32。
- Shen 结构消融固定 unified 严格 IB：S_base（Pre-LN patch）、S_hstem、S_cnnonly、S_cnnparam、S_cnnflops。S_hstem 初始化与原 S03 hstem 精确一致；新训练器不要求完整训练比特相同，监测 batch 改为128，结果不覆盖旧 fit。

成本匹配只使用结构，不读数据：普通 stem width=1..128 中按参数或 torch-counted FLOPs 最近选择，误差必须 <5%，记录在 `cost_matching.json`。不是准确率搜索。
全部实际结构、参数及候选写入 `arms.json`；先提交再训练。

## 源域选择与测试门槛

WiSig 选择阶段 seed2023，只读各 fold 源 RX 第1–2天。ManyRx 留源名单末6台、ManySig末2台为内部跨 RX 验证；其余源 RX 再按官方70/30包划分训练/checkpoint验证。
每折以 W0/W1/W2 内部 RX 等权准确率均值选共同配方，保证结构比较的训练条件相同；C0/CP/CF跟随同配方。
FAN、IB、WaveMLP按各自内部源 RX准确率选配置；同分按源 CE、臂名打破。不能查看目标RX或第3–4天进行选择。

源域决策冻结并提交后，seed2023/2024/2025在该折全部源 RX第1–2天重训（官方70/30划分选checkpoint）。
W0/W1/W2/C0/CP/CF各自报告；FAN两配方均保留以支撑IB初始化及配方比较，IB/WaveMLP使用内部源域选定配置。
CR为目标RX第1–2天，CRD为第3–4天，CRD为主要结果。每种协议同时报告包等权、RX等权准确率及F1。

Shen 先 seed101/102×两fold探索。只按四个fit的源验证CE，在每个方法/配方内选择band/layout/α，不按dev_val选择。
保留所有探索结果。冻结并提交后，对各方法的官方/统一配置和5个结构臂跑seed201/202/203×两fold。confirm RX仅报告，标为已多次使用；不读取blind RX。

源域选择记录包含注册文件、来源结果hash；最终训练器验证允许的臂/折/种子及初始化来源。训练期间只评估源验证，不定期查看目标指标。

## 报告与停止

记录全fit配置、预算、checkpoint、曲线、预测、逐RX混淆矩阵/F1、seed和fold波动、参数、计数FLOPs、同设备端到端单包p50/p95及批吞吐。
在正式训练前测成本，避免与训练争GPU；FFT/逐元素运算计入延迟，但不完整计入FLOPs。
WiSig配对：W1−W0、W2−W1、W1−C0/CP/CF；Shen配对：S_hstem−S_base/cnnonly/cnnparam/cnnflops。
预登记改善描述门槛：均值Δ≥1pp、WiSig至少10/12正配对（Shen至少5/6）、最差RX下降不超过2pp。
没有改善记录unclear或harms；seed不等于额外独立RX，不以fit数量声称独立样本量。两数据集分别报告，不合并掩盖失败。
外部方法作公平系统比较；不以击败两个方法直接宣称全领域SOTA。

完成注册实验后停止报告，按证据判断通用LoRa/WiFi方法、LoRa特定方法或工程优化价值，不自动追加模型/Loss。
测试故障修复可重跑，原失败日志和运行尝试必须保留。

## 执行记录

`bc30820` 已在任何本阶段训练之前提交注册配置。`3649963` 提交源域访问/冻结哈希/最终白名单的门槛补强、69/69 smoke 检查和独占设备上的成本实测。
2026-10-25 已启动正式源域训练：ManyRx、ManySig 各72个基础fit，Shen 68个基础fit；对应32个WiSig及32个Shen IB fit必须等待匹配FAN checkpoint完成。
smoke仅用于检查代码可运行，不选配方或配置，也不作为准确率证据。

```bash
env PYTHONPATH=lib python 06-paper-validation/06-4-summary.py
env PYTHONPATH=lib python 06-paper-validation/06-4-summary.py --require-complete
```

汇总会拒绝损坏/不完整记录，显示缺失fit和失败日志，将fold与seed的波动分开报告。
正式完成后以 `RESULTS.md` 与 `summary/report.json` 为准。FAN上游CUDA median存在非确定性警告，保留上游实现，不宣称固定seed能保证逐比特复现。
