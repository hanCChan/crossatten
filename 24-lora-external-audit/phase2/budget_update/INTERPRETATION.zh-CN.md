# S24 第一批完整外部比较的解释

2026-11-18。依据RESULTS.json、全部270个RX单元及独立预测核验。48个新基线和六个冻结H0已统一评估；ASK尚未完成整组六模型，不能据此宣布S24全部结束。

## 主要结果

H0本次CPU复核Accuracy为97.333%、Macro-F1为97.325%。Native最强已完成外部基线是Shen-DANN，Accuracy为94.208%，H0−DANN为+3.125pp，RX条件95%区间[+1.075,+5.103]pp，6/6个fold/seed均值差为正。Strict-IB最强已完成基线是同骨干Shen CE，Accuracy为94.702%，H0−CE为+2.632pp，条件区间[+0.287,+6.942]pp，5/6配对为正。

这些区间条件于固定模型、种子及历史开发RX，未作多重比较校正；不是新独立测试。H0的历史GPU报告97.335%保留原样，本批为同CPU执行条件下重新推断的97.333%，不能把两者混作同一次运行。本轮未在当前占用的GPU上重跑H0 Dev以定位后端差异，不声称已经找到具体原因或全部逐包差异；六个checkpoint SHA保持冻结值。

## 优势的RX分布需要如实说明

Native下H0并非每台RX都超过Shen：对b200_mini_1和b210_1，H0低于CE/DANN；对rtl_6与DANN接近。当前平均优势来自多个其余RX，不应写成所有接收机全面领先。

Strict-IB下H0−CE在pluto_1上为+21.533pp：H0为93.683%，CE为72.150%。这一RX约贡献总体+2.632pp均值差的82%；其余九台RX平均差约+0.531pp，rtl_6上H0低0.400pp。对DANN也有类似集中现象。该分解是事后描述，不是新的预登记假设；不删RX、不替换主表或重选比较。可以说H0在当前协议中改善平均与最差RX表现，但不能把均值差当作普遍RX不变性的证明。

## 两个控制实验提供了什么

Native DANN−CE为+0.673pp，条件区间[+0.208,+1.233]pp，5/6配对为正；Strict-IB为−0.070pp，区间[−0.247,+0.152]pp。去RX对抗训练在Native配置下有有限增益，在Strict-IB下没有明确增益，不能宣传对抗训练在本协议中普遍有效。

IB Stage2−FAN Stage1在Native下为+0.978pp，区间[−0.037,+2.195]pp；Strict-IB下为+3.983pp，区间[+3.373,+4.582]pp。Stage2在Strict-IB下的增益更清楚，但两者总体都弱于Shen与H0。FAN/IB分数只刻画已冻结的低数据预算Shen/LoRa适配，不能推广为原论文在所有数据与配置中的能力；不根据本批Dev分数重新搜索alpha、patch或模型。

## 剩余工作和稿件边界

保留ASK Native六项：第一项已完成，第二项继续，另四项排队；整组完成后统一封存和评估。Strict-IB ASK六项全部延期，NA不是0分或资源失败。统一同机GPU成本测量待Native训练结束后执行，当前CPU推断时间不作为部署速度对比。

本批支持当前固定协议下的外部竞争力。同步敏感性、非全新Dev、缺少独立新RX、IB移植边界及参数化归因限制，仍按[EVIDENCE_SCOPE.zh-CN.md](EVIDENCE_SCOPE.zh-CN.md)保留。此次输出错误的修复只将NumPy标量转换为JSON原生类型，未重跑推断、改动统计定义或增加优化步。

对比图见[Accuracy](paper/accuracy_comparison.png)与[配对差值](paper/paired_differences.png)；逐RX表见[PER_RX.csv](PER_RX.csv)，核验见[PREDICTION_INTEGRITY.json](PREDICTION_INTEGRITY.json)。
