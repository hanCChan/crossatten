# S15：HStem参数匹配归因

分支 `exp/s15-hstem-attribution`，工作区 `D:\crossatten\worktrees\s15-hstem-attribution`。
本阶段按第二份任务书优先执行Patch-Lite×MHA/HSTU，新增12 fit。可选去BN的12 fit未排入本轮；不重复S13/S14模型。

## 先固定架构，再运行数据

自动枚举groups∈{1,2,4,8,16,32}、中间通道为groups倍数且≤256的504个候选。
每个候选为 `Conv1d(32,mid,256,stride256,groups=g)` 后接线性1×1投影到64，不增加BN或中间激活。
按两个完整模型相对HStem的最大参数误差最小选择，要求均≤1%；同分时更少groups、mid优先。没有读取训练/验证成绩选择结构。

选中groups=8、mid=80，Tokenizer为87184参数，HStem为86608参数。
完整模型：Lite+MHA 134266 vs HStem+MHA 133690；Lite+HSTU 142968 vs HStem+HSTU 142392，误差均不足0.5%。
Patch-Lite仍改变低秩/分组归纳偏置，参数匹配不能唯一隔离“层级结构”的因果贡献。

输入仍为TrueIB五视图，输出仍为32×64 token；CNN步长256、首Token中心127.5与旧模型一致。
CNN理论感受野：Patch/Lite为266，HStem为350。由于全局带限与FFT视图，不能将此数字当作完整模型在原始IQ上的物理感受野。

## 完全配对

2 folds×3 seeds(201/202/203)×2 Mixer，共12 fit。
直接调用冻结S13训练循环；AdamW、batch64、AWGN0–50dB、5轮预热/200轮余弦、源RX等权CE选模与128评估batch均不变。
非Tokenizer初始权重、数据顺序及噪声随机流保持匹配；旧Patch/HStem各12个匹配控制直接复用。
全部新checkpoint先统一提交SHA，再评估dev_val和历史已用confirm；Shen blind始终禁止。Confirm不用于架构搜索或继续搜索的选择。

报告逐RX Accuracy/F1、最差RX、seed变化、HStem−Lite及Lite−Patch配对差值、接收机聚类区间及参数/计算量/本机延迟。
区间条件于现有种子且无多重比较校正。失败/不确定结果保留，不追加seed救结果。

主脚本 `s15_execute.py`；状态 `STATUS.json`；日志 `execution.log` 与 `runs/*/stdout.log`。
完成后写 `RESULTS.zh-CN.md`、`summary.json`、`decision.json`，由S16等待器读取完成状态后顺序进入下一阶段。
