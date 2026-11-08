# S14：把RF-HSTU与普通注意力的比较做完整

独立分支：`exp/s14-stem-mixer`。
隔离工作区：`D:\crossatten\worktrees\s14-stem-mixer`。
阶段目录：`14-stem-mixer-attribution`。用户补充方案保存在 `PROPOSAL_ORIGINAL.txt`。
现有S13仍在 `D:\crossatten\stage_1`，原进程、配方和48/40矩阵保持原样；本阶段对S13只读。

## 已排定的工作

1. S13先完成48个LoRa fit，提交全部checkpoint SHA，再统一评估dev_val和历史已用confirm；随后运行原有40个CFO pilot，按原门槛决定是否确认。
2. 一旦S13完整P1报告提交，S14仅在CPU生成 `ATTENTION_COMPARISON_S13.md`。不拿源验证98%来判断跨RX优劣。
3. 等S13主进程退出、最终状态和决策都完整后，才在本分支合入S13最终证据并提交实际S14注册。
4. 必做Stem×Mixer两因素补充：新增Patch+MHA、Patch+RF-HSTU各2折×3种子，共12 fit；另外两格复用S13的HStem+MHA/HStem+RF-HSTU。
5. 复用S13的CFO模型做分类器级源域诊断，不增加WiSig训练、不删除异常样本。
6. G1/G2只在预设的完整P1门槛通过后新增12 fit；不满足就明确记录NOT_ADMITTED。
7. 浅层无标签适配只做文献/代码可行性审计，另立UDA/TTA协议后才能训练，不混入本轮零样本DG。

S14等待器每30秒读取S13状态。S13有技术错误时停止交接并留记录，不停止、重启或替代S13。

## 公平配对如何保证

训练直接调用未修改的S13 `train.fit` / `evaluate_shen` 函数。
显式适配器只更换输出目录、角色受控的数据读取函数、模型构造和注册检查；优化器、随机流、步序、AWGN、CE、200轮、预热/余弦和源CE选模不改。
种子仍为 `stable_seed('P-H5x160',fold,seed)`，seed取201/202/203。

新增Patch臂从对应S13模型开始构造，仅替换 `cnn_stem.tokenize`。
不能直接分别随机初始化Patch/HStem两个完整模型，否则会因RNG消耗顺序不同，连共同层和Mixer权重也变化。
本实现逐项验证共同卷积、位置、Mixer、分类头等初始参数完全一致；只允许Tokenizer不同。
IQ输入 `[B,2,8192]`，相同TrueIB五视图，输出仍为32×64 Tokens和10类logits。

| Stem | 普通MHA | RF-HSTU |
|---|---|---|
| Patch | E1_patch_mha，新训6个 | E2_patch_hstu，新训6个 |
| HStem | 复用S13 A1_mha，6个 | 复用S13 A3_rfhstu，6个 |

先在同一RX/fold/seed内计算HStem收益，再计算两种Mixer收益之差。
交互差值 `(HStem+HSTU−Patch+HSTU)−(HStem+MHA−Patch+MHA)` 用接收机聚类bootstrap，避免把两项均值当作独立样本。
每臂报告准确率、Macro-F1、最差RX、seed变化，配对差值与条件区间；参数、计数FLOPs、4060实测延迟/吞吐分别报告。
接收机是统计单位；区间条件于现有训练种子，未做多重比较校正。
成本测量另加相同MHA参数的PyTorch SDPA推理实现，避免把手写注意力的算子调度成本误认为注意力机制本身的效率。它不新增训练或精度实验臂；具体后端依赖本机torch/硬件/精度，不预先声称使用FlashAttention。
FLOPs统一为共同前端/卷积计数加Mixer的QKV、QK、AV、FFN稠密公式，FMA计2，避免融合SDPA算子漏计。FFT/逐元素运算仍不完整覆盖，S13旧计数保留但不和新口径混用。
Patch与HStem参数量和感受野不同，这12个fit不替代Stem参数匹配控制，不能完全分离容量和层级归纳偏置的原因。

## HSTU优劣的判断

S13完整主表同时保留普通MHA、自然Transformer及其参数匹配版本、CNN-only、Sigmoid/Softmax/SiLU打分控制。
区间跨0表示方向不确定，不能说等效；普通MHA更好时如实报告，不把“使用HSTU”当作第二篇的新贡献。
自然Transformer FFN256与匹配FFN32分开解读。A4/A5仅隔离RF门控结构中的打分函数，不称为原始HSTU完整复现。

条件G1/G2预算门槛在看到完整P1前登记：
RF-HSTU在dev_val上对自然/匹配MHA和自然/匹配Transformer四个对照均满足平均≥+1pp、至少5/6正向、最差RX平均变化≥−2pp且接收机条件区间下界>0。
这是限制追加预算的描述性门槛，不是校正过的统计显著性结论；confirm不能用于准入。
G1保留QKVU投影但不乘U门控，G2保留bias参数但不加入相对偏置；报告分配参数与实际有用参数，避免拿闲置参数伪装参数效率。

## CFO补充诊断

使用每折留出的源RX第1–2天全部合法包，逐RX/day、TX、coherence高/低、观测总CFO幅度分组。
TX单类组用准确率/召回率；其余组明确固定类别Macro-F1和类别覆盖数。
组内仍按RX等权，空组NA，不把包数当RX数。
所有模型使用预先冻结的±2/5/10kHz与0扰动；分组以未扰动估计为准。
关键比较是Dual(raw,corrected)与Dual(raw,raw)，必须同fold/seed/group配对。

S13列出的长尾cell一律保留；额外用固定起点0/10/20/40/60/80、固定长延迟40/60/80检查窗口敏感性和模2π相位一致性。
这些是分析，不选最好起点、不改校正阈值、不排异常包。没有天然CFO真值，因此不能把观测CFO分组叫作真实估计误差分组。
S13候选没有通过源域门槛就保持STOP，诊断不能借真实WiSig CRD挽救候选。

## 运行与查看

```powershell
& 'D:\miniconda\envs\crossatten4060\python.exe' 'D:\crossatten\worktrees\s14-stem-mixer\14-stem-mixer-attribution\status.py'
```

后台交接日志为 `handoff.log`；等待期间不占用GPU。不要重复启动。
`design.json`是事前设计，`registration.json`仅在S13全部完成后生成并提交。
`planned_jobs.json`只含必做12项；实际 `jobs_p1.json`由完整P1证据决定是否加入G1/G2。
本阶段所有新源域模型统一选模并提交SHA后，才允许评估dev_val/历史confirm。
Shen blind、WiSig真实目标RX和第3–4天始终禁止。
