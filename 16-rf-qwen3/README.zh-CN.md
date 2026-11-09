# S16：RF-Qwen3跨接收机分类探索

最新任务书中的S16优先：本阶段研究Qwen3；第一份复盘中物理接收机响应诊断/增强暂留后续课题。S15完成并退出后，本阶段才使用同一张RTX 4060 Laptop 8GB。

工作区 `D:\crossatten\worktrees\s16-rf-qwen3`，分支 `exp/s16-rf-qwen3`。S00–S14代码、结果和STOP不改。Shen blind、WiSig真实目标RX和date3–4不访问。

## 科学问题与解释范围

1. 小型Qwen3组合结构是否优于同深度、近似参数的普通Transformer？R1−R2回答此问题。
2. 官方文本预训练权重能否迁移到RF连续特征？R3−R4隔离相同冻结配方下的权重作用；R5若进入确认，还须R5−R6。
3. 更大模型相对小型MHA/RF-HSTU的收益是否值得额外计算？比较R0并同时报告成本，但深度、宽度和规模仍有混杂，不能声称单独证明深度贡献。

不预设Qwen3有效，不把权重收益唯一解释为语言知识。随机冻结/随机LoRA都不等价于全参从零训练。

## 输入、输出与共同RF前端

WiSig输入 `[B,2,256]`，沿用S13 B0的RMS归一化、FFT及first-order相位不变特征（5通道）、HStem，得到 `[B,32,64]`。

**所有R0–R6共用同一个在合法源RX上训练、选模后冻结的RF tokenizer。** 本轮隔离的是后端与预训练作用，不是每个模型独立端到端最优训练。R0两种尾部也重新训练；S13端到端B0只作历史背景，不能直接作为同预算对照。

Pilot前端复用S13 B0源checkpoint：first源RX留出组，源训练池70/30随机划分seed16001。尾部也严格使用该70/30划分，避免RF前端训练见过尾部验证包。缓存token只在本阶段生成，旧cache仅读。独立确认改为last源RX留出组，必须重新训练共同RF前端，不能复用见过last组的first组前端。

Qwen连续输入：Linear64→hidden；追加末尾可学习CLS；`inputs_embeds=[B,33,hidden]`；原生因果mask；取末尾CLS隐藏态，分类头输出10类ManyRx或6类ManySig logits，用CE训练。无文本tokenizer、文本生成或KV cache。普通Transformer对照也为因果、末尾CLS。

## 实验臂

| 臂 | 骨干 | 更新范围 |
|---|---|---|
|R0_mha/R0_hstu|历史式64维、2层尾部|整个尾部；共同RF前端冻结|
|R1_qwen_small|随机原生Qwen3：128维、4层、4Q/2KV heads、FFN384|整个尾部|
|R2_transformer|随机普通Transformer：128维、4层、匹配FFN|整个尾部|
|R3_pretrained_frozen|官方Qwen3-0.6B|投影、CLS、分类头|
|R4_random_frozen|同配置随机Qwen3-0.6B|同R3|
|R5_pretrained_lora|官方Qwen3-0.6B+LoRA|投影、CLS、分类头及LoRA|
|R6_random_lora|同配置随机骨干+同LoRA|同R5；条件确认臂|

R1/R2的骨干参数自动匹配误差≤2%，只按参数数目选择FFN，不读成绩。R1使用Qwen3结构，不宣称是0.6B预训练模型。R2使用GELU FFN/LayerNorm/普通MHA/学习位置；R1使用SwiGLU/RMSNorm/GQA/QK norm/RoPE，比较的是组合而非单项机制。

官方模型固定 `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`；28层、hidden1024、16Q/8KV、head_dim128（Q投影宽度2048）。使用bare Qwen3Model，删除词嵌入，LM head不实例化。下载文件含751,632,384个序列化参数，包括两份155,582,464参数的词表矩阵；实际RF骨干440,467,456参数，另加投影、CLS及分类头，详见审计JSON。完整Qwen冻结权重BF16，可训练投影/分类/LoRA及优化器状态FP32。

LoRA固定r8、alpha16、dropout0、q_proj/v_proj、bias none。R3/R4投影与头的初值相同；R5/R6还匹配LoRA初值。骨干冻结仍保留投影与CLS梯度，禁止对整个冻结骨干套no_grad。

## 两阶段内的执行门槛

S16-P0：仅合成输入/随机标签工程检查，验证实际官方权重前反向、连续embedding、CLS、因果性、梯度、移除词表及显存。R5测microbatch16/32/64/128；预留40,000×32×64 FP32缓存空间，峰值allocated≤5.75GiB、reserved≤6.5GiB，选择合格者中吞吐最高的batch。按30,000训练包、20轮、1.4安全系数估计单fit≤2小时才放行大模型。资源不通过时保留记录，仍运行R0/R1/R2的12 fit，不把资源不足解释为准确率失败。P0输出batch1/16的实际GPU尾部及完整RF路径延迟，均标明合成输入。

S16-Pilot：ManyRx四fold、尾部seed17001、固定split16001。R1–R5新增20 fit，加同协议R0_mha/R0_hstu共8 fit，合计28。若大模型资源不通过则12 fit。

所有尾部统一20轮、有效batch128；大模型按P0固定microbatch进行加权梯度累积。AdamW lr1e-3、wd5e-4、warmup2/cos20、clip1、无dropout，BF16 autocast，最小合法源RX等权CE选checkpoint；不按留出源RX选轮数。数据顺序与head seed在各臂配对。无按模型额外搜索、AWGN、TTA或额外训练轮数。

Pilot准入：相对每个规定对照，ManyRx平均≥+1pp，≥3/4 fold为正，最差配对RX不低于−2pp，RX聚类5000次条件bootstrap的95%下界>0。R1须过R2和两R0；R2须过两R0；R3须过R4和两R0；R5须过R3和两R0。每类至多保留1个：小型架构R1/R2、预训练R3/R5；按最小对照均值最高、固定臂次序打破平局。此门槛是探索启发式，无多重比较校正。

S16-独立确认：只对Pilot已准入候选，依次confirm1(seed17011)、confirm2(seed17012)，均使用last源RX组、ManyRx和ManySig四fold。每轮先新增8个合法源B0前端，再训练候选与必需对照。R5须增加R6；最多64个尾部fit/轮。确认控制臂不能变成新候选，某轮失败不得进入下一轮救结果。ManyRx继续满足Pilot门槛，ManySig均值≥0、≥3/4 fold不低于−1pp、两数据最差配对RX均不低于−2pp。两轮分别通过才输出可另行预注册真实目标评估；不自动打开真实目标。

预算上限：Pilot28尾部fit；条件确认最多128尾部fit+16个轻量B0前端。确认通常因源门槛早停，并非无条件执行156个尾部fit。没有科学失败自动重试。任何异常保留attempt、failure、日志并STOP_TECHNICAL_REVIEW。

## 环境与证据

Qwen独立环境 `D:\crossatten\.venv-qwen3`，通过system-site-packages复用旧CUDA torch，新依赖版本见requirements-qwen3.txt与ENVIRONMENT.json；不升级旧环境。s16_core仅为新进程保留旧conda DLL目录句柄，解决torchvision依赖加载。模型资产、特征缓存、checkpoint和逐包预测不入Git，配置、SHA、矩阵、完整汇总与负结果入Git。

入口s16_execute.py；STATUS.json显示等待/工程/训练/完成；runs/*/stdout.log与live.json显示逐轮进度。design.json先于P0提交；P0结果先提交，再冻结Pilot矩阵；每轮确认前固定矩阵及新前端。没有改动S15正在执行的注册代码。

## 技术来源

- [Qwen3官方模型与配置](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca)：实际骨干与权重。
- [Transformers v4.51.3原生Qwen3实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py)：连续inputs_embeds、因果mask、GQA/QK norm/RoPE。
- [PEFT LoRA文档](https://huggingface.co/docs/peft/v0.15.0/en/package_reference/lora)：受限参数适配；LoRA本身不是本课题原创。
- RF输入、相位特征及HStem来自已冻结S09/S13；MHA/RF-HSTU来自当前仓库历史实现。跨领域组合属于待验证研究设计，未预先宣称创新成立。

原始任务书原样保存在proposal_*.txt。附件内时间序列论文只作为用户提供的研究动机，未作为已核验的跨RX证据，也不据此宣称语言预训练有效。
