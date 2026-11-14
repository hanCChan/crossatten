# 外部方法源码与数据权限审计

本表描述固定源码的事实与 P-H5x160 适配边界，不将作者论文分数当作本机实验结果。固定文件及校验值见 [UPSTREAM.json](UPSTREAM.json)。

| 方法 | 固定 commit | 原始输入/任务 | 训练/监督 | 目标数据/额外数据 | P-H5x160 资格 |
|---|---|---|---|---|---|
| Shen TMC2024 | `8910103a3fe953a746f964459a37917c5ff1ce03` | 8192 complex IQ→52×126×1 CIS；10 TX；支持多 RX 训练 | TX CE + RX CE，经 GRL 反转共享特征的 RX 梯度 | 源 RX 标签；代码另有目标微调、5 包融合、协作 RX 分类入口 | 只采用源域 RA 与单包推断，显式声明任务适配；主表禁用微调/融合/协作入口 |
| ASKNet-LoRa | `f00846d68425315b68ae69b8c6b712e08559c2fe` | [B,2,8192]→LSG→SKO→CIS→ResNet | TX CE；单阶段 Adam；无需 RX 标签 | 官方入口为 receiver drift 数据；不需要目标训练样本 | 以固定包适配 loader；完整默认模型资源验证待完成 |
| IB-RFF | `5a03a04da86c4bbf5cf90f41a55c268bf2152203` | 公开 main 写死 [B,2,256]、WiSig ManyRx/ManySig | FAN 的 SAF+FAF+DSCNN；先 CE、后 CE+α HSIC；无需 RX 标签 | stage2 需其自己的 source-only stage1 权重 | 原版不能直接输入8192；缺官方 LoRa patch/α/训练协议，须标“Shen适配” |
| Shen TIFS2022 | `00948769c05487315b88b7edf40d903fcf8e5b7d` | complex8192→102×62×1 CIS；triplet 512维归一化 embedding + 15NN | margin0.1 triplet；训练 TX 与登记 TX 分开 | 原例提取器 TX1–30；登记/识别 TX31–40；另有离线多径/Doppler增强 | source-only闭集适配可设计，但改变原开放集/登记任务；不引入其他 TX 或目标登记样本 |
| Shen TIFS2023 | `8ed263ce895f2bed5e1a7349d995f7eb464982bb` | SF7/8/9，CIS(win64，无crop)→序列；Transformer/GRU/LSTM/FCN | CE，AWGN；单 N210；每TX/SF训练2500、测试500 | 多SF数据相对当前SF7额外 | 可选“SF7跨RX适配”，不声称原多SF复现；本阶段不列默认拟合 |

来源：[TMC作者仓库](https://github.com/gxhen/receiverAgnosticRFFI)、[ASKNet作者仓库](https://github.com/BeechburgPieStar/ASKNet-RFF)、[IB-RFF作者仓库](https://github.com/BeechburgPieStar/IB-RFF)、[TIFS2022作者仓库](https://github.com/gxhen/LoRa_RFFI/tree/main/Openset_RFFI_TIFS)、[TIFS2023作者仓库](https://github.com/gxhen/lengthVersatileRFFI)。TIFS2022应使用 `Openset_RFFI_TIFS`，仓库的 `Closed_set_RFFI` 不是同一个方法。

## Shen-CIS + RA/DANN

`adv_net` 的输入被写死为52×126×1；Conv7/s2(32)→2个32通道残差块→2个64通道残差块→AvgPool2→channels-last Flatten→Dense512→L2 normalize。TX/RX 两个头均为 Dense128/ReLU，再输出10类/源RX类别。GRL系数1；两项CE权重1:1。5源RX时训练总参数13,509,263，移除RX头后的TX推断参数13,442,954。

官方 `advsarial_training` 默认每TX/RX先取800包，池中90%/10%切训练/验证；batch64，SGD1e-3、momentum.9，最多500轮；Plateau因子.2/patience10，early stop20。生成器每batch随机抽包并加AWGN，SNR均匀取0到49dB（源码 `range(0,50)` 取首尾）；验证也动态加噪，监控两头 `val_loss`。原函数没有保存最佳epoch或restore_best_weights，返回停止时模型。采集日期及一套唯一“论文标准训练RX清单”无法仅据这个入口推定；main中有多套注释示例，不等于冻结的可执行实验配置。

发现的具体缺陷：验证生成器接入 `data[num_train_samples:]`、`tx_label[num_train_samples:]`，却接入 `rx_label[:num_train_samples]`。前后不是相同包的RX标签。不能照搬这个错误。候选端口 [s24_dann.py](s24_dann.py) 延用现有保持TF same padding与channels-last Flatten的编码器，补充RX头及GRL；已检验GPU前反向和梯度符号，但尚未完成TF权重转移对比。

正式适配需在训练前写死：160/40固定划分；正确源RX标签；源验证CE的具体组合/噪声规则；最佳checkpoint或停止时checkpoint；是否保留原随机batch抽样。必须逐条披露与官方的差异。不能把历史 non_adv CIS 结果当成新DANN结果，也不能把协议改变归因为DANN本身。

原脚本 `edge_clf` 默认 `fuse_num=5`，另有fine_tune和collaborative_clf；本主任务不调用这些入口。候选推断返回每包10类logits，无多包标签分组融合。

## ASKNet 必须使用 Lora 实现

`Lora/run_CIS.py` 当前默认 **rtl_6 day1训练、n210_1 day1–4测试**，per_class800，训练池70/30随机划分；这与附件概述的训练方向不同，也与当前10台源/开发RX协议不同。loader仅接受这两个RX，并使用`*_day*_wireless_*`文件名，不能原样调用。需替换为已审计固定原包接口，保留每包功率归一化，不自动截断、补零、取最后10类或额外映射标签。

原版为完整8192点、候选k=[128,256,512,1024,2048,4096]，Gumbel-hard训练、argmax推断，tau随epoch从5线性至.1；Adam1e-3/wd0/batch128/200epochs、earlystop20、Plateau(.1,10,min1e-6)，source_val CE选最佳权重。CIS win128/crop.3。ResNet头含AdaptiveAvgPool(13,16)，不是Shen原版的13×31直接Flatten；其torch STFT及epsilon处理也与原Shen scipy实现有细节区别。

完整默认构造在meta设备核算为74,153,872参数元素、141,246,353实自由度；参数约565MB。其中一个8191×8191 complex64矩阵约537MB；仅该矩阵权重、梯度、Adam两个矩就至少2.147GB，另有QR/矩阵乘法反向图及workspace。每次forward重新QR，复杂度O(n³)。这不是附件里暗示的所有分支均小模型；也不能把WiSig PatchNet成本写到LoRa这一行。见 [ASKNET_RESOURCES.json](ASKNET_RESOURCES.json)。

本机RTX4060 Laptop 8GB的完整默认forward/backward与batch128未验证；没有尝试可能长时间占用设备的8191维QR。只运行了原CIS、全尺寸cutoff masks，以及明确标注的7模态算子梯度检查。若需要缩减候选或低秩实现，必须新命名为资源受限适配，与原版分开，不能悄悄替换。

## IB-RFF：公开宣称和实际代码的边界

README宣称LoRa验证；固定tree中的main、run_all和utils/load_data只有WiSig，`choose_backbone` seq_len写死256，dataset choices没有LoRa。当前代码不能提供LoRa官方的日期/RX/TX/包数、patch或α证据。因此本次不臆测这些细节，也不加载已见目标数据训练的发布权重。

公开WiSig模型为SAF复频率滤波+阈值掩码、overlap patch embedding、FAF token频率滤波、DSCNN残差瓶颈和分类头。stage1 α=0，stage2从本折stage1权重开始，以CE+α·normalized HSIC-CCA(H,X)微调。高斯核使用维度缩放的σ=5，中心矩阵与正则求逆的精确定义见原 `utils/hsic.py`；不能换成一般HSIC当作相同损失。公开代码核矩阵移至CPU，需保留/披露这部分训练成本。原默认Adam1e-3、batch128、200epochs、earlystop10、Plateau(.1,5)。

`run_all.sh` 给不同CR/CRD测试任务的best α；这些数值不能按本次目标成绩选择。合法方案是预先固定历史源域选择过的配置并验证初始化证据，或预登记只在本折源RX内部的相同预算搜索。S06确实有Shen8192适配，见历史对照；它不是公开LoRa代码已经齐全的证据。本阶段保留这一项为“适配配置未正式冻结”。

## Shen TIFS2022与2023

TIFS2022的核心是triplet而非CE替代网络。原RMSprop1e-3、margin.1、batch32、最多1000epochs、earlystop20、Plateau(.2,10)，一次AWGN[20,79]dB；原triplet独立采样anchor和positive，可能抽到同一个原包。公开`augmentation.m`另定义Rician信道：RMS delay 10–300ns、Doppler0–10Hz、K0–10、指数PDP至−30dB、Jakes谱、补50点并移除信道滤波时延。不能用AWGN替代整个信道增强而称完整复现。

P-H5x160若纳入，应仅在source训练包上学习embedding，15NN gallery也仅来自这批训练包；源验证只用于选提取器，dev RX只查询。原额外TX预训练、目标域登记都不得混入同预算主表；取消它们会改变原任务，必须命名为闭集源域适配。完整信道增强移植、triplet采样和kNN端到端尚未接入，本阶段只确认原始CIS接口。

TIFS2023的win64、无频率crop，8192输入对应254×64序列。官方有两个Transformer encoder、8 heads、FFN256；不能直接用现有PlainMHA冒充。`model_type == ('lstm' or 'gru' or 'transformer')` 在Python里只比较'lstm'，推断处类似表达式也只比较第一项；正式适配前需修复并记录。SF7/8/9训练数据和单N210接收机与当前SF7跨RX不同，列候选而非默认主表。

## 未核实到代码的候选

[Ma TIFS2025论文](https://arxiv.org/html/2512.12070v1)公开三阶段对比预训练、Siamese分类和推断。方法复现入口为论文IV–VI及VIII：NT-Xent温度.05、原batch32生成64视图，结合CE；Shen数据实验训练RX为B200-1/B210-1，不能直接移用到本协议，尤其B200-1属于历史confirm。额外预训练需单列数据预算。本次检索论文、[作者主页](https://skylarddd.github.io/)及[作者组代码索引](https://junqing-zhang.github.io/research/dataset-code/)，未核实到该篇可运行作者仓库；这是检索范围内的未找到，不是证明不存在。

RICAI2024 gMLP（DOI 10.1109/RICAI64321.2024.10911769）本次未取得可核实的作者代码和完整配置，保留候选，不采用猜测的gMLP结构。联邦学习/目标适应方法若需要额外权限，也不能直接混入source-only主表。

公开仓库能读取不自动等于有明确再分发许可；固定tree有些仓库没有独立LICENSE。作者源码存忽略目录，Git记录来源与hash；新增端口应保留出处与适用许可说明。历史文档中的许可断言不作为本次新的授权证据。
