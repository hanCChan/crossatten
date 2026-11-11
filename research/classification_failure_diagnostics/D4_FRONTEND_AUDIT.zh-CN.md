# D4：输入版本、前端与包对应关系审计

本文件是静态代码与封存元数据审查。**本轮没有打开 compact pickle、IQ、均衡版本、特征缓存或 checkpoint；没有新增前端消融。** 因而不能把以下操作对信息的潜在影响写成已测得的分类因果效应。

## 实际使用的版本

| 链路 | 核查结果 | 依据 |
|---|---|---|
| WiSig compact 元数据 | 提供 `equalized_list=[0,1]`，TX 顺序固定，日期按时间排序 | [历史导出清单](../../05-wisig/out/05-1-ManyRx-manifest.json)、[S13 数据代码](../../13-mixer-cfo-mechanisms/data.py) |
| 当前 S17B–S20 数据 | 使用 `[tx][rx][day][0]`，即未均衡版本；每 TX×RX×Day 取前 **100** 包，不是 compact ManyRx 的上限 200 包 | [wisig_cells](../../13-mixer-cfo-mechanisms/data.py)、[S17 缓存代码](../../17-rx-distortion/s17_data.py) |
| 数据缓存 | 每包先功率归一化，再转为 float32 的 `[N,2,256]`；缺失 cell 保留缺失 | [power_normalize](../../lib/xa/wisig.py)、本轮 [INPUT_REGISTRY](INPUT_REGISTRY.json) 与分组统计 |
| HStem / ASK 输入 | 相同训练/验证索引、相同 source 输入与 held 输入 cell 哈希；ASK 没有额外使用均衡版 | [S19 数据装载](../../19-tx-spectral-evidence/s19_data.py)、[S20 顺序预测](../../20-asknet-source-comparison/s20_run.py)、[S20 协议](../../20-asknet-source-comparison/registration.json) |

正式核验读取了 16 个封存预测文件及 20 个标签/索引文件；预测 NPZ 的实际字段严格为 `logits/y/rx/day`，不含 IQ 或隐藏特征。训练支持由标签和 train_ids 得到；没有训练集逐包 logits，因此不能把验证风险叫作已测得的训练组损失。

## 对两类“配对”的不同结论

**同一输入、不同模型：可按确定的数据血缘重建顺序对应。** source-validation 两模型都调用相同冻结 val_ids，逐段顺序预测；held-source 使用相同顺序的 RX/day 单元和相同单元字节哈希，之后顺序预测。运行时逐行核对 y/rx/day 与 source 索引标签。旧预测缺少独立 immutable packet ID，因此这一结论依赖被审计的提取与预测程序，没有独立重新读取 IQ 来核验。

**同 RX、不同 fold：只有完全相同 source cell 哈希和顺序时才重建对应。** folds1–3 的 6 个 held RX 满足这一条件，可分析相同输入的重复预测；它们不是新增包或新增独立 RX。不同 RX 的第 k 包始终不做配对。

**未均衡 / 均衡版本：本地逐包对应尚未核验。** [WiSig 官方说明](https://cores.ee.ucla.edu/downloads/datasets/wisig/) 描述了同一前导码片段的两个版本，但现有科学缓存只保存版本 0、TX 标签与顺序，没有携带跨版本包 ID、版本 1 的逐包预测或已验收对应表。`equalized_list=[0,1]`、相同 TX/RX/day 标签或相同包数都不足以独立证明本地提取结果逐行相同包。

本轮不新开原始 pickle 来补这项实验。若后续分析均衡前后，需要另行登记具体合法 source RX/day、版本、提取路径、ID/时戳或有来源的映射，并核验同步、裁窗和丢包行为；不能靠两个版本按下标自动连接。即使映射核验通过，均衡也不能解释为“只消除 RX 硬件”。

## HStem 前端保留了什么

冻结 B0 的路径是 [WiSigModel](../../13-mixer-cfo-mechanisms/wifi.py) → [frequency_views](../../lib/xa/explore_models.py) → [first-order phase_features](../../09-phase-invariance/phase_features.py)：

1. 已功率归一化的 IQ 在 tokens 与 features 路径再次调用 RMS；保留历史 eps 和重复调用约定。
2. 256 点默认 FFT 后 fftshift，复谱 Re/Im 按频谱 RMS 缩放。
3. 幅度通道为 `log1p(|FFT|)`，随后包内去均值并除标准差；不是未经变换的加性 `log|X|+log|H|`。
4. 追加相邻复谱归一化乘积的 Re/Im，幅度有效阈值为 0.05，左侧补 1 个频点，得到 5 通道。
5. 原 Re/Im 仍存在，整网不具有自动的全局相位不变性；追加通道的某种不变性不能外推到全部网络。这沿用 S17B 已归档的修正。

功率归一化显式丢弃包总功率尺度；log1p/zscore、相位 mask 与有限窗口也会改变后续分类可用的信息。但本轮只有该输入路径的冻结结果，无法证明某项处理是具体错误的原因，或其取消必然改善未见 RX。

ASK 的官方 FFT、截止选择、复数 Koopman、IFFT/能量归一化与时域 Patch 属于另一表征和训练配方；原输入仍是上述同一份未均衡 IQ。整体对照差异不能归因于均衡、单个频带、Koopman 或训练时长之一。原结构和成本见 [S20 正式归档](../../20-asknet-source-comparison/RESULTS.zh-CN.md)，本次没有重跑成本测量。

## D4 的当前出口

确认了**实际输入版本和处理链路**，确认了可用于 D3 的同输入顺序血缘；**没有确认本地 EQ/non-EQ 逐包映射，也没有测得前端因果效应**。因此目前不能用均衡前后比较为新的接收机扰动族背书，也不能据此开始新输入训练。
