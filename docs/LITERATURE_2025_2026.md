# 2025–2026 竞争方法：指标、方法与可比性（文献审阅）

**2026-10-25 实测更新：** 本文主体保留阶段03之前的审阅快照；“待下载/待重训”等是当时状态，不是当前状态，也不代表穷尽截至该日的全部 SOTA。
ManyRx、ManySig 已备齐并完成官方权重核验，阶段06公平四折×三 seed 重训的 CRD 为：
IB-RFF **88.07 / 91.09**、WaveMLP **88.06 / 91.04**、本方法 W1 **83.56 / 72.33**（ManyRx / ManySig，RX 等权）。
公平 IB 相对实际匹配 FAN 初始化为 **+0.51 / +2.82 pp**，不能由下文的测试集选 α 风险推断其方法没有收益。
阶段03残差/符号堆叠分支未改善，分层 Stem 有效但 WiSig 未证明稳定结构收益；重复结构不能写成已成功的方法贡献。
当前训练结果、源域选择、成本和限制以[阶段06结论](../06-paper-validation/summary/paper/CONCLUSION.md)及[总报告](../RESULTS.md)为准。
Shen 的 IB-RFF/WaveMLP 同协议最终确认已完成；Ma 2025 不在本轮授权矩阵内，未训练。此更新不新增文献搜索或改写历史证据。

更新：2026-10-25。原则是**只写能核实的内容**。每条结论都标出来源等级：
- **[代码+日志]**：读过官方代码和官方训练日志。
- **[代码+README]**：读过官方代码和 README（README 中的数字是作者报告值）。
- **[论文]**：读过论文全文。
- **[摘要]**：读过 OpenAlex/Crossref 摘要。
- **[元数据]**：只核实了题目、作者、刊物。
- **[用户转述，待核实]**：来自用户的检索摘要。

所有下载都由 [external/fetch_refs.sh](../external/fetch_refs.sh) 完成，第三方仓库和 PDF 放在 `external/`（不纳入 git）。

## 1. 总表

| 方法 | 出处 | 数据 / 协议 | 指标（作者报告） | 参数 / 计算量 | 代码 | 核实 |
|---|---|---|---|---|---|---|
| **本工作 TrueIB-RF-HSTU（ep200）** | — | Shen 20-SDR，P-H5x160（5 源 RX × 160 包/TX） | dev_val 95.84 / confirm 93.89（6 次训练） | 580,008；139 MFLOPs/包 | 本仓库 | [代码+日志] |
| Shen-CIS（官方结构移植） | TMC 2024 | 同上，同协议 | 93.12 / 95.20 | 13.44M；582 MFLOPs | 本仓库移植 | [代码+日志] |
| Shen-CIS rank-16 头 | 本仓库对照 | 同上，同协议 | 92.36 / 94.75 | 658K | 本仓库 | [代码+日志] |
| 严格 IB 的 Shen-CIS | 本仓库对照 | 同上，同协议 | 94.62 / 95.72 | 13.44M | 本仓库 | [代码+日志] |
| **IB-RFF**（FAN + HSIC 信息瓶颈），Yu Wang（NJUPT，单作者） | IEEE TIFS 21:8008–8021，2026 | WiSig ManyRx / ManySig，CR/CRD（见 §2） | CRD：ManyRx 89.19，ManySig 93.57；CR：90.68 / 94.62 | 待算（脚本已备好，被权限分类器拦截） | BeechburgPieStar/IB-RFF @5a03a04（MIT） | [代码+日志] |
| **WaveMLP**（可学习复小波 + MLP），Yu Wang、Meiyu Wang、Juzhen Wang（NJUPT / 杭电） | IEEE IMNS 2026（Atlanta） | 与 IB-RFF 相同的 WiSig CRD 协议 | CRD：ManySig 93.13，ManyRx 88.68 | 8.39 万 / 8.45 万参数；1.27 / 1.40 MFLOPs（README） | BeechburgPieStar/WaveMLP-RFF（含 4 轮权重） | [代码+README] |
| **Ma et al.**（谱图 + Siamese + 对比学习） | IEEE TIFS 2025（arXiv 2512.12070v1） | **Shen 数据（Public Dataset 3）：只用 B200-1 和 B210-1 两台 RX 训练（800 包/DUT/RX），在全部 20 台 SDR 上测试**；另有 3 个 LoRa 数据集 | Shen 数据的结果只在柱状图（Fig. 11）中给出，没有数值表 | 13.94M；0.58 GFLOPs；1.13 ms（RTX 4090） | 论文未给代码链接 | [论文] |
| Bothereau et al.（瞬时相位差 + ERT/LDA + 分布匹配） | IEEE ICC 2026；作者含 Junqing Zhang、Alan Marshall | 公开 LoRa 数据集 | 准确率和信道鲁棒性与深度学习 SOTA 相当，计算和内存显著更少 | 非深度 | HAL 开放全文（hal-05721219，下载待完成） | [摘要] |
| RIEI（可分离条件，解耦发射机/接收机特征） | IEEE IoT-J 2025 | 两个真实数据集 | 在 WaveMLP 的表中，CRD 为 76.07 / 72.11 | 4.11M（WaveMLP 表） | arXiv 2411.03636（开放） | [摘要] |
| LightRFF-Net（TQCD + DRCI + KD），Zichuan Yu 等 | Physical Communication 77:103138，2026 | 自采 20 TX / 4 RX LoRa | 严格跨 RX 89.1%，约 1.2M 参数，0.006 GFLOPs，5 ms | — | 未知 | [元数据] + [用户转述，待核实] |
| RIFLE（ResNet + GRL，部分 TX×RX 可见），Minghe Zhang 等（青岛大学） | IEEE ICCCN 2026 | 20 台异构 SDR 的 LoRa | 平均 93.90%（比其基线 +12.06 pp） | — | 未知 | [元数据] + [用户转述，待核实] |
| Triad-GAN（特征级 GAN，多 RX），Shuo Wang、Junqing Zhang、Guanxiong Shen 等 | IEEE ICC 2026 | 多 RX | — | — | 未知 | [元数据] |
| Qi et al.（WiFi 带外特征 + 正交正则），Xinyu Qi、Guyue Li、Guanxiong Shen 等 | IEEE TIFS 2026 | WiFi | — | — | 未知 | [元数据] |
| BERT + 两阶段 KD，Jiawen Shao 等（Guan Gui 组） | IEEE Commun. Lett. 2026 | 跨 RX | — | — | 未知 | [元数据] |
| 即插即用域泛化，Zhi Chai 等 | Ad Hoc Networks 2026 | 冻结骨干后外挂模块 | — | — | 未知 | [元数据] |

**跨论文数字不可直接比较。** 正式对比必须在同协议下重跑（§5）。

## 2. IB-RFF [代码+日志]

### 2.1 方法

- **SAF**：对 256 点复数输入做 FFT，乘以一个可学习的复数全局滤波器；再加一支路，只保留能量高于"中位能量 × 可学习阈值"的频点并乘以第二个滤波器；两路相加后 IFFT。
- **Patch embedding**：Conv1d(2→128)，卷积核为 patch 大小（ManyRx 16，ManySig 32），步长为 patch/2；加可学习位置编码，dropout 0.5。
- **FAF**：LayerNorm 后沿 token 轴做 rFFT，按通道乘可学习复权重，再加高能量掩码支路，然后 irFFT。
- **DSCNN 瓶颈层**：patch 维 Conv1d 和通道维 Conv1d 两支（GELU、dropout）相加，过 1×1 融合层，加残差。最后均值池化，接线性分类器。
- **IB 项**：α·nHSIC-CCA(隐表示, 输入)，高斯核 σ=5，每个 batch 对 128×128 矩阵求逆（在 CPU 上）。不需要 RX 标签。
- **两阶段训练**：先用纯 CE（α=0）训练，再加 HSIC 微调。
- **优化**：Adam 1e-3，batch 128，最多 200 epoch，早停耐心 10，Plateau(0.1, 5)，**只跑 seed 2023**。

### 2.2 协议（WaveMLP 完全相同）

- **数据**：WiSig 非均衡版本，每个 (TX, RX, day) 取前 100 个信号，每个 256 点，按包功率归一化。ManyRx 为 10 TX、32 RX、4 天；ManySig 为 6 TX、12 RX、4 天。
- **RX 划分**：RX 按固定名单顺序切成 4 份，每轮留 1 份测试（ManyRx 8 台，ManySig 3 台）。
- **训练和验证**：训练只用第 1、2 天，从训练池中按 70/30 随机切出同域验证集。
- **测试**：CR 用测试 RX 的第 1、2 天；CRD 用测试 RX 的第 3、4 天。

### 2.3 官方日志结果（%，单 seed；括号内为第 0–3 轮）

| 设置 | ManyRx | ManySig |
|---|---|---|
| FAN 骨干（α=0），CR | 89.68（90.66/85.55/92.80/89.71） | 93.06（95.64/87.94/96.61/92.06） |
| IB 微调，CR | 90.68（93.44/85.84/93.03/90.41） | 94.62（96.67/89.64/98.64/93.53） |
| IB 微调，CRD | 89.19（91.15/87.86/92.15/85.59） | 93.57（95.89/88.53/96.28/93.58） |

### 2.4 可比性问题

1. **α 是按测试集挑的。** `run_all.sh` 写明 "best alpha per (dataset, exp, round)"。同一轮的 CR 和 CRD 用了不同的 α（例如 ManyRx 第 0 轮 CR 用 0.08、CRD 用 0.01），但两者共用同一个源验证集，用验证集不可能选出两个不同的 α。
2. **IB 相对其自身骨干的增益只有** +1.0 pp（ManyRx CR）和 +1.56 pp（ManySig CR）。只有一个 seed，α 又是按测试集挑的，所以这个增益不可靠。
3. **复现方案：**
   - (a) 官方版：每轮用官方 α，用来核对数据管线；
   - (b) 公平版：α 只在源 RX 上选，跑多个 seed，与 FAN（α=0）配对比较。
4. **LoRa 部分不可直接比较。** 仓库里的 LoRa 数据是 IEEE DataPort 的 receiver_drift_dataset，不是 Shen 的 20-SDR，也没有附 LoRa 训练脚本。
5. 代码注释说数据形状是 (N,2,L)，实际上 WiSig 原始数组是 (N,256,2)，转置后才是 (N,2,256)，所以功率归一化是正确的。我们已核对 WiSig pkl 结构：`data[tx][rx][day][eq]`，每个数组为 (n≤200, 256, 2)。

## 3. WaveMLP [代码+README]

- **模型**：
  1. 每个包先做复数去均值和方差归一化；
  2. 进行 J 级可学习复小波分解，分析和合成滤波器都是可学习的复卷积，长度 K，初始化为随机；再逐级重构。整体相当于一个可学习的多分辨率线性滤波；
  3. 乘回原方差、加回均值；
  4. Conv1d patch embedding（k=64，步长 32，256 点得到 7 个 token）；
  5. 一个 Pre-LN MLP 块（加残差）；
  6. 均值池化，接线性分类器。
- **配置**：ManyRx 用 J=5、K=32；ManySig 用 J=4、K=16。按数据集分别设定，怎么选的没有说明。
- **训练**：Adam 1e-3，batch 128，最多 200 epoch，早停耐心 10，Plateau(0.1, 5)，seed 2023，单 seed。数据管线与 IB-RFF 完全相同，代码同源。
- **README 中的 CRD 对比表**（作者自己跑的基线，ManySig / ManyRx）：TFMix 80.95/73.47，SigMix 80.01/73.90，MTL 79.40/72.84，FS 75.92/68.70，RIEI 76.07/72.11，GAN-RXA 78.45/74.33，AWTD 71.05/86.01，WaveMLP 93.13/88.68。**表中没有 IB-RFF**，而 IB-RFF 自己报告的 CRD 是 93.57/89.19，略高于 WaveMLP。
- **结论**：在 WiSig CRD 协议上，当前公开的最好结果大约是 **ManyRx 88.7–89.2%、ManySig 93.1–93.6%**，都是单 seed。要说"超过 SOTA"，必须在同一协议、同一数据管线下，用多 seed、只在源域选超参数的条件下超过这些数字。

## 4. Ma et al. 2025 [论文]

- **三阶段流程**：
  1. （可选）在其他数据集上做无监督对比预训练；
  2. Siamese 训练，损失为 NT-Xent（τ=0.05）+ CE；
  3. 推理时只用一个分支。
- **Siamese 的正样本对**：同一个 DUT、来自两台不同 RX 的包，各自独立做一次增强。增强包括 TDL 信道（指数功率时延谱，RMS 时延扩展 5–300 ns，Jakes 多普勒 0–5 Hz）和 AWGN（SNR 10–40 dB）。
- **表征与网络**：对数幅度谱图（STFT），网络是 Shen 式 ResNet，特征为 512→256 维。
- **优化**：Adam 3e-4，batch 32，Plateau(0.5, 10)，早停 30。
- **Shen 数据上的协议**：只用 B200-1 和 B210-1 两台 RX 训练，在全部 20 台 SDR 上测试（每 DUT 200 包，高 SNR）。结果只在 Fig. 11 柱状图中，作者称 Spec+Siamese 整体最好，CIS+Reverse 和 Spec+Reverse 在多数情况下略差。
- **计算量（Table II）**：Spec+Siamese 13.94M 参数、0.58 GFLOPs、1.13 ms；IQ 方法 132M 参数、1.54 GFLOPs。**不是轻量模型。**
- **对我们的意义**：
  - 它与本文是同一数据集，但协议不同（2 台源 RX，对比我们的 5 台源 RX × 160 包），所以必须在 P-H5x160 上复现，不能引用它的数字。
  - 它的核心训练机制是"跨 RX 正样本对 + NT-Xent + 信道/AWGN 增强"，复现时应作为训练机制类的对照。
  - rl1 的 R0 在 WiSig 上做过类似的"跨 RX support/query"试验，结果是不变性压力损伤了身份信息。这个风险需要在复现中检验。

## 5. 对本工作定位的含义

- **只靠"参数少"站不住。** WaveMLP 只有 8.4 万参数、1.3 MFLOPs，比我们的 58 万参数、139 MFLOPs 小一到两个数量级。本文不能以"轻量"为主卖点，只能在精度–计算量曲线上争取优势，并且必须与它同协议比较。
- **"可学习频域滤波"已被占用**：IB-RFF 的 SAF/FAF 和 WaveMLP 的可学习小波都属于这一类。
- **"无需 RX 标签的不变性正则"已被占用**：IB-RFF 的 HSIC/IB 就是。
- **"跨 RX 对比学习"已被占用**：Ma 2025 的 Siamese + NT-Xent 就是。
- **本工作目前已核实、别人没有的证据**：
  1. 在 Shen 同协议上，严格带内输入 + AWGN 把原始表征的跨 RX 差距从 25–39 pp 降到 0–3 pp；
  2. 8 个 chirp 的相干均值保留了 99.7% 的能量，却丢掉 12–14 pp 准确率，说明能量很小的符号间残差携带身份信息。

  这两点与 LoRa（以及 WiFi 前导码）的**重复符号结构**直接相关，是新方法设计的依据（见 PLAN 阶段 03）。
- Bothereau（ICC 2026）用瞬时相位差揭示"残余硬件相关模式"，与"残差携带指纹"的思路相近。阶段 03 的相位/瞬态分支必须与它对照，说明区别：我们是在深度网络中利用跨符号残差，并且在跨 RX 协议下评估。

## 6. 复现与对比清单（两层）

1. **Shen 同协议（P-H5x160）**：
   - 已完成：TrueIB 系列、Shen-CIS、rank-16 CIS、严格 IB 的 CIS。
   - 待做：FAN/IB-RFF 与 WaveMLP 适配到 8192 点（只改输入长度、patch 大小和小波级数）；Ma 2025 Spec+Siamese 按论文复现。
   - 每个方法跑两套配方：官方配方，以及我们的 AWGN + cos 200 配方。用源验证集决定每个方法采用哪套配方，两套结果都报告。
2. **WiSig CRD / CR（官方协议）**：
   - 先用官方代码复现 FAN、IB-RFF、WaveMLP 的数字，核对管线；
   - 再在同一管线下用 3 个以上 seed 运行全部方法和本方法；
   - α 和 J/K 等超参数只在源 RX 上选。
   - 数据：ManyRx 已在本地（Kaggle 副本，结构与官方一致，未证明逐字节相同）；ManySig 需要从 UCLA 官方页面下载。
