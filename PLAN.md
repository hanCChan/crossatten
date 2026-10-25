# 研究路线：Shen 跨接收机 CrossAttn v2（第二篇论文）

**目标：** 在严格 0-shot 跨 RX 闭集识别中，通过改进信号前处理、RF 表征、时频融合和训练机制，尽可能提升效果。参数量暂不设上限，但每个实验臂都要记录参数、FLOPs、吞吐和端到端延迟。

**原则：**
- 先探索（只看 dev_val），再确认（新 seed，加上 confirm RX）。
- 每次只改一个因素，组合必须能归因到各个因素。
- 所有尝试都要记录，失败的也写进结果。

## 四个技术方向与阶段对应

| 阶段 | 方向 | 内容 | 状态 |
|---|---|---|---|
| [00-signal-audit](00-signal-audit/) | 数据与信号 | 物理核验、梳状谱剖面、各频段信息探针、SNR/CFO 捷径、CIS 坐标、残差 | ✅ 完成，见 [RESULTS](00-signal-audit/RESULTS.md) |
| [01-strong-baselines](01-strong-baselines/) | A + C + D 的起点 | Q1 训练配方阶梯，Q2 Shen-CIS 及其 GAP 版本，Q3 频带，Q4 时频分离 token | ✅ 完成：严格 IB + AWGN 的 CrossAttn 达到 94.7/92.1（CIS 为 93.1/95.2），见 [RESULTS](01-strong-baselines/RESULTS.md) |
| [02-single-factor-screen](02-single-factor-screen/) | A/B/C/D 单因素 | 以 TrueIB+AWGN 为基线：chirp 结构（均值 chirp/梳状/dechirp）、OOB 增量（独立的原始 OOB 路径或 CIS 谱商 OOB 路径）、结构（深度、时频分离）、训练（epoch 数、AWGN 范围），外加 rank-16 CIS 轻量对照 | ✅ 完成（[确认表](02-single-factor-screen/summary/confirm_tables.md)）：只有 ep200 被确认有效（95.84/93.89）；均值 chirp −13.6 pp；原始 OOB 侧路约 −10 pp；depth4 训练失稳（SD 30） |

### 2026-10-25 路线调整（用户决定）

论文主线改为"**面向跨接收机泛化的 Chirp 结构感知轻量化射频指纹识别**"，OOB 不再作为主创新，只保留为负面结果。具体要求：
- 以 TrueIB-RF-HSTU（58 万参数）为底座。
- 必须提出超出常规训练调优的新技术贡献。
- 必须在同协议下与真正强的轻量方法比较：rank-16 CIS、FAN/IB-RFF、WaveMLP、Ma 2025。
- 4 台 confirm RX 已被多次查看，不再用于指导设计；最终的独立证据来自 WiSig 等其他数据。

| 阶段 | 内容 | 状态 |
|---|---|---|
| 03-chirp-structure（Shen，dev_val 筛选） | 第一轮：基线重跑、Pre-LN、depth4、残差输入通道、nofft 对照、共享符号 stem 的等参数 symstack、ifreq、分层 stem；融合和训练目标留待后续 | 8 臂 smoke 与 32 次探索已通过核验；探索冻结提交 `8d482b4`，48 次新 seed 确认运行中；第一轮结束后停止汇报 |
| 04-sota-on-shen | FAN/IB-RFF、WaveMLP 适配到 8192 点，在 P-H5x160 上重跑；Ma 2025 按论文复现 | 文献核实中（[LITERATURE](docs/LITERATURE_2025_2026.md)） |
| 05-wisig | IB-RFF 官方 CR/CRD 协议：先复现 FAN/IB-RFF/WaveMLP 的官方数字，再在同一管线、多 seed 下运行本方法（按 OFDM 前导码的重复结构重新定义） | ManyRx/ManySig 转换计数与发布权重通过，实测 25 MHz；重训待 03 汇报 |
| 06-final | 冻结方法的完整消融、多 seed、精度–计算量 Pareto；历史 14→6 协议标注为复用 | 最后 |

旧的 03–06 规划（组合 / 融合 / 训练 / 确认）由上表取代。**阶段 03–05 的逐项实施规格（交给 Codex 实现）见 [docs/S03_S05_IMPLEMENTATION_SPEC.md](docs/S03_S05_IMPLEMENTATION_SPEC.md)。**

### 阶段 03 第一轮（按实施规格执行）

基线 `s03_base` 就是 s02_ep200（TrueIB + AWGN U[0,50) + cos 200 epoch）。按详细规格重跑，用同 fold/seed 的 s02 结果核对确定性。下面的旧草案由 `03-chirp-structure/README.md` 和 `arms.json` 的预登记取代：残差采用输入通道拼接，symstack 采用共享 k7/k5 stem，并以 nofft 为等参数对照。

| 优先级 | 实验臂 | 相对谁 | 改动 | 物理/工程理由 |
|---|---|---|---|---|
| P1 | `s03_preln` | base | 第一个 mixer 前加 LayerNorm | 阶段 02 实测 token RMS 为 85–134，sigmoid 打分 100% 饱和，depth4 训练崩溃（SD 30）。这是工程修复，不是贡献 |
| P1 | `s03_preln_d4` | preln | 4 层 mixer | 归一化之后，加深能否带来收益 |
| P2 | `s03_resid` | preln | 严格 IB 包减去均值 chirp，单位 RMS 的实虚残差作为 2 个额外输入通道 | 检验低能量残差的身份信息（**候选贡献**） |
| P2 | `s03_nofft` | preln | 去掉整包 FFT 输入，仅保留 IQ/幅相 | symstack 的干净对照 |
| P2 | `s03_symstack` | nofft | 共享符号内 k7/k5 stem，再堆叠 8 个符号特征并按 32 点切 token；与 nofft 参数量相同 | 按 chirp 内位置对齐（**候选贡献**） |
| P3 | `s03_ifreq` | preln | 增加瞬时频率偏差通道（严格 IB 信号的相位差减去理想 chirp 的瞬时频率） | 振荡器/PLL 的非线性会体现在瞬时频率上；需要与 Bothereau 2026 的瞬时相位差方法对照 |
| P4 | `s03_hstem` | preln | 用分层步进卷积 stem 替换 k=256 的 patch 卷积（后者占 52 万参数） | 降参数，并保留局部时间结构 |

- 规模：8 个实验臂（包含重跑基线和 nofft）探索 32 次，确认 48 次，均为 200 epoch。第一轮后必须停下汇报。
- 判定规则与阶段 02 相同。确认阶段只用 dev_val 做结转决策；confirm RX 照常报告，但不参与选择。
- 第二轮：只组合第一轮确认有效的因素，比较融合方式（P5：concat / gate / cross-attn），并加入竞争方法的训练机制作为对照（P6：IB-RFF 的 HSIC 信息瓶颈、Ma 2025 的跨 RX NT-Xent）。

### 阶段 04 / 05：强方法复现（详见 [LITERATURE](docs/LITERATURE_2025_2026.md) §6）

- **04（Shen 同协议）**：FAN/IB-RFF 和 WaveMLP 使用官方模型代码，只改输入长度为 8192；Ma 2025 Spec+Siamese 按论文复现。各系列按探索 dev_val 选择配置；IB-RFF 另给源验证集选 alpha 的公平版。此轮不自动启动 04。
- **05（WiSig）**：
  1. 数据审计：从数据中实测采样率、占用带宽边界、L-STF/L-LTF 的重复周期，据此定义 WiFi 版的"严格带内"和"重复结构"；
  2. 用官方代码复现 FAN、IB-RFF、WaveMLP 在 CRD/CR 上的数字；
  3. 在同一管线、3 个以上 seed 下运行本方法；
  4. 目标是在公平选择超参数的条件下超过 ManyRx 89.2、ManySig 93.6。

各阶段的顺序可以根据证据调整，但任何调整都要在本文件中写明理由。

## 决策门槛（每阶段在运行前冻结）

- **探索：** 4 对配对 fit（2 fold × 2 seed）。判为 "improves" 需同时满足：Δ 均值 ≥ +1 pp、至少 3/4 对为正、最差 RX 的 Δ ≥ −2 pp。
- **确认：** 6 对配对 fit。判定条件相同，但要求至少 5/6 对为正，并且在 confirm RX 上单独报告。
- **组合：** 只有在确认阶段判为 improves 的因素才能进入组合；组合本身也要重新确认，并且保留"组合减去某一项"的对照。

## 预期的论文贡献形态（待实验验证，现在不预设）

以下是待检验的新假设（依据阶段 00–02 的事实）：
1. 8 个重复 chirp 的符号间残差和首符号瞬态只占 0.3% 的能量，却携带可迁移的身份信息。证据：只用均值 chirp 时掉 12–14 pp，IB 残差的跨 RX 线性探针为 52.5%。
2. 在原始波形中，这部分信息会被高能量的均值 chirp 淹没。如果做显式分解并对各分量做能量归一化，再分路编码，有望在不破坏 TrueIB 跨 RX 稳定性的前提下利用它。
3. 同一原理可以推广到 WiFi 前导码的重复结构（L-STF 和 L-LTF），从而在 WiSig 上得到验证。

只有当新组件在同协议下稳定超过 58 万参数的 TrueIB，并且能与 rank-16 CIS、FAN/IB-RFF、WaveMLP 竞争时，才把它写成方法贡献。否则如实报告：强基线 + 机制诊断。

已经否定或降级的假设（保留为负面结果）：
- 远带噪底是主要捷径（阶段 01 否定）。
- 在严格 IB 主路之外，独立的 OOB 路径能带来增量（阶段 02：原始 OOB 侧路约 −10 pp，CIS 谱商 OOB 侧路无增益）。
- 时频分离 token 有帮助（阶段 01、02 均否定）。

## 可复用的已有资产

- 论文 1 的 CrossAttn 代码：从 gai@eb4ce2c 复制到 `lib/legacy_rfhstu/`，未做修改。
- Shen 的 CIS 与网络：用 PyTorch 移植（CC BY-NC-SA 4.0），已与 numpy 参考实现核对，误差小于 1e-5。
- rl1 的折与包：逐字节相同，因此 rl1 的 CE/ccalign 结果可以作为同协议参照（Strict CIS，rank16 头，Adam 3e-4，20 epoch，无 AWGN）：CE 82.67/85.45，ccalign 84.91/89.00（fold0/fold1 的 outer 准确率）。
