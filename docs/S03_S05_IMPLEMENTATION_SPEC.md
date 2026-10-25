# 阶段 03–05 实施规格（交给 Codex 实现）

> 读者：在 `/workspace/crossatten` 中实现并运行实验的编码代理（Codex）。
> 本文是**规格**，不是结果。每个实验臂都写明了改什么、怎么实现、怎么验收、怎么判定。
> 背景结论见 [LITERATURE_2025_2026.md](LITERATURE_2025_2026.md)、[02 确认表](../02-single-factor-screen/summary/confirm_tables.md)、[PLAN.md](../PLAN.md)。

## 0. 必须遵守的规则

### 0.1 环境

```bash
cd /workspace/crossatten
export PYTHONPATH=lib PY=/data/miniforge/envs/env-3.12/bin/python CUBLAS_WORKSPACE_CONFIG=:4096:8
$PY tests/test_core.py && $PY tests/test_models2.py && $PY tests/check_models2_regression.py   # 动手前必须全部通过
```

- 机器：4×A100-40GB。单次 Shen 训练显存不超过 3.1 GB，每卡可并行 5 个任务（`--per-gpu 5`）。
- 外网只能走代理：`export https_proxy=http://star-proxy.oa.com:3128 http_proxy=$https_proxy`。
- `tests/test_models2.py` 中的 `test_preln_control`、`test_null_oob_controls_carry_no_packet_information`、`test_cis_row_options` 写好后**从未运行过**。先跑，有错先修，修完再开始任何训练。

### 0.2 不许改动的内容

- `configs/protocol.json`：已冻结的协议 P-H5x160。
- `lib/legacy_rfhstu/`：论文 1 的原版代码。
- `00-*/`、`01-*/`、`02-*/` 下已有的结果和脚本。
- `lib/xa/guard.py`：拒绝读取 6 台盲测 RX，按用途限制 confirm RX 的读取。
- 不要 push 到 GitHub（由用户自己发布）；不要提交 `model.pt`（已在 .gitignore 中）。

### 0.3 实验纪律（每个阶段都一样）

1. **预登记**：先在 `NN-xxx/arms.json` 和 `NN-xxx/README.md` 中写好实验臂、事先预测和判定规则，提交 git，然后才允许训练。
2. **探索**：seed 101 和 102 × fold 0 和 1，共 4 次配对训练，只看 dev_val。
3. **冻结**：运行 `NN-x-aggregate.py explore --freeze`，生成 `decision_explore.json`（只做记录，所有实验臂都进入确认），提交。
4. **确认**：seed 201、202、203 × 2 fold，共 6 次配对训练。**结转决策只看 dev_val**。confirm RX 只报告，不参与任何选择（这 4 台已被多次查看）。
5. **判定**（与阶段 02 相同，代码见 `lib/xa/aggregate.py::verdict`）：
   - improves：Δ 均值 ≥ +1 pp，且探索时至少 3/4 对为正（确认时至少 5/6），且最差 RX 的 Δ ≥ −2 pp；
   - harms：Δ 均值 ≤ −1 pp，且同样数量的配对为负；
   - 其余记为 unclear。
6. **单因素**：每个实验臂只比它的对照多一个因素。要组合，必须等各因素分别确认之后，并且保留"组合减去某一项"的对照。
7. **全部报告**：失败的实验臂也写进 RESULTS.md，禁止只报告最好的 seed 或 RX。

### 0.4 复用现有工具（不要重写）

| 用途 | 现有代码 |
|---|---|
| 单次训练与评估（输出 `metrics.json`、`history.csv`、`preds_*.npz`、`config.json`） | `lib/xa/train.py::fit(arm, fold, seed, stage, out_dir)` |
| 配方 | `lib/xa/train.py::RECIPES`：本阶段基线用 `long_cos_200` |
| 任务列表 | `lib/xa/jobs.py::make_jobs(stage_dir, stage)`：读取 `<stage_dir>/arms.json` |
| 多卡并行，已完成的任务会自动跳过 | `python -m xa.launch --jobs <jobs.jsonl> --per-gpu 5` |
| 汇总与判定 | `lib/xa/aggregate.py::aggregate(stage_dir, stage, PAIRS, ref_globs)`，用法参照 `02-single-factor-screen/02-3-aggregate.py` |
| 模型分发 | `lib/xa/models.py::build_pipeline`：遇到未知 family 会交给 `models2.build_pipeline2`。新 family 写在 `lib/xa/models3.py`，并把 `build_pipeline2` 最后的 `raise ValueError` 改成继续交给 `models3.build_pipeline3` |
| 计算量与时延 | `01-strong-baselines/01-4-bench.py` 的计时方法（端到端，GPU batch1 p50/p95，batch512 吞吐，CPU 4 线程） |

## 1. 本轮发现（方案的依据）

### 1.1 来自本仓库阶段 00–02 的事实（都是同协议、多 seed 结果）

| # | 事实 | 数据 | 对方案的含义 |
|---|---|---|---|
| F1 | 严格带内输入 + AWGN 是目前的最强配置 | s02_ep200：dev_val 95.84，confirm 93.89（58 万参数）；全频带原始表征的"源→未见 RX"差距为 25–39 pp，严格 IB 只有 0–3 pp | 所有新方法都以严格 IB 为底座，不再回到全频带 |
| F2 | 8 个 chirp 的相干均值保留 99.7% 能量，但准确率掉 12–14 pp | s02_meanchirp：−13.64（dev_val）/ −11.71（confirm），6/6 对为负 | 能量很小的符号间成分携带身份信息，这是新方法的核心依据 |
| F3 | 符号残差不是白噪声 | 相邻符号残差的相关系数为 +0.49 到 +0.67（白噪声应约为 −0.14）；残差能量呈 U 形（第 1 个符号约 0.012，中间约 0.0005，最后约 0.005） | 残差中有确定性的瞬态和慢漂移，应该显式编码，而不是被平均掉 |
| F4 | 原 mixer 的数值尺度失控 | 训练后基线进入第一个 HSTU 的 token RMS 为 85–134，sigmoid 打分 100% 饱和；depth4 出现训练崩溃（67.08±30.42） | 先补 Pre-LN（工程修复），否则任何加深或加分支都不稳定 |
| F5 | 原始 OOB 作为侧路有害，CIS 谱商作为侧路没有增益 | rawoob 三个变体 ≈ −10 pp；cisoob −0.62 | OOB 不再作为主线 |
| F6 | 时频分离 token、梳状 FFT、dechirp 输入都没有帮助 | tfsplit −1.68；combfft 探索时判为 improves，确认时变成 harms（−5.74）；dechirp −3.8 | 探索阶段的结论必须经过确认；token 设计不再沿用"时间/频率分开"这条思路 |
| F7 | CIS 不需要 1344 万参数，但需要保留位置信息 | rank-16 头 − flatten：−0.76 / −0.46（unclear）；GAP 版本崩溃 | 轻量 CIS 对照一律用 rank-16 |
| F8 | 训练轮数 | ep200 相对 ep100：+1.11 / +1.78（improves） | 阶段 03 一律用 200 epoch |

### 1.2 来自文献与官方代码的事实

| # | 事实 | 含义 |
|---|---|---|
| L1 | IB-RFF（TIFS 2026）和 WaveMLP（IMNS 2026）出自同一作者、同一套代码。WiSig 的 CRD 协议是：RX 分 4 折，用第 1–2 天训练、第 3–4 天测试，每个 (TX, RX, 天) 取前 100 个信号，非均衡，单 seed | WiSig 上的对比必须采用这套协议 |
| L2 | 当前公开的最好结果：IB-RFF 的 CRD 为 ManyRx 89.19 / ManySig 93.57；WaveMLP 为 88.68 / 93.13，只有 8.4 万参数 | 我们的目标是在公平条件下超过约 89.2 / 93.6 |
| L3 | IB-RFF 的 α 是按测试集挑的（同一轮的 CR 和 CRD 用了不同的 α）；它相对自身骨干只涨 +1.0 和 +1.6 pp | 复现时要同时给出官方版和公平版（α 只在源域上选） |
| L4 | Ma 2025 在 Shen 数据上只用 2 台源 RX 训练，结果只有柱状图；模型 13.9M 参数，是跨 RX 正样本对 + NT-Xent + 信道/AWGN 增强 | 必须在 P-H5x160 上复现，作为训练机制类的对照 |
| L5 | "可学习频域滤波"（SAF/FAF、可学习小波）、"无需 RX 标签的 IB/HSIC 正则"、"跨 RX 对比学习"都已有人做过 | 本文的贡献点只能落在 LoRa/WiFi 的**重复符号结构**上（F2、F3） |
| L6 | Bothereau（ICC 2026）用瞬时相位差揭示"残余硬件相关模式" | 我们的瞬时频率分支（s03_ifreq）必须与它对照，并在论文中说明区别 |

## 2. 阶段 03：Chirp 结构感知表征（Shen，P-H5x160）

目录 `03-chirp-structure/`，文件按 `03-1-make-jobs.py`、`03-3-aggregate.py`、`arms.json`、`README.md`、`RESULTS.md` 组织，写法照抄阶段 02。
模型代码放在 `lib/xa/models3.py`，测试放在 `tests/test_models3.py`。

### 2.1 公共设置

- 所有实验臂：`recipe="long_cos_200"`，`awgn=[0,50]`，输入一律先做 `rms_normalize`，再做 `band_limit(x, 62.5e3)`（严格 IB）。
- `s03_base` 就是 s02_ep200 的重跑（`family="ib_variant"`）。这个配置没有 MHA，训练是确定性的，所以 `s03_base − s02_ep200` 应该恰好等于 0.00，用作确定性检查。在 03-3-aggregate.py 中通过 `ref_globs` 引入 `02-single-factor-screen/runs/{stage}/s02_ep200__*/metrics.json`。
- 记号：`x_ib` 为 [B,2,8192] 的严格 IB 包，`z = x_ib[:,0] + 1j*x_ib[:,1]`。每个符号 `NS=1024` 点，共 8 个符号。
- 单次训练约 25 分钟。第一轮共 8 个实验臂，探索 32 次训练（约 1.5 小时），确认 48 次（约 2.5 小时）。

### 2.2 实验臂

| 实验臂 | 对照 | 唯一改动 | 预期参数量（实现后以单测为准） | 事先预测 |
|---|---|---|---|---|
| `s03_base` | `s02_ep200` | 无（重跑） | 580,008 | Δ = 0.00 |
| `s03_preln` | base | `ib_variant` 加 `preln=True`（已实现于 `models2.IBVariantPipeline`） | 580,136 | 均值增益小，但同折 seed 间 SD 下降 |
| `s03_preln_d4` | preln | `preln=True, depth=4` | 622,374 | 不再崩溃；Δ 不确定 |
| `s03_resid` | preln | 主 stem 增加 2 个通道：单位 RMS 的残差（实部、虚部） | 约 580,584 | **improves**（主假设，依据 F2、F3） |
| `s03_nofft` | preln | 主 stem 去掉整包 FFT 通道，只保留 I、Q、log1p 幅度、相位/π 四个通道 | 约 579,912 | 不确定（FFT 通道本身就混用了时间轴和频率轴） |
| `s03_symstack` | nofft | 符号堆叠 tokenizer（见 2.3.3）；**参数量与 nofft 完全相同** | 等于 nofft | **improves**（第二假设：token 按 chirp 内位置也就是瞬时频率对齐，可以直接比较不同符号） |
| `s03_ifreq` | preln | 主 stem 增加 1 个瞬时频率偏差通道 | 约 580,360 | 不确定；需要与 Bothereau 2026 对照 |
| `s03_hstem` | preln | 分层步进卷积替换 k=256 的 patch 卷积 | 约 14.2 万 | 不确定；参数减少约 75% |

预登记的对比 PAIRS（Δ = B − A）：

| 组 | B | A |
|---|---|---|
| P1 | s03_preln | s03_base |
| P1 | s03_preln_d4 | s03_preln |
| P2 | s03_resid | s03_preln |
| P2 | s03_nofft | s03_preln |
| P2 | s03_symstack | s03_nofft |
| ref（只报告） | s03_symstack | s03_preln |
| P3 | s03_ifreq | s03_preln |
| P4 | s03_hstem | s03_preln |
| ref（确定性检查） | s03_base | s02_ep200 |

### 2.3 实现细节（`lib/xa/models3.py`）

建一个 `ChirpPipeline(nn.Module)`，结构必须与 `IBVariantPipeline(preln=True)` 一致：legacy `DeviceClassifier` 的 embedder → `encoder.prepare_tokens`（input_proj + position）→ `pre_ln` → `encoder.blocks`（RFHSTUBlock，32 个 token）→ `final_norm` → mean → classifier。**唯一变化是送进 CNN stem 的通道，或 stem 本身**。

验收：`ChirpPipeline(channels="legacy5", preln=True)` 在同一 seed 下，初始化权重和前向输出都必须与 `IBVariantPipeline(preln=True)` 逐元素相同。这条单测保证所有新实验臂都只差一个因素。

#### 2.3.1 `s03_resid`：残差通道

```python
zs = z.view(B, 8, 1024)                                   # 8 个符号
m  = zs.mean(1, keepdim=True)                              # 相干均值 chirp
r  = (zs - m).reshape(B, 8192)                             # 残差：瞬态 + 漂移 + 噪声
r  = r / torch.sqrt((r.abs()**2).mean(-1, keepdim=True) + 1e-12)   # 单位 RMS，否则 -25 dB 的残差会被淹没
stem_in = torch.cat([legacy5_views(x_ib), r.real[:,None], r.imag[:,None]], 1)   # 7 通道 → CNNStemPatchEmbed(in_channels=7)
```

- AWGN 在 `fit()` 中加在原始包上，残差里自然包含噪声，不要另行处理。
- 单测：
  - 周期包（同一个 1024 点符号重复 8 次）的 `r` 为 0，且输出中没有 NaN；
  - 对随机包，`tile(m) + r_raw == z`（atol 1e-6）。

#### 2.3.2 `s03_nofft`

- `CNNStemPatchEmbed(in_channels=4)`，输入通道为 `[I, Q, log1p(amp), phase/π]`。这四个通道与 `torch_rf_views` 返回的 `iq_view` 和 `amp_phase` 定义相同。
- 它是 symstack 的对照。

#### 2.3.3 `s03_symstack`：符号堆叠 tokenizer（参数量与 nofft 相同）

```python
v = nofft_views(x_ib)                       # [B, 4, 8192]
v = v.view(B, 4, 8, 1024).permute(0, 2, 1, 3).reshape(B*8, 4, 1024)
h = stem_convs(v)                           # 与 nofft 相同的 k7→k5 卷积（4→32→32，含 BN 和 LeakyReLU），权重在 8 个符号间共享
h = h.view(B, 8*32, 1024)                   # 8 个符号的特征堆叠成 256 个通道
tokens = Conv1d(256, 64, kernel_size=32, stride=32)(h).transpose(1, 2)   # [B, 32, 64]
```

- 参数核对：nofft 的 tokenizer 是 Conv1d(32→64, k=256)，权重 64×32×256 = 524,288；symstack 的是 Conv1d(256→64, k=32)，权重 64×256×32 = 524,288，**完全相同**；stem 卷积也完全相同。所以两者唯一的差别是 token 覆盖哪些样本：
  - nofft：一个 token 覆盖同一个符号内连续的 256 个样本；
  - symstack：一个 token 覆盖 8 个符号中 chirp 内位置相同的 32 个样本，对应 3.9 kHz 宽的瞬时频率窗口。
- 位置编码和相对位置偏置保持原样。此时它们编码的是"chirp 内频率位置"，要在 README 中说明。
- 单测：
  - 参数量等于 nofft；
  - 扰动 chirp 内位置 p（在任一个符号中），只有第 ⌊(p±5)/32⌋ 号 token 发生变化（±5 是 k7、k5 卷积的感受野）。

#### 2.3.4 `s03_ifreq`：瞬时频率偏差通道

```python
dphi = torch.angle(z[:, 1:] * z[:, :-1].conj())           # 相邻样本相位差
dphi = torch.cat([dphi, dphi[:, -1:]], -1)
f    = dphi * FS / (2*math.pi)                             # Hz
n    = torch.arange(8192) % 1024
f_c  = -BW/2 + BW*(n + 0.5)/1024                           # 理想 SF7 upchirp 的瞬时频率
d    = (f - f_c).clamp(-2*BW, 2*BW) / (2*BW)               # 归一化到 [-1, 1]
stem_in = torch.cat([legacy5_views(x_ib), d[:, None]], 1)  # 6 通道
```

- 单测：无噪声理想 chirp 的 `|d|` 中位数小于 0.005；符号边界附近（±2 个样本）的跳变不计入。
- 在 README 中写明与 Bothereau（ICC 2026，瞬时相位差 + ERT/LDA）的区别：本文是作为深度网络的一个输入通道，并在跨 RX 协议下评估。

#### 2.3.5 `s03_hstem`：分层 stem

把 `CNNStemPatchEmbed.tokenize`（Conv1d(32→64, k=256, s=256)）替换为下面的结构，前面的 k7、k5 两层保持不变：

```python
nn.Sequential(
  nn.Conv1d(32, 48, 8, 4, 2), nn.BatchNorm1d(48), nn.GELU(),   # 8192 → 2048
  nn.Conv1d(48, 64, 8, 4, 2), nn.BatchNorm1d(64), nn.GELU(),   # 2048 → 512
  nn.Conv1d(64, 64, 8, 4, 2), nn.BatchNorm1d(64), nn.GELU(),   # 512 → 128
  nn.Conv1d(64, 64, 4, 4))                                     # 128 → 32 个 token
```

- 单测：输出形状为 [B, 32, 64]，参数量约 14.2 万。

### 2.4 第一轮之后的规则（第二轮内容不在本规格内，由用户决定）

- 只有在确认阶段 dev_val 判为 improves 的因素才能进入第二轮的组合，并且必须保留"组合减去某一项"的对照。
- 如果 `s03_resid` 被确认有效，第二轮比较四种融合方式（P5）：
  1. 输入层通道拼接（第一轮的做法）；
  2. 独立 stem，token 相加；
  3. 独立 stem，token 拼接（用 `models.SegmentHSTUBlock`）；
  4. cross-attention（主流 token 作为 query，残差 token 作为 key/value，复用 `OOBCrossAttentionFusion`）。
- 训练机制（P6）放在最后。只把竞争方法的机制作为对照加入最优模型：IB-RFF 的 HSIC（α 只在源验证集上选）、Ma 2025 的跨 RX NT-Xent。
- RESULTS.md 必须包含：
  - 每个实验臂的参数量、MFLOPs/包、端到端时延（复用 01-4-bench 的方法）；
  - 每台 RX 的准确率；
  - 对 resid 和 symstack 的机制检查：在测试时把残差通道置零或打乱，看精度下降多少，以此证明模型确实用到了残差。

## 3. 阶段 04：在 Shen 同协议上复现强方法

目录 `04-sota-on-shen/`，代码放在 `lib/xa/sota.py`。目标是让 FAN/IB-RFF、WaveMLP、Ma 2025 与 TrueIB 系列在 **P-H5x160 同协议、同数据流**下比较。

### 3.1 代码来源与许可

| 方法 | 来源 | 许可 | 做法 |
|---|---|---|---|
| FAN / IB-RFF | `external/IB-RFF` @5a03a04 | MIT | 运行时从 external 导入（`sys.path` 插入），**不复制进仓库**。timm 只用到 `trunc_normal_`，用 `torch.nn.init.trunc_normal_` 做 shim（写法见 `external/count_sota_params.py`） |
| WaveMLP | `external/WaveMLP-RFF`（clone 时记录 commit SHA） | 仓库中**没有 LICENSE 文件** | 同样只在运行时导入，不复制、不提交 |
| Ma 2025 | 论文（arXiv 2512.12070v1），无代码 | — | 按论文自行实现，代码中注明"re-implementation"及论文章节号 |

两个仓库的包名都叫 `backbones`，同一个进程内导入时要先清除 `sys.modules` 中的 `backbones*`（`count_sota_params.py` 中有示例）。

### 3.2 `train.py` 的扩展（必须向后兼容）

1. 新增配方：
   - `official_ibrff`：Adam 1e-3，batch 128，ReduceLROnPlateau(0.1, 5)，按源验证集 CE 早停（耐心 10），最多 200 epoch，取验证 CE 最低的权重。WaveMLP 共用这个配方。
   - `official_ma`：Adam 3e-4，batch 32 对（即 64 个视图），Plateau(0.5, 10)，早停耐心 30，最多 500 epoch。
2. 把全局 `BATCH=64` 改为每个配方自带 `batch` 字段，现有配方的默认值保持 64。
3. 新增 `arm["objective"]`：
   - `"ce"`：默认，行为不变；
   - `"ce+hsic"`：IB-RFF 阶段 2，见 3.4；
   - `"ce+ntxent_xrx"`：Ma 2025，见 3.6。
4. 新增 `arm["init_from"]`：用于 IB-RFF 阶段 2，加载同一 (fold, seed) 下 FAN 阶段 1 的 `model.pt`。
5. 新增 `arm["aug"]`：Ma 2025 使用 `{"awgn": [10, 40], "fading_doppler_max_hz": 5}`。原有的 `arm["awgn"]` 继续有效。
6. **回归验收**：改完后重跑 `s02_ep200` fold 0 seed 101（smoke 版本即可），输出必须与改动前逐元素相同。

### 3.3 FAN（IB-RFF 的骨干）

- 构造：`FAN(seq_len=8192, in_chans=2, patch_size=512, embed_dim=128, mlp_ratio=2.0, num_classes=10, bottleneck_type="DSCNN", use_SAF=True, use_FAF=True)`。
  - patch 512、步长 256，得到 31 个 token，与我们的 32 个 token 接近。
  - SAF 的复权重长度会随 seq_len 变成 8192。
- 前处理：先 `rms_normalize`，再按输入版本处理：`full` 是官方做法，不限带；`ib` 是我们的严格 IB，`band_limit(62.5e3)`。
- 包装：模型返回 `(x_neck, logits)`。包装层的 `forward` 只返回 logits，另提供 `forward_with_hidden` 供 HSIC 使用。

### 3.4 IB-RFF 阶段 2（HSIC 信息瓶颈）

- 损失：`CE + α · hsic_normalized_cca(hidden.flatten(1), input.flatten(1), sigma=5.0)`。input 就是送进 FAN 的那个张量（RMS 归一化后、按版本处理过的 I/Q）。
- 官方实现在 CPU 上计算（`torch.eye` 和 `FloatTensor`）。要移植成与设备无关的版本，并写单测：随机张量上与原函数的结果相对误差小于 1e-3。
- α 网格就是官方 `run_all.sh` 中出现过的值：{0.001, 0.004, 0.01, 0.04, 0.08, 0.1}。
- **公平选择**：每个 fold 按源验证集 CE 选 α。另外把"按 dev_val 选 α"作为乐观上界单独列出，并明确标注为乐观上界。

### 3.5 WaveMLP

- 主配置（官方绝对超参数）：`MyModel(seq_len=8192, num_classes=10, use_wavelet=True, wavelet_levels=5, wavelet_kernel_size=32, wavelet_init="randn", patch_size=64, stride=32)`，得到 255 个 token。
- 缩放配置（保持官方的相对尺度）：`wavelet_levels=10`、`patch_size=2048`、`stride=1024`，得到 7 个 token，与官方在 256 点上的 token 数相同。
- 模型的 `forward` 返回 `(logits, x, feat)`，包装层只取 logits。

### 3.6 Ma 2025 Spec + Siamese（按论文复现）

- **表征** `shen_spec(x)`：沿用 `frontends.shen_cis` 的 STFT（boxcar 窗 128 点，步长 64，双边谱，fftshift，加 1e-12），但**不做相邻帧谱商**。取 `log10(|S|^2)`，裁剪第 38–89 行，并只保留前 126 帧，使形状与 Shen ResNet 的 [B,1,52,126] 一致。
- **网络**：Shen ResNet 主体（复用 `ShenCISPipeline` 的 stem 和 blocks）→ Flatten → Dense512 → ReLU → Dense256（这就是特征 z）→ Dense10（论文 Fig. 5）。
- **Siamese 采样**：每个锚点包 i，随机选一个同 TX、**不同源 RX** 的包 j，两者分别独立增强，构成正样本对。batch 为 32 对，共 64 个视图。
- **增强**（论文 IV-B）：
  - AWGN，SNR ~ U[10, 40] dB；
  - TDL 信道：RMS 时延扩展 5–300 ns。在 Fs=1 MHz 下它退化为单抽头平坦衰落，所以实现为 Jakes 时变复增益：8 个振荡器，最大多普勒 U[0, 5] Hz，按包重新抽样，增益归一化为单位功率。这一退化要在 README 中说明。
- **损失**：对两个视图都算 CE，再加 NT-Xent（τ=0.05，作用于 L2 归一化后的 z，2B 个视图，其余 62 个都作负样本，按论文实现，不屏蔽同类）。
- **推理**：单分支。
- 不做预训练：协议内没有额外的无标签数据集。在 README 中注明这对应论文中的"w/o pretrain"设置。
- 再加一个 `ma_spec` 变体（同样的增强，只用 CE、不用 Siamese），对应论文自己的消融。

### 3.7 实验臂与选择规则

| 系列 | 实验臂（每个都做 4 次探索训练） | 数量 |
|---|---|---|
| FAN | {full, ib} × {official_ibrff, long_cos_200 + AWGN[0,50]} | 4 |
| IB-RFF 阶段 2 | {full, ib} × 6 个 α，配方为 official_ibrff，从对应的 FAN official 权重初始化 | 12 |
| WaveMLP | {full, ib} × {official_ibrff, long_cos_200 + AWGN} × {abs, scaled} | 8 |
| Ma 2025 | {full, ib} × {specsiam, spec}，配方为 official_ma | 4 |

- 选择规则（**对竞争方法有利，因此是保守的**）：每个系列在探索阶段按 **dev_val 准确率**选出最佳配置，再用确认 seed（201–203）重跑，报告 dev_val 和 confirm。IB-RFF 另外报告"按源验证集选 α"的公平版。
- 我们自己的方法也按 dev_val 选，两边对称。
- 输出总表：每个系列的最佳配置，列出 dev_val、confirm、最差 RX、Macro-F1、参数量、MFLOPs/包、GPU batch1 p50/p95、batch512 吞吐、CPU 4 线程时延。计时代码应从 `01-4-bench.py` 抽成 `lib/xa/bench.py` 共用。WaveMLP 的复数运算可能统计不全，FLOPs 一栏标注"torch-counted"。

## 4. 阶段 05：WiSig 跨数据集验证与 SOTA 复现

目录 `05-wisig/`，代码放在 `lib/xa/wisig.py`。数据放在 `data/wisig/`，这个目录要加进 `.gitignore`。

### 4.1 数据准备：`05-1-build-wisig-cache.py`

- **ManyRx**：解压 `/workspace/gai/data/wisig-manyrx-kaggle/ManyRx.kaggle.v1.zip`，得到 `data/wisig/ManyRx.pkl`（2.0 GB）。
- **ManySig**：用代理从 https://cores.ee.ucla.edu/downloads/datasets/wisig/ 下载官方 compact pkl，记录 URL 和 SHA256。下载不了就停下来报告，**不要换别的来源**。
- **pkl 结构**（已核实）：dict 的键为 `tx_list`、`rx_list`、`capture_date_list`、`equalized_list=[0,1]`、`max_sig=200`、`data`；`data[tx][rx][day][eq]` 是 (n≤200, 256, 2) 的数组。非均衡版本用 `eq` 下标 0。
- **RX 顺序**：必须照抄 IB-RFF/WaveMLP 的名单，并按名字（不是下标）映射到 `rx_list`：
  - ManyRx：`['1-1','1-19','1-20','2-1','2-19','3-19','7-7','7-14','8-7','8-8','8-14','13-7','13-14','14-7','18-2','18-19','19-1','19-2','19-19','19-20','20-1','20-19','20-20','23-1','23-3','23-5','23-6','23-7','24-5','24-6','24-13','24-16']`
  - ManySig：`['1-1','1-19','2-1','2-19','3-19','7-7','7-14','8-8','14-7','18-2','19-2','20-1']`
- **天的顺序**：day 1–4 对应 `capture_date_list` 的时间顺序。
- **每个 (TX, RX, day)**：取前 100 个信号，转置成 (N, 2, 256)，再按单个信号的功率归一化（256 点上 I²+Q² 的均值为 1）。
- **输出两份**：
  1. 我们自己用的 npz，每个 (RX, day) 一个文件；
  2. 官方代码要求的 `dataset/{ManyRx,ManySig}/non_equalized/date{d}/rx_{name}_data.pkl`，格式为 `{'data': [按 tx 的 (N,256,2) 数组]}`，放在 `data/wisig/official_layout/`，然后在 `external/IB-RFF/dataset` 和 `external/WaveMLP-RFF/dataset` 建软链接指向它。这样官方代码无需修改就能运行。
- **验收（只核对数量，不看任何准确率）**：IB-RFF 阶段 1 日志的前几行给出了每轮的训练集和验证集大小，例如 ManyRx 第 0 轮为 32527 + 13941 = 46468。我们按同一划分算出的训练池大小必须与 4 轮 × 2 个数据集的日志**逐一相等**。不相等就停下来报告，可能是天的顺序或 eq 下标的问题。

### 4.2 信号审计：`05-2-wisig-audit.py`

只用第 0 轮训练 RX 的第 1、2 天数据。

1. **采样率**：在前 160 个样本内做自相关，看第一个峰的滞后。L-STF 周期是 0.8 µs，所以 20 Msps 时滞后为 16，25 Msps 时为 20。
2. **占用带宽边界**：计算平均 PSD（256 点，fftshift），找 −10 dB 处的边界。802.11a/g 的理论值是 ±8.125 MHz，即 52 个子载波 × 312.5 kHz / 2。
3. **对齐**：检查 L-STF 的周期性是否从第 0 个样本一直保持到约第 160 个样本，并确定 L-LTF 的 CP（32 点）和第一个 LTF 符号（64 点）的位置。
4. **输出**：`out/05-2-audit.json`，包含 Fs、带宽边界、STF 周期和重复次数、LTF 位置。WiFi 版的"严格带内"截止频率按这里的结果冻结（占用边界加半个子载波），后续不再改。

### 4.3 先复现官方数字（在我们的任何训练之前完成）

1. **用官方发布的权重评估**（`code_state=only_test`）：IB-RFF 和 WaveMLP 仓库都带有各轮的权重。在我们转换的数据上评估，得到的测试准确率与官方日志/README 的差距应在 ±0.5 pp 以内。如果数据完全一致，结果应几乎相同。这一步用来验证数据管线。
2. **官方脚本原样重训**：IB-RFF 阶段 1（α=0）+ 阶段 2（用官方给的每轮 α，跑 CR 和 CRD）；WaveMLP 的 CRD 训练（每个数据集各自的 J/K）。seed 为 2023。torchinfo 只是被导入，没有用到，可以 pip 安装（走代理）或用 shim。
3. **公平多 seed**：用 seed {2023, 2024, 2025} 重训 FAN、IB-RFF、WaveMLP。IB-RFF 的 α 和 WaveMLP 的 J/K 都按 4.4 的内部验证规则选择。

### 4.4 WiSig 上的选择规则（对所有方法一视同仁）

官方协议是 4 折交叉验证，测试 RX 就是目标域，没有独立的开发集。所以超参数和变体都在**训练 RX 内部**选择：

- 每一轮把训练 RX 按名单顺序再切出一部分作内部验证：ManyRx 24 台中取最后 6 台，ManySig 9 台中取最后 2 台。
- 只用第 1、2 天：在其余 RX 上训练，在内部验证 RX 上算 CR 准确率，据此选择变体。**不使用第 3、4 天的任何数据。**
- 选好后，用该轮全部训练 RX 重训，再在测试 RX 上报告官方的 CR 和 CRD。
- 每个方法最多 4 个候选变体：IB-RFF 为 α 网格的子集；WaveMLP 为 {官方 J/K, 另一组 J/K}；我们的方法为 {官方配方, 我们的配方} × {legacy5, 阶段 03 中被确认的组件}。

### 4.5 我们的方法在 WiSig 上的实现：`TrueIB-RF-HSTU-WiFi`

- 与 Shen 版相同的结构：
  - `RFPatchEmbedder(window_size=256, patch_size=8, dim=64, cnn_stem_dim=32, patch_embed_type="cnn_stem", use_oob=False)`，得到 32 个 token；
  - `DeviceClassifier(depth=2)`；
  - 是否加 `preln`，取决于阶段 03 的确认结果；
  - 输入：功率归一化后，用 4.2 冻结的截止频率做 `band_limit`。注意 `band_limit` 中的 `fs` 要改用 WiSig 的实测 Fs。
- 配方：(a) 官方 WiSig 配方（Adam 1e-3，batch 128，Plateau，早停）；(b) 我们的配方（AdamW cos 200 + AWGN U[0,50]）。按 4.4 选择。
- **重复结构的 WiFi 版本**：只有当阶段 03 确认了 resid 或 symstack 有效时才做。
  - L-STF 区段（10 个周期，每个 16 点，假设 20 Msps）：残差 = 该区段减去 10 个周期的均值，并做单位 RMS 归一化；
  - L-LTF 区段：比较 CP 与 LTF 符号尾部的 32 点之差；
  - symstack 的 WiFi 版：把 10 个 STF 周期堆叠为通道。
  - 具体长度以 4.2 的审计结果为准。**不能照搬 LoRa 的 chirp 处理。**
- **报告**：
  - 每个方法、每轮、每个 seed 的 CR 和 CRD 准确率；
  - 官方口径（测试集内所有包合并计算的准确率）和 RX 等权口径两种；
  - 3 个 seed 的均值 ± SD；
  - 256 点输入下的参数量、MFLOPs、GPU/CPU 时延；
  - 同时列出官方报告值与我们的复现值。

## 5. 执行顺序、停止点与交付物

| 里程碑 | 内容 | GPU | 完成后 |
|---|---|---|---|
| M0 | 0.1 中的三组测试全部通过（先修好从未运行过的测试） | 无 | 继续 |
| M1 | 阶段 03：models3 和测试 → 预登记提交 → 探索（32 次）→ 冻结 → 确认（48 次）→ RESULTS.md | 每卡 5 个任务 | **停下来向用户汇报**，由用户决定第二轮做什么 |
| M2（与 M1 并行） | 阶段 05：4.1 数据准备 → 4.2 审计 → 4.3 中的发布权重评估 | 少量 | 数量或发布权重的精度对不上就停下来报告 |
| M3 | 阶段 04：sota.py、train.py 扩展、回归验收 → 预登记 → 探索 → 确认 → RESULTS.md | 每卡 5 个任务 | 汇报 |
| M4 | 阶段 05：4.3 的公平多 seed 复现，4.5 我们的方法 → RESULTS.md | WiSig 模型很小，每卡可跑 8 个以上任务 | 汇报 |

每个阶段的交付物：
- `arms.json`、`README.md`（方法、理由、事先预测、判定规则），在训练前提交；
- 脚本 `NN-1-make-jobs.py`、`NN-3-aggregate.py`；
- `summary/*_tables.md`、`*_pairs.csv`；
- `RESULTS.md`：结论、表格、失败项、局限。

每个阶段一个 git 提交序列：预登记 → 探索 + 冻结 → 确认 + 结果。

## 6. 什么算成功（用户定的三层标准）

1. **方法有效**：至少一个 Chirp 结构组件（resid / symstack / ifreq）在确认阶段相对 `s03_preln` 判为 improves，并且通过机制检查：测试时把残差通道置零或打乱，精度明显下降。
2. **强基线竞争力**：在 Shen 同协议上，最终方法的 dev_val 不低于各竞争系列的最佳配置（rank-16 CIS、严格 IB 的 CIS、FAN/IB-RFF、WaveMLP、Ma 2025），并且在精度–计算量曲线上占优。confirm RX 一并报告（标注为已多次使用）。
3. **跨数据集证据**：在 WiSig 的 CRD 上，用 3 个 seed、按 4.4 公平选择，超过官方报告值（ManyRx 89.19 / ManySig 93.57），也超过我们自己的复现值。

## 7. 已知风险（遇到时如实记录，不要绕开）

- WiSig 的 Kaggle 副本只证明了结构与官方一致，没有证明逐字节相同。如果 4.1 的数量对不上，以官方下载的数据为准。
- IB-RFF 的 HSIC 每个 batch 都要做 128×128 的矩阵求逆，移到 GPU 时要检查数值稳定性（epsilon 1e-5）。
- WaveMLP 没有许可证文件，只能在本地运行，不能把它的代码提交进我们的仓库。
- Ma 2025 是我们按论文复现的（没有预训练，TDL 在 1 MHz 下退化为平坦衰落），表格中必须标注"复现"。
- 含 MHA 的模型（例如 cross-attn 融合）不是比特级确定性的，同 seed 重跑可能有小差异，要在报告中说明。

