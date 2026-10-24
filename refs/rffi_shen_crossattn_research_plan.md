# Shen LoRa 跨接收机轻量 CrossAttn-OOB：科研与 Agent 实施方案

> 审查基线：2026-10-24；用户指定 GitHub `simizhanghao/keyan`，默认分支 `main` SHA `e91eb1d97f521778d075484133a3683e382437f8`。该仓库是 `new_phase` 精简快照，**不是**随附原始 IQ / H5 或所有模型权重的完整可运行仓库。用户另提供 `OOB_IOTJ(7).pdf`。本文只读分析；**未执行新训练、未修改 GitHub、未读新的封存信号**。

## 0. 科学定位（本次唯一主任务）

- 闭集已知 10 TX、目标 RX 完全未见、目标标签 0-shot 的 LoRa RFFI；每个合法 RX 独立报告 accuracy / macro-F1，然后 RX 等权汇总。
- 重点研究在低参数量/低预处理及推理时延预算下，**是否能获取并可靠利用相对强基带内表征的额外 OOB 识别信息**。
- 如果没有相对于充分训练的严格 IB 基线与强 CIS 基线的可重复增量收益，应及时降级“可靠 OOB”主张，转向有清晰效率优势的多视角小模型或停止。
- 1/5-shot 目标接收机校准、无标签目标自适应、开集、WiFi 不能混入主实验；如要研究必须另开冻结协议。
- 第一篇论文6个封存RX已经公开使用（b200_2, b200_mini_2, b210_2, n210_2, n210_3, pluto_2），**不再宣称对第二篇方法独立盲测**。已使用的源14RX也都是研究开发域，需审慎对待重复选择造成的乐观偏差。

## 1. 已有仓库事实

- README 声明核心旧线 `F_cross_attn_chirp_plain`、OSU 24/25 设备跨日/双接收机任务；而沈冠雄20-SDR和论文B依赖 `experiments/paper1_audit/` 里的代码。旧实验的 File-Acc 75.0%、RCPA-T 75.0% 是 OSU file-level 指标，绝不能当 Shen packet-level Acc。
- 现有 Shen C'-OOB `run_final_short.py`: 数据 `*_train.h5`，每样本 H5 data 实虚分离组合为8192点IQ，label值 `31..40` 映射至 0..9，RMS 归一化； `RFPatchEmbedder(window_size=8192, patch_size=256, dim=64, cnn_stem_dim=32, patch_embed_type='cnn_stem', oob_fusion_type='cross_attn_oob', fft_norm='log_zscore', oob_norm='ratio')`，`DeviceClassifier(...depth=2,use_chirp_embedding=False)`；AdamW lr=1e-3、weight_decay=5e-4、batch=64，最后训练4 epoch。重要：4 epoch 是开发阶段上限5 epoch下最佳epoch中位数预冻结，并不自动等于“训练不足”；已有 censoring 证据，应在新的开发协议中独立验证。
- 结构：`src/rfhstu/features.py` 从同一IQ计算时间域 IQ、整段FFT幅值、幅相、OOB幅度（|f|>62.5kHz），OOB ratio相对于IB RMS；主 CNN stem 同时读 IQ + **完整FFT** + 幅相，OOB注意力读32×256个**频域片**；时间Token vs频域Token的轴语义不同。`src/rfhstu/oob_fusion.py` 实现 LayerNorm-MHA-门控残差。`models.py` 实现RF-HSTU-inspired mixer。开启chirp embedding会把同类时间位置编码施加到OOB频率tokens，须显式拆开轴位置编码，Shen历史实验本来禁用了chirp embedding。
- 原论文 Table III/IV/VI：C'-OOB 625,896参数、4 epoch、6RX均值82.32%；B1 539,178参数、5 epoch、61.60%；Shen-CIS 13,510,424参数、74 epoch、99.24%；Shen-RA 99.14%。后两者在不同容量/信号表征下仅是任务可解性对照，不是容量公平实验。报告中的最强准确率也不等于所有RX/采集环境均饱和。
- 旧 `run_x2_formal.py` 的源训练/验证划分用 `hash(pth.name)%10000`；Python进程的内置hash可能受 hash randomization 影响，固定numpy/torch seed并不保证跨进程复现。新框架不得沿用此模式，改稳定SHA/ID派生seed并冻结ID文件。
- `features.py` legacy OOB zscore先masked再减全频均值，IB被填非零常值；开发时 `oob_norm=ratio` 需保持历史定义，并对它的offset、幅度及物理含义做单测。
- 旧审计结论：OSU原CrossAttn 80 epoch Day4窗级 45.5±1.3，CNN 49.9±2.7，Main出现训练塌缩；OOB幅度缩放造成较大下降，synthetic scale-pairing改善人工干预却不能稳定改善真实RX1/RX2，因此禁止直接重复“增强OOB幅度=跨接收机鲁棒”。这一证据来自不同OSU任务，不能与Shen 82.32%直接相减。
- H5原始信号/大量`.pt`/`.npy`不在公开快照，README仍引用 `/data1/hcc/llm4RF/new_phase` 私有路径。新Agent应先查本机真实数据及存档，不得凭空声明可运行。

## 2. 物理和建模问题树

1. **硬件可辨性**：TX振荡器/功放/IQ/滤波等引入硬件缺陷；RX滤波、增益、ADC、采样、时钟也引入失真。仅从复合观测IQ无法唯一因果分解TX/RX。
2. **信道/时变**：幅相频响、多径、衰落、SNR、时间和空间变化。不同 RX/日期与信道常共变，不可把每个误差都归因RX硬件。
3. **预处理双刃剑**：同步/频偏补偿、RMS归一、窗函数、去chirp、CIS都会改变身份特征；CIS的相邻谱商消除部分近似静态乘性响应但可能弱化某些OOB特征。
4. **OOB测量有效性**：采样 1MHz、标称LoRa 125kHz，8192 FFT Δf≈122Hz；|f|>62.5kHz为数字域OOB区域，不等价于被证明来自TX功放。必须审计RX滤波、噪声底、带边泄漏、频偏导致的谱平移、镜像/alias与左右谱不对称。FFT-bin谱比未经RX标定并非真正hardware-invariant。
5. **多视角冗余/捷径**：主路已有全FFT，即使移除显式OOB path也非Strict IB。CrossAttention可能依赖RX相关的OOB幅度或掩码位置，甚至混淆时间与频率轴；门控激活高≠OOB有增量信息。
6. **目标饱和**：Shen-CIS在历史14源/6复用RX协议中99.24%，因此纯准确率提升空间小。高准确率下考虑低源RX预算、源包预算、最差RX、时延和效率，但必须**事先冻结**挑战条件，禁止依据当前结果挑弱RX。
7. **统计独立性/模型选择**：包属于同一RX-TX文件、日期/采集段相关；seed不是独立RX；来源标签、物理RX实例与日期层级不能被误拆。
8. **部署可执行性**：速度以真实 IQ→预处理→推理完整路径实测CPU/GPU p50/p95/p99、吞吐、内存和模型参数；不以参数量替代latency。

## 3. 必须先跑的只读/CPU门槛 P0（暂不训练）

1. `git rev-parse HEAD`、git status、记录代码 SHA 与许可证；对比论文方法与当前 `main` 的差异；定位全部被import的外部路径，补依赖闭包。
2. 只对合法开发域枚举文件元数据：20RX域vs24H5文件、RX型号/实例/日期、每RX×TX数目、sampling/packet长度/iq dtype、标签映射；对本地原始文件计算SHA256、检查数据包重复/近重复/异常NaN、H5索引规律。禁止读取历史6盲RX用于方法发现。
3. 对 `features.py` 写单测：输入输出shape、复数精度、fftshift中心索引、IB/OOB频率划分、Parseval功率守恒、OOB off mask正确、同一原始包在不同feature-view保持时间/频域含义、传递grad且无NaN。
4. 对 `run_x2_formal.py` 修确定性：SHA256(filename,seed)生成固定numpy RNG，显式写入train/val/test packet ID manifest及hash；训练source/val来自source RX且必须拆开包ID；目标RX只用于最终打分，不能用于scheduler/ckpt。
5. 划分两套数据：开发候选只使用原14源RX做 Group-LOORX 或分组 nested CV；历史6RX只能算旧协议复核，不是本文全新独立盲测。独立盲测能力不足须在报告中诚实说明或找合法外部数据。
6. 记录可计算的 `OOB_RMS/IB_RMS`、左右谱能量、噪声底proxy、带边泄漏、F0漂移proxy、谱相似度的TX/RX分组统计；这些只是数字域线索，未经测量不得指称具体模拟器件因果。

P0交付 `docs/SHEN_DATA_CARD.md`、`docs/CODE_AUDIT.md`、`protocol.json`、`manifest/*.csv`、`tests/test_rf_features.py`，以及不能从当前公开仓单独运行的缺件清单。通过后**才另行批准GPU实验**。

## 4. 统一科研基线与实验协议

**主任务**：0-shot未知RX闭集10TX，训练使用源RX标签且禁止目标RX任何样本用于训练/模型选择；主指标 RX等权Accuracy和Macro-F1、最差RX、各RX分布、训练seed方差。

**评测组**（预先冻结、不是根据效果挑）：
- 组 G14：充分源RX（13 train,1 heldout，在14开发RX上group CV）；
- 组 GR：源RX预算 2/5/10或可行的预设水平；每次训练的RX覆盖/型号明确，重复多组source组合，避免把train/source RX型号差异误作算法差异；
- 组 GP：每RX×TX源训练包预算 20/100/500/全部；保持val和test采集独立；为强方法提供相同源包。
- 若14域含多个日期/文件，明确“hold out hardware receiver instance”还是“hold out capture-day file”，不能拿不同类型domain混用。

**基线分层**：
A0 原旧C'-OOB复核（历史参考，不改历史checkpoint）；
A1 同源/同训练步数的优化版C'（无新机制，只调整经source LOORX选定的训练recipe）；
B0 参数/优化预算匹配的纯IQ CNN；
B1 严格TrueIB（先FFT band-limit重建IQ，再生成所有下游视图、与OOB模型采用可比容量）；
B2 主路 IQ+FullFFT+AP、无显式OOB（这是no explicit OOB，不是strict IB）；
B3 main + direct concat/gated/attention，参数/steps配平；
C0 Shen-CIS官方风格PyTorch port（强系统级参考）；
C1 Shen-RA（GRL，RX监督）强参考；
C2 Ma2025对比预训练+孪生式模型（根据公开论文及可获得代码核实协议）；
C3 有适配代码再验证2026 IB-RFF，且其LoRa代码实际数据与Shen20-SDR未核实相同，迁移需标记为新实现；
D1 新提出的可靠性机制（只有在B0~B3证实OOB存在稳定增量后才设计）。

**公平性**：区分(a)整系统对比允许不同架构和输入，和(b)组件消融必须同主干/配方/数据/预算；参数量、训练FLOPs、batch、optimizer、checkpoint规则、输入带宽均报告。不要强制全系统模型参数完全相等，却也不能将54万vs1351万宣传为对等架构对照。

## 5. 逐层改进逻辑（非一口气堆全部Loss）

### 阶段 P1：原模型训练收敛/可重复性
- P1a C'-OOB重新在**仅开发域**做学习率/epoch/EMA/warmup检查，注意旧4epoch是5epoch cap下的开发中位，不能当作显然欠拟合。强制记录train CE、source-held-packet val CE、held-RX score、predictive entropy、梯度norm、seed variance及训练steps；所有配方使用相同数据和合法模型选择，不允许在“held-RX测试”上回挑最佳。
- P1b 单独测strict TrueIB、主路no explicit OOB、C' OOB、简化CNN并行基线。若优化C'不及主路no-OOB，先停止新Loss，解释差距。
- P1c 在固定合法开发交叉验证上计算 paired per-RX delta，检查>=N_RX/2 方向是否支持、worst RX是否劣化；只涨训练acc不能进P2。

### 阶段 P2：物理可证的前处理/Token表示改进
候选顺序：
1. 频谱/输入源 audit：LoRa CFO/时窗、噪声底、泄漏、不同RX的频段增益。不做“默认CFO彻底补偿”因为可能同时删除TX身份特征。
2. OOB token化按**频率位置**标记、左右谱及离带边距离；时间主token使用时间/LoRa chirp位置编码，频域token使用频率/side位置编码。对齐的是语义，不强行逐patch相等。
3. OOB特征对比：legacy ratio、log-ratio、归一后的shape、left/right分路、频谱可靠性估计。所有从同一IQ生成、严格维持两个路径source相同以及模型容量可比。
4. 特定输入信息保持单测/干预：zero/scale/shuffle/neutral/left/right；置换后主路FullFFT仍含OOB，仅为**目标OOB路径依赖**，如要测物理OOB增量必须全谱mask+IFFT生成真正Strict IB对照。
一次只替换一个因素，采用2seed小试验并与配对baseline比较，留有原样的随机种子和训练预算。归一化可能抹去有用TX特征，不能预设shape就是更好。

### 阶段 P3：有证据时才做算法贡献
候选理论：Reliability-Conditioned Selective OOB Fusion；以正常带内主路为可用的预测基础，估计OOB质量/置信度，学习gate在不可靠时下降。但普通gate已经在旧C'存在，必须有新可测量可靠性监督/约束，不能重复。
1. 检测IB-only与OOB-only错误互补性（离线ORACLE仅说明理论上限，不能作为上线预测输入）；
2. gate vs 僵硬concat/fixedgate/原gate 同主干同预算；
3. 只允许来源RX上的物理合理质量扰动和受控学习；合成 OOB-scale 鲁棒不得作为真实receiver泛化代理；必须在未用于选择的真实开发RX heldout独立验证；
4. 若gate学成常数/总压制OOB/主路零信息/最差RX恶化，则机制失败，不继续盲目加SupCon/GRL/KD；
5. KD、SupCon、IB、RX-adversarial等单独做竞争性baseline或消融；只有实测说明哪一种弥补哪个缺陷后才纳入正式配方。

### 阶段 P4：性能/资源与论文证据
- 报告参数、FLOPs/MACs、GPU/batch1单包前处理+模型latency p50/p95/p99、吞吐、CPU推理延迟、模型磁盘大小和Peak CUDA/RAM。热身、同步、固定功率模式、输入shape、测量次数、batch定义记录。
- 强基线对比 + 同主干消融；每RX×seed矩阵、RX类型分组、数据预算曲线、错误类混淆、带内/OOB干预效果；统计单元是RX（seed只是训练重复），不把包当成独立RX。
- 若只在99%已饱和协议中提效，要以可信的Pareto改善（性能-时延-训练样本）为明确贡献；若未改善则放弃方法主张。

## 6. 4 × A100 分配建议（不是自动开跑授权）
- 所有脚本在CPU dry-run、1GPU smoke、训练单步/梯度、无目标数据访问检查通过后，才在收到用户单独授权时启动GPU。
- GPU0..3 各跑一个配置/seed，资源不足时每GPU一进程。每个job独立显存、独立输出目录、`CUDA_VISIBLE_DEVICES`、`torch.set_num_threads`与数据加载workers；避免重复计算CPU FFT和在所有worker重复读H5。
- P1最低配：A1优化CrossAttn vs B1 strict IB vs B2 no-explicit-OOB vs B3 concat；同fold、同3seed配对，在少数预冻结fold做pilot，达成事先成功门槛才扩至全14 group CV。参数/训练成本配平后再纳入Shen-CIS/RA。
- 全试验只用开发RX，最终六旧blind仅做“复用历史外部确认”并标记reused；不能充当从未见过的新论文测试域。
- 运行制品：JSON config、commit SHA、SHA256 manifests、源/val/hold RX IDs、训练指标、目标RX score+packet indices、per RX metrics、latency、JSON decision、kill gate；不能只打印一张最好看的准确率表。

## 7. 必须具备的代码组织与测试

目标新目录（不覆盖历史）：
```
configs/shen_crossattn_v2/{protocol,models,train,bench}.yaml
src/shen_v2/{dataset,registry,guard,rf_frontend,tokenizers,models,losses,train,eval,metrics,benchmark}.py
scripts/shen_v2/{audit,build_manifests,smoke,train_fold,evaluate,aggregate,bench}.py
tests/shen_v2/{test_protocol,test_rf_math,test_oob_pathway,test_model_shapes,test_determinism,test_eval_integrity}.py
reports/shen_v2/{P0,P1,P2,P3,P4}/
```
- `guard`: 目标RX训练读取必须拒绝，旧六RX默认禁止参与方法搜索；结构化日志留证据。
- `registry`: H5文件名、SHA/标签、RX实例/型号/日期，区分物理硬件与多个session。
- `data`: H5 lazy reads + chunked cache + deterministic packet IDs。支持流式计算RF特征、缓存SHA，缓存按train-scoped数据及预处理版本区分。
- `metrics`: TX-balanced、RX-balanced（权重需明确），Macro F1和worstRX，并实现replay分数验证；不将file acc与packet/window acc混算。
- `bench`: preprocessing+model的同设备测速/同步；如果输入H5读取是否计入要分别标注。
- `tests`: 读取前防泄漏、非平凡信号FFT/掩码、谱泄漏/CFO单测、不同随机哈希种子跨进程manifest哈希不变、前向梯度无nan、resumed training可复现。

## 8. 论文及真实链接（确认过的入口）

1. Shen TMC2024： https://arxiv.org/pdf/2207.02999 ，Liverpool 作者稿 https://livrepository.liverpool.ac.uk/3176924/1/TMC_2023_Receiver_Agnostic_RFFI.pdf ，代码/数据 https://github.com/gxhen/receiverAgnosticRFFI ，数据入口见repo README（非本报告保证当前下载站可用）。
2. Ma et al. TIFS2025： https://arxiv.org/pdf/2512.12070 ，DOI https://doi.org/10.1109/TIFS.2025.3630316 ，作者资料页 https://junqing-zhang.github.io/research/rffi/rffi-pub/ ；截至本次核实**未确认公开的可直接复现代码仓**，不得杜撰。
3. IB-RFF TIFS2026： https://doi.org/10.1109/TIFS.2026.3723081 ；代码 https://github.com/BeechburgPieStar/IB-RFF ；它的LoRa数据与Shen20-SDR未必相同，须检查README的receiver_drift_dataset。
4. WiFi OOB orthogonality TIFS2026： https://doi.org/10.1109/TIFS.2026.3723193 ，作者介绍 https://gxhen.github.io/publications/ ，**未核实官方代码仓或公开PDF**，不能虚构。
5. Cross-receiver domain generalization RIEI IoTJ2025： https://doi.org/10.1109/JIOT.2024.3486068 ，讨论TX/RX disentanglement，代码未核实。
6. 2026BERT+2-stage KD IEEE Communications Letters： https://doi.org/10.1109/LCOMM.2026.3656814 ，针对轻量/延迟构成相关工作，代码未核实。
7. 2026 plug-and-play DG： https://doi.org/10.1016/j.adhoc.2026.104199 ，讨论冻结backbone后附加模块，代码未核实。
8. 用户第一篇OOB论文：上传 `OOB_IOTJ(7).pdf`，主要结果为Shen-CIS 99.24%、C'-OOB 82.32%、诊断出OOB接收机依赖；这是当前提出假设的历史证据而不是新方法成绩。
9. 用户仓代码： https://github.com/simizhanghao/keyan/tree/e91eb1d97f521778d075484133a3683e382437f8 ；核心模型 `src/rfhstu/models.py`、`src/rfhstu/oob_fusion.py`、`src/rfhstu/features.py`、`experiments/paper1_audit/scripts/run_final_short.py`。

## 9. 给Agent的第一条明确指令（只读，不启动GPU）

你在`simizhanghao/keyan`当前SHA上进行**P0代码与数据协议审计**，不准修改历史paper_ready/原论文结果，不准使用六个历史盲测RX为新方法调参，不准自动训练，也不准导入OSU旧终局成绩用于Shen数值比较。逐文件给出可复用函数、其输入/输出/归一化/实际使用的训练脚本和每条差异的行号。检验本地H5与README缺失工件，生成无数据访问的初始manifest，给出硬件/日期/packet层级。发现seed不可重复、标签泄漏风险、主路OOB未切断、频率轴和时间轴位置编码混淆必须列P0 blocker。输出 `CODE_AUDIT.md`、`SHEN_DATA_CARD.md`、`PROTOCOL_DRAFT.md`、`IMPLEMENTATION_TASKS.md` 和可独立执行的最小tests。每一个研究假设标记“文献结果 / 旧仓实验事实 / 未验证假设”，不需要创造新模型。等待用户批准P1后再改代码与训练。

## 10. GO/NO-GO门槛（建议，须在新实验开始前冻结）
- P0：数据、协议、tests、定位到所有必需代码文件且无隐藏标签访问 => GO；否则STOP。
- P1：旧模型在确定性协议上能合理复核，且训练曲线/seed受控。若变强只因训练epoch，不写成OOB机制创新。
- P2：至少在事先选定的开发RX folds，真实OOB分支相对于同主干Strict IB和no-explicit-OOB在RX等权Acc/F1上有可重复正增量，最差RX无重大损失 => 才可做P3。
- P3：新方法同时击败普通cross-attn gate、简单拼接、source-only augmentation配对对照，并具有明确诊断解释 => 可成方法候选；否则回退保留负面结果。
- P4：系统级与强对比准确率-效率Pareto、数据预算曲线和外部可用数据复核支撑贡献才进入论文写作。绝不为通过门槛事后改fold、调RX集合和阈值。
