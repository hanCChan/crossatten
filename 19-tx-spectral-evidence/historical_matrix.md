# S00–S18 历史结果矩阵与 S19 继承边界

核验日期：2026-11-10。本文件只读取已有报告、注册、配置、决策与源代码；未打开原始 IQ、checkpoint 或执行目标评分。数字单位除另注均为百分比，差值单位为百分点（pp）。报告中的 source/Dev/Confirm/held-source/CR/CRD 明确区分，不能跨列拼成统一算法成绩。

本地 S18 完整提交为 `933d4a6e05bedb2ad0426c9b99679c07106ccf10`。核对 `18-selective-rx/MANIFEST.json` 的43个归档文件：SHA全部相符，全部存在于该提交；不再是“来源未核验”的用户摘要。此处没有加载权重，S18自身权重核验结论来自已冻结的 `FINAL_ACCESS_AUDIT.json`。

## 评价角色和配方键

- **Shen source_val**：每个源RX/TX保留40包，只用于checkpoint、调度等训练内选择；source训练为160包/TX/RX。
- **Shen Dev**：两fold互换的5台未见源RX，历史上用于研究方向选择；**Confirm**为反复查看过的4台接收机，不能称新盲测；6台blind在本阶段禁止读取。
- **WiSig source validation**：fit RX内的包划分，不是未知RX泛化结果。**held-source**：从该fold官方源RX中另留的接收机，仅是开发泛化代理。
- **WiSig CR**：测试RX、日期1–2；**CRD**：测试RX、日期3–4。下文CR/CRD仅引用已冻结汇总，不打开相应数据或重复评估。
- **Shen200**：P-H5x160，TrueIB、原始包AWGN U[0,50)dB、逐包RMS，AdamW lr1e-3/wd5e-4，batch64，warmup5/cosine，200ep；RX等权源验证CE最低checkpoint。S13的各模型均先封存checkpoint再读Dev/已用Confirm。
- **WiSig-official**：Adam lr1e-3/wd0，batch128，最多200ep，Plateau(.1,patience5,min1e-6)，earlystop10；与各模型公开配方相关的额外步骤保留。
- **WiSig-unified**：AdamW lr1e-3/wd5e-4，batch64，warmup5/cosine200ep，默认AWGN U[0,50)；S07 mild变体为U[10,50)。
- **WiSig-small40**：ManyRx五通道HStem+普通MHA，82,426参数；Adam1e-3/wd0/batch128/FP32，上限40ep，Plateau(.1,5,min1e-6)，earlywait10，RX等权源验证CE选模。S17B seed18101、包split16001；fit18个源RX、first6另留源RX，仅日期1–2。

完整数据与参数定义见 [Shen协议](../docs/PROTOCOL.md)、[WiSig公平协议](../10-tifs-validation/PROTOCOL_AND_LEAKAGE_AUDIT.md)、[recipe代码](../lib/xa/paper_train.py)及各阶段config。相同数据评价协议不等于不同论文方法的优化器、轮数、预训练和选模规则完全相同。

## 阶段矩阵

| 阶段 | 数据、任务、输入与前端 | 模型、参数与训练配方 | 冻结指标及角色 | 有效性 / 出口与证据 |
|---|---|---|---|---|
| S00 | Shen，10台source/Dev RX，每台2000包；8192点IQ、梳状谱、频段、CFO/SNR、chirp残差 | 线性描述性probe，无新深网训练；10类随机率10% | 带内梳状probe单RX/跨RX=71.2/73.8；带内残差50.7/52.5；CFO单RX62.8→跨RX10.0，SNR33.8→8.9 | 支持频带/残差研究线索，不能视为硬件因果识别。**“>200kHz远带是主要捷径”的早期推论已被S01更正为不支持**。[报告](../00-signal-audit/RESULTS.md) |
| S01 | Shen P-H5x160，严格带内与全频带/近带OOB、CIS | CrossAttn TrueIB 580,008；Shen-CIS Flatten 13,442,954；探索76fit/确认114fit。TrueIB100ep warmcos+AWGN；CIS保留各自公开配方 | TrueIB+AWGN Dev94.74、Confirm92.11、source_val94.4；全频CrossAttn+AWGN Dev50.46/Confirm56.75；CIS+AWGN93.12/95.20 | **严格带内有强同模型对照支持**，不是远带单一去噪已被证明。CIS GAP保留位置失败，GAP270,218参数Dev58.67。[报告](../01-strong-baselines/RESULTS.md)、[决定](../01-strong-baselines/decision_explore.json) |
| S02 | Shen同协议；mean-chirp、combFFT、dechirp、Raw-OOB、时频split、深度与轮数 | 基线580,008；depth4 622,246；Raw-OOB625,896；rank16 CIS658,314；主要100ep，预登记ep200臂200ep | 确认：ep200 Dev95.84/Confirm93.89，相对基线+1.11/+1.78pp；mean-chirp81.09/80.40，相对−13.64/−11.71pp；Raw-OOB full83.44/83.63；combFFT88.99/86.48 | 轮数有效；mean-only与Raw-OOB受损；combFFT探索曾+2.72但新seed确认−5.74，不能只留探索正结果；AWGN10确认unclear。[确认表](../02-single-factor-screen/summary/confirm_tables.md)、[探索决定](../02-single-factor-screen/decision_explore.json) |
| S03 | Shen8192点TrueIB，legacy5 RF视图；层级token、残差、ifreq、symstack、preLN | HStem+RF-HSTU 142,392；Shen200，32探索+48确认fit；32×64 tokens | HStem Dev97.48、Confirm96.84、source_val98.25；Confirm最差RX90.22 | **HStem eligible/improves**；preLN/noFFT/ifreq未独立达到正门槛，resid/symstack受损。阶段止步不等于HStem失败。[报告](../03-chirp-structure/RESULTS.md)、[确认决定](../03-chirp-structure/decision_confirm.json) |
| S04 | 当前归档没有独立`04-*`目录或可核验结果 | — | — | **未找到独立阶段证据**；S03决定仅提及未来stage04，不能凭编号补造实验。 |
| S05 | WiSig ManyRx10TX/ManySig6TX，non-equalized、256点、first100/cell、功率归一化；源PSD/STF及发布权重验收 | 无新增公平重训；严格重构官方发布结构并加载原state；这里仅引用其已冻结指标 | **发布权重、包等权**：IB-RFF CRD ManyRx89.1879/ManySig93.5694；WaveMLP88.6759/93.1250；FAN所引89.6840/93.0625是**CR** | 数据/发布权重匹配通过；初始CR对CRD误比已更正。不能与S06多seed RX等权均值混排。256点不能验证完整LTF对齐。[报告](../05-wisig/RESULTS.md) |
| S06 | WiSig CR/CRD与Shen Dev/Confirm公平多seed比较；各体制合法适配 | 588个完成fit（source176/explore100/final240/confirm72）。WiSig W1 HStem+HSTU：ManyRx91,128/ManySig90,868；FAN/IB-RFF151,761/146,605，WaveMLP约84k；Shen HStem142,392。official/unified及IB两阶段分别报告 | RX等权CRD：W1 ManyRx83.56/ManySig72.33；IB-RFF88.07/91.09；WaveMLP88.06/91.04。Shen HStem Dev97.48/Confirm96.84；TrueIB统一IB-RFF94.94/92.21 | 冻结公平参照完成；**Shen结构不能直接证明WiSig有效**。记录10次失败/中断及历史允许的后续完成，不删除失败记录。[报告](../06-paper-validation/RESULTS.md)、[决定](../06-paper-validation/decision_wifi.json) |
| S07 | WiSig频域/时域/时频split、训练与源RX选预算；OSU两RX双方向独立管线 | WiSig频域HStem+HSTU ManyRx90,680；32×64/two HSTU。ManyRx各fold源选official或unified+mild；OSU HStem+HSTU143,367 | 最终**CRD RX等权** ManyRx87.6506/F1 85.4252/最差RX68.0071；ManySig旧方案86.62。OSU跨RX包Acc1.42、TX等权2.00、F1 .78；同RX未来块事后91.24 | 频域优于多项时域方案、训练配方改善；**OSU跨RX验证失败**，不能宣称普适LoRa。S07 ManyRx不是S19 MHA+small40成绩。[报告](../07-wifi-osu/CONCLUSION_ZH.md)、[正式逐fold配方](../07-wifi-osu/S07_WIFI_FINAL_RESULTS.md) |
| S08 | WiSig相邻频点phase及源域选训练轮/EMA策略 | ManySig五通道HStem+HSTU90,868；136fit，含16次完整200ep对照。最终无AWGN；源选exact epochs[10,6,4,13]，checkpoint[EMA,raw,EMA,raw] | ManySig **CRD**93.0116/F1 92.7970/最差RX84.6667；matched control87.56，Δ+5.45pp；ManyRx仅2/4源fold通过，不做新CRD | ManySig组合有效；不能把总收益只归相位，或与S07 ManyRx拼成统一新算法。相对IB-RFF均值更高但RX区间跨零。[报告](../08-wisig-training/RESULTS.md)、[冻结配方](../08-wisig-training/S08_WIFI_FINAL_RESULTS.md) |
| S09 | WiSig五通道first/zero-phase、centered、curvature；新seed同预算对照 | 三通道90,420/五通道90,868（ManySig）；156正式fit+5smoke。固定S08 epochs/EMA与official包早停两策略；新候选96个40ep源fit | ManySig **CRD** first:fixed92.36，null:fixed86.69，Δ+5.68 [2.41,9.12]pp；first:packet92.90，null:packet88.51。centered/curvature各数据0–1/4源折支持 | **相位内容在此条件有效**，不是额外448参数或普遍RX不变性证明；新时延特征未通过，不进新目标评估。[报告](../09-phase-invariance/RESULTS.md)、[决定](../09-phase-invariance/decision_final.json) |
| S10 | WiSig软相位可靠性与ASKNet官方公平重训；重汇总历史CRD | 56正式+4smoke；ASKNet固定官方结构/loader/optimizer，24fit，seed2023–2025，official200。普通自有候选与外部方法配方不强行等同 | 软可靠性源域ManyRx−0.4881pp（1/4正向）、ManySig+0.5651冲突；冻结CRD ASKNet88.7697/93.4838，IB-RFF88.0702/91.0926，WaveMLP88.0562/91.0370 | **STOP**软可靠性候选；没有新90%或全球SOTA证据。[报告](../10-tifs-validation/RESULTS.md)、[官方重训](../10-tifs-validation/ASKNET_FAIR_REPRODUCTION.md)、[决定](../10-tifs-validation/decision_final.json) |
| S11 | WiSig中心频带/F1、TX/RX/day centroid Fisher-like频谱方差比先验/F2 | 2次hard源诊断refit，2次F1 smoke，2次F2预训练工程失败；未启动正式候选网络grid，结构/参数逐诊断config保留 | **centroid probe**（不是深网提升）ManyRx F1/F2+7.9024/+4.1702pp，ManySig+1.4271/+2.3646；TX scatter保留F1约.33，F2 ManyRx.6528/ManySig.6152 vs门槛.65 | **STOP_MECHANISM_GATE**；不是完整训练已证伪。门槛受振幅影响，不能推导真实身份信息消失；散度比不是未来新创新。[报告](../11-spectral-generalization/RESULTS.md)、[新意边界](../11-spectral-generalization/NOVELTY_AND_TIFS_DECISION.md) |
| S12 | WiSig F1/F2训练Pilot，源RX/date1–2，first组seed8101 | 16pilot+4smoke；沿用源域包选模/40ep预算，候选先验仅训练70%估计；个体参数见其封存config，不把统计probe当完整模型 | **held-source** ManyRx hard91.3201、F1 89.5801（−1.7400pp）、F2 91.3951（+.0750）；ManySig hard89.1250、F1 91.2917、F2 92.5313（+3.4062）；F2最差单配对−6.95pp | **STOP_SOURCE_GATE**；F2仅2/4 ManyRx正向、未达+1pp且最差RX退化，不能被ManySig正向补救。无独立确认或新CRD。[报告](../12-spectral-validation/RESULTS.md)、[决定](../12-spectral-validation/decision_pilot.json) |
| S13 | Shen TrueIB Mixer严格消融；WiSig CFO校正/双路；不访问新WiSig目标 | Shen200：MHA133,690，HSTU142,392，自然Transformer200,122；48P1 fit。WiSig CFO40P2 fit，HSTU B0 91,128/90,868、dual149,416/149,156；ordinary MHA初始替换定义固定 | Shen MHA **Dev97.335/Confirm96.508**；HSTU97.460/96.448；自然Transformer97.223/96.733。WiSig **held-source** B0 ManyRx90.133/ManySig90.740；dual90.461/93.479 | **HSTU无稳定独立优势**，CFO **STOP_SOURCE_GATE**；没有新真实WiSig CRD。MHA可成为统一骨干基础，但不同体制的前端仍不同。[Mixer](../13-mixer-cfo-mechanisms/P1_MIXER_RESULTS.md)、[CFO](../13-mixer-cfo-mechanisms/P2_CFO_RESULTS.md)、[结果](../13-mixer-cfo-mechanisms/RESULTS.md) |
| S14 | Shen同TrueIB，Stem×Mixer完整2×2 | 新增12fit Patch+MHA571,434/Patch+HSTU580,136，复用S13 HStem两格；Shen200，非tokenizer初始权重配对 | Dev HStem−Patch：MHA+2.113、HSTU+1.572pp；Confirm+2.823/+2.131。交互Dev−.542[−.743,−.323]pp，Confirm−.692[−.867,−.571] | HStem条件优势跨Mixer存在；**没有正向互补/协同证据**；未做Stem参数匹配，不能唯一归因层级性。[矩阵](../14-stem-mixer-attribution/STEM_MIXER_RESULTS.md)、[结论](../14-stem-mixer-attribution/RESEARCH_CONCLUSION.zh-CN.md) |
| S15 | Shen HStem与参数匹配Patch-Lite归因，继续Shen200 | 新12fit，复用24匹配控制。HStem MHA133,690/HSTU142,392；Lite MHA134,266/HSTU142,968 | Dev HStem−Lite：MHA+2.687[2.375,3.083]pp，HSTU+1.733[1.400,2.150]；Confirm+3.142/+2.806 | **HSTEM_ADVANTAGE_REMAINS_UNDER_MATCHED_PARAMETER_CONTROL**，不是STOP；仍混合感受野/BN/归纳偏置，未唯一识别“层级”的成因。[报告](../15-hstem-attribution/RESULTS.zh-CN.md)、[决定](../15-hstem-attribution/decision.json) |
| S16 | WiSig RF tokenizer先源训练后冻结；小Qwen/匹配Transformer/预训练/LoRA尾部 | 28尾部fit。ManyRx R0 MHA总82,426/可训练40,650，HSTU91,128/49,352；小Qwen839,354；预训练冻结活跃440,587,066/可训练77,834；LoRA441,733,946/1,224,714 | **held-source** ManyRx MHA89.450/HSTU89.929，小Qwen89.535，匹配Transformer89.189，预训练冻结88.759、随机冻结90.055、LoRA89.326 | **STOP_SOURCE_GATE**；MHA尾部不是端到端B0；不能单独归因语言预训练、RoPE或深度，也不是CRD。[Pilot](../16-rf-qwen3/PILOT_RESULTS.zh-CN.md)、[状态](../16-rf-qwen3/STATUS.json) |
| S17 | ManyRx fit源RX/date1–2幅频、相位曲率、IQ描述子与冻结模型错误关联 | 0新增小MHA训练；源训练70%估计，描述性关联依赖既有S16模型误差 | 幅频/相位/IQ满足稳定门槛2/4、3/4、0/4；错误Spearman中位.263/.256/.327 | **STOP_NO_STABLE_ERROR_LINK**针对可自动推进候选；相位保留线索，不构成RX相位硬件因果证明，不否定所有物理增强。[决定](../17-rx-distortion/decision_d0.json)、[解释](../17-rx-distortion/D0_INTERPRETATION.zh-CN.md) |
| S17B | ManyRx平滑相位FIR、同EVM AWGN与全局相位负对照；仅fit源验证 | 4个端到端B0，WiSig-small40；17tap/FFT512/固定强度.25/.5/1与2draw，共72组冻结干预；0新增增强模型训练 | TX保持代理通过，但源RX相位方向对齐仅6.4%–12.8%，未达15%及相对噪声+.02门槛；强度1 EVM q95约10.84%–11.28%超10% | **STOP_D1_GATE**，失败发生在增强训练前代理筛选；不是物理建模普遍无效。[复盘](../17b-phase-intervention/S17_S17B_REVIEW.zh-CN.md)、[决定](../17b-phase-intervention/decision_d1.json) |
| S18 | 已学习HStem32/48/64通道与Mixer特征，TX/RX选择性rank3/alpha.5压缩；训练/源验证隔离 | 复用S17B四个WiSig-small40；**0新fit、20冻结干预**。原完整网格guard停止，特征提取前注册D0S共同支持修正；18源RX保留 | 源验证B0 Acc范围92.4088–97.1170；Tokens跨TX稳定仅1/4；RX散度覆盖7.09%–14.88%低于15%；相对随机方向+0.0297pp，RX描述区间[−.0657,+.1226]，26唯一RX；TX保持4/4 | **STOP_D0_MECHANISM_GATE**；可重复方向线索仍在，但联合稳定性/覆盖及特异性0/4，不启动训练。可逆软压缩不删除互信息；不把此STOP泛化到所有选择性方法。[报告](../18-selective-rx/RESULTS.zh-CN.md)、[决定](../18-selective-rx/d0_support_matched/decision_d0.json)、[审计](../18-selective-rx/FINAL_ACCESS_AUDIT.json) |

以上“未验证”保留两类区别：没有执行（如S04、S11正式候选、S17B增强训练、S18学习门控）与已执行但门槛失败。不能把前者写成实测负准确率，也不能把后者重新命名后自动重试。

## S19 可继承的统一基线定义

### Shen LoRa

物理约定为 **Fs1MHz/BW125kHz/SF7，每包8192=8×1024点**。P-H5x160两fold各5源RX，训练160/源验证40包/TX/RX；Dev为另一组5台，历史Confirm4台，blind6台禁读。TrueIB在`rms_normalize`后用默认无`norm`前向FFT、`|fftfreq|≤62500Hz`掩码和IFFT实现，之后从处理后IQ重建所有RF视图。True In-Band与IB-RFF Information Bottleneck含义不同。

S13 `A1_mha`是可复用的 **133,690参数**统一骨干基线：`ChirpPipeline(stem='hierarchical', preln=True, depth=2)`，把原Mixer替换为两层普通MHA（64维、4头），保留HStem、位置/准备tokens、preLN、末尾norm与分类头。它的历史指标为Dev97.335/Confirm96.508，不能引用S03 HSTU的97.48/96.84作为MHA成绩。

其原五通道是 **I/Q两路 + fftshift后全包FFT的log1p幅度zscore一路 + 时域log1p幅度一路 + 时域phase/pi一路**，不是WiSig五通道。HStem第一层32通道，tokenize的stride4×4×4×4，总步长256，形成32×64 Tokens。代码：[ChirpPipeline](../lib/xa/models3.py)、[RF views](../lib/legacy_rfhstu/features.py)、[MHA替换](../13-mixer-cfo-mechanisms/models.py)。

将来把同一TCSE核心机制移到LoRa时，应在已有TrueIB边界内调制同一复谱，再重建该LoRa原视图；不能以“统一算法”为由把其输入换成WiSig视图、扩大频带或偷换采样率。S19首轮不为获得LoRa分数读取Dev/Confirm/blind。

### WiSig ManyRx / ManySig

物理 **Fs25MHz，256点**，non-equalized、每TX/RX/day前100包，缓存float32 `[N,2,256]`逐包功率归一化。官方RX名单固定四fold；正式历史CRD为测试RX/date3–4，本轮禁止。源开发只使用各fold合法源RX/date1–2，holdout组和包split均须先封存。

已有源模型 `s17_models.build`复用S13 WiSig前端并替换普通MHA：ManyRx **82,426参数**，64维4头、两层，HStem32×64 Tokens。五通道严格为：

1. `fftshift(fft(complex(rms_normalize(x))))`的Re除以每包复谱幅度RMS。
2. 相同复谱Im除以相同RMS。
3. 原幅度`log1p`后按频点zscore（std floor1e-6）。
4. 相邻复频点`z[k]*conj(z[k-1])/(|z[k]||z[k-1]|)`的实部。
5. 同一product的虚部；相邻幅度都≥0.05才有效，左pad1。幅度门限对应归一化Re/Im构成的复谱，不是绝对物理电压。

FFT默认forward无归一化；`fftshift`坐标间隔25MHz/256。**WiSig基线没有Shen的±62.5kHz裁剪**，本轮不能借Shen结论另选截止。相位附加通道不变性不等于保留Re/Im的完整网络不变性。任何频谱调制先作用于同一复谱，再重算Re/Im、LogMag及受幅度阈值影响的phase通道，不能仅缩放某个feature后称真实滤波。

代码：[frequency_views](../lib/xa/explore_models.py)、[first phase](../09-phase-invariance/phase_features.py)、[WiSig adapter](../13-mixer-cfo-mechanisms/wifi.py)、[MHA build](../17-rx-distortion/s17_models.py)。若S19严格复用B0，应核对初始化、数据顺序、split、optimizer、预算与选模规则；S16冻结tokenizer尾部不能替代端到端模型。

## 冻结公平 CRD 参照与统计边界

| 方法 | ManyRx RX Acc / F1 / 最差RX | ManySig RX Acc / F1 / 最差RX | 来源与性质 |
|---|---|---|---|
| IB-RFF | 88.0702 / 85.8595 / 67.1781 | 91.0926 / 90.8187 / 76.7778 | S06多seed公平重训，S10重汇总；FAN预训练+HSIC微调两阶段 |
| ASKNet | 88.7697 / 86.5035 / 65.2028 | 93.4838 / 93.1502 / 82.9167 | S10官方commit `f00846d...`、官方模型/训练/loader，seed2023–2025 |
| WaveMLP | 88.0562 / 85.62 / 68.96 | 91.0370 / 90.41 / 78.75 | S06公平多seed重训；F1/最差RX此处引用S06两位小数汇总 |
| 历史自有S07 / S08 | 87.6506 / 85.4252 / 68.0071 | 93.0116 / 92.7970 / 84.6667 | 数据集各自冻结强方案；不是一个统一MHA+TCSE模型 |

S07−IB-RFF ManyRx差−.4197pp，RX条件区间[−1.8907,+1.0249]；S07−ASKNet−1.1191[−2.5269,+.3268]。S08−IB-RFF ManySig+1.9190[−1.4537,+5.0695]；S08−ASKNet−.4722[−1.9491,+.8796]。不能由均值认证稳定排名。包、同一RX的日期、重复seed和重复fold均不能充作新的独立RX；RX聚类区间不覆盖全部训练随机性。公平外部baseline与自有训练种子未配对。

来源：[S10最终表](../10-tifs-validation/FINAL_CR_CRD_COMPARISONS.md)、[RX区间](../10-tifs-validation/receiver_cluster_intervals.csv)、[S18公平审计](../18-selective-rx/PROTOCOL_AUDIT.zh-CN.md)。上述表只作为历史背景，不为S19重开目标评分。

## TCSE 与已有方法的边界：本次有限主来源核对

2026-11-10只读核验作者官方仓库。以下是方法范围核对，**不是全面文献查新或原创性认证**。

- [IB-RFF官方README](https://github.com/BeechburgPieStar/IB-RFF)定义FAN+HSIC两阶段，不需RX标签；其信息瓶颈不是固定带内裁剪。[SAF/FAF作者代码](https://github.com/BeechburgPieStar/IB-RFF/blob/main/backbones/Adaptive_Filter_Module.py)已有输入复谱可学习滤波与能量mask、特征FFT滤波，因此“给HStem前加频谱可学习门控”不能独立构成创新。该main链接用于只读概念核验；若移植代码训练应锁commit并归属。
- [ASKNet官方README](https://github.com/BeechburgPieStar/ASKNet-RFF)及[固定commit模型代码](https://github.com/BeechburgPieStar/ASKNet-RFF/blob/f00846d68425315b68ae69b8c6b712e08559c2fe/WiSig/backbones/PatchNet.py)已有候选低频区的Gumbel-softmax截止选择及酉Koopman算子。普通频带权重、低频选择或频域相位变换不是TCSE首创。B1只能称参考控制，不能称完整ASKNet/FAN复现。
- S11/S12已有按TX/RX/day统计的频谱先验，S18已有TX保持的方向选择；TCSE若有区别，应落实在**冻结源模型对频带遮蔽的TX margin影响、跨RX保守聚合、该监督相对同结构CE控制的独立贡献**。分类器遮蔽响应只是模型条件下的反事实代理，仍可能反映教师捷径或遮蔽伪迹；不得直接称因果TX硬件证据。
- S19中B4−B3是主要监督贡献检验，同时须通过B0/B1/B2比较及能量匹配遮蔽负对照。若只B4−B0正向而B4−B3不可靠，只能归因一般频谱结构/训练变化，不能声称证据监督有效。不得因为看过held-source或目标结果再修改K、alpha、lambda、mask或分组追分。
- 未找到当前归档中已完成GRL的数值证据；CCAlign只有S01引用的同协议Strict-CIS rank16外部参照（20ep/无AWGN，配方不同）。不能把“出现于计划/文献”写成“已跑HStem配对GRL/CCAlign”。也不能把后来添加一般损失称新创新。

S19首轮只回答受控ManyRx源域问题。独立组/seed、ManySig和Shen都必须以源域GO为前提；没有这些证据，不主张统一跨体制最优、未知RX普遍提升或超过IB-RFF/ASKNet。
