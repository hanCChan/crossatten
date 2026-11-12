# M1-A 共轭镜像：物理与源域可行性结果

**结论：`STOP_EXPLICIT_MIRROR_ESTIMATOR`。本轮分类器拟合和深度更新均为0，M1-B/C未启动。**

主要问题是模型与真实输入失配：WL条件数约1，但拟合/留出NMSE中位数分别为0.480993/0.486520。不是因为SVD病态，也不能据此归因于某一种TX或RX硬件失真。详见[失配、歧义与阶段资格解释](INTERPRETATION.zh-CN.md)。

合成唯一场景包数为1,728；首次JSON输出发生类型错误，保留现场后仅修复布尔序列化，以同一种子重现完整输出，总计计算3,456个合成包。源域仅审计一次，独立QR再读取同一批1,872包，新增唯一源IQ行数为0。原预注册与全部科学阈值未改，工程更正见 [ENGINEERING_AMENDMENT.json](ENGINEERING_AMENDMENT.json)。

## 来源、物理可信度与可辨识性

官方资料支持25 Msps未均衡256点输入；本地继承既有Kaggle来源链，没有宣称与当前UCLA下载逐字节比对。能量检测不提供每包绝对STF起点。只检验预先固定的[20,180)候选内部区间，拟合前四周期、验证后四周期，没有移动区间或通过分类准确率选择模板。见 [来源](SOURCES.md)。

使用完整 TX→信道/CFO→RX 链。RX共轭分支的CFO反向旋转；RX-only即使TX不平衡为0也产生非零有效镜像项。真实TX参数未估计。半周期共轭歧义、未知信道列空间混淆，以及CFO为0时的四列秩亏均显式保留；见 [可辨识性](IDENTIFIABILITY.md)。

合成1728包、有效1665包（96.354%）；8项工程测试通过。已知时序/平坦信道/零CFO的有效系数恢复共288例，最大误差/事前噪声容差=0.343799。这不是在未知信道下恢复真实TX参数。

L-STF共轭与10点平移等式最大误差3.641e-14；12延迟普通信道秩=12，24列广义线性信道秩=12。两列条件数好仍不能消除联合时序或信道歧义。

## 有界源域结果

仅使用两个开发视图共同fit的六台RX、原train_ids、Day1–2，每非空TX×RX×Day最多16包。共1872包，131包满足全部有效条件（6.998%）；失败包没有删除。

| RX | Day | 抽样数 | 有效数 | 有效比例 | 失败原因计数（可重叠） |
|---|---:|---:|---:|---:|---|
| 14-7 | 1 | 160 | 17 | 10.625% | {"fit_template_mismatch": 142, "held_template_mismatch": 141, "CFO_out_of_registered_range": 2, "weak_direct_or_large_effective_mirror": 1} |
| 14-7 | 2 | 160 | 4 | 2.500% | {"fit_template_mismatch": 150, "held_template_mismatch": 155, "CFO_out_of_registered_range": 1, "weak_direct_or_large_effective_mirror": 1} |
| 18-19 | 1 | 160 | 0 | 0.000% | {"CFO_out_of_registered_range": 17, "fit_template_mismatch": 160, "held_template_mismatch": 160, "weak_direct_or_large_effective_mirror": 19} |
| 18-19 | 2 | 160 | 0 | 0.000% | {"fit_template_mismatch": 160, "held_template_mismatch": 160, "CFO_out_of_registered_range": 14, "weak_direct_or_large_effective_mirror": 19} |
| 18-2 | 1 | 160 | 0 | 0.000% | {"fit_template_mismatch": 160, "held_template_mismatch": 160, "CFO_out_of_registered_range": 13, "weak_direct_or_large_effective_mirror": 16} |
| 18-2 | 2 | 160 | 0 | 0.000% | {"CFO_out_of_registered_range": 10, "fit_template_mismatch": 160, "held_template_mismatch": 160, "weak_direct_or_large_effective_mirror": 14} |
| 23-5 | 1 | 144 | 14 | 9.722% | {"fit_template_mismatch": 130, "held_template_mismatch": 130, "weak_direct_or_large_effective_mirror": 4, "CFO_out_of_registered_range": 5} |
| 23-5 | 2 | 144 | 16 | 11.111% | {"fit_template_mismatch": 128, "held_template_mismatch": 128, "CFO_out_of_registered_range": 6, "weak_direct_or_large_effective_mirror": 11} |
| 23-7 | 1 | 144 | 16 | 11.111% | {"fit_template_mismatch": 128, "held_template_mismatch": 128} |
| 23-7 | 2 | 160 | 64 | 40.000% | {"fit_template_mismatch": 96, "held_template_mismatch": 96} |
| 8-7 | 1 | 160 | 0 | 0.000% | {"fit_template_mismatch": 160, "held_template_mismatch": 160, "CFO_out_of_registered_range": 21, "weak_direct_or_large_effective_mirror": 36} |
| 8-7 | 2 | 160 | 0 | 0.000% | {"fit_template_mismatch": 160, "held_template_mismatch": 160, "CFO_out_of_registered_range": 25, "weak_direct_or_large_effective_mirror": 24} |

同自由度、同频带NC相对WL的验证残差优势（除以L残差）RX等权均值=-0.005083，六个已观察RX条件95%区间=[-0.011457,-0.000492]，登记实用门为0.02；无效包记0优势，同时保留原始残差，不能将其排除以提高收益。L→WL相对验证改善=0.001644。这些均不是TX分类准确率。

## 原门槛与解释边界

- overall_valid_at_least90：FAIL。
- every_RX_day_at_least80：FAIL。
- present_cell_support_at_least8：PASS。
- all_RX_days_present：PASS。
- mean_validation_advantage_at_least2percent：FAIL。
- all_RX_advantage_positive：FAIL。
- both_days_advantage_positive：FAIL。
- RX_conditional_interval_lower_positive：FAIL。
- WL_beats_L_on_held_segment：PASS。

可靠性与abs(CFO)/fit残差/channel proxy的Spearman关联：{"abs_CFO": -0.05752728656794221, "fit_nmse": -0.4417800348307062, "channel_proxy_cv": -0.41568322581611755}。仅为描述性混杂线索，不能归因于RX硬件或TX指纹。每个TX×RX×Day单元含空缺均在 SOURCE_VALIDITY.json 列出。

有效q使用预定直接分量占优的参考规范；它不是唯一物理到达时序或真实TX I/Q系数。观测耦合可来自TX、RX、CFO、信道、截窗或其组合；现有数据没有完成这些因素的因果分离。即使A通过，也不证明跨RX TX判别价值，仍须另行授权B。

本候选按冻结门结束；没有为失败修改起点、阈值、对照或估计器，也没有自动启动镜像网络、线性探针或替代表示。F2/P2/固定窗口/S09/S17/S19历史STOP与原报告保持。

独立复核通过20650项，其中同一批1872包用另一份频率多项式实现和QR求解器复算系数与包内验证残差；独立重算覆盖率、RX条件区间、门槛及抽样身份。新增独立IQ行数为0。

没有访问checkpoint_val、source_selection、历史held6、target8、Day3–4、Full UNKNOWN RX或任何模型/预测；没有跨RX包事件配对。访问日志区分整文件字节摘要、选定行科学解码及独立同批回放。

[合成明细](SYNTHETIC_RESULTS.json) · [源域有效率](SOURCE_VALIDITY.json) · [全部逐包诊断](PACKET_DIAGNOSTICS.json) · [机器决策](DECISION.json) · [独立复核](INDEPENDENT_QA.json) · [访问日志](ACCESS_LOG.jsonl)
