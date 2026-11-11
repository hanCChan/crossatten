# M0 完成：当前本地 Raw/EQ 配对证据不足，停止双视图候选

结论为 **`STOP_NO_VERIFIABLE_LOCAL_PAIRING`**。UCLA 官方来源支持存在未均衡和均衡 Id signal 两种处理视图；当前审阅的本地材料不足以验证逐包身份链。按[独立登记](PREREGISTRATION.zh-CN.md)在 G1 结束，TX 信息保留和跨 RX 相关性均为 **NOT_ASSESSED**，不启动双视图训练。

这不是“均衡一定破坏 TX 信息”，也不是“同包配对在原则上不可能”。本次没有打开科学 IQ、compact pickle 载荷、模型或预测，没有科学数据下载、前向或新分类器拟合。[机器决策](DECISION.json)封存了边界。

## 比 D4 新增的具体证据

[官方概述](https://cores.ee.ucla.edu/wisig/overview/)区分未分包的 Raw WiSig 采集流与经过检测得到的 256 点 Id signals。本研究中 Raw/EQ 指后者的未均衡/均衡版本。网页描述同一前导码的两个版本，不能替代本地每行的身份核验。

官方处理链固定到 `wisig-process-raw@6530c6ed6be1d08544ac295040ed50f5babde86a`，compact 代码固定到 `wisig-subset-creation@bc8c6bd352a8c184a571e6b4c2d7a890cdfff3d8`：

| 环节 | 可核对的事实 | 对配对的含义 |
|---|---|---|
| 检测与筛包 | `split_signal_fun.m` 将 start/end 等另存 statistics，输出 packet_log 时只保存筛后信号 | 原始采集端点并未随每个 compact 行一起传递 |
| 均衡函数 | `equalize_channel.m` 在检测失败或长度不足时返回空 | 有可能丢弃输入包；代码本身不证明某个本地包实际被丢弃 |
| 均衡批处理 | `equalize_dataset.m` 只 append 非空输出，最终仅保存 packet_log，没有保存原 pkt_i | 失败后 EQ 行号可能相对未均衡行号移动，不能直接按 k 连接 |
| Python 打包 | `build_dataset_multirx.py` 遍历 packet_log、截取 256 点，输出 data/node_list | 没有传递原包 ID 或跨版本保留索引 |
| compact 创建 | `create_dataset_impl` 按 raw/eq 两种 prefix 分别读取，并分别取 `[:max_sig]` | 相同上限和最终包数也不建立身份对应 |

来源分别为[检测代码](https://github.com/WiSig-dataset/wisig-process-raw/blob/6530c6ed6be1d08544ac295040ed50f5babde86a/matlab_detection_screening/split_signal_fun.m)、[均衡函数](https://github.com/WiSig-dataset/wisig-process-raw/blob/6530c6ed6be1d08544ac295040ed50f5babde86a/matlab_equalization/equalize_channel.m)、[均衡批处理](https://github.com/WiSig-dataset/wisig-process-raw/blob/6530c6ed6be1d08544ac295040ed50f5babde86a/matlab_equalization/equalize_dataset.m)、[Python 打包](https://github.com/WiSig-dataset/wisig-process-raw/blob/6530c6ed6be1d08544ac295040ed50f5babde86a/python_pkl_creation/py/build_dataset_multirx.py)、[compact 创建](https://github.com/WiSig-dataset/wisig-subset-creation/blob/bc8c6bd352a8c184a571e6b4c2d7a890cdfff3d8/data_utilities.py)。逐文件 Git blob SHA、本地审阅副本和许可证见 [PUBLIC_SOURCE_MANIFEST.json](PUBLIC_SOURCE_MANIFEST.json)。8 份公共文本快照已逐一按固定 Git blob SHA 核对，保留原行尾；没有执行这些上游脚本。

示意上，若 Raw 包为 `[a,b,c,d]` 而 b 均衡失败，输出可能为 `[eq(a),eq(c),eq(d)]`。再分别截前两包会得到 `[a,b]` 与 `[eq(a),eq(c)]`。这只是代码允许的反例，**不是对本地某组实际行对应的测量**；它说明下标假设不足。

## 本地来源、sidecar 与身份链

[LOCAL_METADATA_REVIEW.json](LOCAL_METADATA_REVIEW.json)记录 11 个明确文本文件的 SHA256、JSON 字段 schema、3 个现有数据目录的文件名，以及 ZIP 中央目录。重现脚本为 [metadata_review.py](metadata_review.py)，仅使用标准库，不能调用科学数据加载器。

- 历史 `05-1-ManyRx-manifest.json` 与现有 `DOWNLOAD_MANIFEST.json` 将 ManyRx 来源标为 **Kaggle v1 拷贝**，记录 pickle 和 archive 哈希。这里引用的是保存的来源哈希，本轮未重新读科学文件计算哈希，也未证明它逐字节等同于当前 UCLA 下载件。
- 本地 `lib/xa/wisig.py` / `13-mixer-cfo-mechanisms/data.py` 从 `[tx][rx][day][0]` 取前 100 包，功率归一化后保存 x/y。它们没有核验 EQ 行身份；相同标签、shape 和版本列表不构成配对证据。
- fold2 的 `fit_manifest.json` 只记录接收机分组、6 个 NPY 文件摘要与 source_only；没有跨视图映射字段。已审阅 D4/INPUT_REGISTRY 同样明确没有 immutable packet IDs。
- 数据目录没有名称指向 packet-map 的 sidecar；64 个 ManyRx 缓存文件旁无 JSON sidecar；Kaggle ZIP 中央目录只有一个 `ManyRx.pkl` 成员，没有外置映射文件。

本轮**没有解析该 pickle 的内部载荷/全部字段**，因此不声称其中绝无额外信息，也不将有界文件名搜索写成全工作区证明。结论严格限定为：已审阅的来源、schema、代码和 sidecar 无法使当前本地 G1 通过。[D4 历史结论](../classification_failure_diagnostics/D4_FRONTEND_AUDIT.zh-CN.md)保持原样。

并行的[Full-WiSig 元信息任务](../wisig_full_confirmation_metadata/M0_SOURCE_COUNT_SUPPORT.json)提供了官方聚合包数：限定 fit18 × 前两日 × 原十 TX 的 360 个 cell 中，206 个 cell 的 Full 原始 Raw/EQ 总包数不相同（Raw 333,105，EQ 317,149）。[固定来源与摘要](COUNT_SUPPORT_REFERENCE.json)记录来源提交和哈希。它只补充“两版本长度并不总相等”的元信息；**不是本地前 100 包中有 206 组已被证实错配**，也不能证明 EQ 是 Raw 的同索引子集或确定实际丢包位置。M0 仅读取已导出的聚合 JSON，没有解析此公共元信息 pickle。

## 物理与判别解释边界

固定均衡代码先重采样和重新检测/裁窗，临时校正粗细 CFO，依据 L-LTF 估计信道与噪声，进行带噪声正则的频域补偿，只写回 52 个选定频点，逆变换后将估计 CFO 加回并重采样。因此不能简化为“仅去掉 RX 硬件”或“永久去掉 CFO”。这些是代码事实；由此推断它可能改变 TX 判别信息是合理的风险解释，**不是已测得的信息损失**。[处理依据](https://github.com/WiSig-dataset/wisig-process-raw/blob/6530c6ed6be1d08544ac295040ed50f5babde86a/matlab_equalization/equalize_channel.m)。

由于 G1 未通过，M0 没有计算 Raw/EQ 预测、margin、错误翻转、类别对风险或跨 RX 相关性。不能把没有评估写成“安全通过”或“物理变换无效”。从合法原始采集重新运行并保存身份链，原则上可能恢复映射；它不属于这次已结束的元信息范围，不自动重试或改名。

当前双视图路线在现有材料上关闭，C 按独立协议继续。没有新 Loss、新结构、新训练或 CRD 突破声明；S17B、S18、S19 等历史 STOP 均不变。
