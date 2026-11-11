# Full WiSig 新 RX 确认资格：仅元信息结果

状态：`METADATA_SUPPORT_IDENTIFIED_HISTORICAL_EXPOSURE_UNKNOWN`。确认集访问保持 **CLOSED**；本任务没有读取候选 IQ、预测或准确率，没有形成可评分的最终确认集合。

官方公开计数确实给出 Full 41 个 RX，而当前 ManyRx 是其中 32 个。差集的确是下表 9 个官方 RX 节点标识，但它们并非 9 个已经合格、未使用的确认接收机。官方也说明 Full 的 TX×RX×日期数量不均衡，不能由设备总数推导共同 TX 或日期支持。[UCLA WiSig 官方说明](https://cores.ee.ucla.edu/downloads/datasets/wisig/)

## 已验证的身份与计数

从官方 `data_summary.pkl` 提取日期、TX/RX 标识和计数；仅接纳已登记的 459,700 字节 Git blob，限定 3 个 NumPy 数组/类型构造器，随后强制检查字段、`4×41×174` 轴长、有限非负整数计数。SHA-256 为 `072ee403d2eed11a2f960a0cc18a7370afdc60fad96d3252ba83d66a91f34146`。这不是 Full 信号数据文件。[官方计数元信息](https://github.com/WiSig-dataset/wisig-examples/blob/afbf0a95bc888a7e3d9f94fa3442d36dace6e1fb/data_summary.pkl)、[官方计数使用示例](https://github.com/WiSig-dataset/wisig-examples/blob/afbf0a95bc888a7e3d9f94fa3442d36dace6e1fb/py/d001_plot_Full_WiSig_tx_rx_grid.py)

原模型 10 个类按历史顺序映射为：`1-10, 11-1, 14-10, 14-7, 17-11, 20-15, 20-19, 7-11, 7-14, 8-20`。这一顺序同时由本地 `05-wisig/out/05-1-ManyRx-manifest.json` 和官方 ManyRx 创建语句支持；10 个标识均出现在 Full TX 清单中。注意官方创建 notebook 末尾保留了一段与前面语句不一致的旧 `datasets_info` 输出，本任务没有采用该旧输出。[官方创建代码](https://github.com/WiSig-dataset/wisig-subset-creation/blob/bc8c6bd352a8c184a571e6b4c2d7a890cdfff3d8/py/002_FullWiSig_create_compact_datasets.py)

日期依次为 2021-03-01、03-08、03-15、03-23，记为 D1–D4。表中“最少包数”是在原 10 TX 中取最小值；0 表示至少一个原 TX 无对应信号。原历史 CRD 加载器每个 TX/RX/日最多取前 100 个 raw 包，因此以 D3、D4 各 TX 至少 100 个作为**原预算兼容性检查**，没有删除困难 TX 来凑合格集合。各 TX 的 raw/EQ 精确包数见 [CANDIDATE_METADATA.json](CANDIDATE_METADATA.json)。

| 额外 RX | raw 最少包数 D1/D2/D3/D4 | EQ 最少包数 D1/D2/D3/D4 | D3–D4 raw 原10TX各≥100 | 历史未用于方法选择 | 当前确认资格 |
| --- | --- | --- | --- | --- | --- |
| 11-10 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 不满足 | UNKNOWN | 未合格 |
| 13-13 | 604 / 0 / 0 / 580 | 603 / 0 / 0 / 580 | 不满足 | UNKNOWN | 未合格 |
| 14-14 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 不满足 | UNKNOWN | 未合格 |
| 2-20 | 0 / 0 / 551 / 580 | 0 / 0 / 551 / 579 | 满足计数支持 | UNKNOWN | 待证实历史独立性 |
| 24-1 | 0 / 57 / 48 / 0 | 0 / 40 / 23 / 0 | 不满足 | UNKNOWN | 未合格 |
| 24-12 | 0 / 0 / 30 / 0 | 0 / 5 / 7 / 9 | 不满足 | UNKNOWN | 未合格 |
| 24-18 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 不满足 | UNKNOWN | 未合格 |
| 3-2 | 648 / 0 / 62 / 0 | 611 / 0 / 62 / 0 | 不满足 | UNKNOWN | 未合格 |
| 8-13 | 648 / 0 / 609 / 579 | 648 / 0 / 609 / 578 | 满足计数支持 | UNKNOWN | 待证实历史独立性 |

这些是官方 RX 节点身份，尚未核验硬件序列号、设备是否跨日更换，以及它们与采集批次/空间位置等条件的独立性。表格不支持任何硬件优劣结论。

## 历史独立性尚未证实

现存 S10 归档的 ManyRx 接收机身份投影覆盖原 32 RX，与这 9 个差集无交集。本任务只保留 `dataset/rx` 身份信息，没有分析指标列。限定的本地文本协议/代码/manifest 搜索未找到这些候选标识；三个根目录 ZIP 只查看中央目录文件名。以上只能说明当前这些记录没有证明使用过候选，不能证明从未被使用。未归档、删除的实验、其他机器或人工查看仍是 UNKNOWN，见 [HISTORICAL_EXPOSURE.json](HISTORICAL_EXPOSURE.json)。

尤其是 `13-13`、`2-20`、`8-13` 本来就包含在官方 **SingleDay** 创建 RX 清单里；因此“未出现在 ManyRx32”也不能被替换成“未出现在其他 WiSig 子集或历史工作”。这个任务没有读取 SingleDay 或任何候选的科学数据。[官方 SingleDay 创建语句](https://github.com/WiSig-dataset/wisig-subset-creation/blob/bc8c6bd352a8c184a571e6b4c2d7a890cdfff3d8/py/002_FullWiSig_create_compact_datasets.py)

当前结果是两个具有原 CRD 包数支持的候选机会：`2-20`、`8-13`。最终合格 RX 数仍为 **0**，没有冻结确认集合；不能据此宣称已有 2 个盲测 RX。后续只有补足历史使用的可核验记录、另行冻结方法与确认协议后，才可能接纳确认访问，规则见 [QUALIFICATION_PROTOCOL.zh-CN.md](QUALIFICATION_PROTOCOL.zh-CN.md)。这不授权新的科学训练或 Full 源域重拟合，也不改变 C 的关闭确认门。

## 访问与复核

- 范围先登记于 [ACCESS_SCOPE.json](ACCESS_SCOPE.json) 与 [METADATA_FILE_ADMISSION.json](METADATA_FILE_ADMISSION.json)，随后才获取官方计数 blob。
- GitHub 插件拒绝非 UTF-8 二进制，已记录为访问失败；仅这个固定计数 blob 使用直接 HTTPS 读取，按 Git blob SHA 和字节数核对。没有下载 `IdSig_info.pkl`、Full/Raw/compact 信号文件或权重。
- `inspect_counts_metadata.py` 只做固定 blob 静态检查；`decode_counts_metadata.py` 只转换聚合计数；`project_historical_metadata.py` 只投影已有身份/ZIP 文件名。原始计数 blob 存于忽略的 `artifacts/`，公开结果是文本 JSON。
- [M0_SOURCE_COUNT_SUPPORT.json](M0_SOURCE_COUNT_SUPPORT.json) 只提供已有合法 fit18×D1–D2×10TX 的辅助计数：360 个 cell 中 206 个 raw/EQ 数量不同。它不证明同包映射、包含关系、TX 信息安全或性能。

这项元信息资格任务在此结束；不把未知历史独立性转成额外模型诊断，不通过候选性能选择确认集合。
