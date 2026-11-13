# WiSig质量假设 W1 实测结果

判定：`W1_PASS_PROCEED_TO_FROZEN_W2_DESIGN`。深度训练0次，合法源包22968个，目标RX/Day3–4访问0。

|冻结模型|同TX/RX/Day内低一致性额外错误 pp|RX聚类95%条件区间 pp|正向RX|
|---|---:|---:|---:|
|C0 普通频域|+17.5545|[+9.1443, +27.0703]|12/12|
|C1 EMA强参照|+17.5332|[+9.0440, +26.9902]|12/12|

区间以RX为单位，先对三个随机流平均，不将包数量冒充独立接收机数量。两视图、每流和完整辅助代理结果见W1_RESULT.json / W1_VIEW_RESULTS.json，240个条件单元的每模型/代理结果见W1_CELLS.csv。

门槛逐项：

- F0_effect_at_least_2pp: 通过
- F0_rx_ci_lower_positive: 通过
- E0_effect_at_least_2pp: 通过
- E0_rx_ci_lower_positive: 通过
- EMA_both_views_positive: 通过
- EMA_2of3_streams_positive: 通过
- EMA_8of12_rx_positive: 通过

按原授权继续固定W2模型实现、合成验收及匹配训练，W1本身不证明算法收益。

低组是每个TX×RX×Day内部最低四分位，仅用于控制混杂的事后分层，不能作为推断阈值、删包策略或TX无关质量判定。全覆盖识别率保留，所有包进入分母。反例和固定条件均值混杂诊断同时交付，未证明TX硬件成因、B210硬件因果或完整前导码有效性。
