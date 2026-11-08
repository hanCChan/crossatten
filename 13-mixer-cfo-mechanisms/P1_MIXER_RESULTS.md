# P1 LoRa Mixer 同协议结果

完成48/48个200-epoch正式fit；所有checkpoint先在源域冻结提交后，才读取dev_val/已使用confirm。

| Mixer | 参数 | dev_val均值±fit SD % | 已使用confirm均值±fit SD % |
|---|---:|---:|---:|
|A0_cnn|100154|94.253±0.996|93.163±1.226|
|A1_mha|133690|97.335±0.552|96.508±0.227|
|A1_mha_matched|141978|97.343±0.520|96.594±0.469|
|A2_transformer|200122|97.223±0.417|96.733±0.278|
|A2_transformer_matched|142330|97.400±0.411|96.869±0.394|
|A3_rfhstu|142392|97.460±0.342|96.448±0.656|
|A4_softmax_gate|142392|97.252±0.543|96.721±0.391|
|A5_silu_gate|142392|88.695±20.354|86.946±21.191|

| Mixer | 角色 | Macro-F1 % | seed SD pp | 最差RX均值 % | 最差单次RX % |
|---|---|---:|---:|---:|---:|
|A0_cnn|dev_val|94.105|0.515|82.017|79.950|
|A0_cnn|confirm|93.047|0.472|83.317|81.350|
|A1_mha|dev_val|97.326|0.308|93.667|90.500|
|A1_mha|confirm|96.464|0.134|90.867|88.300|
|A1_mha_matched|dev_val|97.334|0.213|94.050|92.350|
|A1_mha_matched|confirm|96.540|0.234|91.192|86.450|
|A2_transformer|dev_val|97.214|0.190|93.800|92.200|
|A2_transformer|confirm|96.662|0.112|92.217|88.450|
|A2_transformer_matched|dev_val|97.390|0.118|94.517|93.800|
|A2_transformer_matched|confirm|96.801|0.254|92.817|90.150|
|A3_rfhstu|dev_val|97.457|0.269|95.500|94.800|
|A3_rfhstu|confirm|96.382|0.641|90.775|86.300|
|A4_softmax_gate|dev_val|97.241|0.209|93.550|93.050|
|A4_softmax_gate|confirm|96.649|0.325|92.058|89.150|
|A5_silu_gate|dev_val|88.244|14.377|77.650|37.200|
|A5_silu_gate|confirm|86.399|14.850|76.842|24.050|

配对方向是RF-HSTU减去对照；完整逐折/seed、逐RX、F1和接收机聚类区间见CSV。
参数匹配Transformer采用FFN32瓶颈，不能替代自然FFN256对照；匹配MHA内部投影80维，输入输出仍32×64。
A3/A4/A5初始完整state逐项相同，唯一结构开关为Sigmoid/Softmax/SiLU打分。A5不是完整原版HSTU复现。
多项比较未做多重校正；区间条件于已拟合种子。若普通MHA/Transformer接近或更好，应收缩HSTU特定优越性。
成本见costs.csv：不能因参数更少就宣称更快，也不能使用A100历史速度替代4060测量。
