# S23 数学等价与同步敏感性结果

本轮新增训练、优化步、架构搜索和保留RX/WiSig目标样本访问均为0。全部结果来自36个冻结模型和合法源验证包。

|模型|条件|Accuracy %|Macro-F1 %|CE|margin|
|---|---|---:|---:|---:|---:|
|H0|circular_+1|75.933|75.333|0.8462|3.002|
|H0|circular_+4|17.775|8.405|6.7154|-5.874|
|H0|circular_+4_fft_locked|17.775|8.405|6.7154|-5.874|
|H0|circular_-1|81.167|81.269|0.6584|3.621|
|H0|circular_-4|21.858|14.160|4.8232|-3.819|
|H0|circular_-4_fft_locked|21.850|14.155|4.8232|-3.819|
|H0|clean|98.175|98.174|0.0590|6.586|
|H0|zero_+1|83.633|83.433|0.5889|3.764|
|H0|zero_+4|17.383|7.631|6.4674|-5.593|
|H0|zero_-1|81.008|81.112|0.6650|3.585|
|H0|zero_-4|21.867|14.476|4.7158|-3.796|
|P0|circular_+1|70.908|69.208|1.2679|2.200|
|P0|circular_+4|16.983|8.101|7.0505|-6.129|
|P0|circular_+4_fft_locked|16.983|8.101|7.0505|-6.129|
|P0|circular_-1|79.417|78.967|0.6536|3.503|
|P0|circular_-4|22.567|13.352|5.2836|-4.405|
|P0|circular_-4_fft_locked|22.567|13.352|5.2836|-4.405|
|P0|clean|95.925|95.909|0.1313|5.729|
|P0|zero_+1|74.450|73.414|1.0339|2.644|
|P0|zero_+4|17.233|7.694|6.8956|-6.030|
|P0|zero_-1|79.767|79.299|0.6552|3.457|
|P0|zero_-4|21.092|12.369|5.4462|-4.672|

全部预注册零填充、循环与FFT幅度锁定条件同时报告。FFT锁定是内部视图诊断，不代表物理信号。时移实验不是独立确认，不能把循环移位等同真实空口同步误差。

数值等价验证：{"time": "2026-11-12T22:26:11+08:00", "models_recovered": 36, "equivalence_checks": 12, "equivalence_all_pass": false, "prediction_disagreements": 0, "new_fits": 0, "optimizer_constructions": 0, "all_reference_files_unchanged": true}

预注册描述信号：{"H0": {"mean_circular4_drop_pp": 78.35833333333333, "mean_zero4_drop_pp": 78.55, "circular_drop_positive_pairs_by_sign": [6, 6], "boundary_dominant": false, "zero_boundary_alone_insufficient": true}, "P0": {"mean_circular4_drop_pp": 76.15, "mean_zero4_drop_pp": 76.76249999999999, "circular_drop_positive_pairs_by_sign": [6, 6], "boundary_dominant": false, "zero_boundary_alone_insufficient": true}}

FFT输入明确是幅度，不是显式频谱相位；时域atan2相位与频域相位须区分。有效核频谱坐标是前端特征索引，不是TX硬件频响。T的可逆性证明表达空间未缩小；轨迹、局部梯度和频谱仅提供优化相关线索，SGD预条件公式不是AdamW精确更新式。

当前数据只包含固定8192点窗口。SOURCE_METADATA.json给出已授权源HDF5的schema；缺少邻接波形或同步误差分布时，不能证明人工时移具备部署物理有效性。论文1重合审查及最终研究建议另列，停止于S23，不自动追加任何拟合。
