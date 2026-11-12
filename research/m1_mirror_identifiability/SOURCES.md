# A0 来源与可确认范围

- [WiSig 作者论文 v2](https://arxiv.org/html/2112.15363)：III-B 给出 25 Msps；IV-A 说明能量检测未使用已知 Wi-Fi 前导码；IV-B 区分直接截取256点与均衡版本的25→20→25 Msps流程。本轮只沿用未均衡版本。
- [UCLA 官方数据入口](https://cores.ee.ucla.edu/downloads/datasets/wisig/)：区分 raw continuous capture 与两种256点 Id signal，链接官方处理仓库。不是将1.4TB原始连续采集当作当前输入。
- [固定官方检测代码](https://github.com/WiSig-dataset/wisig-process-raw/blob/6530c6ed6be1d08544ac295040ed50f5babde86a/matlab_detection_screening/split_signal_fun.m)：25e6采集标记、能量/时长筛选，packet_log=x(s:f)，没有协议级精确STF起点证明。
- [固定均衡代码](https://github.com/WiSig-dataset/wisig-process-raw/blob/6530c6ed6be1d08544ac295040ed50f5babde86a/matlab_equalization/equalize_channel.m)：resample(4,5)、重新检测、CFO校正、LTF/MMSE处理、CFO加回、resample(5,4)。不运行此代码，不把它当作去除RX硬件的干预。
- [打包代码](https://github.com/WiSig-dataset/wisig-process-raw/blob/6530c6ed6be1d08544ac295040ed50f5babde86a/python_pkl_creation/py/build_dataset_multirx.py)：slice(0,256)。既有 M0 固定了这些公开文本与 Git blob SHA，本轮引用其快照并核验哈希。
- [MathWorks wlanLSTF 算法说明](https://www.mathworks.com/help/wlan/ref/wlanlstf.html)：20MHz模式的12个占用子载波、8us字段及FFT零填充过采样；[原始专利 US7039000B2](https://patents.google.com/patent/US7039000B2/en) 的式(3)公开列出 IEEE 802.11a 短训练符号向量及0.8us周期。这里只复现数值参考，不复现其专利同步算法，不宣称替代标准全文合规认证。
- 用户给出的 [Huang/Zhang/Guo 联合估计论文](https://opus.lib.uts.edu.au/bitstream/10453/43235/4/Joint%20Transmitter%20and%20Receiver%20IQ%20Imbalance%20Estimation%20in%20Presence%20of%20Carrier%20Frequency%20Offset.pdf) 本轮机构端点返回429，未取得全文，不声称逐页核验其算法。我们的级联推导与反例独立列出；不凭不可访问论文假定 WiSig 能唯一估计 TX/RX。

本地来源核查使用 M0 的 LOCAL_METADATA_REVIEW.json、原 loader 和 fit_manifest 文本。缓存属于既有 ManyRx Kaggle v1 未均衡拷贝，loader取 [tx][rx][day][0] 并做功率归一化，没有新增重采样。25Msps是继承官方定义与现有处理链的来源判断，不是从单个窄带256点波形独立测出的绝对时钟。也没有验证跨 RX 同一发射事件身份。

因此 A0 **不确认每包绝对 STF 起点**；登记一个不重定的内部候选区间并用包内留片残差检验，失败完整保留。0.9972 的历史 lag20相关性仅为背景，不用于选取新算法或放宽覆盖门。
