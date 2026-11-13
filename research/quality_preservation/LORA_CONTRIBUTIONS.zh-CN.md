# LoRa稿件定位：冻结模型，收拢贡献和证据

建议题目：**Hierarchical RF Tokenization for Parameter-Efficient Cross-Receiver LoRa Identification**。

本轮不新增LoRa训练，不扩大S23已经停止的机制拆解。论文围绕“已对齐、严格带内LoRa、有限源预算下的token构造”展开；WiSig另作独立研究，不写成统一LoRa/WiFi SOTA。

## 一项核心方法贡献

在继承的严格带内五通道表示和32×64 token接口中，用四阶段层级卷积替换大核Patch构造，后接普通MHA。HStem+MHA为133,690参数，完整Patch约减少76.6%；同协议Dev 97.335%、历史Confirm 96.508%。完整Patch的95.222%与参数匹配Patch-Lite的94.648%须分列：两者回答不同问题，不混用比较对象。与Patch-Lite的Dev差值为+2.687pp。

层级卷积、CNN前端、注意力、五通道、32×64接口和严格带内处理均有继承或通用先例；具体受控配置的参数效率和同协议效果是可主张部分，不宣称首次层级tokenization或全新射频前端。普通MHA结果不借用HSTU的97.48% / 96.84%最优数字。

## 两项支撑性贡献

1. **参数化与局部支持的实证区分。** S22无新增支持的满秩往返插值相对Patch-Lite约+0.878pp，Overlap-Patch再约+0.818pp。S23将六个S1物化为普通卷积，12,000源验证包logits逐比特一致，说明训练参数化收益可在无插值的推理函数中保留。它不证明抗混叠、TX指纹保留的唯一物理机制，SGD几何解释也不是AdamW精确更新式。
2. **配对消融与适用边界。** 参数匹配、两个Mixer中的结构效果、BN/affine区分、逐RX风险和成本共同支撑可信度；不分别包装成独立新算法。BN冻结统计可折叠，但默认TF32设置下六个H0均未通过原定数值容差（最大logits差约0.01957，分类差异0）；禁用TF32的补充对照六个模型才全部通过（最大差约5.39e-5）。报告补充诊断的时序，不能删去原数值失败。

## 必须写入摘要或适用范围的限制

- S23 H0源验证clean 98.175%，循环−1/+1点约81.167%/75.933%，循环±4点平均损失78.358pp。Patch-Lite也严重敏感。不能宣称同步鲁棒，不能把人工循环移位当成真实部署同步误差。
- 当前定长IQ没有连续邻接波形或实际同步误差分布。频谱核和局部梯度支持参数化/优化相关解释，不能当TX硬件频响或跨RX硬件不变性证明。
- 历史Confirm和六台保留RX已有使用记录，不具备新盲测资格。新的外部泛化证据仍缺失。
- 已有OOB论文中的严格带内表示、CNN前端、数据和token接口须在相关工作与贡献边界明确披露。当前可以整理限定范围稿件；尚不能保证独立发表新颖性或TIFS/IoT-J投稿就绪。

## 建议的正文与表格

正文顺序：任务和继承范围→RF token构造→参数匹配主结果→局部支持/参数化对照→BN及推理导出→逐RX风险/实际成本→时间敏感性和独立验证限制。

主表区分完整Patch、Patch-Lite、HStem及Mixer；机制表区分无新增支持S1、Overlap和层级结构；导出表并列严格精度/默认精度条件；风险表保留正常与时移结果。所有Dev、历史Confirm、source-validation数字列清楚数据角色，不能跨角色相减。

## 可追溯原始证据

- [S15结果](../../../s15-hstem-attribution/15-hstem-attribution/RESULTS.zh-CN.md)
- [S22结果](../../../s22-mechanism-closure/22-mechanism-closure/RESULTS.zh-CN.md)
- [S23研究解释](../../../s23-mechanism-evidence/23-hstem-mechanism-evidence/RESEARCH_INTERPRETATION.zh-CN.md)
- [S23论文重合审查](../../../s23-mechanism-evidence/23-hstem-mechanism-evidence/PAPER_OVERLAP_REVIEW.zh-CN.md)

上述相对路径以本机worktrees布局定位，历史实验文件保持原样；本页是投稿组织建议，不是新增实验结果。
