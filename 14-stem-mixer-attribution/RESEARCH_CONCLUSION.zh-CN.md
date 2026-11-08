# 最终研究判断与贡献边界

完整对照见ATTENTION_COMPARISON_S13.md、STEM_MIXER_RESULTS.md、costs.csv和CFO_MECHANISM_REPORT.md。
所有判断条件于这套LoRa/WiSig协议、有限接收机和既有种子；未做多重比较校正。

## HSTU与普通注意力
RF-HSTU未满足事前登记的独立优势门槛；不追加G1/G2，不预设HSTU更优。
自然MHA/Transformer、参数匹配控制和同门控换打分全部保留。精度方向、最差RX与参数/延迟要分别解读，不能从其中一个数概括全面优越。

## 分层RF Token与交互
HStem在两种Mixer上均有正向条件区间证据。
交互差值不支持确定的正向互补性；不能只凭均值说有协同。
Patch/HStem容量和感受野也不同，未做Stem参数匹配；不能宣称已唯一分离层级结构成因。

## CFO路线
沿用S13决策：{'decision_p2.json': 'STOP_SOURCE_GATE'}。
分类器级分组和异常诊断不重新选择候选、不删除样本、不修改门槛。未通过源域确认就保持STOP；即使源域确认通过，也只构成冻结独立CRD的前提，本阶段没有新增真实WiSig CRD证据。

## 创新边界
RF-HSTU承接第一篇；本轮训练沿用CE、AdamW和预热/余弦，没有新训练机制。第二篇的新增价值要由受控HStem效果和结构交互证明，不能把已有HSTU首次引入重新作为贡献。
LoRa与WiSig前端不同，不声称一个跨无线体制统一最优模型。浅层无标签适配仅有可行性审计，训练预算0，不能报告适配优势或与零样本DG混排。
审计：没有Shen blind访问、没有WiSig目标RX/date3–4访问，没有修改S00–S13历史。
