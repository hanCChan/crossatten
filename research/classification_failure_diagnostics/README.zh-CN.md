# 冻结 HStem / ASKNet 分类功能失效诊断

基于 `7ae38a7`，研究分支 `research/classification-failure-diagnostics`。

本阶段只重分析合法 source 的封存预测与标签/索引，不打开 IQ、checkpoint 或特征缓存，不执行模型 forward 或训练；目标 CRD、日期 3–4、ManySig/Shen 科研数据均不访问。历史 STOP 和原代码保持原样。

分析在打开预测数组前冻结于 [G0_PROTOCOL.json](G0_PROTOCOL.json) 和 [INPUT_REGISTRY.json](INPUT_REGISTRY.json)。这些是探索性诊断登记，不是新算法的正式效力预注册。已查看的 held-source 组不称为独立确认。

入口为 `diagnose.py`；仅接受登记的 16 个预测文件与 20 个标签/索引文件。7 项合成工程检查通过，见 [ENGINEERING_TESTS.json](ENGINEERING_TESTS.json)。结果与三个竞争解释将在本分支的后续归档中给出。
