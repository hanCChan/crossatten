# S24：LoRa 外部方法对照，第一阶段交付

**当前执行入口（2026-11-18更新）**：第一阶段历史快照如下；第二阶段已启动真实训练。最新用户预算修订见 [phase2/budget_update/README.zh-CN.md](phase2/budget_update/README.zh-CN.md)：保留54项，ASK Native六项继续、Strict-IB六项延期；48个已完成基线与冻结H0先整体封存、统一报告。原60项草案和本页的fit=0描述不代表当前执行状态。实时进度见 [phase2/LIVE_STATUS.json](phase2/LIVE_STATUS.json)。

2026-11-14；独立分支 `exp/s24-lora-external-audit`，基线 `e041f09`。
执行范围来自用户附件：**第一阶段只做源码审计和数据适配，不启动训练**。
本阶段 fit=0、optimizer step=0。没有新增准确率结果，没有改写 S15/S21/S22/S23 或旧论文。

已完成：

- 在线核实并锁定 5 个作者仓库、29 个文件的 commit、Git blob SHA1 和字节 SHA256，见 [UPSTREAM.json](UPSTREAM.json)。第三方原文仅存本地忽略目录，可按固定 URL 恢复。
- 对 10 台合法源 RX 的 20,000 个固定包建立 [PACKETS.csv](PACKETS.csv)。每折 8,000 训练包、2,000 源验证包；逐包 ID、原始行号、TX 映射、源域编号、波形 SHA 均可核验。
- 逐方法原始前处理接口在真实源包上运行；19 项权限/数值/梯度工程检查通过，见 [FRONTEND_AUDIT.json](FRONTEND_AUDIT.json)。DANN 已有候选 PyTorch 移植与 GPU 前反向检查，但尚未做 TensorFlow 权重迁移等价核验。
- 10 台源 RX 的行号/IQ 哈希与 S13 历史缓存相同；6 个冻结 HStem+PlainMHA 模型及 config/metrics/history SHA 与 S23 登记相同，见 [HISTORY_AUDIT.json](HISTORY_AUDIT.json)。
- 仅打开每一折自身的 source 文件的 `label`、选定 `data` 行；没有以评估角色加载数据，没有读取 confirm/blind 波形或 CFO/SNR 字段。见 [ACCESS_AUDIT.json](ACCESS_AUDIT.json)。

审计发现了会影响正式复现的实质问题，**不能宣布所有训练前检查通过，也没有冻结或启动拟合计划**：

| 优先级 | 方法 | 当前结论 |
|---|---|---|
| P0 | Shen TMC2024 CIS+RA/DANN | 数据和候选移植已接通；官方验证 RX 标签切片有误；需完成跨框架等价核验并冻结修复与选模规则 |
| P0 | ASKNet-LoRa | 原版 LoRa 前处理和模型定义已核实；默认模型为 74,153,872 参数元素、141,246,353 实自由度，完整 8191 模态 QR 的本机可运行性尚未确认 |
| P0 | IB-RFF | 公开训练入口只有 WiSig，缺 LoRa 完整训练配置；历史 Shen 8192 适配可以参考，但不能称官方 LoRa 复现 |
| P1 | Shen TIFS2022 | 官方 triplet+CIS+kNN 与原论文的辅助 TX/登记设定需改成源域闭集适配；不能用另一个 CE-CNN 冒充 |
| 候选 | Shen TIFS2023 | 官方多 SF/单 RX Transformer；可运行前处理，但不是原生跨 RX 方案 |
| 候选 | Ma TIFS2025、RICAI gMLP | 本次没有核实到可用的作者实现；不以近似网络填表 |

逐方法算法、权限和阻塞条件见 [METHOD_AUDIT.zh-CN.md](METHOD_AUDIT.zh-CN.md)；前处理数学定义见 [PREPROCESSING.zh-CN.md](PREPROCESSING.zh-CN.md)；完整配对计划草案见 [DRAFT_PLAN.json](DRAFT_PLAN.json)。
两张后续论文主表与历史结果单列在 [COMPARISON.zh-CN.md](COMPARISON.zh-CN.md)。

复查命令（在本 worktree 根目录的 PowerShell 执行）：

```powershell
$s24Python = 'D:/crossatten/.venv/Scripts/python.exe'
& $s24Python 24-lora-external-audit/s24_upstream.py --restore
& $s24Python 24-lora-external-audit/s24_data_audit.py --raw-dir D:/crossatten/stage_1/data/raw
& $s24Python 24-lora-external-audit/s24_history_audit.py
& $s24Python 24-lora-external-audit/s24_verify.py
& $s24Python 24-lora-external-audit/s24_package.py --check
```

`s24_history_audit.py` 当前使用本机历史工件位置；跨机器需显式恢复同 SHA 工件并调整路径。其余脚本不依赖历史权重。
本目录无训练启动器；复查命令不会运行第三方 main、训练、目标 RX 适应或评估。
工程检查中的随机初始化输出不作为识别分数；7 模态 Koopman 检查只是算子梯度测试，没有替代默认 ASKNet。

统计边界：这批 RX 经过长期开发，新增比较不会构成新的独立盲测；confirm 仅可作为历史已曝光报告角色。两折中的同一 RX 可在另一折担任源域，这是原协议预定的交叉划分，必须逐折隔离；统计时不能当作更多独立设备。
