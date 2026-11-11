# F2 受限执行器工程附录：工程通过，科学执行关闭

本附录承接 `21212557782a96382446be0181c648bf7b5678b8` 的[设计预注册](../spectral_scale_review/PREREGISTRATION.zh-CN.md)。模型、kernel 3/33、3,201 参数、alpha 初值 0、优化配方、12 次预算及全部晋级门槛均保持原样。未修改原设计目录或两个已 STOP 阶段的任何封存文件。

本轮用户要求“先补齐执行器工程证据，得到明确执行授权后完成原定 12 次侧路拟合”。据此，本轮开放十二个登记的源模型 checkpoint 恢复与合成验收；没有开放源 IQ、源元数据数组、真实预测解码或科学拟合。`EXECUTION_AUTHORIZATION.json` 不存在，科学入口在数组读取前拒绝执行。

## 已完成的验收

| 项目 | 证据与结论 |
|---|---|
| 六个 F0 和六个 E0 恢复 | 从 P0/C 阶段封存记录登记身份，十二份原件与本地副本 SHA-256 全部一致；未替换 seed 或端点。见 [恢复记录](REFERENCE_RECOVERY.json)。 |
| 真实权重的 CUDA 合成前向 | 每个模型使用同一批随机 128 包；原实现与冻结 F0 最大 logit 绝对差 ≤1e-6 且逐包 argmax 一致；六个 P0 对应的 F1/F2 在 alpha=0 时与冻结 F0 位级相同，输出 tokens 为 32×64。 |
| CPU 执行器测试 | 8 项通过；包括授权先于 IO、固定预算、checkpoint 最早同值规则、身份/数值漂移拒绝、同一训练内核、EMA 联合门及辅助端点不得取代主端点。最后另复查 2 项无优化器测试，拒绝非有限的封存 logits。 |
| CUDA 完整循环 | 随机冻结骨干和随机输入，F1/F2 各 40 epoch、120 更新，共 80 epoch / 240 更新；31 项运行验收全部通过。它们是两条合成工程轨迹，不占用或冒充 12 次科学拟合。 |
| 冻结与配对 | 每 epoch 骨干全部参数和 BN buffer 的语义 hash 不变；optimizer 仅含侧路 3,201 参数；没有骨干梯度。两臂共同侧路初始化、初始 RNG/scheduler 与 40 个 epoch 的实际包顺序相同。 |
| 梯度与 alpha | 240 次更新逐步记录有限梯度、alpha 更新前后值和参数移动。首步侧路任务梯度为 0，alpha 任务梯度非零，后续侧路任务梯度出现；coupled Adam weight decay 在首步仍使侧路权重移动，已与任务梯度分开记录。 |
| 回退与选模 | 每 epoch 另拷贝模型将 alpha 归零，在合成探针上位级回退 F0；验证/探针不推进训练 RNG。epoch0 与 1–40 按 RX 等权 CE 严格取最小，同值最早；epoch0 不推进 Plateau。epoch0、selected、epoch10、final40 的保存预测逐项精确回放。 |
| 派生证据复核 | [589 项工程复核](ENGINEERING_QA.json)通过；直接核验 80 份 epoch 全状态、2 份初始状态、240 条梯度、实际顺序、checkpoint hash，以及旧阶段全部封存文件。复核器未做优化器更新。 |

CUDA 环境：PyTorch 2.5.0+cu124、NumPy 1.26.4、CUDA 12.4、cuDNN 90100、RTX 4060 Laptop GPU；FP32，关闭 TF32，确定性执行。合成验收的耗时只描述该小型工程 fixture，不能外推真实训练或模型推断速度。十二份恢复的源模型仅用于前向核验，合成优化器使用随机构造的骨干。

本轮合计 **科学拟合 0、源 IQ 读取 0、源预测解码 0**。CPU 内核测试另有 12 次合成更新；CUDA 合成更新为 240。二者分别登记，不计入未来 60,960 次科学更新。

## 受限执行与尚未完成的真实源验收

`executor.py` 是科学入口；`kernel.py` 是 CPU/CUDA 工程验收与未来科学拟合共用的内核。授权必须明确对应原设计 SHA、本附录注册 SHA 和最多 12 次预算，并先提交记录。授权文件不能由工程验收自动生成。

授权后的固定顺序如下：

1. `preflight`：只读原合法 fit18 / D1–D2 的 checkpoint_val 包；恢复十二个参照，核对 keys、pool_ids、TX/RX/day、logits ≤1e-6 和逐包 argmax；并要求每个 P0 对应的 F1/F2 在 alpha=0 时与冻结 F0 位级一致。通过后提交 `SOURCE_PREFLIGHT.json`，才允许训练。**当前这一真实源数据预测一致性尚未验证。**失败保留原尝试，停止并报告工程偏差，不放宽阈值或换参照。
2. `train-all`：严格执行原 V0/V1 × S0/S1/S2 × F1/F2 十二次，各 40 epoch / 5,080 更新。每个科学尝试只可创建一次；失败保留状态并停止，不自动续跑、补 seed 或解冻骨干。记录全 alpha/梯度、每 epoch 全状态与实际样本顺序、训练/验证/回退及总耗时。
3. 全部十二次完成，复核预算、原 P0 身份、冻结/BN 和配对顺序，生成并提交 `TRAINING_SEAL.json` 与 `RUN_EVIDENCE.json`。**此前不读取 source_selection IQ。**
4. `evaluate`：只在全部提交封存后一次性评价所有预定端点，同时回放 F0/E0 与历史 source_selection 预测。任一身份或预测一致性失败就阻断解释，不事后更换基线。保存 36 份侧路端点预测与 12 份参照预测，共 48 份。
5. `analyze.py`：仅 selected 决定 F2 是否同时通过 F1/F0/E0 三组联合门；epoch10/final40 完整报告但不能晋级。bootstrap 固定 12 个已观察 RX、先平均三流、10,000 次、seed24501。不扫描 alpha、分支、epoch、kernel、Loss 或额外 seed。

以上命令均需在本工程注册已提交、另有明确用户授权后运行。正式 target8、历史 held6、日期3–4及新 RX 保持关闭。原 source-CV 是开发研究，不能称为独立确认或正式 CRD 突破。

## 可复核材料

- [执行注册](EXECUTION_REGISTRATION.json)：冻结工程代码、依赖、环境与证据 SHA；它本身不授权科学拟合。
- [完整 CUDA 验收](CUDA_ACCEPTANCE.json)、[CPU 验收](CPU_ACCEPTANCE.json)、[工程访问日志](ENGINEERING_ACCESS_LOG.jsonl)。
- [逐步合成轨迹归档](SYNTHETIC_TRACES.zip)：240 条更新的 alpha/梯度、80 个 epoch 的历史与实际顺序；归档不含源权重或 IQ。
- [执行代码字节归档](EXECUTION_CODE_BYTES.zip)：保留实际运行的旧依赖和新内核字节，避免 Windows 换行转换影响复核。
- [本地合成文件清单](LOCAL_SYNTHETIC_MANIFEST.json)：全状态和随机模型 checkpoint 留在忽略的 artifacts 目录，可按 SHA 校验；十二份源 checkpoint 副本同样留在本地，不上传 Git。

工程通过只支持执行器遵守登记协议；真实源数据一致性仍是授权后的首道门。F2 的成败解释限于冻结骨干、3,201 参数和登记配方。F1/F2 等参数与最大名义跨度并不等于相同拓扑、非线性或计算量，也没有新增机制或原创性的性能证据。
