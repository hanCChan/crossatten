# S24 计算预算修订：54项保留、ASK Strict-IB六项暂缓

**当前恢复入口**：使用`./24-lora-external-audit/phase2/budget_update/resume_001.ps1`。54模型推理完成后的原汇总发生NumPy整数JSON序列化错误，追加修复见[TECHNICAL_FIX_001.json](TECHNICAL_FIX_001.json)。原冻结训练、统计、控制器和launch.ps1均未改写；新入口只为原汇总输出转换标量类型，继续接管同一Native worker。`CONTROLLER_FAILURE.json`及attempt1日志是保留的历史失败，当前状态见LIVE_STATUS与RECOVERY_001.json。不要与旧入口同时启动。

本修订来自2026-11-18用户指示：保留当前正在运行的完整ASKNet；继续Native两折×三种子；六次Strict-IB ASK不再自动启动；优先统一评估已完成CE/DANN/FAN/IB的48个模型。依据是计算预算和研究优先级，变更前未读取本轮Dev。此前已查看的源验证成绩和历史开发信息明确保留暴露记录，不称完全未知配置。

原pilot_registry.json、科学训练脚本、所有模型超参/种子/源验证选择规则及H0六个冻结模型均保持原样。AMENDMENT.json作为追加预算政策：原60项中保留54项，整体暂缓ASK Strict-IB的6项。DEFERRED.json将其标为未运行/NA，而非失败或0分；后续需要用户明确授权才恢复。没有按某个Dev分数选择性停止种子，没有增加HStem拟合或新结构搜索。

控制器交接仅停止旧的排队进程，保留现有runner/GPU子进程自然完成，原stdout/history/checkpoint不重建、不丢弃。HANDOFF.json记录进程身份和交接状态。新controller.py接管其完成状态，再通过train_next.py启动原配置的剩余Native任务；旧训练实现仍按原SHA核查。控制器运行期间持有临时Windows防空闲睡眠请求，结束后释放。

## 先封存、后分批统一报告

1. baseline48：CE、DANN、FAN、IB的Native/Strict-IB全部48项与H0六个冻结权重，在第一次Dev推断前整体核验、封存并提交seal_baseline48.json。
2. 先用CPU并行评估上述全部模型，避免当前ASK占用8GB GPU时显存竞争。网络FP32，既有CIS/投影FP64保留；eval/no_grad、无噪声、无目标适应/多包融合。纯推断不参与任何选模。CPU时间不冒充GPU部署延迟。
3. ASK Native六项全部完成后整体封存seal_native_ask6.json和保留矩阵seal_retained54.json，使用空闲GPU统一评估六个ASK模型，再补齐Native主表。没有仅报告第一个或最佳种子的ASK结果。
4. Strict-IB主表包含已有完整基线，但ASK行明确为DEFERRED_BY_USER_BUDGET/NA。不能宣称已完成ASK同带内信息预算比较。

全部Dev访问通过新的已提交预算/群组封存门控：验证完整成员、checkpoint/config/metrics SHA、固定packet rows和label mapping、每RX2000包与原始IQ hash；仅允许该fold的五台Dev RX。源RX、未知RX、confirm/blind以及未封存/暂缓任务在波形IO前拒绝。访问记录保存在本目录access.jsonl。原始科学训练守卫不被绕用作Dev入口。

## 统计与成本

完整报告两折×三seed的逐RX Accuracy/Macro-F1、fit SD、seed SD、折内seed波动、最差RX种子均值。配对包括H0与各基线、DANN−CE和IB−FAN。唯一RX先平均种子再聚类bootstrap10000次（seed24016），报告条件95%区间，不把包/seed/fold当新增独立设备，不作多重比较校正。所有注册对照均报告，不在Dev上挑模型/配方。FAN是IB Stage1复用，不增加拟合。

成本在Native六项完成后同一RTX4060、FP32、原始IQ到logits完整推理链上测量；保留ASK每次QR。MACs只报告计数器覆盖部分，明确排除未覆盖FFT/QR/复数操作。新结果是长期开发RX上的报告，不恢复独立盲测资格，也不消除原有±4样点同步敏感性证据。

## 产物和恢复

AMENDMENT.json、TESTS.json、seal_*.json是前置证据；RESULTS.zh-CN.md、RESULTS.json、PER_RX.csv是统一评估输出；成本见COSTS.json。逐包预测保存在本地忽略目录predictions/。运行进度由phase2/LIVE_STATUS.json以及EVALUATION_*.json给出。

controller.py支持接收HANDOFF.json中现有worker；禁止同时运行旧supervisor.py。CPU报告可按`evaluate.py --cohort baseline48 --device cpu`恢复，已完成模型按seal SHA复用。Native ASK报告仅在六项全部封存后运行`evaluate.py --cohort native_ask6 --device cuda`。任何失败保留日志，不通过调模型或改变RX修复结果。

Windows工程单测发现默认numpy int32标签不能直接传给Torch交叉熵；正式评估前已在metrics入口显式转int64，并用手算10类混淆矩阵与样本顺序不变性检查，未读取任何Dev样本来调整该实现。
