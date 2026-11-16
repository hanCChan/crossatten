# S24 第二阶段：方法级验证与真实拟合

用户于2026-11-16授权推进训练，扩展Non-adv CE对照。本目录是独立第二阶段；父目录的DRAFT_PLAN与STATUS保留第一阶段历史快照，不能将其中的“授权0次”误读为当前执行状态。

三项P0结果：

- DANN：隔离环境TensorFlow2.15.1原作者模型→PyTorch，逐Conv、pool、Flatten、512维归一化特征、两个概率头、TX/RX/联合损失及其输入/参数梯度共103项比较通过。预设forward atol2e-5/rtol2e-4，gradient atol2e-5/rtol2e-3，没有放宽。详见DANN_PARITY.json。
- ASKNet：**完整8191模态、原候选k、原每次QR、batch128、FP32/complex64**在RTX4060 Laptop上完成前向、反向及一次沙盒Adam更新，峰值已分配显存4,930,779,648字节，QR/前反向计时见ASK_GATE_b128.json。不能据先前理论下界就判定不可行；没有缩小网络。
- IB-RFF：完整8192、SAF8192、patch512/stride256、31×128位置编码、FAF/DSCNN/dropout.5，共310,481参数，保留原作者CPU normalized HSIC-CCA；前反向通过。alpha .004固定两视图，来自S06原预登记候选的最小非零值；不重新搜索，也不称最佳alpha。历史开发结果已曝光，明确保留选择偏差边界。

科学矩阵为5条训练路径（DANN、同骨干Non-adv CE、FAN Stage1、IB Stage2、ASKNet）×2视图×2折×3seed＝**最多60个初始化fit**。FAN作为对照直接复用每个IB的Stage1，不另加fit。5个native/fold0/seed201完整源域Pilot就是对应预登记矩阵中的首个单元；它们在optimizer step前已有pilot_registry与代码hash冻结。只有代码/配置完全不变，且未接触Dev时才在方法级freeze中复用，避免再训练5次。不能按Pilot得分换配置。

所有Gate技术step、极小沙盒恢复测试独立计数，不混入科学fit。ASK完整训练步约7秒，batch128一轮有63个训练batch并有16个源验证batch；预计完整矩阵需长时间运行，以真实日志为准，不通过减epoch、缩小k或更换seed加速。队列优先较轻的IB与CIS方法，再执行完整ASK。

共同规则：固定PACKETS.csv原包、160/40源划分、同(fold,seed)数据顺序；DANN/CE共享完全相同encoder/TX初始权重和AWGN流。修复官方RX验证标签错位；两者均按源RX等权**TX CE**选最佳checkpoint和调度，DANN额外源RX CE仅参与训练，不参与选择。原SGD1e-3/momentum.9、batch64、500epochs上限、earlystop20、Plateau(.2,10)保留；改为epoch无放回遍历及固定一次完整源验证噪声。相对作者随机batch/动态验证噪声/停止时权重的区别已登记。

IB/FAN保留Adam1e-3/batch128/200epochs/earlystop10/Plateau(.1,5)，从本fold/seed/view的FAN最佳checkpoint开始Stage2，重新初始化Adam，禁止发布权重或跨fold初始化。保留原HSIC CPU执行，包括alpha0诊断与成本。ASK保持作者Adam/200epochs/batch128/tau5→.1/earlystop20及原CIS/ResNet。Strict先原包训练AWGN（仅DANN/CE）再统一IB投影，然后方法原处理；未增加额外原始带外信息。

`runner.py`只读该fold的source缓存并逐项验证PACKETS与波形hash。原子best.pt；每5epoch及结束保存完整模型、优化器、scheduler、各RNG、计数，异常保留独立attempt日志，恢复相同初始化与流。`INITIALIZATIONS.json`计数不超过60。方法失败停止该方法并继续其他方法；禁止自动换结构/参数/种子重训。冻结代码若变动必须技术修订，原证据保留。

`supervisor.py`完成方法Pilot后自动提交freeze，按原配置执行剩余单元，每fit提交真实配置/日志并尝试推送本分支；网络故障保留本地提交，不改变训练。Dev读取需要**全部60个模型与配置封存且global_seal.json已提交**。随后report.py统一输出两张主表、逐RX/seed指标、按唯一RX聚类的配对bootstrap条件区间；bench.py同机测原始推理链及各法前处理成本。Blind/Confirm始终禁止。H0只复用六冻结checkpoint，源模型/历史结果不修改。

当前真实进度看LIVE_STATUS.json、WORKER.json、INITIALIZATIONS.json和runs/*/progress.json；未完成单元没有识别分数，source指标不冒充Dev。推送状态看PUBLICATION.json；大权重和逐包logits留本地忽略目录。

PowerShell启动/恢复入口：`./24-lora-external-audit/phase2/launch.ps1`。已存在锁时先核对PID存活，不盲目清锁或启动第二个GPU队列。正常重启继续原单元；失败单元不会被按得分自动重启。

复核Gate：parity_gate.py/ib_gate.py/test_engineering.py；复核配置与代码：execution.validate_registry。隔离TF环境为D:/crossatten/.venv-s24-parity，现有torch环境未安装或替换TensorFlow依赖。没有启动额外HStem层级数实验。
