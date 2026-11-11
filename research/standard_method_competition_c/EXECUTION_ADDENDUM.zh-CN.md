# C 执行工程附录：执行既有24作业，不改变科学设计

日期2026-11-11。原科学协议与全部源成员/方法/seed/weight_decay/选模/预算固定于 `bebf6bb`，原文件不修改。用户本次明确要求补齐执行器并执行C，因而原 `scope_this_turn=PREREGISTRATION_ONLY_NO_TRAINING` 仅描述上次登记任务，不是对本次已授权执行的永久禁止。

本附录只增加 [engine.py](engine.py)、[analyze.py](analyze.py)、工程合成验收、执行登记和日志。C仍为合法fit18内部开发，不打开historical held6、target8、日期3–4或新确认候选，不报告为正式CRD突破。LoRa结构和历史STOP不改。

## 先验验收与执行次序

34项纯合成CPU检查通过（原12方法/划分、9整环路、13完整分析）：覆盖两梯度族、40epoch固定预算、全部端点、EMA及BN状态保护、拒绝未封存评分、概率范围/缺失组、配对bootstrap和完整晋级判断。整环路使用合成模型/输入，与科学数据完全隔离。

另对原82426参数HStem+MHA使用随机生成的(2,256)输入作3条CUDA工程轨迹：ERM_EMA、相同设置重放、RX_GDRO；每条40轮、每轮2次更新，共240次合成更新。GPU ERM学生/EMA完整逐轮重放逐位相同、跨族初始化/相对顺序相同、全部readout推理重放精确，校准前后训练/RNG状态保持相同。见 [ENGINEERING_GPU_QA.json](ENGINEERING_GPU_QA.json)。这是工程合成数据，不是额外科学拟合、训练结果或性能选择；科学预算执行仍从0开始。

执行顺序：提交本附录与代码/验收 → `preflight` 仅核对登记标签/索引和角色原始SHA → 提交 `SOURCE_PREFLIGHT.json` → 按 `JOBS.json` 顺序执行24个作业 → 核对共同初始化/逐轮序列/标签/校准保护与全部预算 → 提交全作业封存 → 一次性评分全部108份source_selection预测 → 按原规则选wd、比较与晋级 → 固定模型延迟测量/归档。任何失败保留并停止，不自动重试或加seed。

## 访问与训练隔离

`PRE_REGISTRATION.json`、原 `SOURCE_PLAN.json` 与 `JOBS.json` 保持不变。新登记同时固定原科学文件哈希、原67个代码依赖和新增执行代码哈希。Git提交状态批量校验，运行时文件原始字节SHA逐一验证；PyTorch、NumPy、CUDA/cuDNN、GPU名称和本机SWA/EMA代码摘要核对。

训练时仅将 `fit_train` 和 `checkpoint_val` 的IQ行从已登记source池复制到内存。计算全部角色的标签/规范键属于既有登记元信息，不对source_selection做前向或BN。只有24个作业及全部C0/C1/C2 selected/epoch10/final40参数、BN和完整轨迹封存并提交，`evaluate` 才允许复制各视图source_selection行并评分。模型只接收IQ，RX标签仅用于训练分组及派生评价。

每轮保存完整学生/optimizer/scheduler/RNG、order generator、EMA影子或GroupDRO logq、选模状态和实际相对置换；保存每臂三个部署端点和原始checkpoint_val logits。C1保存经过该轮fit-only BN重估的完整部署状态，不在评分时再校准。每轮校准对学生/影子/全局RNG/order的保护以实际摘要复核。

两族共享初始化、优化器初态和随机流；同视图/流/wd逐轮实际规范键、相对位置和TX/day顺序一致。C0/C1本就是一个ERM学生轨迹的两个预定读出，不计为两个独立拟合。GroupDRO记录每一步q、组损失和支持，缺失batch和权重范围可复核，数值异常立即失败。

## 资源与部署测量

实际科学上限仍为24梯度fit、960epoch、121920更新；12个EMA影子最多480遍fit-only BN重估。完整fit wall time包含读取、训练、校准、验证、状态/预测保存及最后摘要；另报source读取、训练含EMA wall、EMA host调用、EMA CUDA event kernel时间、raw/EMA验证、BN校准样本/时间和显存。EMA CUDA时间是训练内子集，host调用也是子项，不能与训练含EMA wall重复加总。

只对36个selected部署读出做已登记的batch1/128延迟测量：每轮每batch100次warmup，1000/200次CUDA event测量，3轮旋转C0/C1/C2顺序。只使用对应view训练输入，包含完整RF前端，GPU resident，不含磁盘与主机拷贝。epoch10/final40仅补充科学指标，不新增第二套延迟选择。

## 分析冻结与科研出口

评分清单须精确包含24作业×各臂×3端点的108份输出，分析前逐文件核对全封存及哈希。先在每RX内平均3个固定流，再对12个不同source-selection RX做配对cluster bootstrap，10000次、seed24501。相同wd的对照完整呈现；各方法按selected CE选wd后，epoch10/final40沿用该主要端点wd，不再选择辅助端点的独立最优配置。

原全部晋级条件共同适用，不因某个坏例、辅助快照或普通方法的局部收益改规则。结果只能决定下一轮标准基线的地位，EMA/GroupDRO自身不是第三项创新。M0和确认资格是独立元信息任务；它们不增加C预算或授权新Loss/目标访问。C无论结果如何到此结束普通方法排除，不再追加D/E/F普通因素训练。
