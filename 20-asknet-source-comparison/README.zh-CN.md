# S20：HStem+MHA 与 ASKNet 的同数据源域正面对照

本阶段不提出新模块。冻结S18 `933d4a6`、S19 `9641b7b`及所有历史STOP；只在`20-asknet-source-comparison/`写入。主要问题：在**完全相同S19数据和源RX划分**下，官方ASKNet整体模型/配方能否稳定优于冻结HStem+MHA？不能用历史S10 CRD数字与本轮源域数字作排名。

## 数据与既有模型

沿用S19 ManyRx四fold，官方每fold24个source RX，first6留出、余18个fit，日期仅1–2，10TX、256点、non-equalized、first100/TX/RX/day，逐包功率归一化。固定原训练70%/源验证30%索引，split16001。训练输入按原缓存直接索引，不调用官方全数据loader，不重新normalize或改变样本顺序。原缺失cell保留，不插补或删除TX/RX。

B0复用S19四个固定HStem+MHA checkpoint和已保存source/held logits；SHA必须一致，推理结果逐项复算相同，禁止修改或续训B0。S10的12个ManyRx ASKNet checkpoint均在全部24个源RX训练，包含本次6个held-source且split/seed不同，因此不能严格复用；不加载它们的权重。本轮新增**最多四次正式训练**，每fold一个ASKNet，不挑seed、不追加消融训练。

数据与S19一样，不等于所有配方相同；held-source身份曾在历史阶段被使用，不能称历史上首次接触的独立RX。本轮新ASKNet所有checkpoint及配置/指标先封存并commit，再一次性读held-source。不访问该fold真实目标RX，不读任何日期3–4、完整原始pickle、Shen或ManySig。

## 官方ASKNet路径与不可归因的差异

固定官方commit `f00846d68425315b68ae69b8c6b712e08559c2fe`，仅下载并验证main、PatchNet和loader三个Git blob的字节SHA，不转载作者源代码。来源见[UPSTREAM.json](UPSTREAM.json)。执行官方`build_model`、`setup_seed`、`train_epoch`、`evaluate_epoch`、`train_and_evaluate`；禁止调用官方main/prepare_dataset或直接load_single_dataset。包装仅记录训练顺序、验证输出和曲线，不更改loss、梯度、优化器、LR、earlystop或checkpoint规则。

| 项 | 冻结HStem+MHA B0 | 本轮ASKNet |
|---|---|---|
| 模型输入 | IQ→原双层RMS→FFT五通道→HStem→2层普通MHA | 同IQ缓存→原复FFT/LSG→127模态unitary Koopman→IFFT/总能量归一化→overlap Patch/MLP |
| 频带/尺度 | 完整WiSig谱；32×64 tokens | 官方候选k=[8,16,24,32,48,64]；最大127bins；Patch64/stride32→7×128 tokens |
| seed | macro18101，对应原各fold data_seed | 与对应B0相同有效data_seed，官方setup_seed；不同架构不声称相同随机流 |
| 训练/验证样本 | 原70/30 split16001 | 完全相同数组、索引、RX、date、TX |
| 优化器 | Adam1e-3/wd0/batch128 | 相同基础数值，官方CPU DataLoader/global RNG |
| LR/earlystop | Plateau(.1,5,min1e-6)/10 | 同官方规则 |
| 最大轮数 | 40 | **官方200** |
| checkpoint/LR依据 | RX等权源验证CE | **官方包等权源验证CE** |
| 参数/精度 | 82,426实参数，FP32 | 100,240参数元素，其中16,129复数；116,369实自由度，FP32 |

因此结果是**同数据/同评估的整体模型与官方配方比较**，不是严格隔离架构或Koopman的消融。保留官方最大200轮和checkpoint规则，避免为比较任意削弱ASKNet。每epoch另记录RX等权验证CE的最佳轮数作选择规则审计，但不另存/测试替代checkpoint、不依据held改变选择。

只使用已有默认超参：patch64、embed128、MLP ratio2、Gumbel tau1/hard、complex QR/SKO、dropout0、无grad clip。复数谱处理与WiSig Fs25MHz/256点沿用原方法；不使用Shen的±62.5kHz裁剪。

## 预先固定的结果分类

主要比较ASK−B0的四fold RX等权Accuracy差、F1、CE、margin、最差配对RX，以及按唯一RX身份聚类的条件区间。固定bootstrap5000次、seed20101；同RX跨fold先平均，再对唯一身份等权抽样。本pilot只有12个唯一held-source RX，fold1–3的6个held RX重复，不能当24个独立接收机。fold均值和唯一RX均值加权不同，分别报告。

将“明显优势”操作性定义为：fold均值≥+1pp、至少3/4 folds正向、最差配对RX不低于−2pp、唯一RX条件95%CI下界>0。对ASK和HStem两个方向对称检查。若ASK满足，状态`ASKNET_CLEAR_SOURCE_ADVANTAGE`；若HStem满足则`HSTEM_CLEAR_SOURCE_ADVANTAGE`。

只有|fold均值|<1pp且唯一RX95%区间完全落在[−1,+1]pp内，才标记`PRACTICALLY_CLOSE_ON_THIS_SOURCE_PROTOCOL`，它是描述性接近，不是总体等价检验。其余为`INCONCLUSIVE_SOURCE_COMPARISON`。**CI跨零不等于方法等价，更不能据此证明source pilot失效**。单seed、RX支持和模型共享依赖仍限制区间解释。

## 固定出口与归因审计

无论胜负均输出四个ASK checkpoint的截止k/概率、复数K的unitarity误差/离identity距离/off-diagonal能量、Patch与HStem token/感受野/归一化差异、逐RX/TX错误和成本。这些是代码及已学习权重的描述，不通过改变模型参数/替换算子来选优。

只有ASK达到明显优势，才形成后续“频带/复数交互/Patch表征/训练配方”分开控制的候选问题；S20不训练新结构、不将全部差距归功Koopman。如果没有清晰差距，审视source覆盖、分组重复、样本量与seed不确定性，不能立即扩大模型或增加损失。本轮不自动开始S21、更多seed/分组或目标CRD测试。

## 科研证据

先完成合成工程与GPU前反向检查、历史复用审计，再冻结代码/协议并commit注册。正式训练上限4；失败attempt保留，不自动重复训练。全量logits只留本地，Git保留配置、训练曲线/顺序SHA、官方k trace、日志、checkpoint SHA、配对指标/CI、访问日志及失败记录。独立QA从logits重算Accuracy/Macro-F1/CE/margin/最差RX/TX和所有门槛。

成本在同机GPU、相同随机IQ batch1/128上测量两个**原始推理路径**，不缓存/消除ASKNet每次forward的QR，不改变截止或精度。参数同时报告元素与实自由度。LoRa、目标CRD和SOTA比较均不在本轮证据范围内。
