"""Assemble the claim/evidence matrix, bounded decisions and bilingual research notes."""
import csv
from s25_common import HERE, ROOT, digest, now, read, write


def main():
    matrix=[
      dict(claim='The complete HStem tokenizer improves this parameter-matched system.',
           definition='Four spatial stages, same input/32x64/MHA;133690 vs134266 parameters.',
           evidence=['15-hstem-attribution','22-mechanism-closure/summary.json'],
           result='H0-P0 +2.687pp;6/6 paired means positive.',status='SUPPORTED_WITHIN_PROTOCOL',
           limit='Support, normalization, groups, nonlinearities and compute remain different; not a pure hierarchy effect.'),
      dict(claim='Support-related design and training parameterization matter.',
           definition='P0/S1 share266 local feature-coordinate support;H1 has350;all134266 parameters.',
           evidence=['22-mechanism-closure/summary.json','23-hstem-mechanism-evidence/kernel_bands.csv','25-tifs-integration/kernel_geometry_join.csv'],
           result='S1-P0 +0.878pp;H1-S1 +0.818pp;effective-kernel high-energy21.808/7.692/4.043%.',
           status='INCREMENTAL_EFFECTS_AND_ASSOCIATION',limit='Smoothing/support/frequency response/optimization are not causally separated; not hardware spectra.'),
      dict(claim='Final source representations show better TX separation.',
           definition='Per-packet L2 final64D embedding;balanced scatter andall6pairs;all3taps retained.',
           evidence=['25-tifs-integration/MECHANISM_EXTENSION.json','25-tifs-integration/source_tx_rx_margins.csv'],
           result='Final Jw P0/S1/H1/H0=4.270/4.435/4.925/5.449;H0-P0 positive6/6.',
           status='SUPPORTED_SOURCE_DIAGNOSTIC',limit='Source validation was used for checkpoint selection; not new held-RX confirmation or identified cause.'),
      dict(claim='The tokenizer itself removes RX information monotonically.',
           definition='Would require consistent layer-level RX reduction, not only final classification.',
           evidence=['25-tifs-integration/mechanism_geometry.csv'],
           result='Counterexample: tokenizer absolute conditional RX scatter larger for H0 in6/6;pre-MHA meanJc H0=33.983 vsP0=41.241.',
           status='NOT_ESTABLISHED_COUNTEREXAMPLES',limit='Pooled centroid scatter is not mutual information.'),
      dict(claim='BN affects this training/design, beyond only affine parameter count.',
           definition='H2 deletes3BN;S2 keeps352 affine parameters without statistics.',
           evidence=['22-mechanism-closure/summary.json','23-hstem-mechanism-evidence/RESEARCH_INTERPRETATION.zh-CN.md'],
           result='H0-H2 +1.190pp;H0-S2 +.918pp butonly4/6 positive;S2-H2 +.272pp.',
           status='BOUNDED_NORMALIZATION_OPTIMIZATION_EVIDENCE',limit='Registered stability criterion fails forH0-S2;no independent separation of statistics/optimization.'),
      dict(claim='Competitive cross-RX identification and parameter efficiency.',
           definition='S24 same source packet permissions;Native andStrict-IB separated;all seeds retained.',
           evidence=['24-lora-external-audit/phase2/budget_update/RESULTS.json','24-lora-external-audit/phase2/budget_update/COSTS.json'],
           result='97.333%;Native H0-DANN+3.125pp;Strict H0-CE+2.632pp;about1/101 DANN parameter elements.',
           status='SUPPORTED_SYSTEM_EFFECT',limit='Native loses toShen onsomeRX;Strict mean gain concentrated atPluto;batch1 slower;historicalDev only.'),
      dict(claim='A causal CFO/SNR or hardware explanation for Pluto errors.',
           definition='Only legal source metadata diagnostics are permitted;target fields not loaded.',
           evidence=['25-tifs-integration/metadata_correlations.csv','25-tifs-integration/per_tx_rx.csv'],
           result='Small within-cell source correlations andclass-dependent target errors;no unique cause identified.',
           status='NOT_ESTABLISHED',limit='Group confounding, metadata meaning andsource/target mismatch prevent causal attribution.'),
      dict(claim='New-independent-RX, sync robustness or full authentication.',
           definition='Needs new physical RX/acquisition provenance, same enrolledTX identities andsecurity-specific evaluation.',
           evidence=['23-hstem-mechanism-evidence/PAPER_OVERLAP_REVIEW.zh-CN.md','25-tifs-integration/INDEPENDENT_DATA_STATUS.json'],
           result='User confirms no new capture;source circular4-sample shifts cause severe failures.',
           status='NOT_CLAIMED',limit='No unknownTX rejection,FAR/FRR,active spoofing orreal synchronization-offset evaluation.')
    ]
    write(HERE/'EVIDENCE_MATRIX.json',dict(time=now(),rows=matrix,new_fits=0))
    write(HERE/'INDEPENDENT_DATA_STATUS.json',dict(status='NOT_AVAILABLE_NOT_RUN',time=now(),
        authority='User explicitly selected no new acquisition; complete existing evidence only.',
        new_independent_rx_evaluations=0,new_confirm_blind_waveform_reads=0,
        future_requirements=['Same enrolled physical transmitter identities, not merely10 new labels.',
                             'New physical receiver/acquisition provenance andprior-use ledger.',
                             'Sample rate/bandwidth/window extraction/real alignment metadata.',
                             'Fresh frozen cohort/config before one-shot evaluation.'],
        existing_location_variants='Not reclassified as fresh independent devices; no waveform opened.'))
    write(HERE/'GO_STOP.json',dict(status='GO_BOUNDED_MANUSCRIPT_STOP_AUTOMATIC_NEW_FITS',time=now(),
        go=['Complete A/B/C evidence figures andEnglish draft withnegative evidence.',
            'Claim parameter-efficient design andwithin-protocol system effects.',
            'Report final source feature separation asan exploratory diagnostic.'],
        stop=['No automaticLong-Patch orASKStrict fits.','No uniqueRFhardware/receiver-invariance theorem.',
              'No independent newRX orcomplete authentication claim.','No selective seed/RX/layer/projection removal.'],
        long_patch=read(HERE/'LONG_PATCH_AUDIT.json')['decision'],new_fit_budget=0,
        remaining_scientific_risks=['NovelRF-specific mechanism not uniquelyidentified.','Independent acquisition unavailable.']))
    overlap=[
      ['TrueIB andfiveviews','Inherited fromcompanion','Notanew contribution'],
      ['Kernel7/kernel5 localCNN','Inherited fromcompanion','Reuse explicitlycited'],
      ['32x64token geometry','Shared interface','Not new'],
      ['Four-stage tokenizer withordinaryMHA','Current controlled design','Specificdesign/evidence,notconvolutional tokenization first'],
      ['Source160/40,twofold3seed budget','Current protocol','Different fromcompanion;cross-paperaccuracy subtractionnot causal'],
      ['Same20RX/10TXcorpus','Shared data origin','Oldsixblind notfresh independent'],
      ['S21-S23mechanism controls','Current archived evidence','Do notrename asnewS25 experiments']
    ]
    write(HERE/'OOB_OVERLAP.json',dict(status='AUDITED',source_pdf='refs/OOB_IOTJ.pdf',
        source_pdf_sha256=digest(ROOT/'refs/OOB_IOTJ.pdf'),
        prior_review_sha256=digest(ROOT/'23-hstem-mechanism-evidence/PAPER_OVERLAP_REVIEW.zh-CN.md'),
        columns=['component','relationship','claim_boundary'],rows=overlap))
    write(HERE/'LITERATURE_AUDIT.json',dict(time=now(),status='AUDIT_COMPLETE_NO_NEW_BASELINE_SCORE',
      ma2025=dict(title='Towards Channel-Robust and Receiver-Independent Radio Frequency Fingerprint Identification',
        doi='10.1109/TIFS.2025.3630316',paper='https://arxiv.org/html/2512.12070v1',
        checked_author_code_index='https://junqing-zhang.github.io/research/dataset-code/',
        checked_publication_index='https://gxhen.github.io/publications/',
        author_implementation='Not located in checked paper, author indexes andexact-title public searches;not a proof of globalnonexistence.',
        reproducible_method_facts=dict(stages=['contrastivepretrain','Siameseclassifier','inference'],
            embedding_dim=256,temperature=.05,batch_pairs=32,views=64),
        budget_mismatch='Original pretraining usesadditionalpublicdata;originalfine-tuningRXB200-1 isConfirm undercurrentprotocol.',
        qualification='Source-only closed-set adaptation andpretrainingpermissions wouldneedseparateregistration;no inferred originalscores.',
        new_fits=0),
      verified_primary_citations=[
        {'title':'Shen receiver-agnosticRFFI','doi':'10.1109/TMC.2023.3340039','url':'https://livrepository.liverpool.ac.uk/3176924/'},
        {'title':'IB-RFF','doi':'10.1109/TIFS.2026.3723081','url':'https://github.com/BeechburgPieStar/IB-RFF/blob/main/README.md'},
        {'title':'ASKNet','doi':'10.1109/LCOMM.2026.3718894','url':'https://github.com/BeechburgPieStar/ASKNet-RFF/blob/main/README.md'},
        {'title':'CvT','url':'https://openaccess.thecvf.com/content/ICCV2021/html/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.html'}],
      code_policy='S24 pinnedimplementations unchanged;current authorREADMEs usedforcitationmetadata,notforretroactivecode/recipechanges.'))
    questions=[
      ('是否只是四层CNN加Transformer？','一般卷积Token与层级结构已有先例；本文贡献限定为RF流水线中的具体设计、参数匹配效果与可复现控制。尚未建立独特RF物理机制。','EVIDENCE_MATRIX.json;paper/main.tex'),
      ('是否重复论文1？','TrueIB、五通道、局部CNN、token几何与数据源明确继承；新问题是同MHA下的token形成与参数预算。论文1的OOB融合、混合器和预算不同，不作跨论文因果差值。','OOB_OVERLAP.json'),
      ('是否只因参数更多或算力更高？','P0与H0参数相差约0.43%，H0仍高2.687pp；但H0相对P0的MAC/延迟更高，必须写parameter efficiency而非全面compute efficiency。','S15/S22;MATCHED_COSTS.json'),
      ('层级效应与局部支持是否已分离？','S1/P0和H1/S1有增量证据，但参数化与频响仍混杂。Long-Patch审计无法同时匹配容量、BN和MAC；只报告方案，不启动训练。','LONG_PATCH_AUDIT.json'),
      ('是否证明Tokenizer去除了RX信息？','没有。H0在Tokenizer端绝对RX离散更大，Pre-MHA均值比值也不领先；最终特征更有利，只支持联合系统的表征结果。','MECHANISM_EXTENSION.json'),
      ('BN是否等于接收机归一化？','不能。H0-S2只有4/6正向并未满足原稳定门槛，统计量和优化效应未唯一分离；数值折叠也不等价于从头去BN训练。','S22/S23;EVIDENCE_MATRIX.json'),
      ('为何Strict优势主要来自Pluto？','这是已保留的RX×TX异质性。CE投影后在TX32/34/35/39严重混淆；原因尚未识别。B200-mini-1、B210-1的Native负差同样展示，不删RX。','per_tx_rx.csv;confusion_pairs.csv'),
      ('ASKNet低均值是否是不公平失败？','完整LoRa代码与六个种子均保留；部分源验证已弱拟合，不能全解释为未知RX泛化失败，不将+32pp当主要竞争证据。','S24RESULTS/INTERPRETATION'),
      ('SNR/CFO和同步实验支持什么？','源元数据的组内相关很小且无稳定方向，不能识别目标RX原因；循环±4点严重失败必须保留，不能宣传同步鲁棒或硬件不变。','metadata_correlations.csv;S23'),
      ('是否已独立验证或形成认证系统？','没有新采集，旧Confirm/blind都有历史；同闭集新测试还需同一批物理TX。未知拒识、FAR/FRR和主动伪装未测，不由Accuracy推出认证保证。','INDEPENDENT_DATA_STATUS.json')
    ]
    intro='在跨接收机LoRa闭集识别中，接收端、信道与数字前处理共同影响接收到的波形。较高分类准确率并不自动证明模型提取了接收机不变的物理指纹。本研究把问题收窄为：在严格带内、已截取对齐的8192点输入、相同五通道表示与普通MHA后端下，Token形成过程的局部支持、训练参数化与归一化如何影响发射机身份可分性？\n\n卷积Token和层级网络已有成熟先例。我们不把它们、TrueIB、五通道或已有局部CNN重命名为新贡献，而是研究一个参数高效的四阶段RF Tokenizer实例。参数匹配的Patch-Lite、平滑与支持相关控制以及BN对照提供系统效果证据；已有开发接收机的长期使用、Source验证选模、同步失败和计算代价均限制其外推。\n\n本轮新增工作是对冻结H0/P0/S1/H1的只读源域诊断和既有RX×TX错误的完整整理。最终特征的分离比改善与分类效果一致，但中间层反例否定了“Tokenizer逐级消除RX信息”的简单解释；核谱相关性和源SNR/CFO诊断不能升级为硬件因果证明。\n'
    contributions=['方法设计：在沿用的严格带内RF流水线上，研究四阶段、32×64输出的参数高效Tokenizer与普通MHA组合；不是宣称卷积Token或新Transformer首创。',
      '结构与机制证据：用参数匹配、平滑/支持、BN和冻结表征分析约束解释，完整保留层间反例、未通过门槛与源验证偏差。',
      '联合实证：固定源预算下与Shen CE/DANN、FAN/IB适配和完整ASK比较，同时报告RX等权精度、最差RX、全部种子和GPU成本；该项是实验贡献而非独立算法。']
    text='S25 科研材料与GO/STOP说明\n\n日期：'+now()+'\n新增训练：0；HStem及历史模型保持冻结；独立新采集：用户确认暂无。\n\n一、中文Introduction\n\n'+intro+'\n二、中文Contributions\n\n'
    text+='\n'.join(f'{i+1}. {v}' for i,v in enumerate(contributions))
    text+='\n\n英文Introduction与Contributions：见paper/main.tex的Introduction。\n\n三、GO / STOP\n\nGO：以参数效率、同协议系统效果和有边界的表征证据完成英文工作稿。\nSTOP：不自动训练Long-Patch或ASK Strict-IB；不宣称新独立RX、完整认证、普遍低延迟或唯一物理机制。\nLong-Patch保留原组数/宽度时161146参数，较H0约多20.5%；1%内候选的秩8瓶颈不适合作为便利弱对照。\n\n四、十个审稿质疑与证据回答\n\n'
    for i,(q,a,e) in enumerate(questions,1):text+=f'{i}. {q}\n回答：{a}\n证据：{e}\n\n'
    text+='五、复现与限制\n\n先恢复registration.json中同SHA的历史checkpoint和原始缓存到登记位置；这些路径是本机只读工件路径，不能在跨机器复现时静默改写科学注册。代码、源预算、随机种子、权限和校验值提供复现实验依据，但没有授权重新分发大波形或全部权重。PCA包含全部六组、四模型，不挑图，且每幅有独立坐标系。当前英文稿为内部匿名工作稿，不代表创新性已通过审稿或投稿准备充分。\n'
    (HERE/'RESEARCH_NOTES.zh-CN.txt').write_text(text,encoding='utf-8',newline='\n')
    write(HERE/'REVIEWER_QA.json',dict(questions=[dict(question=q,answer=a,evidence=e) for q,a,e in questions]))
    print('Evidence matrix, GO/STOP, overlap, literature audit, bilingual notes andtenrevieweranswers generated.')


if __name__=='__main__':main()
