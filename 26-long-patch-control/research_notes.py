"""Outcome-dependent evidence matrix, bilingual contribution notes and answers."""
from s26_common import HERE,ROOT,read,write,sha,now
from paper_revision import INTRO


def literature():
    write(HERE/'LITERATURE_EXTENSION.json',dict(time=now(),status='PRIMARY_SOURCES_CHECKED_NO_NEW_METHOD_SCORE',
        ma2025=dict(paper='https://arxiv.org/html/2512.12070v1',pdf='https://arxiv.org/pdf/2512.12070',
            checked_fig5_pdf_page=6,author_code_index='https://junqing-zhang.github.io/research/dataset-code/',
            bounded_code_search=['Exact title','arXiv2512.12070','DOI3630316','LoRa/Siamese/JieMa combinations'],
            author_code='Not located in checked sources; not a proof of global nonexistence.',
            corrected_scope='Unpretrained Siamese can in principle use current legal Source RX. External pretraining is an optional separate permission/budget question.',
            current_decision='No qualified frozen adapter or score added in this six-fit control stage; implementation choices remain to be audited.'),
        primary_bibliography_sources=[
            'https://junqing-zhang.github.io/research/rffi/rffi-pub-topics/',
            'https://arxiv.org/abs/2506.09807','https://arxiv.org/abs/2207.03001',
            'https://proceedings.mlr.press/v119/chen20j.html',
            'https://ieeexplore.ieee.org/document/11421903/',
            'https://sites.google.com/ohtsuki.ics.keio.ac.jp/www/publications/journals'],
        new_method_fits=0,new_method_scores=0))


def main():
    result=read(HERE/'RESULTS.json');pair=result['paired_comparisons'][0];other=result['paired_comparisons'][1]
    summary={r['arm']:r for r in result['summary']};ci=pair['conditional_rx_ci95_pp']
    quantitative=f"H0−LP {pair['mean_delta_pp']:+.3f} pp，条件 RX 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]，{pair['positive_fit_pairs']}/6 配对正向；LP−P0 {other['mean_delta_pp']:+.3f} pp。"
    if pair['primary_practical_rule_met']:
        interpretation='通过预先冻结的一致性门槛；在这个候选上，单个自由长核投影未恢复完整HStem结果。但宽度、BN、非线性深度、边界参数化和MAC仍混杂，不能归因为纯层级性。'
    else:
        interpretation='未通过预先冻结的一致性门槛；当前控制不能确立稳定的HStem优势，层级独立收益仍未建立。小差或区间含零也不等于已证明等效，应收窄贡献。'
    matrix=read(ROOT/'25-tifs-integration/EVIDENCE_MATRIX.json')['rows']
    matrix+=[dict(claim='A same-support direct long projection is sufficient to recover HStem.',
        definition='Registered groups8/width64/k340/stride256/pad42;138346 params, same nominal350 support and32x64 tokens.',
        evidence=['26-long-patch-control/registration.json','26-long-patch-control/RESULTS.json','26-long-patch-control/EXPERIMENT_INTEGRITY.json'],
        result=quantitative,status='H0_CONSISTENCY_RULE_MET' if pair['primary_practical_rule_met'] else 'H0_CONSISTENCY_RULE_NOT_MET',
        limit=interpretation),dict(claim='The architectural accounting is a novel RF generalization theorem.',
        definition='Support/center/coordinate-count/parameter/MAC identities and frozen-BN Jacobian factorization.',
        evidence=['26-long-patch-control/METHOD_DERIVATION.json'],result='Accounting verified against actual models.',
        status='ARCHITECTURAL_IDENTITIES_NOT_A_NEW_THEOREM',
        limit='No TX-information preservation, antialiasing, physical locality or receiver-invariance guarantee follows.')]
    write(HERE/'EVIDENCE_MATRIX.json',dict(time=now(),rows=matrix,prior_rows_retained=True,new_fits=6))
    questions=[
        ('相对已有CNN Tokenizer，新增贡献是什么？','四阶段RF设计、明确的接口/支持/预算定义和受控经验结论；CvT/PVT/Swin、BN和MHA都不是首创。数学递推是结构核算，不是新的RF泛化理论。'),
        ('TrueIB、五通道和前端是否与论文1重叠？','这些及数据源明确继承，不能列为新增贡献。论文1的OOB融合、混合器与预算不同，不用跨论文准确率相减证明架构因果。'),
        ('是否只是参数更多或算力更多？','H0只有133690参数，P0为134266，LP为138346；但H0部分MAC约95.028M，LP约55.477M。参数效率与计算效率须分开，延迟以同一S26会话报告。'),
        ('Long-Patch排除了单纯扩大支持的解释吗？',quantitative+interpretation),
        ('是否已经证明RX不变性？','没有。旧源域最终Jw为H0 5.449/P0 4.270，但Tokenizer绝对RX离散更大，Pre-MHA指标也不占优。它支持联合网络最终判别结果，不支持逐层去RX或互信息结论。'),
        ('条件置信区间和六种子是否等于独立泛化？','没有。RX内平均三种子，再bootstrap十个唯一RX；区间条件于当前模型与历史协议。架构/超参数已有Dev开发历史，同一RX在另一折是源域，不是新采集。'),
        ('强基线和Ma方法的覆盖是否公平？','Shen CE/DANN、FAN/IB和完整ASK的全部既有结果保留。Ma不预训练版本原则上可源域适配，不能整体用预训练预算排除；但合格冻结实现尚未完成，不填入虚构或论文原分数。'),
        ('为何保留ASK弱种子与Pluto集中优势？','它们属于完整结果。部分ASK源验证已弱拟合，不能全归为跨RX失败；Strict对CE约82%总体优势来自Pluto，Native也有输给Shen的RX，不删种子或接收机。'),
        ('同步敏感性对安全结论有什么影响？','±4点循环移位严重退化仍保留，不能宣传同步鲁棒。闭集Accuracy不提供未知拒识、FAR/FRR、主动伪装或完整认证保证。'),
        ('实现修复是否改过训练或数据权限？','没有。原科学文件SHA保持不变；TECHNICAL_FIX_001只删除评估脚本一个多余括号，另存修复模块并在Dev前登记。恢复入口复用全部六组和封存记录，不补种子。')]
    write(HERE/'REVIEWER_QA.json',dict(time=now(),questions=[dict(question=q,answer=a) for q,a in questions]))
    text='S26 创新论证、结果解释与稿件材料\n\n'+now()+'\n\n'
    text+='一、本轮实质执行\n\n仅新增LP一个控制臂，2折×3种子，共6次、每次200轮；历史HStem与全部S15–S25结果不变。全部6组封存后才统一评估LP/H0/P0，共18模型、180000次Dev预测。\n\n'
    text+='二、核心结果与GO/STOP\n\n'+quantitative+'\n'+interpretation+'\n\n'
    text+='GO：将实际结果用于有边界的RF Token形成研究与英文稿重写。STOP：本阶段6/6完成后不自动再加模型、种子、ASK Strict-IB或新超参数；不宣称独立采集或RX不变性。\n\n'
    text+='三、中文Introduction与Contributions\n\n'
    text+='跨接收机LoRa识别不仅取决于输入信息和注意力后端，还取决于这些RF坐标如何形成Token。固定Token数量和参数量不会自动固定局部支持、非线性、归一化或计算量。本文在继承的严格带内五视图前端上，研究四阶段局部处理与单次大核投影的差异。\n\n'
    text+='首先，给出HStem的固定接口、逐级支持/中心、坐标压缩和参数/MAC定义，避免把一般卷积结构和数学核算说成新的物理理论。其次，以参数匹配、平滑/支持、BN与新增同支持Long-Patch检验不同解释，保留中间层反例。最后，在固定源数据权限下与代表性外部方法竞争，同时报告全部种子、RX差异和成本。\n\n'
    text+='本轮结果：'+quantitative+interpretation+'没有新独立采集，已对齐窗口、闭集已知TX和历史开发RX的范围不变。\n\n'
    text+='四、英文Introduction与Contributions\n\n'+INTRO+'\n\n五、十个审稿质疑\n\n'
    for i,(q,a) in enumerate(questions,1):text+=f'{i}. {q}\n{a}\n\n'
    text+='六、限制与后续边界\n\n新控制不是纯层级因果消融；支持是冻结BN下的卷积坐标跨度，TrueIB/FFT具有全局依赖，训练BN也跨批次/空间统计。坐标维数不等于信息量。没有新数据，不重命名旧Confirm/blind。论文仍是匿名内部工作稿，实验数量不保证TIFS创新门槛。\n'
    (HERE/'RESEARCH_NOTES.zh-CN.txt').write_text(text,encoding='utf-8',newline='\n')
    write(HERE/'GO_STOP.json',dict(time=now(),status='GO_BOUNDED_REVISED_MANUSCRIPT_STOP_FURTHER_AUTOMATIC_FITS',
        completed_new_fits=6,remaining_authorized_new_fits=0,primary_control_rule_met=pair['primary_practical_rule_met'],
        interpretation=interpretation,independent_acquisition='NOT_AVAILABLE_NOT_RUN',ma_adapter='NOT_QUALIFIED_NO_SCORE',
        new_physical_invariance_claim=False,new_training_required_by_this_stage=False))
    literature()
    print('Outcome-based matrix, bilingual introduction/contributions,10 reviewer answers and GO/STOP ready.',flush=True)


if __name__=='__main__':main()
