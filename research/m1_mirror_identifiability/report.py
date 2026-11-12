"""Physics-only report and manifest. Never loads IQ, weights or classifiers."""
import json
from pathlib import Path
import runner as R

HERE=R.HERE


def main():
    s=R.read(HERE/'SYNTHETIC_RESULTS.json');v=R.read(HERE/'SOURCE_VALIDITY.json');q=R.read(HERE/'INDEPENDENT_QA.json')
    if q['status']!='PASS':raise ValueError('Independent verification required')
    lines=['# M1-A 共轭镜像：物理与源域可行性结果','',f"**结论：`{v['decision']}`。本轮分类器拟合和深度更新均为0，M1-B/C未启动。**",'',
        '## 来源、物理可信度与可辨识性','',
        '官方资料支持25 Msps未均衡256点输入；本地继承既有Kaggle来源链，没有宣称与当前UCLA下载逐字节比对。能量检测不提供每包绝对STF起点。只检验预先固定的[20,180)候选内部区间，拟合前四周期、验证后四周期，没有移动区间或通过分类准确率选择模板。见 [来源](SOURCES.md)。','',
        '使用完整 TX→信道/CFO→RX 链。RX共轭分支的CFO反向旋转；RX-only即使TX不平衡为0也产生非零有效镜像项。真实TX参数未估计。半周期共轭歧义、未知信道列空间混淆，以及CFO为0时的四列秩亏均显式保留；见 [可辨识性](IDENTIFIABILITY.md)。','',
        f"合成{s['packets']}包、有效{s['valid']}包（{s['valid_fraction']*100:.3f}%）；{s['engineering_tests']}项工程测试通过。已知时序/平坦信道/零CFO的有效系数恢复共{s['effective_coefficient_recovery_cases']}例，最大误差/事前噪声容差={s['max_error_over_registered_tolerance']:.6f}。这不是在未知信道下恢复真实TX参数。",'',
        f"L-STF共轭与10点平移等式最大误差{s['structural_proofs']['short_conjugate_halfshift_max_error']:.3e}；12延迟普通信道秩={s['structural_proofs']['ordinary_channel_rank']}，24列广义线性信道秩={s['structural_proofs']['widely_linear_channel_rank']}。两列条件数好仍不能消除联合时序或信道歧义。",'',
        '## 有界源域结果','',
        f"仅使用两个开发视图共同fit的六台RX、原train_ids、Day1–2，每非空TX×RX×Day最多16包。共{v['overall']['packets']}包，{v['overall']['valid']}包满足全部有效条件（{v['overall']['valid_fraction']*100:.3f}%）；失败包没有删除。",'',
        '| RX | Day | 抽样数 | 有效数 | 有效比例 | 失败原因计数（可重叠） |','|---|---:|---:|---:|---:|---|']
    for x in v['per_rx_day']:lines.append(f"| {x['rx']} | {x['day']} | {x['packets']} | {x['valid']} | {x['valid_fraction']*100:.3f}% | {json.dumps(x['failure_reasons'],ensure_ascii=False)} |")
    ci=v['advantage_RX_conditional95'];lines+=['',
        f"同自由度、同频带NC相对WL的验证残差优势（除以L残差）RX等权均值={v['mean_advantage_RX_equal']:.6f}，六个已观察RX条件95%区间=[{ci[0]:.6f},{ci[1]:.6f}]，登记实用门为0.02；无效包记0优势，同时保留原始残差，不能将其排除以提高收益。L→WL相对验证改善={v['linear_to_WL_RX_equal_gain']:.6f}。这些均不是TX分类准确率。",'',
        '## 原门槛与解释边界','']
    for name,value in v['gates'].items():lines.append(f"- {name}：{'PASS' if value else 'FAIL'}。")
    lines+=['',f"可靠性与abs(CFO)/fit残差/channel proxy的Spearman关联：{json.dumps(v['correlations_with_validity_descriptive'],ensure_ascii=False)}。仅为描述性混杂线索，不能归因于RX硬件或TX指纹。每个TX×RX×Day单元含空缺均在 SOURCE_VALIDITY.json 列出。",'',
        '有效q使用预定直接分量占优的参考规范；它不是唯一物理到达时序或真实TX I/Q系数。观测耦合可来自TX、RX、CFO、信道、截窗或其组合；现有数据没有完成这些因素的因果分离。即使A通过，也不证明跨RX TX判别价值，仍须另行授权B。',
        '', '本候选按冻结门结束；没有为失败修改起点、阈值、对照或估计器，也没有自动启动镜像网络、线性探针或替代表示。F2/P2/固定窗口/S09/S17/S19历史STOP与原报告保持。',
        '',f"独立复核通过{q['check_count']}项，其中同一批{q['QR_replay_rows']}包用另一份频率多项式实现和QR求解器复算系数与包内验证残差；独立重算覆盖率、RX条件区间、门槛及抽样身份。新增独立IQ行数为0。",'',
        '没有访问checkpoint_val、source_selection、历史held6、target8、Day3–4、Full UNKNOWN RX或任何模型/预测；没有跨RX包事件配对。访问日志区分整文件字节摘要、选定行科学解码及独立同批回放。', '',
        '[合成明细](SYNTHETIC_RESULTS.json) · [源域有效率](SOURCE_VALIDITY.json) · [全部逐包诊断](PACKET_DIAGNOSTICS.json) · [机器决策](DECISION.json) · [独立复核](INDEPENDENT_QA.json) · [访问日志](ACCESS_LOG.jsonl)', '']
    (HERE/'RESULTS.zh-CN.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    files={p.name:dict(bytes=p.stat().st_size,sha256=R.sha(p)) for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='ARTIFACT_MANIFEST.json' and p.suffix!='.log'}
    R.new(HERE/'ARTIFACT_MANIFEST.json',dict(status=v['decision'],files=files,raw_IQ_included=False,classifier_fits=0,neural_updates=0))
    print(dict(status='REPORT_AND_MANIFEST_COMPLETE'))


if __name__=='__main__':main()
