"""Derived, descriptive reporting only; cannot select or fit any model."""
import csv
from pathlib import Path
import numpy as np
from s18_core import HERE, read, write, sha, now

RESULTS = HERE/'d0_support_matched'


def make_figure(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = ['#2962aa', '#d55e00', '#37966f', '#9973ac']
    layers = ['stem1', 'stem_mid', 'tokens', 'mixer']
    for f, row in enumerate(rows):
        axes[0, 0].plot(range(4), [row['layers'][l]['stability']['days'] for l in layers], marker='o', color=colors[f], label=f'fold{f}')
    axes[0, 0].axhline(.35, ls='--', color='gray', lw=1, label='primary tokens threshold')
    axes[0, 0].set(xticks=range(4), xticklabels=layers, ylim=(0, 1), ylabel='Mean squared principal cosine', title='A. Cross-day subspace stability')
    axes[0, 0].legend(fontsize=8)
    x = np.arange(4)
    axes[0, 1].bar(x-.16, [r['layers']['tokens']['val_metrics']['rx_fraction'] for r in rows], width=.3, label='RX scatter fraction', color='#2962aa')
    axes[0, 1].bar(x+.16, [r['layers']['tokens']['val_metrics']['tx_fraction'] for r in rows], width=.3, label='TX scatter fraction', color='#d55e00')
    axes[0, 1].axhline(.15, ls='--', color='gray', lw=1)
    axes[0, 1].set(xticks=x, xticklabels=[f'fold{f}' for f in range(4)], ylabel='Fraction of total scatter', title='B. Fixed selective directions on validation')
    axes[0, 1].legend(fontsize=8)
    arms = ['selective', 'tx_dominant', 'random_0', 'random_1', 'random_2']
    labels = ['Selective', 'TX control', 'Random 0', 'Random 1', 'Random 2']
    for j, key in enumerate(['delta_acc_pp', 'delta_ce']):
        ax = axes[1, j]
        for f, row in enumerate(rows):
            lookup = {r['arm']:r for r in row['interventions']}
            ax.plot(np.arange(5)+(f-1.5)*.045, [lookup[a][key] for a in arms], 'o-', lw=.7, color=colors[f], label=f'fold{f}')
        ax.axhline(0, color='gray', lw=1)
        ax.set(xticks=range(5), xticklabels=labels, ylabel='Accuracy change (pp)' if j == 0 else 'Cross-entropy change',
               title='C. Frozen TX classifier preservation' if j == 0 else 'D. Frozen TX classifier loss')
        ax.tick_params(axis='x', labelrotation=12)
    fig.suptitle('S18-D0S | fixed rank3, alpha=0.5 | source validation only | no new training', fontsize=13)
    fig.savefig(HERE/'S18_D0S_OVERVIEW.png', dpi=180)
    plt.close(fig)


def run():
    rows = [read(RESULTS/f'D0_fold{f}.json') for f in range(4)]
    decision = read(RESULTS/'decision_d0.json')
    audit = read(HERE/'FINAL_ACCESS_AUDIT.json')
    if audit['status'] != 'PASS': raise ValueError('Final scientific audit must pass before reporting')
    flattened = []
    for row in rows:
        for arm in row['interventions']:
            flattened.append(dict(fold=row['fold'], arm=arm['arm'], accuracy=arm['score']['summary']['acc'], macro_f1=arm['score']['summary']['macro_f1'],
                                  ce=arm['score']['summary']['ce'], margin=arm['score']['summary']['margin'],
                                  **{k:arm[k] for k in ['delta_acc_pp', 'delta_f1_pp', 'delta_ce', 'delta_margin', 'worst_rx_paired_pp', 'worst_tx_paired_pp', 'rx_scatter_reduction', 'tx_scatter_reduction']}))
    with (HERE/'D0_interventions.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(flattened[0]), lineterminator='\n'); w.writeheader(); w.writerows(flattened)
    make_figure(rows)
    interval = decision['receiver_cluster_selective_vs_random']
    passed_stable = sum(f['stable'] for f in decision['folds'])
    passed_specific = sum(f['specific'] for f in decision['folds'])
    passed_joint = sum(f['stable'] and f['specific'] for f in decision['folds'])
    lines = ['# S18 HStem选择性接收机表征：机制报告', '',
             f"源域决定：**{decision['status']}**。四fold共20组冻结方向干预，新增模型训练0；真实目标CRD、held-source、ManySig和独立seed均未读取或启动。", '',
             f"主要Tokens方向的稳定性与RX覆盖联合条件通过{passed_stable}/4 folds，分类特异性通过{passed_specific}/4，二者同时通过{passed_joint}/4。准入需要至少3/4且全部fold保持TX指标；这里的0/4不能解释成所有稳定RX差异均不存在。", '',
             '![固定机制审计总览](S18_D0S_OVERVIEW.png)', '',
             '图中仅是冻结模型对源验证数据的响应，各fold点不是独立随机种子，也不代表目标泛化收益。', '',
             '## 结论与边界', '',
             ('当前固定rank3、alpha=.5、Tokens通道子空间方案未通过预先固定的联合门槛。保留负结果，停止B1–B4训练，不因本次结果继续搜索层、rank或强度。' if not decision['admitted'] else
              '当前固定方案通过D0S操作性筛选，仅准入进一步受控源域Pilot；仍需在任何新fit前封存训练配方。'), '',
             '这次观察支持的范围较窄：方向可以跨日和训练/验证重复，但跨TX稳定性只通过1/4；方向内TX类均值散度很低，却没有覆盖足够多的RX散度。高RX/TX比值本身不等于找到了分类器依赖的主要RX子空间。所有fold的源TX保持条件通过，仍未形成相对随机方向的稳定功能优势。', '',
             f"预选方向的验证RX散度占比为{100*min(r['layers']['tokens']['val_metrics']['rx_fraction'] for r in rows):.2f}%–{100*max(r['layers']['tokens']['val_metrics']['rx_fraction'] for r in rows):.2f}%，均低于15%门槛；TX散度占比仅{100*min(r['layers']['tokens']['val_metrics']['tx_fraction'] for r in rows):.4f}%–{100*max(r['layers']['tokens']['val_metrics']['tx_fraction'] for r in rows):.4f}%。随机平均与选择性CE差没有任何fold达到0.005门槛。", '',
             '本结果不证明全部选择性表征或物理建模无效，也不证明TX/RX硬件已分离。Shen True In-Band与IB-RFF Information Bottleneck的区别、S10公平数字及已有S11/S12重叠见[协议审计](PROTOCOL_AUDIT.zh-CN.md)。', '',
             '## 数据支持修正与可追溯性', '',
             '原D0因完整cell假设在特征提取前触发支持检查，状态为`STOP_PRE_FEATURE_SUPPORT`，原注册和attempt保留。随后只核查源标签支持，在任何方向或干预结果产生前单独注册D0S；未改变模型、数据split、设备集合、rank、alpha、对照或门槛。', '',
             '四fold有13/15/13/8个空cell（各360个），训练/验证空缺模式相同。D0S不插补也不删除整RX，使用TX条件RX散度和成对TX共同RX散度；完整矩形时后者严格还原原定义，缺失时估计的是共同支持条件下的差异。详见[修正说明](SUPPORT_AMENDMENT.zh-CN.md)与[支持计数](SUPPORT_AUDIT.json)。', '',
             '## 冻结模型基线', '',
             '| fold | 估计训练包 | 源验证包 | Accuracy % | Macro-F1 % | 最差RX % | 最差TX % | CE | margin |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        b=r['baseline']['summary']
        lines.append(f"| {r['fold']} | {r['train_packets']} | {r['validation_packets']} | {100*b['acc']:.4f} | {100*b['macro_f1']:.4f} | {100*b['worst_rx']:.4f} | {100*b['worst_tx']:.4f} | {b['ce']:.5f} | {b['margin']:.4f} |")
    lines += ['', '各fold源RX等权Accuracy严格复现S17B冻结checkpoint，不能与S10真实CRD数字直接比较。', '',
              '## 逐层方向稳定性', '',
              '| fold | 层 | 包二分 | 跨日 | 跨TX半数 | 训练/源验证 | 验证RX散度占比 | 验证TX散度占比 | RX probe % |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        for layer,v in r['layers'].items():
            s=v['stability']
            lines.append(f"| {r['fold']} | {layer} | {s['packet_halves']:.3f} | {s['days']:.3f} | {s['tx_halves']:.3f} | {s['source_validation']:.3f} | {v['val_metrics']['rx_fraction']:.4f} | {v['val_metrics']['tx_fraction']:.4f} | {100*v['rx_probe']['accuracy_rx_equal']:.2f} |")
    lines += ['', '只预选Tokens作为主要干预位置；其余层不用于事后挑选。RX probe用TX/day标签作离线诊断，随机率1/18；该probe不进入分类推理。', '',
              '## 冻结干预与对照', '',
              '| fold | 方向 | ΔAcc pp | ΔF1 pp | ΔCE | Δmargin | 最差配对RX pp | 最差配对TX pp | RX散度减少 | TX散度减少 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in flattened:
        lines.append(f"| {r['fold']} | {r['arm']} | {r['delta_acc_pp']:+.4f} | {r['delta_f1_pp']:+.4f} | {r['delta_ce']:+.5f} | {r['delta_margin']:+.4f} | {r['worst_rx_paired_pp']:+.3f} | {r['worst_tx_paired_pp']:+.3f} | {r['rx_scatter_reduction']:.4f} | {r['tx_scatter_reduction']:.4f} |")
    lines += ['', f"选择性相对三个随机对照平均的Accuracy差，按同一RX跨fold先平均：{interval['mean_pp']:+.4f}pp，描述性RX聚类95%区间[{interval['ci95_pp'][0]:+.4f},{interval['ci95_pp'][1]:+.4f}]pp，唯一RX身份数{interval['independent_rx']}。区间不涵盖训练随机性，也不能消除共享训练集造成的相关性。", '',
              '所有干预rank/alpha一致，未匹配实际特征扰动能量；不能将随机方向与选择性方向的差异归因于相同能量下的完整因果机制。', '',
              '## 逐fold联合门槛', '',
              '| fold | 稳定性 | 特异性 | TX保持 | 未通过条件 |', '|---|---|---|---|---|']
    for r in decision['folds']:
        failed = [k for group in ['stability_checks','specificity_checks','preservation_checks'] for k,v in r[group].items() if not v]
        lines.append(f"| {r['fold']} | {r['stable']} | {r['specific']} | {r['preserved']} | {', '.join(failed) or '无'} |")
    lines += ['', '门槛原值与所有布尔量见[机器决策](d0_support_matched/decision_d0.json)，没有放宽门槛或选取更有利的结果。', '',
              '## 算法定义与未执行的Pilot', '',
              '候选定义为训练源cell统计得到的U/中心，加有界残差 `z−0.5 U diag(sigmoid(theta)) Uᵀ(z−mu)`；推理不需RX/TX真值。D0S只检验固定g=1。alpha=.5时映射可逆，理想精确算术下不改变互信息；“软瓶颈”只描述有限网络中的几何抑制与归纳偏置。', '',
              '| 臂 | 方法 | 本轮状态 |', '|---|---|---|',
              '| B0 | 既有HStem+MHA端到端模型 | 四fold冻结复用，仅诊断 |',
              '| B1 | HStem+普通HSIC | 未训练；与完整IB-RFF分别报告 |',
              '| B2 | 静态选择性方向 | 仅冻结干预，不等于训练后的B2 |',
              '| B3 | 三参数有界学习门控 | 未训练 |',
              '| B4 | 同参数随机方向门控 | 未训练；本轮只有静态随机方向干预 |', '',
              '若将来提出新的研究，需重新注册并明确方向估计时点、所有臂对称预训练/warm-start、HSIC核及权重；不能把候选单独预训练后的收益归因于结构。当前STOP不自动开展新的方案。', '',
              '## 参数与计算成本', '',
              '| fold | 参数 | 核心dense FLOPs/包 | 静态投影额外FLOPs/包 | B0 batch1 ms | 静态 batch1 ms | B0 batch128 ms | 静态 batch128 ms |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        c=r['costs']; l={(v['arm'],v['batch']):v['median_batch_ms'] for v in c['latency']}
        lines.append(f"| {r['fold']} | {c['parameters']} | {c['dense_core_flops']:,} | {c['bottleneck_dense_flops']:,} | {l['B0',1]:.3f} | {l['static_selective',1]:.3f} | {l['B0',128]:.3f} | {l['static_selective',128]:.3f} |")
    lines += ['', 'FLOPs按2/MAC，只计Conv/Linear/attention乘积，不含FFT、norm、激活、softmax与逐元素算子。GPU为RTX4060 Laptop，FP32、每配置10次warm-up/30次同步实测；延迟含Python hook及同步方式，是本机描述性测量。静态模块新增训练参数0、固定basis/center256 floats；B3/B4设计各3参数，未把设计值伪装为已训练实测。', '',
              '## 证据与复现', '',
              '- [原注册](registration.json)、[D0S注册](registration_support.json)、[原支持STOP](D0_SUPPORT_STOP.json)',
              '- [逐fold原始JSON](d0_support_matched/D0_fold0.json)、[全部干预CSV](D0_interventions.csv)、[最终审计](FINAL_ACCESS_AUDIT.json)',
              '- 本地`d0_support_matched/cache/features_f*.npz`保存源训练/验证池化特征及索引，SHA在原始JSON；原始IQ、checkpoint和逐包特征不进Git。',
              '- 工程测试包含16项原测试及支持修正测试；实际结果复核检查注册代码/历史证据SHA、混淆矩阵指标、散度/方向、门槛和源访问范围。', '',
              '证据限制：本地特征缓存没有逐包logits；独立QA从混淆矩阵重新计算Accuracy/F1/TX指标，并重新估计散度和方向。CE/margin核验覆盖RX聚合、差值与门槛一致性，未独立从逐包logits再算。原始源IQ与冻结权重的SHA及代码仍提供完整计算复现入口。', '',
              '现有模型已在方向估计训练样本上训练；源验证已参与历史checkpoint选择，因此本轮不是独立确认。mean pooling未覆盖完整Token位置交互，RX散度混有信道/同步，TX类均值散度不穷尽非线性判别信息。完整限制见[方法定义](README.zh-CN.md)。', '',
              '没有新增真实CRD比较结果，不能宣称超越IB-RFF或ASKNet。']
    (HERE/'RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    write(HERE/'STATUS.json', dict(time=now(), stage='S18', status=decision['status'], admitted=decision['admitted'],
          completed_folds=4, frozen_interventions=20, new_fits=0, held_source_read=False, real_target_access=False,
          original_d0='STOP_PRE_FEATURE_SUPPORT', support_amendment=True, audit='PASS'))
    evidence = [p for p in HERE.rglob('*') if p.is_file() and 'cache' not in p.parts and '__pycache__' not in p.parts and p.suffix in ['.json','.jsonl','.md','.csv','.py','.png'] and p.name != 'MANIFEST.json']
    write(HERE/'MANIFEST.json', dict(time=now(), files_sha256={p.relative_to(HERE).as_posix():sha(p) for p in sorted(evidence)},
          local_feature_sha256={str(r['fold']):r['feature_cache_sha256'] for r in rows}, checkpoint_sha256={str(r['fold']):r['checkpoint_sha256'] for r in rows}))
    print(decision['status'], 'report and evidence generated', flush=True)


if __name__ == '__main__': run()
