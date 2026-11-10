"""Derived S19 source-only reports; no model/data loading or result selection.

All decisions come from the frozen machine-readable source gate. This module
only renders aggregate evidence and checkpoints' saved metrics. It never reads
raw IQ, model tensors or per-packet prediction/evidence caches.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from s19_core import HERE, job_id, now, read, sha, write


ARMS = ('B0', 'B1', 'B2', 'B3', 'B4')
CONTROLS = ('B0', 'B1', 'B2', 'B3')
ARM_NAMES = {
    'B0': '原HStem+MHA',
    'B1': '静态可学习8band滤波',
    'B2': '参数匹配power门控',
    'B3': 'TCSE结构+CE',
    'B4': 'TCSE结构+CE+证据监督',
}
COLORS = {'B0': '#616161', 'B1': '#d55e00', 'B2': '#359968', 'B3': '#8662aa', 'B4': '#2466a8'}


def load_records():
    records = []
    for fold in range(4):
        for arm in ARMS:
            out = HERE/'runs'/job_id(fold, arm)
            config = read(out/'config.json')
            fitted = read(out/'metrics.json')
            held = read(out/'held_metrics.json')
            if fitted['status'] != 'COMPLETE':
                raise ValueError('All20 source models must complete before reporting')
            records.append(dict(fold=fold, arm=arm, config=config, fitted=fitted, held=held))
    return records


def arm_records(records, arm):
    return sorted([r for r in records if r['arm'] == arm], key=lambda r: r['fold'])


def domain_score(record, role):
    return record['fitted']['source_validation'] if role == 'source_validation' else record['held']['held_source']


def f4(value):
    return f'{float(value):.4f}'


def signed(value):
    return f'{float(value):+.4f}'


def ci_text(value):
    return f"[{float(value[0]):+.4f}, {float(value[1]):+.4f}]"


def mean_arm(records, arm, role):
    summaries = [domain_score(r, role)['summary'] for r in arm_records(records, arm)]
    return {key:float(np.mean([s[key] for s in summaries])) for key in ['acc', 'macro_f1', 'ce', 'margin', 'worst_rx', 'worst_tx']}


def comparison_csv(decision):
    rows = []
    for control in CONTROLS:
        comp = decision['comparisons'][control]
        for fold in comp['folds']:
            rows.append(dict(candidate='B4', control=control, fold=fold['fold'],
                             delta_accuracy_pp=fold['delta_acc_pp'], delta_macro_f1_pp=fold['delta_f1_pp'],
                             delta_ce=fold['delta_ce'], worst_paired_rx_pp=fold['worst_paired_rx_pp'],
                             mean_fold_gain_pp=comp['mean_fold_gain_pp'], positive_folds=comp['positive_folds'],
                             unique_rx=comp['unique_rx'], unique_rx_mean_pp=comp['unique_rx_mean_pp'],
                             conditional_rx_ci95_lower_pp=comp['ci95_pp'][0], conditional_rx_ci95_upper_pp=comp['ci95_pp'][1],
                             all_control_checks_passed=all(decision['checks'][control].values()),
                             scope='Held-source RX/date1-2; conditional single-seed comparison; not real target CRD'))
    with (HERE/'PILOT_COMPARISONS.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def plot_overview(records, decision):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), constrained_layout=True)
    x = np.arange(len(ARMS))
    for role, offset, color, label in [('source_validation', -.15, '#7c9ab8', 'Fit-RX validation'),
                                        ('held_source', .15, '#2466a8', 'Held-source RX')]:
        means = [100*mean_arm(records, arm, role)['acc'] for arm in ARMS]
        axes[0, 0].bar(x+offset, means, width=.28, color=color, label=label, alpha=.85)
        for i, arm in enumerate(ARMS):
            points = [100*domain_score(r, role)['summary']['acc'] for r in arm_records(records, arm)]
            axes[0, 0].scatter(np.full(4, i+offset), points, s=15, color='#222222', alpha=.65)
    values = [100*domain_score(r, role)['summary']['acc'] for r in records for role in ['source_validation', 'held_source']]
    axes[0, 0].set(xticks=x, xticklabels=ARMS, ylabel='TX accuracy (%)', ylim=(max(0, min(values)-3), min(100, max(values)+2)),
                   title='A. Fixed source protocol: mean folds and fold points')
    axes[0, 0].legend(fontsize=8)
    for control in CONTROLS:
        comp = decision['comparisons'][control]
        axes[0, 1].plot(range(4), [r['delta_acc_pp'] for r in comp['folds']], 'o-', color=COLORS[control], label=f'B4 - {control}')
    axes[0, 1].axhline(0, color='gray', lw=1)
    axes[0, 1].axhline(1, color='gray', lw=1, ls='--', label='B4-B0 mean gate: +1pp')
    axes[0, 1].set(xticks=range(4), xticklabels=[f'fold{i}' for i in range(4)], ylabel='Held-source accuracy gain (pp)',
                   title='B. Primary candidate against all fixed controls')
    axes[0, 1].legend(fontsize=8)
    for i, control in enumerate(CONTROLS):
        comp = decision['comparisons'][control]
        center = comp['unique_rx_mean_pp']; lo, hi = comp['ci95_pp']
        axes[1, 0].errorbar(center, i, xerr=np.array([[max(0, center-lo)], [max(0, hi-center)]]),
                            fmt='o', capsize=4, color=COLORS[control], lw=1.6)
        axes[1, 0].annotate(f"fold mean {comp['mean_fold_gain_pp']:+.2f}", (hi, i), xytext=(6, 0), textcoords='offset points', va='center', fontsize=8)
    axes[1, 0].axvline(0, color='gray', lw=1)
    axes[1, 0].set(yticks=range(4), yticklabels=[f'B4 - {c}' for c in CONTROLS], xlabel='Accuracy gain (pp)',
                   title='C. Unique-RX mean and conditional 95% interval')
    axes[1, 0].invert_yaxis()
    # Space for fold-mean annotations without changing the scientific limits.
    xlo, xhi = axes[1, 0].get_xlim(); axes[1, 0].set_xlim(xlo, xhi+.30*max(xhi-xlo, .5))
    for fold in range(4):
        points = [100*next(r for r in records if r['fold'] == fold and r['arm'] == arm)['held']['held_source']['summary']['worst_rx'] for arm in ARMS]
        axes[1, 1].plot(x, points, 'o-', lw=1, label=f'fold{fold}')
    axes[1, 1].set(xticks=x, xticklabels=ARMS, ylabel='Worst held-source RX accuracy (%)',
                   title='D. Absolute worst-RX accuracy per fold')
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(f"S19 | {decision['status']} | ManyRx source only | 4 folds x 1 seed", fontsize=13)
    fig.savefig(HERE/'S19_OVERVIEW.png', dpi=180)
    plt.close(fig)


def plot_evidence(evidence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13.3, 8.2), constrained_layout=True)
    locations = [(0, 0), (0, 1), (0, 2), (1, 0)]
    maximum = max(float(np.asarray(r['summary']['bar_q']).max()) for r in evidence)
    heat_axes = []
    im = None
    for record, location in zip(evidence, locations):
        ax = axes[location]
        target = np.asarray(record['summary']['bar_q'])
        im = ax.imshow(target, aspect='auto', vmin=0, vmax=maximum, cmap='viridis')
        ax.set(xticks=range(8), xticklabels=[f'band{i}' for i in range(8)],
               yticks=range(len(target)), yticklabels=record['summary']['tx_labels'],
               xlabel='Fixed spectral band', ylabel='Source TX class', title=f"fold{record['fold']}: conservative target bar_q")
        ax.tick_params(axis='x', labelrotation=40, labelsize=8)
        heat_axes.append(ax)
    fig.colorbar(im, ax=heat_axes, shrink=.75, label='Evidence target probability')
    x = np.arange(4)
    amplitude = [r['summary']['controls']['amplitude_signed_mean'] for r in evidence]
    phase = [r['summary']['controls']['phase_signed_mean'] for r in evidence]
    axes[1, 1].bar(x-.17, amplitude, width=.32, color='#359968', label='Same-EVM global amplitude')
    axes[1, 1].bar(x+.17, phase, width=.32, color='#8662aa', label='Same-EVM global phase')
    axes[1, 1].axhline(0, color='gray', lw=1)
    axes[1, 1].set(xticks=x, xticklabels=[f'fold{i}' for i in x], ylabel='Mean signed teacher margin drop',
                   title='Controls: descriptive, not used to tune q')
    axes[1, 1].legend(fontsize=7)
    axes[1, 2].plot(x, [r['summary']['artifacts']['evm_mean'] for r in evidence], 'o-', label='Mean EVM', color='#2466a8')
    axes[1, 2].plot(x, [r['summary']['artifacts']['evm_q95'] for r in evidence], 's-', label='EVM q95', color='#d55e00')
    axes[1, 2].set(xticks=x, xticklabels=[f'fold{i}' for i in x], ylabel='Relative complex-signal distortion',
                   title='Fixed 50% band attenuation; circular filtering')
    axes[1, 2].legend(fontsize=8)
    fig.suptitle('S19 teacher evidence | source-training packets only | 8 fixed smooth bands', fontsize=13)
    fig.savefig(HERE/'EVIDENCE_DIAGNOSTICS.png', dpi=180)
    plt.close(fig)


def report(records, decision, evidence, costs, audit):
    comparisons = decision['comparisons']
    main, supervision = comparisons['B0'], comparisons['B3']
    checks_supervision = all(decision['checks']['B3'].values())
    conclusion = ('本轮通过预先固定的源域联合门槛，仅准入另行冻结的独立源RX/seed确认。尚无跨数据集或真实目标CRD收益结论。' if decision['admitted'] else
                  '本轮未通过预先固定的源域联合门槛，停止当前固定TCSE方案；不追加band、门控结构、调制强度或loss系数搜索，不启动条件扩展。')
    independent = ('B4相对同结构B3通过本轮操作性门槛，但该证据仍条件于单seed、固定teacher和已使用过的源域划分，不能写成统一跨RX方法已经成立。' if checks_supervision else
                   'B4相对同结构B3没有通过独立收益门槛，因此不能宣称新的TX证据监督带来正向收益。即便某个其他对照正向，也不改推其他实验臂。')
    lines = [
        '# S19：统一RF表征与TX频谱证据学习结果', '',
        f"源域决定：**{decision['status']}**。完成ManyRx四fold×五臂，16次新增正式训练、4个严格复用B0。", '',
        conclusion, '',
        f"主要比较B4−B0：fold平均{signed(main['mean_fold_gain_pp'])}pp，正向{main['positive_folds']}/4 folds，最差配对RX{signed(main['worst_paired_rx_pp'])}pp；按唯一RX身份汇总的均值{signed(main['unique_rx_mean_pp'])}pp，描述性条件95%区间{ci_text(main['ci95_pp'])}pp。", '',
        f"监督独立比较B4−B3：fold平均{signed(supervision['mean_fold_gain_pp'])}pp，正向{supervision['positive_folds']}/4 folds，条件95%区间{ci_text(supervision['ci95_pp'])}pp。{independent}", '',
        '![S19源域对照总览](S19_OVERVIEW.png)', '',
        '图表只描述本轮固定源域评估，不参与重新选参或换臂。Source validation是fit RX包留出，held-source是本fold训练时留出的源RX；两者均不是实际目标CRD。', '',
        '## 固定设计与实际执行', '',
        '本研究继承S18本地提交`933d4a6`，完整43个归档文件已核验；S00–S18有效证据、负结果和STOP未改，见[historical_matrix.md](historical_matrix.md)和[S18_VERIFIED.json](S18_VERIFIED.json)。S03的LoRa HSTU、S07 ManyRx和S08 ManySig历史成绩不是一个统一算法的最终成绩。', '',
        '| 臂 | 固定定义 | 参数 | 本轮训练角色 |',
        '|---|---|---:|---|',
    ]
    for arm in ARMS:
        cfg = arm_records(records, arm)[0]['config']
        role = '历史源域checkpoint严格复用；同时作为固定teacher' if arm == 'B0' else ('从匹配随机主干初值训练，CE+0.1证据CE' if arm == 'B4' else '从匹配随机主干初值训练，仅CE')
        lines.append(f"| {arm} | {ARM_NAMES[arm]} | {cfg['active_params']:,} | {role} |")
    lines += [
        '', 'B2/B3/B4均新增272参数，B3/B4同结构、同初值；仅B4增加监督。B1新增8个参数，是一般可学习频谱滤波的机制控制，不是FAN/ASKNet完整模型复现。普通频谱模块不能重新包装成原创贡献。', '',
        '每fold官方24个源RX中first6作held-source、余18个fit；日期1–2、10TX、256点、non-equalized、first100/TX/RX/day。fit70/30 split16001，seed18101及B0原data_seed，Adam1e-3/wd0/batch128/FP32/最多40epochs，Plateau(.1,5,min1e-6)、earlywait10。所有新臂匹配主干初值与每epoch包顺序，仅fit-RX等权验证CE选checkpoint/调LR，不用辅助损失选模。', '',
        '全部20个checkpoint及元数据在读取本轮held-source前封存并commit，见[seal_pilot.json](seal_pilot.json)。held-source仅在封存后一次性评价；同名源接收机曾被历史研究使用，本轮held-source不能称为历史上从未看过的独立RX。新确认分组/seed仍是后续独立阶段，不能由本次数据替代。', '',
        '## 预注册联合门槛逐项结果', '',
        '| 比较 | fold均值 ΔAcc pp | 正向fold | 最差配对RX pp | 唯一RX数 | 唯一RX均值 pp | 条件95%区间 pp | 门槛检查 |',
        '|---|---:|---:|---:|---:|---:|---|---|',
    ]
    for control in CONTROLS:
        c = comparisons[control]
        outcome = '；'.join(f"{key}={'PASS' if value else 'FAIL'}" for key, value in decision['checks'][control].items())
        lines.append(f"| B4−{control} | {signed(c['mean_fold_gain_pp'])} | {c['positive_folds']}/4 | {signed(c['worst_paired_rx_pp'])} | {c['unique_rx']} | {signed(c['unique_rx_mean_pp'])} | {ci_text(c['ci95_pp'])} | {outcome} |")
    lines += [
        '', 'B4−B0需fold均值≥+1pp、至少3/4正向、最差配对RX≥−2pp、条件CI下界>0。B4−B1/B2/B3各需fold均值>0、至少3/4正向及条件CI下界>0；所有条件共同通过才准入。只有B4预选为主要候选，未依据结果改推其他臂。机器决定见[decision_pilot.json](decision_pilot.json)。', '',
        'fold均值先在各fold对RX等权再平均，会重复计入跨fold出现的同一RX；bootstrap则先将同一RX跨fold的差值平均，再对唯一身份等权重抽样5000次（seed19191）。因此fold均值和唯一RX均值可能不同，不能将一种加权的点估计与另一种区间混说。该区间不消除共享训练集/模型依赖，也不覆盖训练随机种子不确定性；包、日期、TX和重复fold不是新增独立RX。', '',
        '## 所有实验臂的配对指标', '',
        '| 评估角色 | 臂 | 四fold平均Acc % | 四fold平均Macro-F1 % | CE | TX margin | fold最差RX准确率的平均 % |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for role, name in [('source_validation', 'fit-RX验证'), ('held_source', 'held-source')]:
        for arm in ARMS:
            v = mean_arm(records, arm, role)
            lines.append(f"| {name} | {arm} | {100*v['acc']:.4f} | {100*v['macro_f1']:.4f} | {v['ce']:.5f} | {v['margin']:.4f} | {100*v['worst_rx']:.4f} |")
    lines += ['', '上表最差RX列是“每fold最差RX准确率”的四fold平均，不能替代门槛中的所有配对RX变化最小值。逐fold全量如下。', '',
              '| fold | 臂 | 角色 | Acc % | Macro-F1 % | CE | TX margin | 最差RX % | 最差TX % |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        for role, label in [('source_validation', 'fit验证'), ('held_source', 'held源')]:
            v = domain_score(r, role)['summary']
            lines.append(f"| {r['fold']} | {r['arm']} | {label} | {100*v['acc']:.4f} | {100*v['macro_f1']:.4f} | {v['ce']:.5f} | {v['margin']:.4f} | {100*v['worst_rx']:.4f} | {100*v['worst_tx']:.4f} |")
    lines += ['', '每RX Macro-F1按模型声明的10个TX类别计算；原支持中的缺失类不被删除。TX真值仅用于训练损失和离线评分，分类前向只输入IQ。', '',
              '## 训练频谱证据、能量匹配对照与伪迹', '',
              '![训练源频谱证据与负对照](EVIDENCE_DIAGNOSTICS.png)', '',
              '| fold | 证据训练包 | 每TX源RX数范围 | q数值范围 | EVM均值 | EVM q95 | 平均保留谱能量 | 幅度控制Δmargin均值 | 相位控制Δmargin均值 | EVM匹配最大误差 |',
              '|---|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for r in evidence:
        s = r['summary']; a = s['artifacts']; c = s['controls']; q = np.asarray(s['q'])
        rx_range = f"{min(s['rx_support_per_tx'])}–{max(s['rx_support_per_tx'])}"
        error = max(a['amplitude_evm_max_error'], a['phase_evm_max_error'])
        lines.append(f"| {r['fold']} | {r['packets']} | {rx_range} | {q.min():.5f}–{q.max():.5f} | {a['evm_mean']:.5f} | {a['evm_q95']:.5f} | {a['retained_energy_mean']:.5f} | {c['amplitude_signed_mean']:+.6f} | {c['phase_signed_mean']:+.6f} | {error:.3g} |")
    q95 = [r['summary']['artifacts']['evm_q95'] for r in evidence]
    error_max = max(max(r['summary']['artifacts'][key] for key in ['amplitude_evm_max_error', 'phase_evm_max_error']) for r in evidence)
    tails = np.concatenate([np.asarray(r['summary']['artifacts']['residual_impulse_tail_fraction']) for r in evidence])
    lines += [
        '', 'q只来自固定teacher在fit训练70%中每非空TX/RX/day cell最多16包的输出。先cell均值、再对可观测day等权、最后跨有支持RX取线性Q0.25；缺失cell不插补、不删除整RX。不同TX的RX/day支持可以不同，完整counts/mask均保留在[evidence_f0.json](evidence_f0.json)及其他fold记录中。', '',
        'teacher已在这些源训练样本上训练，其checkpoint曾用历史源验证CE选择。因此q是该teacher依赖关系的代理，不是独立于选模的TX硬件真值；低分位数也不能保证去除了所有接收机捷径。B4没有额外held或目标监督，但确实使用了一个预训练teacher的输出，成本和信息来源单独报告。', '',
        f'证据估计固定削弱单band 50%，得到EVM q95范围{min(q95):.4f}–{max(q95):.4f}；训练门控的归一化前幅度调制仅±20%。两者扰动尺度不同，较大遮蔽下的teacher依赖不保证是局部调制下的最优证据。该差异作为设计限制披露，没有据此改参数或增删正式fit。', '',
        f'全谱幅度缩放和全局相位旋转与每包遮蔽EVM匹配，最大绝对匹配误差{error_max:.3g}。幅度控制检查总能量/归一化效应；相位控制检查非频段特异的相位敏感性，不能假设原Re/Im网络对全局相位完全不变。两种控制均只描述，没有从q中扣除、重选band或修改遮蔽强度。', '',
        f'遮蔽通过复数FFT→平滑频谱幅度mask→IFFT实现，是圆周滤波，可能有时域ringing；残差filter impulse在±4sample外的能量比例范围{tails.min():.4f}–{tails.max():.4f}。这不是识别RX硬件传递函数，也不证明无伪迹。热图为固定bar_q目标，不是目标域后验重要性图。', '',
        '所有动态/静态gain先在同一复谱上作用，再重算Re/Im、LogMag和相位valid mask；±20%界只约束归一化前的谱幅度gain，不适用于重新归一化后的每个RF通道。生产推理用Parseval相对RMS校正避免IFFT往返，teacher遮蔽则保留显式IFFT以保持反事实路径清楚。', '',
        '## 参数、推理路径与训练成本', '',
    ]
    cost_rows = {r.get('arm'):r for r in costs.get('arms', costs.get('rows', []))}
    lines += ['| 臂 | 参数 | 相对B0新增 | 部分算术估计/包 | batch1中位延迟 ms | batch128中位延迟 ms |',
              '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        r = cost_rows.get(arm, {})
        params = r.get('parameters', arm_records(records, arm)[0]['config']['active_params'])
        arithmetic = r.get('arithmetic_proxy', {}).get('partial_arithmetic_subtotal')
        latency = {v.get('batch'):v for v in r.get('latency', [])}
        fmt_latency = lambda n: f"{latency[n]['median_batch_ms']:.4f}" if n in latency and 'median_batch_ms' in latency[n] else '见COSTS.json'
        arithmetic_text = f'{arithmetic:,}' if arithmetic is not None else '见COSTS.json'
        lines.append(f"| {arm} | {params:,} | {params-82426:,} | {arithmetic_text} | {fmt_latency(1)} | {fmt_latency(128)} |")
    training = costs.get('training_costs', {})
    new_seconds = training.get('new_fit_seconds', sum(r['fitted'].get('wall_seconds', 0.) for r in records if r['arm'] != 'B0'))
    teacher_seconds = training.get('historical_reused_fit_seconds', sum(r['fitted'].get('historical_teacher_fit_seconds', 0.) for r in records if r['arm'] == 'B0'))
    evidence_seconds = training.get('teacher_evidence_inference_seconds', sum(r['seconds'] for r in evidence))
    lines += [
        '', f'16个新增fit累计计时{new_seconds:.2f}s（{new_seconds/60:.2f}min）；4个复用B0/teacher历史训练累计{teacher_seconds:.2f}s，按同一checkpoint只计一次；本轮teacher证据推理累计{evidence_seconds:.2f}s。若从零构建本研究，需同时计入teacher训练，不能只计新增272参数或B4推理开销。计时不等于完整阶段墙钟时间，封存、held评价、复核和报告等未完全包含。', '',
        '每包证据估计有1次clean加8×3次teacher forward，即25次；这些只在训练证据构造时发生，推理不运行teacher。各臂训练epoch、早停点、日志和配置完整保留，并没有把较早停止的臂补训或重复试种子。', '',
        '推理路径包含原RMS、一次FFT256、band描述子/有界gain、复谱调制后全部五通道重建、HStem、普通两层MHA和分类头。B3/B4另需一次用于phase描述子的原五通道构造，不能只用参数数目推断真实成本。', '',
        f"[COSTS.json](COSTS.json)保存完整路径、算子清单、计时样本和SHA来源。实测设备为{costs.get('device', '见COSTS.json')}，FP32、固定合成输入，计时包含Python dispatch和CUDA同步。部分算术subtotal只覆盖声明的Conv/Linear/attention/频带矩阵乘积和FFT代理；未计齐norm、log1p、softmax、elementwise等，**不能称完整总FLOPs**。算子profile只作清单，不能补成完整总FLOPs。", '',
        'B3/B4架构相同，其延迟数值差异是本机单次benchmark的描述性波动，不能归因于架构差异。', '',
        '## 访问审计、验证范围与后续出口', '',
        f"[FINAL_ACCESS_AUDIT.json](FINAL_ACCESS_AUDIT.json)状态为{audit['status']}。从保存的逐包teacher/source/held logits重新计算q、CE、margin、Accuracy、Macro-F1、最差RX/TX和配对门槛；核对主干初值SHA、B3/B4完整初值、共用epoch顺序、checkpoint/代码/协议SHA及held读取晚于封存。审计本身不再运行模型或打开原始IQ。", '',
        '本轮没有访问真实目标RX/date3–4，没有训练或评估ManySig、Shen TrueIB、Shen blind、额外seed或更换源RX确认分组。S10官方公平重训IB-RFF/ASKNet结果保留在历史矩阵中，不能拿本轮held-source准确率直接声称超过其CRD结果。', '',
        '已核对的LoRa统一基线为Shen TrueIB+HStem+MHA：Fs1MHz、BW125kHz、8192点、|f|≤62.5kHz，133,690参数及原legacy5输入。WiSig使用Fs25MHz、256点和原频域五通道；本轮没有把LoRa裁频界复制到WiSig，也没有将S03 HSTU成绩替代MHA基线。LoRa适配仍待源域准入后独立验证。', '',
        ('源域GO只允许另行冻结至少3个seed与source-last分组确认；之后才按同一机制及主要超参数扩展ManySig，再核对Shen TrueIB+HStem+MHA。真实CRD需最后单独冻结协议，不能从目标成绩返回选参。' if decision['admitted'] else
         '按预注册STOP，不开展source-last/新seed确认、ManySig或Shen训练，不重新搜索gate/lambda/band。保留HStem已有有效证据和本轮负结果，不能将未通过的TCSE候选写成论文贡献。'), '',
        '## 复核入口', '',
        '- [设计与算法定义](README.zh-CN.md)、[历史矩阵](historical_matrix.md)、[预注册](registration.json)',
        '- [全部配对比较CSV](PILOT_COMPARISONS.csv)、[机器决定](decision_pilot.json)、[checkpoint封存](seal_pilot.json)',
        '- [训练证据fold0](evidence_f0.json)、其余evidence_f1–3.json，以及runs/下全部配置、曲线、日志、源/held混淆矩阵',
        '- [成本](COSTS.json)、[最终审计](FINAL_ACCESS_AUDIT.json)、[归档SHA](MANIFEST.json)', '',
        '原始IQ、模型权重、逐包预测及证据数组只保留本地并由SHA引用；Git归档包含代码、设计、聚合统计、日志、配置、checkpoint引用、决策和图表。派生报告不参与选模。',
    ]
    (HERE/'RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')


def manifest(records, decision):
    allowed = {'.json', '.jsonl', '.md', '.csv', '.py', '.png', '.log'}
    paths = []
    for path in HERE.rglob('*'):
        if not path.is_file() or path.name == 'MANIFEST.json' or any(part in {'cache', '__pycache__'} for part in path.relative_to(HERE).parts):
            continue
        if path.suffix in allowed or path.name in {'.gitignore', '.gitattributes'}:
            paths.append(path)
    write(HERE/'MANIFEST.json', dict(time=now(), status=decision['status'],
          files_sha256={path.relative_to(HERE).as_posix():sha(path) for path in sorted(paths)},
          local_checkpoints={job_id(r['fold'], r['arm']):r['fitted']['model_sha256'] for r in records},
          local_predictions={job_id(r['fold'], r['arm']):dict(source=r['fitted']['prediction_sha256'], held=r['held']['prediction_sha256']) for r in records},
          local_evidence_caches={str(f):read(HERE/f'evidence_f{f}.json')['cache_sha256'] for f in range(4)},
          excluded='raw IQ, tensor checkpoints, per-packet logits/evidence arrays, cache and Python bytecode; all retained locally by hash',
          primary_results_selected_by_report=False))


def run():
    decision = read(HERE/'decision_pilot.json')
    audit = read(HERE/'FINAL_ACCESS_AUDIT.json')
    if audit['status'] != 'PASS' or audit['decision_status'] != decision['status']:
        raise ValueError('Passing final QA and matching source decision required before reports')
    records = load_records()
    evidence = [read(HERE/f'evidence_f{fold}.json') for fold in range(4)]
    costs = read(HERE/'COSTS.json')
    if costs.get('status') not in {'COMPLETE', 'PASS'}:
        raise ValueError('Derived cost benchmark must finish before reporting')
    comparison_csv(decision)
    plot_overview(records, decision)
    plot_evidence(evidence)
    report(records, decision, evidence, costs, audit)
    write(HERE/'STATUS.json', dict(time=now(), stage='S19', status=decision['status'], admitted=decision['admitted'],
          primary='B4', evidence_supervision_supported=decision['evidence_supervision_supported'],
          completed_source_comparisons=20, formal_new_fits=16, reused_B0=4, maximum_new_fits=20,
          dataset='ManyRx', folds=4, seeds=1, real_target_access=False, held_source_after_seal=True,
          independent_source_confirmation_started=False, ManySig_read=False, Shen_read=False,
          audit='PASS', report_is_derived=True))
    manifest(records, decision)
    print(decision['status'], 'derived report, CSV, figures and manifest generated', flush=True)


if __name__ == '__main__': run()
