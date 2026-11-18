"""Render existing frozen report statistics; no training or waveform access."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
NAMES = {'h0': 'HStem (frozen)', 'ce': 'Shen CE', 'dann': 'Shen DANN',
         'fan': 'FAN / IB stage 1', 'ib': 'IB-RFF stage 2', 'ask': 'ASKNet'}
METHODS = list(NAMES)
COLORS = {'h0': '#1b6b8f', 'ce': '#727984', 'dann': '#905b9c',
          'fan': '#bd7c39', 'ib': '#329575', 'ask': '#c15a63'}


def save(figure, name):
    for extension in ('png', 'svg'):
        figure.savefig(OUT / f'{name}.{extension}', dpi=250, facecolor='white')
    plt.close(figure)


def main():
    result = json.loads((HERE / 'RESULTS.json').read_text(encoding='utf-8'))
    cells = []
    for path in sorted((HERE / 'evaluations').glob('*/metrics.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        cells.extend(record['rows'])
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False})
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 5.0), sharey=True)
    for axis, view in zip(axes, ('native', 'strict_ib')):
        for index, method in enumerate(METHODS):
            group = next(r for r in result['summary'] if r['view'] == view and r['method'] == method)
            if group['status'] != 'COMPLETE_REPORT_ONLY':
                label = 'Deferred by budget' if view == 'strict_ib' else 'Pending full cohort'
                axis.text(3, index, label, va='center', color='#737373', style='italic')
                continue
            selected = [r for r in cells if r['method'] == method and r['view'] == ('native' if method == 'h0' else view)]
            fits = [100 * np.mean([r['acc'] for r in selected if r['fold'] == f and r['seed'] == s])
                    for f in (0, 1) for s in (201, 202, 203)]
            axis.scatter(fits, index + np.linspace(-0.12, 0.12, 6), s=24,
                         facecolors='none', edgecolors=COLORS[method], zorder=3)
            axis.scatter(group['acc'] * 100, index, s=44, marker='D', color=COLORS[method], zorder=4)
            axis.text(101, index, f"{group['acc'] * 100:.3f}", va='center', ha='left', fontsize=9,
                      color=COLORS[method], clip_on=False)
        axis.set(xlim=(0, 100), ylim=(5.6, -0.6), xlabel='RX-equal packet accuracy (%)',
                 title='Native method pipeline' if view == 'native' else 'Strict-IB information control')
        axis.grid(axis='x', color='#e2e5e8', linewidth=0.6)
        axis.set_yticks(range(6), [NAMES[m] for m in METHODS])
    figure.suptitle('LoRa cross-RX comparison on historically developed Dev receivers', y=0.98)
    figure.text(0.02, 0.04, 'Open circles: six fold/seed fits; diamond: mean. All complete baseline/HStem results use CPU FP32.', fontsize=9)
    figure.text(0.02, 0.01, 'FAN checkpoints are reused stage 1. ASK Native appears only after all six models are evaluated; Strict-IB ASK is deferred.', fontsize=8)
    figure.subplots_adjust(left=0.17, right=0.94, bottom=0.17, top=0.87, wspace=0.28)
    save(figure, 'accuracy_comparison')

    comparisons = [('h0', m) for m in ('ce', 'dann', 'fan', 'ib', 'ask')] + [('dann', 'ce'), ('ib', 'fan')]
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.6), sharey=True)
    for axis, view in zip(axes, ('native', 'strict_ib')):
        for index, (aa, bb) in enumerate(comparisons):
            pair = next((p for p in result['paired_comparisons'] if p['view'] == view and
                         p['a'] == aa and p['b'] == bb and p['metric'] == 'acc'), None)
            if pair is None:
                axis.text(0.03, index, 'Pending / deferred', transform=axis.get_yaxis_transform(),
                          va='center', color='#737373', style='italic')
                continue
            delta = pair['delta'] * 100
            lower, upper = np.asarray(pair['conditional_rx_cluster_ci']) * 100
            axis.plot([lower, upper], [index, index], color=COLORS[aa], linewidth=2)
            axis.scatter([delta], [index], s=35, color=COLORS[aa], zorder=3)
        axis.axvline(0, color='#737373', linestyle='--', linewidth=0.8)
        axis.set(ylim=(6.6, -0.6), xlabel='Accuracy difference (percentage points)', title=view)
        axis.grid(axis='x', color='#e2e5e8', linewidth=0.6)
        axis.set_yticks(range(7), [f"{NAMES[a]} minus {NAMES[b]}" for a, b in comparisons])
    limits = [axis.get_xlim() for axis in axes]
    for axis in axes:
        axis.set_xlim(min(v[0] for v in limits), max(v[1] for v in limits))
    figure.suptitle('All preregistered paired accuracy comparisons', y=0.98)
    figure.text(0.02, 0.04, '95% conditional intervals: bootstrap 10 unique RX after averaging three seeds; 10,000 replicates, fixed seed 24016.', fontsize=9)
    figure.text(0.02, 0.01, 'No multiplicity adjustment. Historically developed RX; intervals do not establish new independent generalization.', fontsize=9)
    figure.subplots_adjust(left=0.29, right=0.97, bottom=0.16, top=0.88, wspace=0.16)
    save(figure, 'paired_differences')


if __name__ == '__main__':
    main()
