"""Standalone research figures from sealed summary tables, no new model scoring."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
colors={'B0':'#6b7280','B1':'#3b82b5','B2':'#d17a22','N0':'#7b609d'}


def main():
    with (HERE/'W1_RECEIVERS.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    rows=[r for r in rows if r['feature']=='q20_mean'];names=sorted({r['rx'] for r in rows})
    fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained')
    for model,offset,color,label in [('F0',-.13,colors['B1'],'Frozen frequency model'),('E0',.13,colors['B2'],'Frozen EMA reference')]:
        delta=[float(next(r['difference_pp'] for r in rows if r['model']==model and r['rx']==name)) for name in names]
        ax.scatter(delta,np.arange(len(names))+offset,s=35,color=color,label=label)
    ax.axvline(0,color='#374151',lw=.8);ax.set_yticks(np.arange(len(names)),names);ax.invert_yaxis()
    ax.set_xlabel('Low-repeatability quartile minus remaining error rate (pp)')
    ax.set_ylabel('Source-held receiver');ax.grid(axis='x',alpha=.18);ax.legend(loc='lower right')
    ax.set_title('W1: error association within TX / RX / day\nThree streams averaged; no packet rejection')
    for ext in ('png','svg'):fig.savefig(HERE/f'W1_QUALITY_ASSOCIATION.{ext}',dpi=200)
    plt.close(fig)
    if not (HERE/'W2_RESULT.json').exists():return
    r=json.loads((HERE/'W2_RESULT.json').read_text(encoding='utf8'))
    totals=r['totals'];arms=['B0','B1','B2','N0'];labels=['EMA base','Feature gate','Quality gate','Shuffled quality']
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    clean=[next(t['accuracy_pct'] for t in totals if t['arm']==a and t['condition']=='clean' and t['subset']=='all') for a in arms]
    axes[0].bar(np.arange(4),clean,color=[colors[a] for a in arms],width=.65)
    axes[0].set_xticks(np.arange(4),labels,rotation=25,ha='right');axes[0].set_ylim(0,100);axes[0].set_ylabel('RX-equal accuracy (%)')
    axes[0].set_title('Clean source-held evaluation')
    for i,value in enumerate(clean):axes[0].text(i,value+1,f'{value:.2f}',ha='center',fontsize=9)
    cc=r['comparisons']
    for i,c in enumerate(cc):
        lo,hi=c['conditional_rx_ci95_pp'];d=c['mean_delta_pp']
        axes[1].errorbar(d,i,xerr=np.array([[d-lo],[hi-d]]),fmt='o',color=colors['B2'],capsize=4)
    axes[1].axvline(0,color='#374151',lw=.8);axes[1].set_yticks(np.arange(3),['B2 - '+c['control'] for c in cc]);axes[1].invert_yaxis()
    axes[1].set_xlabel('Accuracy difference (pp)');axes[1].set_title('95% conditional RX-cluster intervals');axes[1].grid(axis='x',alpha=.2)
    for a,label in zip(arms,labels):
        values=[next(t['accuracy_pct'] for t in totals if t['arm']==a and t['condition']==c and t['subset']=='all') for c in ['clean','noise20','noise10','noise0']]
        axes[2].plot(np.arange(4),values,'o-',color=colors[a],label=label)
    axes[2].set_xticks(np.arange(4),['Clean','+20 dB','+10 dB','+0 dB']);axes[2].set_ylim(0,100);axes[2].set_ylabel('RX-equal accuracy (%)')
    axes[2].set_title('Added-noise ratio to observed power');axes[2].legend(fontsize=8);axes[2].grid(alpha=.18)
    fig.suptitle('W2: matched end-to-end gates, EMA selected; 18 new fits\nExisting development receivers, full packet coverage',fontsize=12)
    for ext in ('png','svg'):fig.savefig(HERE/f'W2_COMPARISON.{ext}',dpi=200)
    plt.close(fig)


if __name__=='__main__':main()
