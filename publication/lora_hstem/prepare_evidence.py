"""Prepare manuscript tables/figures from frozen aggregates only; no IQ or weights."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
S23=Path('D:/crossatten/worktrees/s23-mechanism-evidence/23-hstem-mechanism-evidence')
S22=S23.parent/'22-mechanism-closure'
S15=Path('D:/crossatten/worktrees/s15-hstem-attribution/15-hstem-attribution')
LABELS={'H0':'HStem','P0':'Patch-Lite','H1':'Overlap-Patch','H2':'NoMidBN','S1':'Smooth-RoundTrip','S2':'AffineOnly'}
ORDER=['P0','S1','H1','H2','S2','H0']
PROVENANCE=[]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def copy_file(p,name=None):
    p=Path(p);target=HERE/'data'/(name or p.name);shutil.copyfile(p,target)
    PROVENANCE.append(dict(source=str(p),destination=target.relative_to(HERE).as_posix(),sha256=sha(p)))
    return target


def csvrows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def tex(name,text):
    (HERE/'tables'/name).write_text(text,encoding='utf8')


def numeric(value,digits=3):return f'{float(value):.{digits}f}'


def main():
    for folder in ('data','tables','figures'): (HERE/folder).mkdir(parents=True,exist_ok=True)
    for name in ('main_ablation_table.csv','paired_contrasts.csv','cost_table.csv','receiver_metrics.csv','tx_rx_deltas.csv','robustness_source.csv','historical_mixer_patch_anchors.csv','historical_shen_external_methods.csv'):
        copy_file(S22/'paper'/name)
    for name in ('equivalence.csv','precision_equivalence.csv','timing_summary.csv','conversion_latency.csv','OPERATOR.json','PAPER_OVERLAP_REVIEW.zh-CN.md','MATHEMATICAL_SCOPE.zh-CN.md','RESEARCH_INTERPRETATION.zh-CN.md','FINAL_AUDIT.json','registration.json'):
        copy_file(S23/name,'s23_'+name)
    copy_file(S15/'summary.json','s15_summary.json')
    copy_file(ROOT/'configs/protocol.json','protocol.json')
    copy_file(S22/'costs.json','s22_costs.json')
    registration=json.loads((S23/'registration.json').read_text(encoding='utf8'))
    copy_file(Path(registration['references']['H0_f0_s201']['path'])/'config.json','baseline_training_config.json')
    # Include the actual model and front-end definitions for component-level review.
    code_paths=['lib/xa/frontends.py','lib/xa/models3.py','lib/xa/models2.py','lib/legacy_rfhstu/features.py',
        'lib/legacy_rfhstu/cnn_stem.py','lib/legacy_rfhstu/models.py','13-mixer-cfo-mechanisms/models.py','13-mixer-cfo-mechanisms/train.py',
        '15-hstem-attribution/s15_models.py']
    for relative in code_paths:copy_file(ROOT/relative,'code_'+relative.replace('/','__'))
    for relative in ('21-hstem-mechanisms/s21_models.py','22-mechanism-closure/s22_models.py','22-mechanism-closure/s22_train.py'):
        copy_file(S23.parent/relative,'code_'+relative.replace('/','__'))
    m=csvrows(HERE/'data/main_ablation_table.csv');cost=csvrows(HERE/'data/cost_table.csv')
    lookup={(r['arm'],r['role']):r for r in m}
    s15=json.loads((HERE/'data/s15_summary.json').read_text(encoding='utf8'))['arm_summary']
    patch=next(r for r in s15 if r['arm']=='E1_patch_mha' and r['role']=='dev_val')
    patchh=next(r for r in s15 if r['arm']=='E1_patch_mha' and r['role']=='confirm')
    lines=[r'\begin{table}[t]',r'\centering\caption{Consolidated MHA results. Dev: development RX; Hist.: historical confirmation with prior exposure. SD is across three seed means, each averaged over folds.}',r'\label{tab:main}',r'\setlength{\tabcolsep}{3pt}\footnotesize',r'\begin{tabular}{lrrrr}',r'\toprule Model & Params & Dev (\%) & SD & Hist. (\%)\\\midrule',
        f"Full Patch & 571,434 & {numeric(patch['acc'])} & {numeric(patch['seed_sd_pp'])} & {numeric(patchh['acc'])}\\\\"]
    params={r['arm']:int(r['parameters']) for r in cost}
    for arm in ORDER:
        a,b=lookup[arm,'dev_val'],lookup[arm,'confirm']
        lines.append(f"{LABELS[arm]} & {params[arm]:,} & {numeric(a['acc'])} & {numeric(a['seed_sd_pp'])} & {numeric(b['acc'])}\\\\")
    lines += [r'\bottomrule\end{tabular}',r'\end{table}']
    tex('main_results.tex','\n'.join(lines)+'\n')
    contrasts=csvrows(HERE/'data/paired_contrasts.csv')
    lines=[r'\begin{table}[t]',r'\centering\caption{Four planned development contrasts; intervals cluster 10 RX and condition on the fitted seeds. Pos. denotes positive fold/seed pairs out of six.}',r'\label{tab:contrasts}',r'\setlength{\tabcolsep}{3pt}\footnotesize',r'\begin{tabular}{lrrrc}',r'\toprule Contrast & $\Delta$ pp & RX 95\% interval & Pos. & Signal\\\midrule']
    short={'H1-S1':'Overlap -- Smooth','S1-P0':'Smooth -- Patch-Lite','H0-S2':'HStem -- Affine','S2-H2':'Affine -- NoMidBN'}
    for r in contrasts:
        if r['role']!='dev_val':continue
        lo,hi=ast.literal_eval(r['ci95_pp'])
        lines.append(f"{short[r['contrast']]} & {float(r['mean_delta_pp']):+.3f} & [{lo:+.3f}, {hi:+.3f}] & {r['positive_pairs']}/6 & {'Yes' if r['practical_signal']=='True' else 'No'}\\\\")
    lines += [r'\bottomrule\end{tabular}',r'\end{table}'];tex('contrasts.tex','\n'.join(lines)+'\n')
    lines=[r'\begin{table}[t]',r'\centering\caption{Recorded same-GPU forward costs. MACs incompletely cover FFT/interpolation/elementwise work. Latency is resident-input p50, without I/O.}',r'\label{tab:costs}',r'\setlength{\tabcolsep}{4pt}\footnotesize',r'\begin{tabular}{lrrr}',r'\toprule Model & MACs (M) & Batch 1 (ms) & Batch 128 (ms)\\\midrule']
    for arm in ORDER:
        r=next(r for r in cost if r['arm']==arm)
        lines.append(f"{LABELS[arm]} & {float(r['counted_macs'])/1e6:.3f} & {numeric(r['batch1_p50_ms'])} & {numeric(r['batch128_p50_ms'])}\\\\")
    lines += [r'\bottomrule\end{tabular}',r'\end{table}'];tex('costs.tex','\n'.join(lines)+'\n')
    timing=csvrows(HERE/'data/s23_timing_summary.csv')
    td={(r['arm'],r['condition']):float(r['accuracy_pct']) for r in timing}
    robustness=csvrows(HERE/'data/robustness_source.csv')
    # Noise is from S22, timing from S23; retain the roles and separate provenance.
    noise={}
    for r in robustness:
        cond=r.get('condition','')
        if 'awgn' in cond.lower():noise[(r['arm'],cond)]=r
    lines=[r'\begin{table}[t]',r'\centering\caption{Frozen source-validation digital-shift tests (\%). No receiver-generalization or physical synchronization claim is inferred from these interventions.}',r'\label{tab:stress}',r'\setlength{\tabcolsep}{6pt}\footnotesize',r'\begin{tabular}{lrr}',r'\toprule Condition & HStem & Patch-Lite\\\midrule']
    for cond,label in [('clean','Clean'),('circular_-1','Circular $-1$'),('circular_+1','Circular $+1$'),('circular_-4','Circular $-4$'),('circular_+4','Circular $+4$'),('zero_-4','Zero-filled $-4$'),('zero_+4','Zero-filled $+4$'),('circular_-4_fft_locked','Circular $-4$, FFT locked'),('circular_+4_fft_locked','Circular $+4$, FFT locked')]:
        lines.append(f"{label} & {td['H0',cond]:.3f} & {td['P0',cond]:.3f}\\\\")
    lines += [r'\bottomrule\end{tabular}',r'\end{table}'];tex('stress.tex','\n'.join(lines)+'\n')
    # Reproducible final-scope figures. No historical Chinese plot is relabeled.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11.5,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(4.8,3.5),layout='constrained');x=np.arange(6)
    ax.bar(x-.18,[float(lookup[a,'dev_val']['acc']) for a in ORDER],width=.36,label='Development RX',color='#245b83')
    ax.bar(x+.18,[float(lookup[a,'confirm']['acc']) for a in ORDER],width=.36,label='Historical confirmation',color='#9ab6ca')
    ax.set_xticks(x,['Patch-\nLite','Smooth\nRT','Overlap\nPatch','NoMid\nBN','Affine\nonly','HStem'])
    ax.set_ylabel('RX-equal accuracy (%)');ax.set_ylim(90,100);ax.legend(frameon=False,fontsize=11);ax.grid(axis='y',alpha=.16)
    fig.savefig(HERE/'figures/mechanism.png',dpi=300);fig.savefig(HERE/'figures/mechanism.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(13.6,3.4));ax.set(xlim=(0,14),ylim=(0,3.4));ax.axis('off')
    def box(x,y,w,h,text,color='#e8f0f5'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',linewidth=.8,edgecolor='#35586e',facecolor=color))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=13)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,color='#35586e',lw=1))
    box(.1,1.25,1.45,1,'Complex IQ\n8,192 samples')
    box(1.95,1.25,1.8,1,'Inherited front-end\nRMS + strict IB\n5 feature channels')
    box(4.15,1.25,1.45,1,'Shared CNN\nk7, k5\n32 channels')
    box(6.05,.4,4.05,2.6,'Token construction (one alternative)\n\nFull Patch: k256 / s256\nPatch-Lite: grouped k256 + 1x1\nHStem: (k8 / s4) x3, k4 / s4\n\nCommon output: 32 x 64',color='#e3eee5')
    box(10.55,1.25,1.5,1,'Position + Pre-LN\n2 MHA blocks\n4 heads / 64 dim')
    box(12.45,1.25,1.4,1,'Final LN\nMean pooling\n10-class head')
    for a,b in [((1.55,1.75),(1.95,1.75)),((3.75,1.75),(4.15,1.75)),((5.6,1.75),(6.05,1.75)),((10.1,1.75),(10.55,1.75)),((12.05,1.75),(12.45,1.75))]:arrow(a,b)
    ax.text(7,.03,'No OOB branch. Shared FFT/temporal feature coordinates do not imply physical time-local support.',ha='center',fontsize=12,color='#344955')
    fig.tight_layout(pad=.2);fig.savefig(HERE/'figures/architecture.png',dpi=300,bbox_inches='tight');fig.savefig(HERE/'figures/architecture.svg',bbox_inches='tight');plt.close(fig)
    # Receiver aggregates and local-risk summaries for supplementary review.
    receiver=csvrows(HERE/'data/receiver_metrics.csv');ragg=[]
    for role in ('dev_val','confirm'):
        for rx in sorted({r['rx'] for r in receiver if r['role']==role}):
            row=dict(role=role,rx=rx)
            for a in ORDER:row[a]=float(np.mean([float(r['acc']) for r in receiver if r['role']==role and r['rx']==rx and r['arm']==a]))
            ragg.append(row)
    with (HERE/'data/receiver_aggregates.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ragg[0]));w.writeheader();w.writerows(ragg)
    lines=[r'\begin{longtable}{llrrrrrr}',r'\caption{Accuracy by receiver, averaged over the applicable fitted repeats (\%). Dev receivers are held out from their evaluated fit; historical confirmation is previously exposed.}\label{tab:rx}\\',r'\toprule Role & RX & Patch-Lite & Smooth & Overlap & NoMidBN & Affine & HStem\\\midrule\endfirsthead',r'\toprule Role & RX & Patch-Lite & Smooth & Overlap & NoMidBN & Affine & HStem\\\midrule\endhead']
    for r in ragg:lines.append(('Dev' if r['role']=='dev_val' else 'Hist.')+' & '+r['rx'].replace('_',r'\_')+' & '+' & '.join(f"{r[a]:.3f}" for a in ORDER)+r'\\')
    lines += [r'\bottomrule\end{longtable}'];tex('receivers.tex','\n'.join(lines)+'\n')
    # Exact same-stage mixer comparisons; do not substitute historical best HSTU.
    lines=[r'\begin{tabular}{llrrr}',r'\toprule Mixer & Tokenizer & Parameters & Dev (\%) & Hist. (\%)\\\midrule']
    p15={'E1_patch_mha':571434,'C2_lite_mha':134266,'A1_mha':133690,'E2_patch_hstu':580136,'C2_lite_hstu':142968,'A3_rfhstu':142392}
    for a,label,mixer in [('E1_patch_mha','Full Patch','MHA'),('C2_lite_mha','Patch-Lite','MHA'),('A1_mha','HStem','MHA'),('E2_patch_hstu','Full Patch','RF-HSTU'),('C2_lite_hstu','Patch-Lite','RF-HSTU'),('A3_rfhstu','HStem','RF-HSTU')]:
        d=next(r for r in s15 if r['arm']==a and r['role']=='dev_val');h=next(r for r in s15 if r['arm']==a and r['role']=='confirm')
        lines.append(f"{mixer} & {label} & {p15[a]:,} & {d['acc']:.3f} & {h['acc']:.3f}\\\\")
    lines += [r'\bottomrule\end{tabular}'];tex('mixers.tex','\n'.join(lines)+'\n')
    ext=csvrows(HERE/'data/historical_shen_external_methods.csv')
    models=sorted({r['model'] for r in ext if r['model'].startswith(('FAN','IB-RFF','WaveMLP'))})
    lines=[r'\begin{longtable}{p{8.4cm}rrr}',r'\caption{Historical same-corpus external-method results. Complete recorded recipes are retained; these are not newly paired tokenizer fits. Accuracy is RX-equal (\%).}\\',r'\toprule Recorded configuration & Dev & Hist. & Parameters\\\midrule\endfirsthead',r'\toprule Recorded configuration & Dev & Hist. & Parameters\\\midrule\endhead']
    for model in models:
        d=next(r for r in ext if r['model']==model and r['task']=='dev_val');h=next(r for r in ext if r['model']==model and r['task']=='confirm')
        lines.append(model.replace('_',r'\_')+f" & {float(d['rx_accuracy_pct']):.3f} & {float(h['rx_accuracy_pct']):.3f} & {int(d['parameters']):,}\\\\")
    lines += [r'\bottomrule\end{longtable}'];tex('historical_external.tex','\n'.join(lines)+'\n')
    # Copy provenance before deriving checks; all source files remain unmodified.
    checks={'all_development_means_reproduced':all(abs(np.mean([r[a] for r in ragg if r['role']=='dev_val'])-float(lookup[a,'dev_val']['acc']))<1e-9 for a in ORDER),
        'all_historical_means_reproduced':all(abs(np.mean([r[a] for r in ragg if r['role']=='confirm'])-float(lookup[a,'confirm']['acc']))<1e-9 for a in ORDER),
        'HStem_parameter_count':params['H0']==133690,'matched_control_parameter_count':params['P0']==134266,
        'all_source_copies_exact':all(sha(HERE/r['destination'])==r['sha256'] for r in PROVENANCE)}
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,source_files=PROVENANCE,new_fits=0,IQ_reads=0,checkpoint_loads=0)
    (HERE/'EVIDENCE_MANIFEST.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    if not all(checks.values()):raise ValueError('Aggregate consistency failed')
    print('PASS',len(PROVENANCE),'exact source copies; no IQ or checkpoint reads',flush=True)


if __name__=='__main__':main()
