"""Reproducible figures/tables from fixed evidence, including counterexamples."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
from s25_common import HERE, ROOT, P, digest, read, write

PAPER=HERE/'paper'; FIG=PAPER/'figures'; TABLE=PAPER/'tables'
COLORS={'H0':'#1b6b8f','P0':'#747b87','S1':'#309a83','H1':'#be8641',
        'h0':'#1b6b8f','ce':'#747b87','dann':'#9564a7','fan':'#be8641','ib':'#309a83','ask':'#bd5962'}
NAMES={'h0':'HStem','ce':'Shen CE','dann':'Shen DANN','fan':'FAN','ib':'IB-RFF','ask':'ASKNet'}

def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',pad_inches=.035)
    fig.savefig(FIG/(name+'.png'),bbox_inches='tight',pad_inches=.035,dpi=230)
    plt.close(fig)

def box(ax,x,y,w,h,text,color='#e5edf3',size=8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',facecolor=color,edgecolor='#617182',linewidth=.7,clip_on=False))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)

def arrow(ax,a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,linewidth=.8,color='#4c5c6b'))

def figures():
    FIG.mkdir(parents=True,exist_ok=True);TABLE.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    f,ax=plt.subplots(figsize=(7.0,2.25));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    for x,text in [(.015,'10 enrolled TX\nfixed device IDs'),(.27,'5 source RX / fold\n160 train/TX/RX\n40 val/TX/RX'),(.535,'Source validation\nselect checkpoint\nthen freeze'),(.80,'Six paired fits\n2 folds x 3 seeds')]:
        box(ax,x,.65,.18,.25,text,size=7.5)
    for x in [.20,.455,.72]:arrow(ax,(x,.775),(x+.065,.775))
    for x,text in [(.015,'Different RX\nsame enrolled TX'),(.27,'8192 aligned IQ\nTrueIB\nfive RF views'),(.535,'Frozen HStem\nordinary MHA'),(.80,'10-way ID\nsingle packet')]:
        box(ax,x,.15,.18,.27,text,'#f0f4f7',size=7.5)
    for x in [.20,.455,.72]:arrow(ax,(x,.285),(x+.065,.285))
    arrow(ax,(.625,.64),(.625,.43))
    ax.text(.5,.015,'No target adaptation. Closed-set identification; no unknown-TX rejection or attack-resistance claim.',ha='center',fontsize=7.3)
    save(f,'fig01_scope')

    f,ax=plt.subplots(figsize=(7,2.55));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,.01,.33,.17,.34,'Shared front-end\nConv k7 / k5\n32 channels\nBN + LeakyReLU',size=7.6)
    box(ax,.245,.63,.38,.28,'HStem: (kernel, stride, channels)\n(8,4,48) -> (8,4,64)\n(8,4,64) -> (4,4,64)\nBN + GELU after first three stages',size=7.1)
    box(ax,.245,.13,.38,.28,'Patch-Lite\nk256 / s256, groups8, width80\npointwise 80 -> 64\nSingle spatial step',color='#f0f0f1',size=7.2)
    arrow(ax,(.18,.5),(.24,.76));arrow(ax,(.18,.5),(.24,.27))
    box(ax,.705,.33,.28,.34,'32 x 64 tokens\nposition / pre-LN\n2 ordinary MHA blocks\nfinal LN / mean / Linear')
    arrow(ax,(.63,.76),(.70,.52));arrow(ax,(.63,.27),(.70,.49))
    ax.text(.435,.51,'Total parameters: 133,690 / 134,266',ha='center',fontsize=7.1)
    ax.text(.5,.015,'Local feature-coordinate support: 350 vs 266. Global FFT/TrueIB prevent a physical receptive-field interpretation.',ha='center',fontsize=7)
    save(f,'fig02_architecture')

    geo=read(HERE/'MECHANISM_EXTENSION.json')['geometry']
    f,axs=plt.subplots(1,2,figsize=(7,2.65),gridspec_kw={'width_ratios':[1,1.18]})
    arms=['P0','S1','H1','H0']
    for fold in (0,1):
        for seed in (201,202,203):
            vals=[next(r['tx_to_rx_centroid_ratio'] for r in geo if r['arm']==a and r['layer']=='final' and r['fold']==fold and r['seed']==seed) for a in arms]
            axs[0].plot(range(4),vals,color='#b5bdc5',lw=.7,alpha=.75)
    for i,a in enumerate(arms):
        vals=[r['tx_to_rx_centroid_ratio'] for r in geo if r['arm']==a and r['layer']=='final']
        axs[0].scatter(np.full(6,i),vals,color=COLORS[a],s=18,zorder=4)
        axs[0].scatter(i,np.mean(vals),color=COLORS[a],marker='D',s=35,zorder=5)
    axs[0].set(xticks=range(4),xticklabels=arms,ylabel=r'Final centroid ratio $B_{\rm TX}/D_{\rm RX|TX}$',title='A: six fixed source cohorts')
    for a in arms:
        groups=[[r['tx_to_within_class_ratio'] for r in geo if r['arm']==a and r['layer']==layer] for layer in ['tokens','pre_mha','final']]
        axs[1].errorbar(range(3),[np.mean(v) for v in groups],yerr=[np.std(v,ddof=1) for v in groups],marker='o',ms=3,lw=1,capsize=2,label=a,color=COLORS[a])
    axs[1].set(xticks=range(3),xticklabels=['Tokenizer','Pre-MHA','Final'],ylabel=r'$B_{\rm TX}/(D_{\rm RX|TX}+W)$',title='All registered taps: mean +/- fit SD')
    axs[1].legend(ncol=2,fontsize=6.8,frameon=False)
    for ax in axs:ax.grid(axis='y',color='#e6e9ec',lw=.5)
    f.tight_layout(w_pad=1.1);save(f,'fig03_mechanism_A')

    bands=list(csv.DictReader((ROOT/'23-hstem-mechanism-evidence/kernel_bands.csv').open(encoding='utf-8-sig')))
    joined=list(csv.DictReader((HERE/'kernel_geometry_join.csv').open(encoding='utf-8')))
    f,axs=plt.subplots(1,2,figsize=(7,2.45))
    for i,a in enumerate(['P0','S1','H1']):
        values=[float(r['high_energy'])*100 for r in bands if r['arm']==a]
        axs[0].scatter(i+np.linspace(-.08,.08,6),values,color=COLORS[a],s=17)
        axs[0].plot([i-.19,i+.19],[np.mean(values)]*2,color=COLORS[a],lw=2)
        selected=[r for r in joined if r['arm']==a and r['layer']=='final']
        axs[1].scatter([float(r['kernel_high_energy'])*100 for r in selected],[float(r['tx_to_within_class_ratio']) for r in selected],s=24,color=COLORS[a],label=a)
    axs[0].set(xticks=range(3),xticklabels=['P0\nsupport266','S1\nsupport266','H1\nsupport350'],ylabel='High-frequency energy (%)',title='B: existing S23 kernel evidence')
    axs[1].set(xlabel='Kernel high-frequency energy (%)',ylabel='Final class-separation ratio',title='18 fits: descriptive Spearman = -0.556')
    axs[1].legend(frameon=False,fontsize=7)
    f.tight_layout(w_pad=1);save(f,'fig04_mechanism_B')

    per=read(HERE/'PER_RX_ANALYSIS.json');rxs=sorted({r['rx'] for r in per['receivers']})
    legacy=list(csv.DictReader((HERE/'matched_tx_rx_comparisons.csv').open(encoding='utf-8')))
    data=[];labels=[]
    for view in ['native','strict_ib']:
        p=next(r for r in per['paired_rx_contributions'] if r['view']==view and r['baseline']=='dann')
        data.append([p['per_rx_delta_pp'][rx] for rx in rxs]);labels.append('S24 H0-DANN '+('Native' if view=='native' else 'Strict-IB'))
    data.append([np.mean([float(r['h0_minus_baseline_pp']) for r in legacy if r['rx']==rx and r['baseline']=='P0']) for rx in rxs]);labels.append('S22 H0-P0 (archival GPU)')
    f,ax=plt.subplots(figsize=(7,2.35))
    image=ax.imshow(data,cmap='RdBu',vmin=-22,vmax=22,aspect='auto')
    for i,row in enumerate(data):
        for j,value in enumerate(row):ax.text(j,i,f'{value:+.1f}',ha='center',va='center',fontsize=6.7,color='white' if abs(value)>14 else '#23313c')
    ax.set(xticks=range(10),xticklabels=rxs,yticks=range(3),yticklabels=labels)
    plt.setp(ax.get_xticklabels(),rotation=32,ha='right',fontsize=6.8)
    ax.tick_params(axis='y',labelsize=6.8)
    f.colorbar(image,ax=ax,fraction=.022,pad=.02,label='Accuracy difference (pp)')
    f.tight_layout();save(f,'fig05_mechanism_C')

    result=read(ROOT/'24-lora-external-audit/phase2/budget_update/RESULTS.json')
    costs=read(ROOT/'24-lora-external-audit/phase2/budget_update/COSTS.json')['rows']
    f,axs=plt.subplots(1,2,figsize=(7,2.6))
    offsets={'h0':(4,3),'ce':(4,-10),'dann':(4,3),'fan':(4,-10),'ib':(4,3),'ask':(4,3)}
    for c in [v for v in costs if v['view']=='native']:
        a=next(r['acc']*100 for r in result['summary'] if r['view']=='native' and r['method']==c['method'])
        method=c['method'];pps=128000/c['batches']['128']['p50_ms']
        for ax,x in [(axs[0],c['params']),(axs[1],pps)]:
            ax.scatter(x,a,color=COLORS[method],s=28)
            ax.annotate(NAMES[method],(x,a),xytext=offsets[method],textcoords='offset points',fontsize=6.4)
    axs[0].set(xscale='log',xlabel='Parameter elements (complex ASK noted in text)',ylabel='Native Dev accuracy (%)',ylim=(59,102))
    axs[1].set(xscale='log',xlabel='Batch128 throughput (packets/s)',ylabel='Native Dev accuracy (%)',ylim=(59,102))
    for ax in axs:ax.grid(color='#e6e9ec',lw=.5)
    f.tight_layout(w_pad=1);save(f,'fig06_costs')

    f,axs=plt.subplots(1,2,figsize=(7,2.9),gridspec_kw={'width_ratios':[.85,1.25]})
    for arm,values in [('H0',[98.175,21.858,17.775]),('P0',[95.925,22.567,16.983])]:
        axs[0].plot(range(3),values,marker='o',ms=4,label=arm,color=COLORS[arm])
    axs[0].set(xticks=range(3),xticklabels=['Aligned','Circular -4','Circular +4'],ylabel='Source accuracy (%)',ylim=(0,105),title='Synchronization stress (S23)')
    axs[0].tick_params(axis='x',labelsize=6.4);axs[0].legend(frameon=False,fontsize=7)
    tx=list(csv.DictReader((HERE/'per_tx_rx.csv').open(encoding='utf-8')))
    for method,view,label,color in [('h0','native','HStem',COLORS['H0']),('ce','native','CE Native','#777b82'),('ce','strict_ib','CE Strict-IB','#bd8640'),('dann','strict_ib','DANN Strict-IB','#9564a7')]:
        vals=[float(next(r['accuracy_pct'] for r in tx if r['rx']=='pluto_1' and r['method']==method and r['view']==view and int(r['tx_raw_id'])==t)) for t in range(31,41)]
        axs[1].plot(range(31,41),vals,marker='o',ms=2.4,lw=.9,label=label,color=color)
    axs[1].set(xticks=range(31,41),xlabel='TX raw identity',ylabel='Pluto-1 Dev accuracy (%)',ylim=(0,105),title='All TX at the difficult RX')
    axs[1].tick_params(axis='x',labelsize=6.3)
    axs[1].legend(frameon=False,ncol=2,fontsize=6.3,loc='upper center',bbox_to_anchor=(.5,-.31))
    f.subplots_adjust(left=.08,right=.995,bottom=.29,top=.86,wspace=.36)
    save(f,'fig07_limits')
    return result,costs

def tables(result,costs):
    lines=[r'\begin{tabular}{lrrrrrr}',r'\toprule Method & Native Acc. & Strict-IB Acc. & Native F1 & Strict F1 & Native Worst & Strict Worst\\ \midrule']
    for method in ['h0','ce','dann','fan','ib','ask']:
        a=next(r for r in result['summary'] if r['view']=='native' and r['method']==method)
        b=next(r for r in result['summary'] if r['view']=='strict_ib' and r['method']==method)
        complete=b['status']=='COMPLETE_REPORT_ONLY'
        strict=f"{b['acc']*100:.3f} $\\pm$ {b['fit_sd']*100:.3f}" if complete else 'Deferred'
        strict_f1=f"{b['macro_f1']*100:.3f}" if complete else '--'
        strict_worst=f"{b['worst_rx_acc']*100:.3f}" if complete else '--'
        lines.append(f"{NAMES[method]} & {a['acc']*100:.3f} $\\pm$ {a['fit_sd']*100:.3f} & {strict} & {a['macro_f1']*100:.3f} & {strict_f1} & {a['worst_rx_acc']*100:.3f} & {strict_worst} \\\\")
    lines += [r'\bottomrule\end{tabular}']
    (TABLE/'external.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    lines=[r'\begin{tabular}{lrrr}',r'\toprule Method & Parameters & B1 p50 (ms) & B128 p50 (ms)\\ \midrule']
    for row in costs:
        if row['view']!='native':continue
        lines.append(f"{NAMES[row['method']]} & {row['params']:,} & {row['batches']['1']['p50_ms']:.3f} & {row['batches']['128']['p50_ms']:.3f} \\\\")
    lines += [r'\bottomrule\end{tabular}']
    (TABLE/'costs.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    g=read(HERE/'MECHANISM_EXTENSION.json')['geometry']
    margin=list(csv.DictReader((HERE/'source_tx_rx_margins.csv').open(encoding='utf-8')))
    lines=[r'\begin{tabular}{lrrrrr}',r'\toprule Arm & Source Acc. & $B_{\rm TX}$ & $D_{\rm RX|TX}$ & $J_c$ & $J_w$\\ \midrule']
    for a in ['P0','S1','H1','H0']:
        rows=[r for r in g if r['arm']==a and r['layer']=='final']
        values={key:np.mean([r[key] for r in rows]) for key in ['between_tx','conditional_rx','tx_to_rx_centroid_ratio','tx_to_within_class_ratio']}
        acc=np.mean([float(r['accuracy_pct']) for r in margin if r['arm']==a])
        lines.append(f"{a} & {acc:.3f} & {values['between_tx']:.3f} & {values['conditional_rx']:.4f} & {values['tx_to_rx_centroid_ratio']:.3f} & {values['tx_to_within_class_ratio']:.3f} \\\\")
    lines += [r'\bottomrule\end{tabular}']
    (TABLE/'source_geometry.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')

def pca_supplement():
    evidence=[]
    for fold in (0,1):
        for seed in (201,202,203):
            f,axs=plt.subplots(1,4,figsize=(7,2.8))
            for ax,arm in zip(axs,['P0','S1','H1','H0']):
                path=HERE/'features'/f'{arm}_f{fold}_s{seed}.npz'
                with np.load(path,allow_pickle=False) as data:
                    z=data['final'].astype(float);tx=data['y'];rx=data['rx']
                z/=np.linalg.norm(z,axis=1,keepdims=True);z-=z.mean(0)
                values,vectors=np.linalg.eigh(z.T@z);order=np.argsort(values)[::-1]
                basis=vectors[:,order[:2]]
                for j in range(2):
                    if basis[np.argmax(np.abs(basis[:,j])),j]<0:basis[:,j]*=-1
                projection=z@basis
                for t in range(10):
                    for r in range(5):
                        center=projection[(tx==t)&(rx==r)].mean(0)
                        ax.scatter(*center,s=15,color=plt.cm.tab10(t),marker=['o','s','^','D','x'][r],alpha=.85,linewidths=.6)
                explained=float(values[order[:2]].sum()/values.sum())
                ax.set(title=f'{arm}: PC1+2 {explained*100:.1f}%',xlabel='PC1');ax.tick_params(labelsize=6)
                evidence.append(dict(arm=arm,fold=fold,seed=seed,pc12_explained_variance=explained,feature_sha256=digest(path)))
            axs[0].set_ylabel('PC2')
            f.suptitle(f'Source validation, fold {fold}, seed {seed}: all 50 TX x RX centroids',fontsize=8)
            tx_handles=[Line2D([],[],marker='o',ls='',color=plt.cm.tab10(t),label=f'TX{t+31}',ms=3) for t in range(10)]
            rx_handles=[Line2D([],[],marker=['o','s','^','D','x'][r],ls='',color='#3d4852',label=P['folds'][fold]['source'][r],ms=3) for r in range(5)]
            f.legend(handles=tx_handles,ncol=10,loc='lower center',bbox_to_anchor=(.5,.105),fontsize=5.6,frameon=False,handletextpad=.25,columnspacing=.65)
            f.legend(handles=rx_handles,ncol=5,loc='lower center',bbox_to_anchor=(.5,.025),fontsize=5.8,frameon=False,handletextpad=.35,columnspacing=1)
            f.subplots_adjust(left=.07,right=.995,bottom=.32,top=.80,wspace=.37)
            save(f,f'supp_pca_f{fold}_s{seed}')
    write(HERE/'PCA_PROVENANCE.json',dict(status='COMPLETE',selection='All six fold/seed cohorts, all four arms, all50 cell centroids; independent per-model PCA, no panel selection.',normalization='Registered final per-packet L2 embeddings, mean-centered; top2 eigenvectors, deterministic signs.',scope='Separate coordinate systems; 2D appearance is not quantitative cross-model evidence or new validation.',rows=evidence))

def main():
    result,costs=figures();tables(result,costs);pca_supplement()
    write(HERE/'FIGURE_PROVENANCE.json',dict(status='COMPLETE',main_figures=7,pca_cohorts=6,new_fits=0,new_waveform_reads=0,
        plot_code_sha256=digest(__file__),evidence=[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in [HERE/'MECHANISM_EXTENSION.json',HERE/'PER_RX_ANALYSIS.json',ROOT/'24-lora-external-audit/phase2/budget_update/RESULTS.json',ROOT/'23-hstem-mechanism-evidence/kernel_bands.csv']]))
    print('Seven main figures, six complete-cohort PCA supplements and formal English tables generated.')

if __name__=='__main__':main()
