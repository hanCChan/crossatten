"""Evidence-driven manuscript rewrite; final build requires all six results."""
from pathlib import Path
import re
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from s26_common import HERE,ROOT,read,sha,write,now

PAPER=HERE/'paper'

INTRO=r'''
\section{Introduction}
\IEEEPARstart{R}{adio-frequency} fingerprint identification (RFFI) predicts an enrolled transmitter from received waveforms. Receiver hardware, propagation, and acquisition can change the same transmitter's observation. The resulting identification problem matters to device monitoring, while remaining distinct from unknown-device rejection or attack-resistant authentication \cite{zhang2025survey}. Cross-receiver learning must therefore state both what information the learner receives and what a held receiver means.

Existing approaches address channel effects through signal representations, receiver differences through adversarial objectives, and training-data limitations through contrastive learning \cite{shen2022scalable,shen2024receiver,ma2025robust}. These choices leave another design decision: how the available RF representation becomes the token sequence consumed by a classifier. Equal token counts and parameter totals do not imply equal local support, nonlinear processing, or computational cost. Conversely, a better accuracy does not identify which of these differences caused the gain.

We study token formation under fixed in-band information, a 32-by-64 token interface, and an ordinary two-block attention backend. Our HStem design performs four local convolutional stages before the final sequence reaches global attention. Its 133,690 parameters are comparable to a single spatial-step Patch-Lite projection. The design allocates computation across intermediate lengths and nonlinear transformations instead of spending the budget on one large projection. This is a concrete RF architecture and empirical study; convolutional tokens and hierarchical processing are established \cite{wu2021cvt,wang2021pvt,liu2021swin}.

The RF setting requires care with architectural explanations. Five inherited views concatenate time-indexed and frequency-indexed coordinates. In-band projection and spectral construction have global waveform dependencies. Thus a local convolutional support is not an isolated physical RF interval, and progressively reducing coordinate counts does not establish preservation of transmitter information. We formalize these distinctions and test explanations through fixed controls, including a freely learned long patch with the same local support as HStem.

The contributions are:
\begin{enumerate}
\item A parameter-efficient four-stage RF tokenizer at a fixed input/output and ordinary MHA interface. Explicit support, center, dimensional, parameter, and MAC accounting distinguishes staged local processing from a direct grouped projection. These are architectural identities and design definitions, not new generalization guarantees.
\item Controlled evidence about alternative explanations: parameter matching, smooth parameterization, support-related interventions, normalization controls, and a separately frozen six-model same-support Long-Patch control. All-layer frozen-feature diagnostics retain observations that contradict simple progressive receiver suppression.
\item Competition under fixed source-packet permissions with Shen CE/DANN, FAN, an IB-RFF adaptation, and complete LoRa ASKNet. Native and common-band results are reported with every seed, unfavorable receivers, and separate parameter, latency, and throughput measurements.
\end{enumerate}

TrueIB, five views, and the shared kernel-7/kernel-5 front-end are inherited from the companion OOB study \cite{oobcompanion}. They are not new contributions. The present receivers have a development history, and no fresh independent acquisition is available. Results concern aligned extracted windows and ten enrolled TXs; these boundaries guide the claims and the interpretation of negative synchronization tests.
'''

RELATED=r'''
\subsection{Relation to Receiver-Robust and Tokenization Methods}
Channel-independent signal representations provide an important LoRa baseline \cite{shen2022scalable}. Length/noise robustness is another related problem \cite{shen2023length}, but does not by itself establish transfer to new receiver hardware. Shen et al.\ combine CIS with receiver-adversarial learning \cite{shen2024receiver}, related to gradient-reversal domain training \cite{ganin2016domain}. Our CE and DANN controls share the same encoder and transmitter head; their contrast separates the adversarial objective from the backbone.

Contrastive learning connects augmentation and representation geometry \cite{chen2020simclr}. Ma et al.\ use spectrograms and same-TX cross-RX Siamese training, both from scratch and with optional contrastive pretraining \cite{ma2025wcnc,ma2025robust}. The unpretrained variant could in principle use only our legal source RXs. External pretraining is a separate permission question. We did not locate an author implementation in the checked author-code sources and have not qualified a frozen same-protocol adapter. This coverage gap is disclosed, rather than treating the whole method as incompatible or borrowing its scores.

FAN/IB-RFF and ASKNet represent complementary frequency-aware, bottleneck, and spectral-dynamics approaches \cite{wang2026ib,liu2026ask}. We retain the two-stage IB adaptation and the complete LoRa ASK implementation; neither its low scores nor ASK's unstable seeds establish the general limits of the original methods. Federated contrastive learning and federated feature alignment address distributed clients and heterogeneous representations \cite{shen2024federated,wang2026fedalign}. Their client/data/aggregation protocol would need a separate adapter before a numerical comparison; their paper scores are not comparable measurements here.

CvT, PVT, and Swin establish convolutional and hierarchical token precedents \cite{wu2021cvt,wang2021pvt,liu2021swin}. Attention, BN, and LN are standard components \cite{vaswani2017attention,ioffe2015batch,ba2016layer}. Our question is their controlled RF-system behavior at a fixed interface. Shift-stability work \cite{zhang2019shift} motivates tests, but does not confer guarantees on our mixed-coordinate representation.
'''

MATHEMATICS=r'''
\subsection{Support Growth and Coordinate Compression}
Let $N_l,C_l$ denote length and channels after stage $l$, and $J_l,R_l,c_l$ denote its jump, convolution-path support, and first-center coordinate. With kernel $k_l$, stride $s_l$, dilation one, and padding $p_l$,
\begin{align}
N_l&=\left\lfloor\frac{N_{l-1}+2p_l-k_l}{s_l}\right\rfloor+1,\\
J_l&=s_lJ_{l-1},\quad R_l=R_{l-1}+(k_l-1)J_{l-1},\\
c_l&=c_{l-1}+\left((k_l-1)/2-p_l\right)J_{l-1}.
\end{align}
The common front-end gives $R_0=11,J_0=1,c_0=0$. HStem consequently has supports $18,46,158,350$, lengths $2048,512,128,32$, and final center 127.5. Both P0 and Long-Patch use jump 256; their supports are 266 and 350. Center matching avoids a hidden token-grid shift.

\begin{table}[t]\centering
\caption{HStem coordinate reduction after the shared front-end. These counts are dimensions, not information estimates.}
\label{tab:compression}
\begin{tabular}{lrrr}\toprule
Position & Length & Channels & Coordinates\\\midrule
Shared input &8192&32&262144\\
Stage 1&2048&48&98304\\
Stage 2&512&64&32768\\
Stage 3&128&64&8192\\
Stage 4&32&64&2048\\\bottomrule
\end{tabular}\end{table}

Support here describes the convolutional path with frozen BN. Training BN also uses batch/spatial statistics, and TrueIB/FFT have global raw-waveform dependencies. Table~\ref{tab:compression} describes progressive coordinate reduction; it does not show that an equal fraction of TX information survives each stage. Learned strided filters also have no established antialiasing guarantee.

\subsection{Long Projection, Nonlinearity, and Budget}
For fixed frontend features $h$, the two-convolution P0/Long-Patch tokenizer is affine: its effective kernel is a product of the grouped projection and pointwise mixing. HStem interleaves three BN/GELU stages before its final convolution. With frozen BN, its Jacobian can be written
\begin{equation}
 J_H(h)=C_4D_3(h)B_3C_3D_2(h)B_2C_2D_1(h)B_1C_1,
\end{equation}
where $D_l$ contains GELU derivatives and $B_l$ frozen BN scales. Its local mixing is input dependent; the full Patch-based models are also nonlinear because the shared frontend and backend are nonlinear. This distinction motivates a hypothesis about joint processing, not a receiver-invariance theorem.

A grouped long projection with intermediate width $m$ and $g$ groups has
\begin{equation}
 P_{\rm patch}=P_{\rm shared}+32mk/g+m+64m+64,
\end{equation}
with $P_{\rm shared}=47,082$. P0 uses $m=80,g=8,k=256$; the fixed Long-Patch uses $m=64,g=8,k=340$, yielding 138,346 parameters, 3.48\% above HStem. Its output can have rank 64, though its per-input-group factorization differs from P0's. Equal output rank does not imply identical function classes.

Convolution MACs are $\sum_lN_lk_lC_{l-1}C_l/g_l$. HStem spends 42.467M tokenizer MACs versus 2.916M for Long-Patch, while both share 51.118M frontend convolution MACs. Supported dispatcher totals are 95.028M versus 55.477M per packet. FFT, nonlinearities, and unsupported work are excluded, so these are partial MACs, not complete FLOPs.

Staged mixing therefore trades more computation for a smaller parameter total and a different nonlinear parameterization. A same-support Long-Patch asks whether a direct freely learned projection suffices at a nearby parameter budget. It still differs in internal width, BN, nonlinear depth, and compute. None of these comparisons alone identifies pure hierarchy causality or a physical hardware mechanism.
'''

NEW_REFS=r'''
@article{zhang2025survey,
 author={Junqing Zhang and Francesco Ardizzon and Mattia Piana and Guanxiong Shen and Stefano Tomasin},
 title={Physical Layer-Based Device Fingerprinting for Wireless Security: From Theory to Practice},
 journal={IEEE Transactions on Information Forensics and Security},year={2025},volume={20},pages={5296--5325},
 doi={10.1109/TIFS.2025.3570118},url={https://arxiv.org/abs/2506.09807}}
@article{shen2022scalable,
 author={Guanxiong Shen and Junqing Zhang and Alan Marshall and Joseph R. Cavallaro},
 title={Towards Scalable and Channel-Robust Radio Frequency Fingerprint Identification for {LoRa}},
 journal={IEEE Transactions on Information Forensics and Security},year={2022},volume={17},pages={774--787},doi={10.1109/TIFS.2022.3152404}}
@article{shen2023length,
 author={Guanxiong Shen and Junqing Zhang and Alan Marshall and Mikko Valkama and Joseph R. Cavallaro},
 title={Towards Length-Versatile and Noise-Robust Radio Frequency Fingerprint Identification},
 journal={IEEE Transactions on Information Forensics and Security},year={2023},volume={18},pages={2355--2367},doi={10.1109/TIFS.2023.3266626}}
@article{shen2024federated,
 author={Guanxiong Shen and Junqing Zhang and Xuyu Wang and Shiwen Mao},
 title={Federated Radio Frequency Fingerprint Identification Powered by Unsupervised Contrastive Learning},
 journal={IEEE Transactions on Information Forensics and Security},year={2024},volume={19},pages={9204--9215},doi={10.1109/TIFS.2024.3469820}}
@article{wang2026fedalign,
 author={Yuteng Wang and Zhenxin Cai and Tiantian Tang and Tomoaki Ohtsuki and Guan Gui},
 title={Toward Robust Radio Frequency Fingerprint Identification: A Federated Learning Framework With Feature Alignment},
 journal={IEEE Transactions on Information Forensics and Security},year={2026},volume={21},pages={2641--2651},doi={10.1109/TIFS.2026.3671107}}
@inproceedings{ma2025wcnc,
 author={Jie Ma and Junqing Zhang and Guanxiong Shen and Linning Peng and Alan Marshall},
 title={Towards Channel-Robust Radio Frequency Fingerprint Identification Using Contrastive Learning},
 booktitle={IEEE Wireless Communications and Networking Conference},year={2025},pages={1--6},doi={10.1109/WCNC61545.2025.10978330}}
@inproceedings{chen2020simclr,
 author={Ting Chen and Simon Kornblith and Mohammad Norouzi and Geoffrey Hinton},
 title={A Simple Framework for Contrastive Learning of Visual Representations},
 booktitle={Proceedings of the International Conference on Machine Learning},year={2020},volume={119},pages={1597--1607},
 url={https://proceedings.mlr.press/v119/chen20j.html}}
'''


def replacement(text,start,end,new):
    a=text.index(start);b=text.index(end,a)
    return text[:a]+new+'\n\n'+text[b:]


def figure(result):
    records=read(HERE/'EVALUATION.json')['records']
    colors={'P0':'#747b87','LP':'#bb7c36','H0':'#1b6b8f'}
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    f,axes=plt.subplots(1,2,figsize=(7,2.8),gridspec_kw={'width_ratios':[.85,1.3]})
    for fold in (0,1):
        for seed in (201,202,203):
            values=[np.mean([r['acc'] for r in records if r['arm']==a and r['fold']==fold and r['seed']==seed])*100 for a in ['P0','LP','H0']]
            axes[0].plot(range(3),values,color='#b4bbc4',lw=.7,alpha=.7)
    for j,a in enumerate(['P0','LP','H0']):
        s=next(r for r in result['summary'] if r['arm']==a)
        axes[0].scatter(np.full(6,j),s['fit_accuracies_pct'],color=colors[a],s=17)
        axes[0].scatter(j,s['accuracy_pct'],marker='D',color=colors[a],s=30,zorder=5)
    axes[0].set(xticks=range(3),xticklabels=['P0','Long-Patch','HStem'],ylabel='Dev accuracy (%)',title='All six paired fits')
    p=result['paired_comparisons'][0];q=result['paired_comparisons'][1]
    names=list(p['per_rx_delta_pp']);xx=np.arange(len(names))
    axes[1].axhline(0,color='#6e7680',lw=.7)
    axes[1].plot(xx,[p['per_rx_delta_pp'][r] for r in names],marker='o',lw=.9,ms=3,label='H0 - Long-Patch',color=colors['H0'])
    axes[1].plot(xx,[q['per_rx_delta_pp'][r] for r in names],marker='s',lw=.9,ms=3,label='Long-Patch - P0',color=colors['LP'])
    axes[1].set(xticks=xx,xticklabels=names,ylabel='Difference (pp)',title='Every held RX: three-seed mean')
    axes[1].tick_params(axis='x',rotation=35,labelsize=6)
    for label in axes[1].get_xticklabels():label.set_ha('right')
    axes[1].legend(frameon=False,fontsize=6.7)
    f.tight_layout(w_pad=1.2)
    for extension in ['pdf','png']:
        f.savefig(PAPER/'figures'/('fig08_long_patch.'+extension),bbox_inches='tight',pad_inches=.035,dpi=230)
    plt.close(f)


def main():
    if not (HERE/'RESULTS.json').exists():
        raise RuntimeError('Manuscript finalization waits for all6 fits/sealed evaluation; no fabricated table values.')
    result=read(HERE/'RESULTS.json');cost=read(HERE/'COSTS.json');pair=result['paired_comparisons'][0];secondary=result['paired_comparisons'][1]
    assert result['new_fits']==6 and len(result['summary'])==3
    src=ROOT/'25-tifs-integration/paper';PAPER.mkdir(exist_ok=True)
    shutil.copytree(src/'figures',PAPER/'figures',dirs_exist_ok=True)
    shutil.copytree(src/'tables',PAPER/'tables',dirs_exist_ok=True)
    figure(result)
    s={r['arm']:r for r in result['summary']};delta=pair['mean_delta_pp'];ci=pair['conditional_rx_ci95_pp']
    if pair['primary_practical_rule_met']:
        implication='The registered consistency rule is met. A freely learned long projection at this nearby parameter budget does not recover the full HStem result. Width, normalization, nonlinear depth, and compute still prevent attribution to hierarchy alone.'
        short='The same-support control narrows a support-only explanation for this selected arm.'
    else:
        implication='The registered consistency rule is not met. This control does not establish a stable HStem advantage; a small difference or an interval including zero is not proof of equivalence. The claim of an independent hierarchy benefit must remain unestablished.'
        short='The same-support control does not establish a stable independent hierarchy advantage.'
    abstract=r'''\begin{abstract}
How a fixed RF representation becomes a token sequence can affect cross-receiver identification even when token geometry and the classifier are unchanged. We study a four-stage convolutional RF tokenizer with two ordinary attention blocks and 133,690 parameters. Explicit support and budget accounting motivates comparisons with direct projections. Existing parameter-matched, smooth-parameterization, support, and normalization controls are complemented by a separately registered six-model Long-Patch control with equal local support and 3.48\% more parameters. @@LONG@@ In a frozen external comparison, HStem obtains 97.333\% RX-equal accuracy versus 94.208\% for Shen-DANN under native preprocessing. Common-band results reveal receiver heterogeneity. Frozen source features improve final TX separation while intermediate-layer counterexamples constrain receiver-suppression explanations. Parameters, latency, throughput, and negative synchronization tests are jointly reported. The evidence concerns an aligned-window, historically developed closed-set protocol; fresh acquisition and hardware invariance remain unverified.
\end{abstract}'''
    abstract=abstract.replace('@@LONG@@',f"Long-Patch reaches {s['LP']['accuracy_pct']:.3f}\\%; HStem minus Long-Patch is {delta:+.3f} percentage points.")
    text=(src/'main.tex').read_text(encoding='utf-8')
    text=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:abstract,text,flags=re.S)
    text=replacement(text,r'\section{Introduction}',r'\begin{figure*}[t]',INTRO)
    text=replacement(text,r'\subsection{Relation to Cross-Receiver and Tokenization Methods}',r'\section{Hierarchical RF Token Formation}',RELATED)
    text=text.replace(r'\section{Experimental Protocol and Reproducibility}',MATHEMATICS+'\n\n'+r'\section{Experimental Protocol and Reproducibility}')
    table=[r'\begin{tabular}{lrrrrrr}',r'\toprule Arm & Params & Support & BN & Acc. $\pm$ fit SD & F1 & Worst RX\\\midrule']
    for a,param,support,bn in [('P0',134266,266,2),('LP',138346,350,2),('H0',133690,350,5)]:
        row=s[a];name='Long-Patch' if a=='LP' else a
        table.append(f"{name}&{param:,}&{support}&{bn}&{row['accuracy_pct']:.3f} $\\pm$ {row['fit_sd_pp']:.3f}&{row['macro_f1_pct']:.3f}&{row['worst_rx_accuracy_pct']:.3f}\\\\")
    table.append(r'\bottomrule\end{tabular}')
    (PAPER/'tables/long_patch.tex').write_text('\n'.join(table)+'\n',encoding='utf-8',newline='\n')
    lptext=r'''\subsection{A Separately Registered Same-Support Long Patch}
The single groups-8/width-64 Long-Patch design was proposed without accuracy measurements and then registered before its six fits. Source rows, seeds, augmentation, 200-epoch AdamW updates, and minimum source RX-equal CE selection are retained. Shared frontend/backend initial states match P0/H0. The new tokenizer needs its own fixed initializer because its shape differs. Atomic epoch recovery preserves model, optimizer, permutation/noise generators, and global random state; a data-free resumed-step check was bit exact.

All six checkpoints/configurations/histories were committed before new Dev inference. Table~\ref{tab:long} uses a common fresh S26 GPU execution for H0/P0/Long-Patch: FP32 tensors, CUDA matmul TF32 off, cuDNN TF32 on, batch 128. It is separate from the S24 CPU external table and its TF32-off cost session. No Confirm/blind packet is opened. All six new fits use 200 epochs; no seed is replaced.

\begin{table*}[t]\centering
\caption{Same-support developmental control: common S26 GPU inference of all arms, six paired fits each. Only Long-Patch is newly fitted. Accuracy/F1/worst RX are percentages; SD is over fold/seed means.}
\label{tab:long}\input{tables/long_patch.tex}
\end{table*}
\begin{figure*}[t]\centering
\includegraphics[width=.96\textwidth]{fig08_long_patch.pdf}
\caption{The registered Long-Patch comparison retains every fold/seed (left) and all ten RXs (right). No independent acquisition is implied.}
\label{fig:long}
\end{figure*}

@@PAIRED@@

@@IMPLICATION@@

The primary practical rule required at least 0.5\,pp mean gain, a positive conditional RX lower bound, at least five of six positive fit differences, and positive means in both folds. These are developmental descriptive checks. Historical target exposure, reused source selection, and residual structural differences remain limitations. The full per-fit source/Dev results, selected epochs, and frozen registration appear in the supplement.
'''
    lptext=lptext.replace('@@PAIRED@@',f"H0--Long-Patch is {delta:+.3f}\\,pp, conditional RX 95\\% interval $[{ci[0]:.3f},{ci[1]:.3f}]$, with {pair['positive_fit_pairs']}/6 positive paired means. Long-Patch--P0 is {secondary['mean_delta_pp']:+.3f}\\,pp, interval $[{secondary['conditional_rx_ci95_pp'][0]:.3f},{secondary['conditional_rx_ci95_pp'][1]:.3f}]$. Figure~\\ref{{fig:long}} shows the complete cohorts.")
    lptext=lptext.replace('@@IMPLICATION@@',implication)
    text=text.replace(r'\subsection{Frozen Feature Geometry}',lptext+'\n'+r'\subsection{Frozen Feature Geometry}')
    # Detailed kernel/equivalence and metadata diagnostics move to the supplement.
    kernel_start=text.index(r'\subsection{Kernel Evidence and Its Limits}')
    kernel_end=text.index(r'\section{External Results, RX Heterogeneity, and Cost}',kernel_start)
    kernel=text[kernel_start:kernel_end]
    text=text[:kernel_start]+r'''\subsection{Supporting Parameterization Diagnostics}
Existing effective-kernel and materialization tests support a training-parameterization explanation without identifying a physical transmitter spectrum. The full-rank S1 round trip retains its inference function class; its optimization coordinates differ. Across 18 P0/S1/H1 fits, kernel high-frequency energy has descriptive Spearman correlation $-.556$ with final source $J_w$, confounded by arm and optimization. Source CFO/SNR associations and numerical equivalence/failure records are reported in the supplement. No target metadata was read, and no causal CFO/SNR explanation is established.

'''+text[kernel_end:]
    # RX heatmap is complete but secondary to the new registered control.
    rxfig=re.search(r'\\begin\{figure\*\}\[t\]\\centering\s*\\includegraphics\[width=\.97\\textwidth\]\{fig05_mechanism_C.pdf\}.*?\\end\{figure\*\}',text,re.S)
    assert rxfig
    rxfigure=rxfig.group();text=text.replace(rxfigure,'')
    text=text.replace(r'Fig.~\ref{fig:rx}',r'the complete RX heatmap in the supplement')
    text=replacement(text,r'\subsection{An Unfitted Long-Patch Audit}',r'\subsection{Independence and Security Scope}',r'''\subsection{Interpretation of the Long-Patch Control}
The new control tests one freely learned support-matched projection, not every large-kernel or hierarchical design. Both alternatives retain an inherited nonlinear RF frontend and MHA backend. Different intermediate width, group factorization, training BN, nonlinear depth, and MACs remain. The new result cannot uniquely identify hierarchy as a causal mechanism, nor turn historically developed RXs into independent validation. No further arm or replacement seed is chosen from this result.''')
    costtable=[r'\begin{tabular}{lrrr}',r'\toprule Arm & Partial MACs & B1 p50 (ms) & B128 p50 (ms)\\\midrule']
    audit=read(HERE/'GATE.json')['architecture']
    for row in cost['rows']:
        a=row['arm'];costtable.append(f"{a}&{audit[a]['partial_macs_per_packet']/1e6:.3f}M&{row['batches']['1']['p50_ms']:.3f}&{row['batches']['128']['p50_ms']:.3f}\\\\")
    costtable.append(r'\bottomrule\end{tabular}')
    (PAPER/'tables/long_cost.tex').write_text('\n'.join(costtable)+'\n',encoding='utf-8',newline='\n')
    extra=r'''\begin{table}[t]\centering
\caption{Separate S26 common GPU-resident IQ-to-logits session, matching S26 precision flags. Partial MACs exclude FFT and unsupported work. Warmup 3, repetitions 15; B128 is whole-batch time.}
\label{tab:longcost}\input{tables/long_cost.tex}\end{table}

Table~\ref{tab:longcost} compares the new control in one timing session. Its smaller convolution workload does not establish equal effective capacity or a physical mechanism. Training used at most two independent CUDA jobs with unchanged per-job batch and RNG rules. Summed fit wall times overlap and must not be labeled active GPU hours.
'''
    text=text.replace(r'\section{Failure Boundaries and Innovation Limits}',extra+'\n'+r'\section{Failure Boundaries and Innovation Limits}')
    conclusion=r'''\section{Conclusion}
RF token formation changes this fixed-interface identification system beyond parameter count. HStem combines staged local support, nonlinear processing, and normalization at a small parameter budget. Parameter-matched controls, a separately frozen same-support projection, and all-layer source diagnostics constrain explanations. @@SHORT@@ External competition remains strongest against the stable Shen baselines, with receiver heterogeneity and real compute costs visible.

The evidence concerns this aligned-window developmental protocol. It does not identify a unique RF hardware mechanism, establish fresh acquisition generalization, or provide a complete security authenticator. Severe synchronization failure, weak baseline seeds, and unfavorable receivers remain part of the result.
'''.replace('@@SHORT@@',short)
    text=replacement(text,r'\section{Conclusion}',r'\appendices',conclusion)
    appendix=r'''\appendices
\section{Reproduction and Scope}
Repository \href{https://github.com/hanCChan/crossatten}{\texttt{hanCChan/crossatten}}, S26 branch \texttt{exp/s26-long-patch-control}, builds on S25 \texttt{657bdaa}. The initial S26 scientific code/registration commit is \texttt{0935eb8}. All six source-selected checkpoints were sealed before report-only target IO. The supplement supplies selected epochs, source/Dev pairs, numeric recovery, complete PCA cohorts, detailed equivalence/metadata checks, and the evidence map. Large waveforms, checkpoints, resume states, and sample predictions remain local; aggregate records and generation code are provided.

TrueIB, five views, kernel-7/kernel-5 frontend, token geometry, and data origin are inherited from the unpublished companion \cite{oobcompanion}. The new question concerns token formation under a different mixer/budget protocol. Former Confirm/blind RXs have historical use; no fresh independent capture is reported. Cross-paper accuracy subtraction is not architectural causal evidence. This remains an anonymous internal draft; submission readiness and novelty are not certified by experiment count.
\balance
\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
'''
    text=text[:text.index(r'\appendices')]+appendix
    text=text.replace('Working Research Manuscript, November 22, 2026','Revised Working Research Manuscript, November 22, 2026')
    assert '@@' not in text
    (PAPER/'main.tex').write_text(text,encoding='utf-8',newline='\n')
    (PAPER/'references.bib').write_text((src/'references.bib').read_text(encoding='utf-8')+NEW_REFS,encoding='utf-8',newline='\n')
    supplement(rxs=rxfigure,kernels=kernel,result=result)
    write(HERE/'MANUSCRIPT_PROVENANCE.json',dict(time=now(),status='GENERATED_FROM_COMPLETE_RESULTS',
        rewrite_code_sha256=sha(__file__),results_sha256=sha(HERE/'RESULTS.json'),evaluation_sha256=sha(HERE/'EVALUATION.json'),
        main_source_sha256=sha(PAPER/'main.tex'),supplement_source_sha256=sha(PAPER/'supplement.tex'),
        primary_rule_met=pair['primary_practical_rule_met'],source_legacy='S25 and prior evidence remain unchanged; copied figures are reused, not new experiments.'))
    print('Rewritten manuscript and supplement generated from all6 completed fits.',flush=True)


def supplement(rxs,kernels,result):
    pre=r'''\documentclass[journal]{IEEEtran}
\usepackage{amsmath,amssymb,booktabs,array,graphicx,cite,xurl,balance}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdftitle={Supplementary Evidence and Reproduction Records: Hierarchical RF Tokenization},pdfauthor={Anonymous working draft}}
\graphicspath{{figures/}}
\begin{document}
\title{Supplementary Evidence and Reproduction Records}
\author{Anonymous working draft}\maketitle
\section{All New Fits and Common Inference}
Only Long-Patch is newly trained. Every fit uses 200 epochs and minimum source RX-equal CE selection. The six source/Dev pairs below are complete, without deleting a weak seed. Target inference occurs after the full source seal. H0/P0 are reused weights; the primary comparison re-evaluates all three arms with identical current GPU precision flags. Historical CPU/GPU values remain separately archived.
\begin{table*}[t]\centering\caption{All six Long-Patch fits; no replacement seeds.}\label{tab:allfits}
\begin{tabular}{rrrrrrr}\toprule
Fold&Seed&Epochs&Best epoch&Source acc.(\%)&Dev acc.(\%)&Fit wall(s)\\\midrule
'''
    rows=[]
    for fold in (0,1):
        for seed in (201,202,203):
            fit=read(HERE/'runs'/f'LP_f{fold}_s{seed}'/'metrics.json')
            dev=next(r['fit_accuracies_pct'][fold*3+[201,202,203].index(seed)] for r in result['summary'] if r['arm']=='LP')
            rows.append(f"{fold}&{seed}&200&{fit['best_epoch']}&{fit['source_validation']['summary']['acc_mean']*100:.3f}&{dev:.3f}&{fit['wall_seconds']:.1f}\\\\")
    text=pre+'\n'.join(rows)+r'''
\bottomrule\end{tabular}\end{table*}
The original S26 registration, data-free resource/recovery gate, source recovery, checkpoint/configuration/history seal, access log, saved predictions, and final recomputation audit provide the evidence chain. Summed per-fit wall times overlap under concurrent jobs and are not active GPU time. Atomic recovery restores optimizer, global and data-stream RNG states at a completed epoch; it does not reset a seed after observing a score.

\section{Kernel, Equivalence, and Metadata Evidence}
'''+kernels+r'''
\section{Complete External RX Attribution}
These are the original S24 external and S22 matched archival GPU cohorts. The heatmap retains the two receivers where H0 loses to Native Shen-DANN and the concentration at Pluto-1. Its H0/P0 archival cohort is explicitly distinct from the new S26 common inference.
'''+rxs+r'''
\section{All Frozen-Source PCA Cohorts}
All four arms and all six fold/seed cohorts are included. Each model uses its own deterministic top-two PCA components of final per-packet L2 features. Colors identify TX31--40; shapes identify source RXs. These are 50 cell centroids, not the full sample cloud; two-dimensional appearance is not quantitative evidence or independent generalization.
'''
    for fold in (0,1):
        for seed in (201,202,203):
            text+=r'\begin{figure*}[t]\centering'+'\n'+f'\\includegraphics[width=.98\\textwidth]{{supp_pca_f{fold}_s{seed}.pdf}}\n'+f'\\caption{{All source TX-by-RX centroids, fold {fold}, seed {seed}. Separate model coordinate systems.}}\n'+r'\end{figure*}'+'\n'
    text+=r'''
\section{Audit Boundaries and Companion Overlap}
The existing source validation packets already selected checkpoints; feature diagnostics are exploratory. TrueIB, five views, local frontend, token geometry, and data origin are inherited from the companion OOB manuscript \cite{oobcompanion}. The old six-RX blind set and Confirm RXs have research history and are not newly independent. No new capture is available. Unknown-TX rejection, FAR/FRR, active spoofing, and calibrated real synchronization-offset robustness are untested.

The S25 TF32-off source analysis retained two net one-packet accuracy changes and one additional changed confusion pattern despite identical weights. Its final ratios are computed from that single analysis execution. S26's separate original-precision source recovery and common target-inference comparisons are recorded without overwriting these historical files. Numeric recovery is an execution observation, not permission to select a better result.

Ma's unpretrained Siamese variant is source-compatible in principle \cite{ma2025robust}; the optional externally pretrained variant requires a separate budget. A checked author implementation and a frozen source-only adapter were not qualified here. No simplified network or paper score substitutes for it. This is a remaining coverage limitation, not proof that the method cannot be adapted.

The eight-claim evidence matrix and ten reviewer answers are updated in the repository. They separate measured system effects, exploratory associations, architectural identities, and unresolved mechanisms. Full negative shift and weak-fit results are retained.
\clearpage
\balance
\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
'''
    # References in inherited diagnostics are self-contained and resolved here.
    (PAPER/'supplement.tex').write_text(text,encoding='utf-8',newline='\n')


if __name__=='__main__':main()
