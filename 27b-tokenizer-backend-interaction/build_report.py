"""English B mechanism record; compile/render from public aggregate evidence."""
import csv
from pathlib import Path
import subprocess
from pypdf import PdfReader
from s27b_common import HERE,read,write,sha,now

LABEL={'H0':'HS/MHA','LP':'LP/MHA','H0_P':'HS/MLP','LP_P':'LP/MLP'}


def table(rows,columns):
    return '\n'.join(' & '.join(str(q[k]) for k in columns)+r'\\' for q in rows)+'\n'


def main():
    result=read(HERE/'RESULTS.json');decision=read(HERE/'MECHANISM_DECISION.json');audit=read(HERE/'EXPERIMENT_INTEGRITY.json')
    assert audit['status']=='PASS';primary=result['primary_interaction'];ci=primary['conditional_rx_ci95_pp']
    source=read(HERE/'SOURCE_EVALUATION.json');src={q['arm']:q for q in result['source_summary']};gate=read(HERE/'GATE.json')['architecture']
    body=r'''\documentclass[10pt]{article}
\usepackage[letterpaper,margin=.78in]{geometry}
\usepackage{amsmath,booktabs,array,graphicx,hyperref,xurl}
\hypersetup{hidelinks,pdftitle={S27-B: Tokenizer by Backend Interaction},pdfauthor={Research execution record}}
\graphicspath{{../figures/}}\setlength{\parskip}{4pt}
\title{S27-B: Tokenizer by Backend Interaction}
\author{Research execution record}\date{23 November 2026}
\begin{document}\maketitle
\section{Question and Frozen Execution}
Does the HStem-versus-Long-Patch accuracy difference depend on the selected backend? This experiment uses the S27 B design frozen before its execution, on top of S27-A commit \texttt{636da47}. Twelve new source-only fits replace the two MHA blocks by the already defined two-block positionwise residual TokenMLP; twelve original HStem/Long-Patch MHA checkpoints are reused. No source packet, seed, readout width or optimizer is selected from B target scores.

Each MLP block is $h'=\operatorname{LN}(h+W_2\operatorname{GELU}(W_1h))$, with width128 and16,704 parameters versus16,768 for MHA. The full-model difference is128 parameters. The 32-by-64 interface, input projection, position terms, pre-LN, final-LN, mean pooling and classifier remain unchanged. Corresponding surviving initial parameters and paired data/noise streams match; the two new heads share a fixed CPU initialization stream. Different nonlinear forms, token interactions, expressivity and computation remain; parameter matching does not isolate only attention.

The new fits use two folds, seeds201/202/203, five legal source RXs, 8,000 train and2,000 source-validation packets,200epochs, batch64, AdamW, five warmup epochs/cosine schedule and AWGN0--50dB. Only clean source RX-equal CE selects checkpoints. All twelve new checkpoint/config/history hashes are sealed and committed before any new Dev inference; both new arms are completed before target results are observed. C, IQ-only, ASK Strict-IB and further readout search remain unexecuted.
\section{All Four Cells and Primary Interaction}
\begin{center}\small\begin{tabular}{lrrrrrrr}\toprule
Cell&Params&Dev acc.&Fit SD&Macro-F1&Worst RX&Source acc.&Source CE\\\midrule
'''
    for q in result['summary']:
        arm=q['arm'];ce=sum(v['ce'] for v in source['records'] if v['arm']==arm)/30
        body+=f"{LABEL[arm]}&{gate[arm]['params']:,}&{q['accuracy_pct']:.3f}&{q['fit_sd_pp']:.3f}&{q['macro_f1_pct']:.3f}&{q['worst_rx_accuracy_pct']:.3f}&{src[arm]['accuracy_pct']:.3f}&{ce:.5f}\\\\\n"
    body+=r'''\bottomrule\end{tabular}\end{center}
Accuracy/F1 are percentages, fit SD is over six fold/seed means in pp, and worst RX is the minimum three-seed RX mean. Source validation has already participated in selection. HS denotes HStem; LP denotes the fixed freely learned long projection.
\begin{align*}
\Delta_M&=A_{HS,MHA}-A_{LP,MHA},\\
\Delta_P&=A_{HS,MLP}-A_{LP,MLP},\\
\Delta_{int}&=\Delta_M-\Delta_P.
\end{align*}
'''
    m,p=result['secondary_contrasts'][:2]
    body+=f"The complete-cohort tokenizer contrasts are $\\Delta_M={m['mean_delta_pp']:+.3f}$pp and $\\Delta_P={p['mean_delta_pp']:+.3f}$pp. The primary interaction is {primary['mean_delta_pp']:+.3f}pp, conditional RX95\\% interval [{ci[0]:.3f},{ci[1]:.3f}], with {primary['positive_fit_pairs']}/6 positive paired interactions. Fold means are {primary['fold_deltas_pp'][0]:+.3f} and {primary['fold_deltas_pp'][1]:+.3f}pp.\n"
    if decision['positive_interaction_rule_met']:
        body+='The registered positive rule is met. The HStem-relative advantage is larger under this MHA backend than under this fixed MLP; this is conditional architectural interaction, not a universal attention-necessity result.\n'
    elif primary['mean_delta_pp']<0 and ci[1]<0:
        body+='The estimate and conditional interval are below zero. The selected MLP has the larger tokenizer contrast, so the hypothesized MHA amplification is not supported. This does not show that every attention mechanism lowers performance.\n'
    else:body+='The registered positive rule is not met. A small difference or an interval containing zero does not prove equivalence or that attention is useless.\n'
    body+=r'''The unchanged rule requires mean at least .5pp, positive conditional lower bound, at least5/6 positive fit interactions and positive means in both folds. Three seeds are first averaged within each of ten unique RXs;10,000 RX bootstrap repetitions use seed27028. Intervals omit new-seed, acquisition and architecture-selection uncertainty. Secondary contrasts are descriptive, without multiplicity-corrected confirmatory claims.
\begin{figure}[htbp]\centering\includegraphics[width=.98\linewidth]{fig01_interaction.pdf}
\caption{Every paired tokenizer difference and every RX interaction. Negative and zero effects remain visible.}\end{figure}
\section{Source Learning, Historical Dev, and Scope}
'''
    body+=f"Source interaction is {result['source_interaction']['mean_delta_pp']:+.3f}pp, compared with {primary['mean_delta_pp']:+.3f}pp on Dev. Source-minus-Dev interaction is {result['source_minus_dev_interaction']['mean_pp']:+.3f}pp. All six paired values are archived.\n"
    backend=result['secondary_contrasts'][2]
    body+=f"Both chosen backends preserve positive HStem-minus-LP differences in all six fit pairs and all ten RX means. This secondary observation does not meet the primary MHA-amplification criterion or establish equivalence. The HStem MHA-minus-MLP difference is {backend['mean_delta_pp']:+.3f}pp; Pluto's {backend['per_rx_delta_pp']['pluto_1']:+.3f}pp contributes {backend['per_rx_delta_pp']['pluto_1']/10:+.3f}pp to the unchanged ten-RX mean. RTL5 and RTL6 favor MLP. No RX is removed from the primary estimate.\n"
    body+=r'''Source validation uses40 selected packets/TX/RX and already selects checkpoints; Dev uses200 and has a development history. Source/Dev differences do not replace a freshly fitted four-source/one-held-source study. Poor source learning by an MLP control also limits assigning its errors entirely to receiver transfer. A positive interaction cannot alone establish receiver-specific robustness.
\begin{figure}[htbp]\centering\includegraphics[width=.98\linewidth]{fig02_fourcells.pdf}
\caption{Complete source and Dev accuracies of all24 frozen models; no weak fit is excluded.}\end{figure}
\clearpage
\section{Fixed-Layer Source Geometry}
All24 models use the same2,000 legal source-validation packets/fold. The taps are mean tokenizer output, mean after input projection/position/pre-LN and final classifier input, each64D. Pooling precedes per-packet L2 normalization. Balanced40-packet TX-by-RX cells use the original S25 between-TX scatter $B$, conditional RX scatter $D$, within-cell scatter $W$, $J_c=B/(D+10^{-12})$ and $J_w=B/(D+W+10^{-12})$ definitions. The table averages six per-fit statistics, not pooled features or a ratio of overall means. If any fit is invalid, its entire arm/layer aggregate is NA; no replacement normalization is chosen.
\begin{center}\small\begin{tabular}{llrrrrr}\toprule
Cell&Tap&$B$&$D$&$W$&$J_c$&$J_w$\\\midrule
'''
    for q in read(HERE/'GEOMETRY_SUMMARY.json')['rows']:
        values=['NA' if q[k] is None else f'{q[k]:.4f}' for k in ['between_tx','conditional_rx','within_tx_rx','tx_to_rx_centroid_ratio','tx_to_within_class_ratio']]
        body+=' & '.join([LABEL[q['arm']],q['layer'].replace('_',r'\_')]+values)+r'\\'+'\n'
    body+=r'''\bottomrule\end{tabular}\end{center}
\begin{figure}[htbp]\centering\includegraphics[width=.98\linewidth]{fig03_source_geometry.pdf}
\caption{All three fixed taps and all four cells. Error bars are fit SD. Mean64D diagnostics do not summarize all information in the full token sequence.}\end{figure}
The models are trained end to end; changing the backend also changes the learned tokenizer. Thus these comparisons are descriptive representations of different trained systems, not causal information flow through a fixed representation. Source-selected geometry, scatter and ratios do not establish mutual information, receiver removal or RF hardware invariance. This source execution preserves the original FP32/TF32 flags and is not numerically pooled with old TF32-off diagnostics.

The complete table contains counterexamples to a single-geometry explanation. MHA's pre-backend $J_w$ is lower for HStem than LP. For both tokenizers, the MLP final-source $J_w$ is higher than MHA's, while Dev accuracy is lower. Under MLP, final conditional RX scatter is higher for HStem than LP despite HStem's accuracy advantage. These fixed diagnostics therefore do not monotonically rank Dev accuracy or show that HStem removes RX information at every layer. The mean64D taps do not characterize all information in the full token sequence.
\clearpage
\section{Every New Fit and Same-Session Cost}
\begin{center}\small\begin{tabular}{lrrrrrrr}\toprule
Cell&Fold&Seed&Best ep.&Source CE&Source acc.&Dev acc.&Fit wall(s)\\\midrule
'''
    with (HERE/'all_fit_source_dev.csv').open(encoding='utf-8',newline='') as stream:fits=list(csv.DictReader(stream))
    for q in fits:
        if not q['arm'].endswith('_P'):continue
        body+=' & '.join([LABEL[q['arm']],q['fold'],q['seed'],q['best_epoch'],f"{float(q['source_ce']):.5f}",
            f"{float(q['source_accuracy_pct']):.3f}",f"{float(q['dev_accuracy_pct']):.3f}",f"{float(q['fit_wall_seconds']):.1f}"])+r'\\'+'\n'
    body+=r'''\bottomrule\end{tabular}\end{center}
Every new fit completes200epochs and25,000 retained updates, without early stopping or target-driven reseeding. The twelve final trajectories total300,000 scientific updates. After a user-requested pause, continuation was explicitly authorized on 23 November 2026. Three fits were complete; LP/MLP fold0 seed202 and HS/MLP fold0 seed203 resumed their saved complete-epoch states at epochs180 and3. Model, optimizer, RNG, registration and best-source-checkpoint consistency were verified before resuming; snapshot hashes and recovery receipts are archived. Work within an unsealed partial epoch may have been discarded, so300,000 counts retained updates rather than all physical executions. Original and resumed controller logs are preserved separately. Summed fit wall times cover active execution segments, overlap under at most two CUDA jobs and are not exclusive GPU hours.
\begin{figure}[htbp]\centering\includegraphics[width=.98\linewidth]{fig04_all_training.pdf}
\caption{All twelve source CE histories, all200epochs, no seed removed.}\end{figure}
\begin{center}\small\begin{tabular}{lrrr}\toprule
Cell&Partial MACs(M)&B1 p50(ms)&B128 p50(ms)\\\midrule
'''
    for q in read(HERE/'COSTS.json')['rows']:
        body+=f"{LABEL[q['arm']]}&{gate[q['arm']]['partial_macs_per_packet']/1e6:.3f}&{q['batches']['1']['p50_ms']:.3f}&{q['batches']['128']['p50_ms']:.3f}\\\\\n"
    body+=r'''\bottomrule\end{tabular}\end{center}
This is one common GPU-resident IQ-to-logits timing session, batch1/128, three warmups and15 repetitions, original FP32 tensors with matmul TF32 off/cuDNN TF32 on. B128 is whole-batch latency. The latency includes FFT/frontend work and excludes host transfer. Partial MACs omit FFT and unsupported operations. Parameters, compute, latency and throughput are distinct axes; no old timing session is pooled here.
\section{Mechanism Judgment and Unexecuted C}
The registered positive interaction is not established. The current evidence supports a HStem-relative accuracy advantage under both selected backends; it does not support MHA amplification as an established explanation. An interval crossing zero does not demonstrate backend equivalence. Tokenwise MLPs also introduce different nonlinear forms, capacity and optimization paths. The source-geometry counterexamples prevent a single pooled scatter ratio from serving as a sufficient mechanism explanation.

If a receiver-specific gap claim is pursued, C remains necessary: B's five-source factorial training does not fit the predeclared four-source/one-held-source split. The fixed b210\_1/rtl\_5 pilot has only two held RX units and no population-RX guarantee. Current C, IQ-only and ASK Strict-IB fit counts are zero; no automatic expansion or new readout search follows these results. The existing innovation evidence matrix is updated without renaming old experiments. The formal TIFS manuscript is not rewritten in this stage.
\section{Integrity and Reproduction}
The engineering gate uses no RF data and separately records eight synthetic optimizer updates. It checks model shape/capacity, matching surviving initial states and MLP heads, finite gradients, exact optimizer/model/all-RNG recovery for both arms, forbidden data roles, batch64/128 resources, bit-exact feature-hook forward results and four-cell interaction arithmetic. Unique-process JSON temporary files, bounded Windows rename retry and shared-delete progress reads preserve scientific update order.

Independent verification recomputes240,000 Dev predictions,48,000 source predictions and72 geometry records, including balanced row/label identities, confusion matrices, accuracy/F1/CE, conditional interaction intervals, all200-epoch minimum-source-CE histories, the original paired permutation and AWGN streams and historical model, configuration and history hashes. All24 models and all RXs remain reported. No Confirm or old-blind packet, new target adaptation or fresh capture is introduced. Larger waveforms, checkpoints, resume states and per-packet predictions/features remain local; aggregate evidence, code and real logs are committed.

\href{https://github.com/hanCChan/crossatten/tree/exp/s27b-backend-interaction/27b-tokenizer-backend-interaction}{Repository evidence and Chinese mechanism report} accompanies this execution record. It supplies bounded architecture evidence, not submission-readiness or independent-generalization certification.
\end{document}
'''
    for old,new in {'width128':'width 128','and16,704':'and 16,704','versus16,768':'versus 16,768','is128':'is 128','seeds201':'seeds 201',
        'and2,000':'and 2,000',',200epochs':', 200 epochs','batch64':'batch 64','AWGN0':'AWGN 0','least5/6':'least 5/6','RXs;10,000':'RXs; 10,000',
        'seed27028':'seed 27028','uses40':'uses 40','uses200':'uses 200','all24':'all 24','All24':'All 24','same2,000':'same 2,000','each64D':'each 64D',
        'Balanced40':'Balanced 40','Mean64D':'Mean 64D','completes200epochs':'completes 200 epochs','and25,000':'and 25,000','total300,000':'total 300,000',
        'all200epochs':'all 200 epochs','batch1/128':'batch 1/128','and15':'and 15','recomputes240,000':'recomputes 240,000',
        'fold0 seed202':'fold 0 seed 202','fold0 seed203':'fold 0 seed 203','epochs180 and3':'epochs 180 and 3','so300,000':'so 300,000','mean64D':'mean 64D',
        'predictions,48,000':'predictions, 48,000','and72':'and 72','all200-epoch':'all 200-epoch'}.items():body=body.replace(old,new)
    folder=HERE/'paper';folder.mkdir(exist_ok=True);tex=folder/'s27b_report.tex';tex.write_text(body,encoding='utf-8',newline='\n')
    temp=HERE/'tmp/pdfs/report';temp.mkdir(parents=True,exist_ok=True)
    r=subprocess.run(['D:/bianyiqi/texlive/2026/bin/windows/latexmk.exe','-pdf','-interaction=nonstopmode','-halt-on-error','-outdir='+temp.as_posix(),tex.name],cwd=folder,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (temp/'build_stdout.txt').write_bytes(r.stdout);assert r.returncode==0,'See report build_stdout.txt'
    out=HERE/'output/pdf/s27b_mechanism_report.pdf';out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes((temp/'s27b_report.pdf').read_bytes())
    render=temp/'render';render.mkdir(exist_ok=True)
    subprocess.run(['D:/bianyiqi/texlive/2026/bin/windows/pdftoppm.exe','-png','-r','115',str(out),str(render/'page')],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    d=dict(path=str(out),sha256=sha(out),pages=len(PdfReader(out).pages),render=str(render),log=str(temp/'s27b_report.log'))
    write(HERE/'DOCUMENT_BUILD.json',dict(time=now(),status='COMPILED_RENDERED_REVIEW_PENDING',documents=[d]));print(d,flush=True)


if __name__=='__main__':main()
