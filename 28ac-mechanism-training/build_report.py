"""English PDF from actual fully verified A/C aggregates; no fitting or RF IO."""
import csv
import subprocess
from pypdf import PdfReader
from ac_common import HERE,read,write,sha,now

def main():
    integrity=read(HERE/'EXPERIMENT_INTEGRITY.json');assert integrity['status']=='PASS'
    r=read(HERE/'RESULTS.json');decision=read(HERE/'MECHANISM_DECISION.json');a=r['A'];c=r['C'];effect=a['primary_interaction'];ci=effect['conditional_rx_ci95_pp'];pilot=c['pilot']
    src={q['arm']:q for q in a['source_summary']}
    body=r'''\documentclass[10pt]{article}
\usepackage[letterpaper,margin=.8in]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,hyperref,xurl,float}
\emergencystretch=2em
\hypersetup{hidelinks}\graphicspath{{../figures/}}\setlength{\parskip}{4pt}
\title{S28-A and Original S27-C: Joint Mechanism Evidence}
\author{Completed registered execution record}\date{24 November 2026}
\begin{document}\maketitle
\section{Execution and Scientific Questions}
Starting from the S28-0 commit \texttt{de7d010}, explicit user authorization activates six BN-by-GELU control fits (A), followed by twelve fresh four-Source/one-held-RX fits (C). Both original JSON plans remain byte-identical. All eighteen fits complete 200 epochs. C is launched unconditionally with its original splits and seeds; all A Dev and C held inference is delayed until both complete cohorts are sealed and committed. No A outcome changes a C decision.

A asks whether the contribution of three tokenizer GELUs depends on three middle BN modules. C asks whether higher held accuracy coincides with a smaller Source-to-held loss. These questions and decisions are distinct; shared data and development history do not create statistically independent datasets. No S28-B, IQ-only, ASK Strict-IB, seed replacement or architecture search is run. After this report, the remaining automatic scientific fit budget is zero.
\section{A: All Four Cells and the Registered Interaction}
\begin{center}\small\begin{tabular}{lrrrrrr}\toprule
Arm&Params&Dev acc.&Fit SD&Macro-F1&Worst RX&Source acc.\\\midrule
'''
    for q,param in zip(a['dev_summary'],[133690,133690,133338,133338,138346]):
        name=q['arm'].replace('_',r'\_')
        body+=f"{name}&{param:,}&{q['accuracy_pct']:.3f}&{q['fit_sd_pp']:.3f}&{q['macro_f1_pct']:.3f}&{q['worst_rx_accuracy_pct']:.3f}&{src[q['arm']]['accuracy_pct']:.3f}\\\\\n"
    body+=r'''\bottomrule\end{tabular}\end{center}
H0 keeps BN/GELU; LA removes only the three GELUs; H2 removes only the three BN modules; H2\_LA removes both. LP is a reused secondary Long-Patch comparator. Accuracy/F1 are percentages and SD describes six fold/seed means in pp.
\[
\Delta_{BN\times GELU}=(A_{H0}-A_{LA})-(A_{H2}-A_{H2\_LA}).
\]
'''
    body+=f"The primary interaction is {effect['mean_delta_pp']:+.3f}pp, conditional RX95\\% interval [{ci[0]:.3f},{ci[1]:.3f}]. {effect['same_direction_fit_pairs']}/6 paired effects share its direction. Fold means are {effect['fold_deltas_pp'][0]:+.3f} and {effect['fold_deltas_pp'][1]:+.3f}pp. The registered two-sided practical rule is {'met' if effect['two_sided_practical_rule_met'] else 'not met'}.\n"
    if effect['two_sided_practical_rule_met']:
        body+=('The GELU benefit is larger with middle BN under this complete training procedure.\n' if effect['mean_delta_pp']>0 else 'The GELU benefit is larger without middle BN under this complete training procedure. This negative interaction must not be rewritten as positive synergy.\n')
    else:body+='A stable practically large interaction is not established; this is neither an equivalence result nor proof that BN or GELU is unimportant.\n'
    body+=r'''The unchanged rule requires magnitude at least .5pp, a conditional interval excluding zero, at least5/6 same-direction pairs and both fold means in that direction. Three seeds are averaged within each of ten unique RXs before10,000 bootstrap repetitions, seed28028. The intervals condition on the observed seeds, packets and architecture choices; they do not include fresh-acquisition or selection uncertainty or multiplicity-corrected confirmatory claims.
\clearpage
\begin{figure}[H]\centering\includegraphics[width=.98\linewidth]{A_interaction.pdf}
\caption{All paired A accuracies and every RX interaction, including counterexamples.}\end{figure}
\section{A: Simple Effects and Causal Limits}
\begin{center}\small\begin{tabular}{lrr}\toprule
Descriptive contrast&Mean(pp)&Conditional95\% RX interval\\\midrule
'''
    for q in a['secondary']:
        body+=q['name'].replace('_',r'\_')+f"&{q['mean_delta_pp']:+.3f}&[{q['conditional_rx_ci95_pp'][0]:.3f},{q['conditional_rx_ci95_pp'][1]:.3f}]\\\\\n"
    body+=r'''\bottomrule\end{tabular}\end{center}
The two shared-stem BN modules, two LeakyReLUs and MHA remain. Removing middle BN removes352 affine parameters and changes batch statistics, running statistics and the optimization trajectory jointly. The intervention compares complete training procedures, not a unique normalization or factorization cause. LP still differs in grouping, free parameters, initialization and computation. Source validation already selects checkpoints. Simple effects, absolute Source learning and the factorial interaction must be read together; secondary contrasts do not replace the primary rule.
With BN present, GELU adds .752pp on average; without BN its mean effect is -.598pp, with a conditional interval crossing zero. The latter does not by itself establish harm from GELU. The double-removal model still reaches 96.743\% and exceeds LP by 2.243pp on average; all six paired differences are positive. Thus these BN/GELU interventions do not exhaust the observed LP difference; that residual still cannot uniquely identify factorized parameterization as the cause.
\clearpage
\section{C: Absolute Held Performance and Gap Reduction}
The original fold0 training RXs are rtl1/rtl3/rtl6/rtl9 with b210\_1 held; fold1 trains b200\_mini\_1/pluto\_1/rtl2/rtl7 with rtl\_5 held. Fresh H0/LP use the fixed S27C-H4x160 seed domain and6400/1600 train/validation packets. No five-RX checkpoint is loaded. Only the four training RXs choose checkpoints; all2000 held packets are evaluated after all twelve C models seal.
\[
G=A_{\rm SourceVal}-A_{\rm Held},\qquad\Delta G=G_{H0}-G_{LP}.
\]
\begin{center}\small\begin{tabular}{lrrr}\toprule
Held RX&Source H0-LP(pp)&Held H0-LP(pp)&$\Delta G$(pp)\\\midrule
'''
    for q in pilot['fold_means']:body+=q['held_rx'].replace('_',r'\_')+f"&{q['source_delta_pp']:+.3f}&{q['held_delta_pp']:+.3f}&{q['delta_gap_pp']:+.3f}\\\\\n"
    body+=r'''\bottomrule\end{tabular}\end{center}
'''
    body+=f"The original consistent-pilot rule is {'met' if pilot['consistent_pilot_signal'] else 'not met'}. {pilot['favorable_fit_pairs']}/6 pairs simultaneously have positive held improvement and negative gap difference. Across six repeats, mean held improvement is {pilot['means']['held_delta_pp']:+.3f}pp, Source improvement {pilot['means']['source_delta_pp']:+.3f}pp and gap difference {pilot['means']['delta_gap_pp']:+.3f}pp.\n"
    body+=r'''The unchanged rule additionally requires held improvement at least .5pp and $\Delta G\leq-.5$pp in both RX/fold means. A smaller gap alone is insufficient if held identification is worse. Source ceilings can distort gaps. Two held RXs do not become six independent receiver units: no population-RX significance or bootstrap is reported. Higher held accuracy without stable gap reduction supports stronger overall discrimination, rather than an established RX-specific suppression mechanism.
All six held-accuracy differences favor HStem. Mean HStem/LP held accuracy is 89.367/84.783\% on b210\_1 and 95.167/93.783\% on rtl\_5. The rtl\_5 gap reduction is .342pp, below the frozen .5pp threshold; it remains a favorable mean direction, without passing the registered criterion. The b210\_1 seed 202 counterexample improves held accuracy by 1.150pp but improves Source by 2.000pp, enlarging the gap by .850pp. Neither the threshold nor this repeat is changed after evaluation.
\begin{figure}[H]\centering\includegraphics[width=.98\linewidth]{C_source_held.pdf}
\caption{All twelve new C models, Source validation to held accuracy. Both RXs and all seeds remain visible.}\end{figure}
\clearpage
\section{C: Every Paired Repeat and Fixed Diagnostics}
\begin{center}\small\begin{tabular}{rrrrrrrr}\toprule
Fold&Seed&H0 Source&H0 Held&LP Source&LP Held&Held diff.&$\Delta G$\\\midrule
'''
    for q in pilot['pairs']:
        body+=f"{q['fold']}&{q['seed']}&{q['H0']['source_accuracy_pct']:.3f}&{q['H0']['held_accuracy_pct']:.3f}&{q['LP']['source_accuracy_pct']:.3f}&{q['LP']['held_accuracy_pct']:.3f}&{q['held_delta_pp']:+.3f}&{q['delta_gap_pp']:+.3f}\\\\\n"
    body+=r'''\bottomrule\end{tabular}\end{center}
\begin{figure}[H]\centering\includegraphics[width=.80\linewidth]{C_gap_pairs.pdf}
\caption{Held accuracy gain and gap reduction jointly. Outlined points are means of the same two RX units, not new independent observations.}\end{figure}
All C models use three predeclared64D taps: tokenizer mean, pre-backend mean and final classifier input. Pooling precedes per-packet L2 normalization. Source geometry uses all1600 validation packets,40/TX/RX; held geometry uses only the original last40/TX,400 packets, while classification uses all2000 held packets. All72 layer records and72,000 vectors remain saved. All layers, normalization, class errors, CE/F1 and true-class margins are retained.

Held-only geometry has a single RX; conditional across-RX scatter is zero by construction. Its $J_c$ is marked not identifiable and the algebraic epsilon ratio is saved separately, without an invariance interpretation. $J_w$, scatter and margins are descriptive: Source selected the checkpoints, Source/Held RX counts differ, and jointly trained representations do not establish causal information flow. S27-B's counterexample to a sufficient final-$J_w$ explanation remains valid.
With unit-normalized packets, let $B$ be TX-centroid scatter about the global centroid, $D$ the TX/RX-centroid scatter about each TX centroid, and $W$ the packet scatter within TX/RX cells. Balanced cell averages retain the S25 definitions $J_c=B/(D+10^{-12})$ and $J_w=B/(D+W+10^{-12})$; the latter denominator includes both receiver-centroid and within-cell dispersion.

All three-seed layer means are retained below; held $J_c$ is undefined and omitted.
\begin{center}\scriptsize\begin{tabular}{llrrrrrr}\toprule
Fold&Tap&Source $J_c$ H0&LP&Source $J_w$ H0&LP&Held $J_w$ H0&LP\\\midrule
'''
    with (HERE/'C_geometry_all_layer_means.csv').open(encoding='utf-8',newline='') as f:
        for q in csv.DictReader(f):
            label=q['layer'].replace('_',r'\_')
            body+=f"{q['fold']}&{label}"+''.join(f"&{float(q[k]):.3f}" for k in ['source_Jc_H0','source_Jc_LP','source_Jw_H0','source_Jw_LP','held_Jw_H0','held_Jw_LP'])+r'\\'+'\n'
    body+=r'''\bottomrule\end{tabular}\end{center}
In fold 1 at the pre-backend tap, HStem has lower Source $J_c$, Source $J_w$ and held $J_w$ than LP, despite higher mean held accuracy. The same fold also has slightly lower tokenizer Source $J_c$. Higher final $J_w$ in both folds does not erase these counterexamples or prove monotonic receiver-information removal.
\section{Precision and Research History}
S28-0 established a finite affine tokenizer only conditional on the shared nonlinear stem and frozen evaluation BN, with explicit first/last-token corrections. Its native TF32 runtime failed the original tolerance and changed1/12000 predictions. The separate TF32-off300-packet probe qualified numerical export; it did not replace that failure or the historical recipe. This execution trains original architectures in the original FP32-tensor/matmul-TF32-off/cuDNN-TF32-on mode, never materialized substitutes.

All receiver data have development history. A uses the historical five-Source protocol, including physical RXs later held from the distinct freshly fitted C models. C is a Source-internal held-RX pilot; it is not a new independent acquisition or blind test. No unknown-TX rejection or security attack claim follows from closed-set accuracy.
\section{Every New Fit, Training Cost and Audit}
\begin{center}\small\begin{tabular}{llrrrrr}\toprule
Stage&Arm/Fold/Seed&Best ep.&Source CE&Source acc.&Updates&Wall(s)\\\midrule
'''
    with (HERE/'all_new_fits.csv').open(encoding='utf-8',newline='') as f:fits=list(csv.DictReader(f))
    for q in fits:
        label=(q['arm']+'/'+q['fold']+'/'+q['seed']).replace('_',r'\_')
        body+=f"{q['stage']}&{label}&{q['best_epoch']}&{float(q['source_ce']):.5f}&{float(q['source_accuracy_pct']):.3f}&{q['updates']}&{float(q['wall_seconds']):.1f}\\\\\n"
    body+=r'''\bottomrule\end{tabular}\end{center}
All eighteen fits run200epochs without early stopping, reseeding or target-driven tuning. A has125updates/epoch; C has100, yielding390,000 retained updates. Atomic full-epoch snapshots include current and best models, optimizer, all RNG states and history. Any discarded partial-epoch work after a pause is not counted as retained updates. Summed fit wall times overlap under isolated CUDA workers and are not exclusive GPU hours. Engineering synthetic updates are counted separately from scientific fits.
\begin{figure}[H]\centering\includegraphics[width=.98\linewidth]{all_training.pdf}
\caption{Every new source-CE trajectory, all18fits and all200epochs.}\end{figure}
\clearpage
\section{Evidence Boundaries and Stop}
'''
    body+=('A supports a signed interaction of the selected BN/GELU training procedures.\n' if effect['two_sided_practical_rule_met'] else 'A does not meet the practical signed-interaction rule; this does not prove equality or irrelevance.\n')
    body+=('C meets its bounded two-RX pilot criterion for higher held accuracy with smaller degradation.\n' if pilot['consistent_pilot_signal'] else 'C does not meet its joint two-RX criterion; absolute discrimination and receiver-specific gap reduction must remain distinct.\n')
    body+=r'''Neither outcome uniquely identifies factorized optimization or physical receiver invariance. The evidence matrix updates the completed A/C claims while retaining previous mathematical, preprocessing and historical-data limits. No favorable result automatically rewrites TIFS contributions. The remaining automatic fitting budget is zero. Any future same-function factorized-versus-direct optimization study needs matched boundary constraints, freedom, initialization and optimizer-state auditing; no S28-B arm is started here.
\section{Reproduction and Integrity}
The activation, original plans, engineering gate, seeds, permissions and execution hashes were committed and pushed before Source IO. A5RX uses8,000/2,000 packets and P-H5x160 seeds; C4RX uses6,400/1,600 and the frozen new seed domain. Minimum clean Source RX-equal CE alone selects checkpoints. Data order/noise and surviving initialization pairings are checked. Both complete cohort seals precede every new Dev/held read.

The independent audit recomputes metrics from324,000 saved target and79,200 Source predictions, all C geometry, the A conditional interaction and C gap rule. It separately reruns all eighteen new models on permitted Source validation and restores their saved logits exactly. It checks all200-epoch histories, best-source checkpoints, actual RF streams and hashes. Twenty-four old A baseline models reproduce archived Dev logits elementwise; their weights/configurations/histories remain unchanged. No Confirm/blind data or target adaptation is introduced. Larger waveforms, checkpoint snapshots and per-packet arrays stay local; code, full logs, aggregate tables and audit records are published.

After all eighteen models sealed, the first evaluation encountered a filename/payload argument-name collision in prediction saving, before any Dev or held read. A separate committed IO-only wrapper changed only that storage parameter, preserving all original files, errors, models, formulas and plans. A synthetic writer fixture passed and the same sealed cohort continued; no scientific fit or optimizer step was repeated by the repair. The original and recovery logs and hashes are retained.

The Chinese report supplies all numerical contrasts, per-RX/TX tables, limitations and next-step recommendations. The task stops at this complete registered evidence record, without extra fits or independent-generalization certification.
\end{document}
'''
    for old,new in {'least5/6':'least 5/6','before10,000':'before 10,000','seed28028':'seed 28028','RX95':'RX 95',
        'Conditional95':'Conditional 95','removes352':'removes 352','fold0':'fold 0','fold1':'fold 1','and6400/1600':'and 6,400/1,600',
        'all2000':'all 2,000','predeclared64D':'predeclared 64D','all1600':'all 1,600','packets,40':'packets, 40','last40':'last 40',
        'TX,400':'TX, 400','All72':'All 72','and72,000':'and 72,000','changed1/12000':'changed 1/12000','TF32-off300':'TF32-off 300',
        'run200epochs':'run 200 epochs','has125updates':'has 125 updates','has100':'has 100','yielding390,000':'yielding 390,000',
        'all18fits':'all 18 fits','all200epochs':'all 200 epochs','A5RX':'A 5-RX','uses8,000':'uses 8,000','C4RX':'C 4-RX','uses6,400':'uses 6,400',
        'from324,000':'from 324,000','and79,200':'and 79,200','all200-epoch':'all 200-epoch'}.items():body=body.replace(old,new)
    folder=HERE/'paper';folder.mkdir(exist_ok=True);tex=folder/'s28_ac_results.tex';tex.write_text(body,encoding='utf-8',newline='\n')
    temp=HERE/'tmp/pdfs/report';temp.mkdir(parents=True,exist_ok=True)
    p=subprocess.run(['D:/bianyiqi/texlive/2026/bin/windows/latexmk.exe','-pdf','-interaction=nonstopmode','-halt-on-error','-outdir='+temp.as_posix(),tex.name],cwd=folder,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (temp/'build_stdout.txt').write_bytes(p.stdout);assert p.returncode==0,'See build_stdout.txt'
    out=HERE/'output/pdf/s28_ac_mechanism_results.pdf';out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes((temp/'s28_ac_results.pdf').read_bytes())
    render=temp/'render';render.mkdir(exist_ok=True)
    subprocess.run(['D:/bianyiqi/texlive/2026/bin/windows/pdftoppm.exe','-png','-r','115',str(out),str(render/'page')],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    write(HERE/'DOCUMENT_BUILD.json',dict(time=now(),documents=[dict(path=str(out),sha256=sha(out),pages=len(PdfReader(out).pages),render=str(render),log=str(temp/'s28_ac_results.log'))]))
    print('Complete A/C PDF built and rendered; visual review pending.',flush=True)

if __name__=='__main__':main()
