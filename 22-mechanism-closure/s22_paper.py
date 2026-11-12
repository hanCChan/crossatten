"""Paper assets from fixed archived results; never opens blind or raw data."""
import argparse
import csv
import itertools
from pathlib import Path
import re
import numpy as np
from s22_core import HERE,ROOT,S13,read,write,sha,now,job_id

PAPER=HERE/'paper'
S21_CORRECTION='Overlap-Patch恢复了原HStem优势的63.151%，且两个开发折方向一致；插值同时改变局部支持、平滑和频响，尚不能唯一归因于重叠。去除中间BN导致6/6配对退化、开发准确率下降1.190pp，说明这些层对当前结构的训练或泛化有重要作用。两种因素均有证据支持，其贡献不可直接相加，不能强行归因于单一机制。'

def save_csv(name,rows):
    with (PAPER/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows)

def repair_display():
    changed=[]
    for name in ['MANUSCRIPT_DRAFT.zh-CN.md','RESULTS.zh-CN.md']:
        p=ROOT/'21-hstem-mechanisms'/name;text=p.read_text(encoding='utf-8');lines=text.splitlines()
        bad=[i for i,line in enumerate(lines) if 'Overlap-Patch' in line and re.search(r'\?{3,}',line)]
        if bad:
            before=sha(p)
            for i in bad:lines[i]=S21_CORRECTION
            p.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n');changed.append(dict(file=p.relative_to(ROOT).as_posix(),before_sha256=before,after_sha256=sha(p),semantic_change=False))
    write(HERE/'ENCODING_REPAIR.json',dict(time=now(),scope='Presentation only in new S22 branch; S21 branch, raw metrics and science unchanged',changed=changed))

def eligibility():
    protocol=read(ROOT/'configs/protocol.json')
    write(HERE/'BLIND_QUALIFICATION.json',dict(time=now(),status='NOT_QUALIFIED_AS_NEW_UNTOUCHED_HOLDOUT',receivers=protocol['blind_receivers_never_open'],reason=protocol['blind_note'],source_files={name:sha(ROOT/name) for name in ['configs/protocol.json','docs/PROTOCOL.md']},data_opened=False,new_evaluation_authorized=False))
    (PAPER/'BLIND_QUALIFICATION.zh-CN.md').write_text('# 最终blind评估资格检查\n\n本轮未读取六台blind RX的原始文件、cache、样本或模型输出，也未执行任何blind评价。\n\n现有configs/protocol.json的blind_note与docs/PROTOCOL.md第23行明确记载：六台RX曾被论文1作为盲测使用。因此，本协议中未打开它们，不等于它们是整个研究历史中从未使用的独立测试集。不得把未来重复使用称作全新盲测或新的独立确认。\n\n这项检查只依据已有协议和文档，不能证明其他项目中所有历史暴露均已穷尽。未来若评价这些保留RX，需要单独授权并披露历史使用；严格的全新独立确认仍需未暴露的数据或新采集。当前权限维持关闭，不因模型或论文冻结而自动开放。\n',encoding='utf-8',newline='\n')

def setup_plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','savefig.dpi':180})
    return plt

def architecture():
    plt=setup_plots()
    from matplotlib.patches import FancyBboxPatch
    fig,ax=plt.subplots(figsize=(13,3.8));ax.set(xlim=(0,13),ylim=(0,4));ax.axis('off')
    labels=['IQ 2 x 8192\nRMS + TrueIB','5 RF views\nIQ / FFT / AP','Local frontend\nk7 + k5, BN','HStem\n4 stride-4 stages','32 x 64 tokens\nposition + Pre-LN','2 MHA blocks\nmean + classifier']
    for i,label in enumerate(labels):
        x=.1+i*2.15
        ax.add_patch(FancyBboxPatch((x,1.5),1.9,1.2,boxstyle='round,pad=0.06',facecolor='#e8f1f8' if i!=3 else '#cfe8da',edgecolor='#31536d'))
        ax.text(x+.95,2.1,label,ha='center',va='center')
        if i<5:ax.annotate('',(x+2.12,2.1),(x+1.95,2.1),arrowprops=dict(arrowstyle='->',color='#31536d'))
    ax.text(6.5,3.45,'Frozen main model: HStem + MHA, 133,690 parameters',ha='center',fontsize=13,weight='bold')
    ax.text(6.5,.75,'Tokenizer: Conv(32,48,8/4/2)-BN-GELU -> Conv(48,64,8/4/2)-BN-GELU',ha='center')
    ax.text(6.5,.35,'-> Conv(64,64,8/4/2)-BN-GELU -> Conv(64,64,4/4/0); local CNN support 350, stride 256',ha='center')
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(PAPER/f'model_structure.{ext}',bbox_inches='tight')
    plt.close(fig)

def historical_tables():
    rows=[];registration=read(ROOT/'15-hstem-attribution/registration.json')
    for arm in ['E1_patch_mha','E2_patch_hstu','A0_cnn','A3_rfhstu']:
        for role in ['dev_val','confirm']:
            observations=[];params=[];provenance=[]
            for fold,seed in itertools.product([0,1],[201,202,203]):
                name=job_id(dict(arm=arm,fold=fold,seed=seed))
                root=Path(registration['reused'][name]['root']) if arm.startswith('E') else S13
                path=root/'runs'/name
                if not (path/'target_metrics.json').exists():raise ValueError('Missing historical anchor: '+str(path))
                d=read(path/'target_metrics.json')[role];observations.extend(d['per_rx'].values());params.append(read(path/'config.json')['params']);provenance.append(dict(path=str(path),metrics_sha256=sha(path/'target_metrics.json')))
            rows.append(dict(arm=arm,role=role,accuracy_pct=float(np.mean([m['acc']*100 for m in observations])),macro_f1_pct=float(np.mean([m['macro_f1']*100 for m in observations])),parameters=params[0],fits=6,scope='S13/S14 historical same protocol anchors; mixers differ; no new forward'))
            write(PAPER/f'provenance_{arm}_{role}.json',provenance)
    save_csv('historical_mixer_patch_anchors.csv',rows)
    source=ROOT/'06-paper-validation/summary/paper/main_metrics.csv'
    with source.open(encoding='utf-8-sig') as f:external=[dict(r,scope='Historical S06; implementation/selection differs; do not pool as S22 paired fits') for r in csv.DictReader(f) if r['dataset']=='Shen']
    if not external:raise ValueError('No archived Shen comparisons')
    save_csv('historical_shen_external_methods.csv',external)

def write_draft():
    base=(ROOT/'21-hstem-mechanisms/MANUSCRIPT_DRAFT.zh-CN.md').read_text(encoding='utf-8')
    base=base.replace('S21的实际结果及贡献边界如下；训练完成前不预填机制结论。','S21的已完成结果及贡献边界如下。')
    addition='\n\nS22将H0 HStem+MHA固定为唯一主模型，RF-HSTU仅作Mixer对照。S1使用256→340→256固定线性插值与sqrt(256/340)尺度，全部256个基核位置有效，局部CNN支持为266；它是训练前针对中心裁剪秩亏问题作出的设计修订。S2将三个中间BN替换为初始化为1/0的通道仿射，保留352个参数、去除对应统计量，局部支持仍为350。两个新结构共12次拟合，按源CE选模后统一封存。\n\n主要比较为H1−S1、S1−P0、H0−S2、S2−H2，采用事前固定的0.5pp实用门槛、两折方向、配对、RX区间与风险规则。全部比较共同报告，未作多重比较校正。时移和AWGN仅在冻结H0/P0的源验证包上进行，是有限压力测试，不是新的跨RX或盲测证据。\n\n模型结构图见model_structure.svg；主消融、RX表、成本和机制图由同一冻结输出生成。历史S06外部方法独立列出，训练实现和选择方式的差异必须保留，不与S22配对样本合并。\n\n[卷积token已有CvT先例](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.html)；[抗混叠与平移稳定性已有研究](https://proceedings.mlr.press/v97/zhang19a.html)；[IB-RFF官方代码](https://github.com/BeechburgPieStar/IB-RFF)是现有跨RX对照。本文差异仅可限定为严格带内LoRa、有限源预算下的RF token形成与受控归一化证据，不能把卷积token、平滑、BN、普通MHA或信息瓶颈称作首创。当前证据不足以宣称普遍SOTA或完整物理机制。\n\n六台保留RX有论文1使用历史。本轮不打开它们，未来评价也不自动构成全新独立验证，见BLIND_QUALIFICATION.zh-CN.md。\n'
    if (HERE/'summary.json').exists():
        summary=read(HERE/'summary.json');dev={r['arm']:r for r in summary['arms'] if r['role']=='dev_val'}
        addition+='\nS22已完成：S1开发准确率'+f"{dev['S1']['acc']:.3f}%"+'，S2为'+f"{dev['S2']['acc']:.3f}%"+'。四个主要比较如下，未触发门槛不等于无效。\n\n'
        for r in summary['contrasts']:
            if r['role']=='dev_val':addition+=f"- {r['contrast']}: {r['mean_delta_pp']:+.3f}pp，条件RX95%区间[{r['ci95_pp'][0]:+.3f},{r['ci95_pp'][1]:+.3f}]，正配对{r['positive_pairs']}/6，实用信号{r['practical_signal']}。\n"
        addition+='\n已证实的是这些固定系统与干预的实际效果；仍未隔离局部支持与频响、统计归一化与优化动力学，也没有识别交互贡献。所有结论须连同逐RX风险和真实成本一起陈述。本轮停止新结构搜索。\n'
    else:addition+='\nS22结果待12次拟合及严格恢复评价全部通过后填入。当前稿件不预设哪种机制成立。\n'
    (PAPER/'MANUSCRIPT_DRAFT.zh-CN.md').write_text(base+addition,encoding='utf-8',newline='\n')

def result_assets():
    import shutil
    summary=read(HERE/'summary.json');save_csv('main_ablation_table.csv',summary['arms'])
    for name in ['receiver_metrics.csv','tx_rx_deltas.csv','paired_contrasts.csv','class_geometry.csv','robustness_source.csv']:
        shutil.copyfile(HERE/name,PAPER/name)
    costs=read(HERE/'costs.json')['rows'];aggregated=[]
    for arm in ['H0','P0','H1','H2','S1','S2']:
        a=[r for r in costs if r['arm']==arm]
        aggregated.append(dict(arm=arm,parameters=a[0]['params'],counted_macs=a[0]['counted_macs'],batch1_p50_ms=float(np.median([r['p50_ms'] for r in a if r['batch']==1])),batch128_p50_ms=float(np.median([r['p50_ms'] for r in a if r['batch']==128])),scope='Same-GPU resident FP32;3 repeats;FFT/interpolation/elementwise incomplete in MACs'))
    save_csv('cost_table.csv',aggregated)
    plt=setup_plots();fig,axes=plt.subplots(1,3,figsize=(14,4.2));arms=['P0','S1','H1','H2','S2','H0'];dev={r['arm']:r for r in summary['arms'] if r['role']=='dev_val'}
    axes[0].bar(arms,[dev[a]['acc'] for a in arms],color=['#7892a4','#d19a46','#4f8e86','#a684aa','#a36b50','#426a9d']);axes[0].set(ylim=(min(dev[a]['acc'] for a in arms)-1,100),ylabel='Dev RX-equal accuracy (%)',title='Frozen models; six fits each')
    contrasts=[r for r in summary['contrasts'] if r['role']=='dev_val']
    for i,r in enumerate(contrasts):
        mean=r['mean_delta_pp'];lo,hi=r['ci95_pp'];axes[1].errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=4,color='#426a9d')
    axes[1].set(yticks=range(4),yticklabels=[r['contrast'] for r in contrasts],xlabel='Accuracy difference (pp)',title='Conditional RX95% intervals');axes[1].axvline(0,color='gray',lw=1);axes[1].axvline(.5,color='#b38336',ls='--',lw=1)
    byarm={r['arm']:r for r in aggregated};axes[2].scatter([byarm[a]['batch128_p50_ms'] for a in arms],[dev[a]['acc'] for a in arms],s=60,color='#426a9d')
    for a in arms:axes[2].annotate(a,(byarm[a]['batch128_p50_ms'],dev[a]['acc']),xytext=(4,4),textcoords='offset points')
    axes[2].set(xlabel='Batch128 latency p50 (ms)',ylabel='Dev accuracy (%)',title='Measured cost, not parameter proxy')
    fig.tight_layout()
    for ext in ['svg','png']:fig.savefig(PAPER/f'mechanism_results.{ext}',bbox_inches='tight')
    plt.close(fig)

def run(prepare=False):
    PAPER.mkdir(exist_ok=True)
    if not (HERE/'ENCODING_REPAIR.json').exists():repair_display()
    eligibility();architecture();historical_tables();write_draft()
    if not prepare:result_assets()
    checked=[]
    for p in list(PAPER.glob('*.md'))+[ROOT/'21-hstem-mechanisms'/n for n in ['RESULTS.zh-CN.md','MANUSCRIPT_DRAFT.zh-CN.md']]:
        text=p.read_text(encoding='utf-8');assert '\ufffd' not in text and not re.search(r'\?{3,}',text),p;checked.append(p.name)
    write(HERE/'PAPER_QA.json',dict(time=now(),status='PREPARED' if prepare else 'COMPLETE',unicode_display_checked=checked,main_model='H0 HStem+MHA',blind_data_opened=False,files={p.relative_to(HERE).as_posix():sha(p) for p in PAPER.iterdir() if p.is_file()}))
    print('Paper assets prepared' if prepare else 'Paper tables, diagrams, draft and blind qualification complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args();run(a.prepare)
