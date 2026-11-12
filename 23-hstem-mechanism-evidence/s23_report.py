"""Exploratory conclusions and paper revisions from the fixed zero-fit analysis."""
import csv
import json
import itertools
from pathlib import Path
import numpy as np
from s23_core import HERE,ROOT,read,write,sha,now,validate

def rows(name):
    with (HERE/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def mean(data,key):return float(np.mean([float(r[key]) for r in data]))
def save_csv(name,data):
    with (HERE/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in data for k in r)),lineterminator='\n');w.writeheader();w.writerows(data)

def metadata_review():
    # Schemas only for explicitly authorized source receivers; no waveform reads.
    import h5py
    from s23_data import PROTOCOL
    records=[]
    for rx in sorted(set(r for fold in PROTOCOL['folds'] for r in fold['source'])):
        p=Path('D:/crossatten/stage_1/data/raw')/f'{rx}_train.h5'
        if not p.exists():records.append(dict(rx=rx,path=str(p),available=False));continue
        with h5py.File(p,'r') as f:
            fields={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in f.items() if isinstance(v,h5py.Dataset)}
            record=dict(rx=rx,path=str(p),available=True,fields=fields,root_attribute_names=list(f.attrs.keys()),waveform_arrays_read=False)
        records.append(record)
        with (HERE/'access.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=now(),dataset='Shen',role='source_metadata_schema',rx=rx,selected_packets=0,mode='read_only_no_waveform'))+'\n')
    write(HERE/'SOURCE_METADATA.json',dict(time=now(),records=records,cache_fields=['x','y','rows','hash manifest'],interpretation='Schema audit only;window rows do not supply adjacent continuous samples or a validated deployment synchronization-error distribution'))

def figures(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','savefig.dpi':180})
    folder=HERE/'figures';folder.mkdir(exist_ok=True)
    data=rows('kernel_spectra.csv');bands=rows('kernel_bands.csv');trajectories=rows('archived_trajectories.csv')
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    for arm in ['P0','S1','H1']:
        local=[r for r in data if r['arm']==arm];freq=sorted(set(float(r['frequency_cycles_per_frontend_sample']) for r in local))
        power=[mean([r for r in local if float(r['frequency_cycles_per_frontend_sample'])==f],'normalized_power') for f in freq]
        ax[0].plot(freq,10*np.log10(np.maximum(power,1e-15)),label=arm)
    ax[0].set(xlabel='Cycles per frontend feature sample',ylabel='Normalized mean kernel power (dB)',title='Learned effective grouped kernels');ax[0].legend()
    for arm in ['P0','S1','H0','S2','H2']:
        values=[mean([r for r in trajectories if r['model'].startswith(arm+'_') and int(r['epoch'])==e],'source_val_ce') for e in range(1,201)]
        ax[1].plot(range(1,201),values,label=arm)
    ax[1].set(xlabel='Archived epoch',ylabel='Source validation CE',yscale='log',title='Existing trajectories, no refitting');ax[1].legend()
    operator_data=rows('operator_spectrum.csv');ax[2].plot([float(r['cycles_per_weight_index']) for r in operator_data[:129]],[float(r['fourier_direction_gain']) for r in operator_data[:129]])
    ax[2].set(xlabel='Kernel-coordinate Fourier direction',ylabel='Direction gain of T T^T',title='SGD geometry; not an AdamW formula')
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(folder/f'parameterization.{ext}',bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4.2));timing=summary['timing'];shifts=[-4,-1,0,1,4]
    for arm in ['H0','P0']:
        for kind,style in [('zero','--'),('circular','-')]:
            values=[next(r['accuracy_pct'] for r in timing if r['arm']==arm and r['condition']==('clean' if s==0 else f'{kind}_{s:+d}')) for s in shifts]
            ax[0].plot(shifts,values,style,marker='o',label=arm+' '+kind)
    ax[0].set(xlabel='Input sample shift',ylabel='Source RX-equal accuracy (%)',ylim=(0,102),title='All predefined shift types reported');ax[0].legend(fontsize=8)
    equivalence_rows=rows('equivalence.csv')
    for i,arm in enumerate(['S1','H0']):
        values=[float(r['max_abs']) for r in equivalence_rows if r['arm']==arm]
        ax[1].scatter(np.full(len(values),i)+np.linspace(-.08,.08,len(values)),values,label=arm)
    ax[1].set(xticks=[0,1],xticklabels=['S1 to standard Conv','H0 Conv-BN folding'],ylabel='Maximum absolute logit difference',title='All six source-only conversions')
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(folder/f'equivalence_and_timing.{ext}',bbox_inches='tight')
    plt.close(fig)

def report():
    complete=read(HERE/'ANALYSIS_COMPLETE.json');timing=rows('timing_metrics.csv');summary=[]
    for arm in ['H0','P0']:
        for condition in sorted(set(r['condition'] for r in timing)):
            local=[r for r in timing if r['arm']==arm and r['condition']==condition]
            summary.append(dict(arm=arm,condition=condition,accuracy_pct=mean(local,'accuracy_pct'),macro_f1_pct=mean(local,'macro_f1_pct'),ce=mean(local,'ce'),margin=mean(local,'margin')))
    save_csv('timing_summary.csv',summary)
    signals={}
    for arm in ['H0','P0']:
        clean=next(r['accuracy_pct'] for r in summary if r['arm']==arm and r['condition']=='clean')
        circular=[];zero=[];positive=[]
        for amount in [-4,4]:
            circular.append(clean-next(r['accuracy_pct'] for r in summary if r['arm']==arm and r['condition']==f'circular_{amount:+d}'))
            zero.append(clean-next(r['accuracy_pct'] for r in summary if r['arm']==arm and r['condition']==f'zero_{amount:+d}'))
            counts=0
            for fold,seed in itertools.product([0,1],[201,202,203]):
                base=[r for r in timing if r['arm']==arm and int(r['fold'])==fold and int(r['seed'])==seed and r['condition']=='clean']
                changed=[r for r in timing if r['arm']==arm and int(r['fold'])==fold and int(r['seed'])==seed and r['condition']==f'circular_{amount:+d}']
                counts+=int(mean(base,'accuracy_pct')>mean(changed,'accuracy_pct'))
            positive.append(counts)
        signals[arm]=dict(mean_circular4_drop_pp=float(np.mean(circular)),mean_zero4_drop_pp=float(np.mean(zero)),circular_drop_positive_pairs_by_sign=positive,boundary_dominant=bool(np.mean(circular)<=5 and np.mean(circular)<=.2*np.mean(zero)),zero_boundary_alone_insufficient=bool(np.mean(circular)>=20 and min(positive)>=5))
    write(HERE/'summary.json',dict(time=now(),timing=summary,signals=signals,equivalence=complete,operator=read(HERE/'OPERATOR.json')))
    metadata_review();figures(read(HERE/'summary.json'))
    lines=['# S23 数学等价与同步敏感性结果','', '本轮新增训练、优化步、架构搜索和保留RX/WiSig目标样本访问均为0。全部结果来自36个冻结模型和合法源验证包。','', '|模型|条件|Accuracy %|Macro-F1 %|CE|margin|','|---|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['arm']}|{r['condition']}|{r['accuracy_pct']:.3f}|{r['macro_f1_pct']:.3f}|{r['ce']:.4f}|{r['margin']:.3f}|")
    lines+=['','全部预注册零填充、循环与FFT幅度锁定条件同时报告。FFT锁定是内部视图诊断，不代表物理信号。时移实验不是独立确认，不能把循环移位等同真实空口同步误差。','', '数值等价验证：'+json.dumps(complete,ensure_ascii=False), '', '预注册描述信号：'+json.dumps(signals,ensure_ascii=False),'', 'FFT输入明确是幅度，不是显式频谱相位；时域atan2相位与频域相位须区分。有效核频谱坐标是前端特征索引，不是TX硬件频响。T的可逆性证明表达空间未缩小；轨迹、局部梯度和频谱仅提供优化相关线索，SGD预条件公式不是AdamW精确更新式。','', '当前数据只包含固定8192点窗口。SOURCE_METADATA.json给出已授权源HDF5的schema；缺少邻接波形或同步误差分布时，不能证明人工时移具备部署物理有效性。论文1重合审查及最终研究建议另列，停止于S23，不自动追加任何拟合。']
    (HERE/'RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')

def audit():
    import shutil
    reg=read(HERE/'registration.json');validate();complete=read(HERE/'ANALYSIS_COMPLETE.json')
    assert complete['models_recovered']==36 and complete['equivalence_checks']==12 and complete['new_fits']==0 and complete['optimizer_constructions']==0
    for key,ref in reg['references'].items():
        for name,digest in ref['sha256'].items():assert sha(Path(ref['path'])/name)==digest
    from s23_data import PROTOCOL,roles
    legal=set(r for fold in PROTOCOL['folds'] for r in fold['source'])
    entries=[json.loads(line) for line in (HERE/'access.jsonl').read_text(encoding='utf-8').splitlines()]
    for e in entries:
        assert e['dataset']=='Shen' and e['rx'] in legal and e['rx'] not in PROTOCOL['blind_receivers_never_open']
        assert e['role'] in ['source_val','source_metadata_schema']
        if e['role']=='source_val':assert e['rx'] in roles(e['fold'])['source'] and e['selected_packets']==400
        else:assert e['selected_packets']==0
    evidence=HERE/'evidence';evidence.mkdir(exist_ok=True);shutil.copyfile(HERE/'access.jsonl',evidence/'access.jsonl')
    manifest={p.relative_to(HERE).as_posix():sha(p) for p in HERE.rglob('*') if p.is_file() and p.suffix in ['.csv','.png','.svg'] and 'tmp' not in p.relative_to(HERE).parts}
    manifest['evidence/access.jsonl']=sha(evidence/'access.jsonl')
    write(HERE/'FINAL_AUDIT.json',dict(time=now(),status='PASS',new_fits=0,new_architecture_search=0,source_models_exact_recovered=36,equivalence_checks=12,equivalence_all_pass=complete['equivalence_all_pass'],prediction_disagreements=complete['prediction_disagreements'],all_original_artifacts_unchanged=True,source_only_access=True,blind_or_wisig_access=0,artifacts_sha256=manifest))

if __name__=='__main__':
    import sys
    audit() if '--audit' in sys.argv else report()
