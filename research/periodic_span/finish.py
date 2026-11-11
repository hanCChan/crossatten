"""Deployment cost measurement, complete archive and reports after sealed scoring."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import engine as E


def benchmark():
    if (HERE/'LATENCY.json').exists():raise FileExistsError('Keep original latency measurement')
    reg=E.read(HERE/'REGISTRATION.json')
    E.A.setup(24301)
    generator=torch.Generator(device='cuda').manual_seed(75002)
    x=torch.randn(128,2,256,generator=generator,device='cuda')
    models={}
    for arm in E.M.ARMS:
        ckpath=HERE/'artifacts'/f'V0_S0_{arm}'/'selected.pt'
        expected=E.read(HERE/'TRAINING_SEAL.json')['runs'][f'V0_S0_{arm}']['checkpoints']['selected']
        if E.sha(ckpath)!=expected:raise ValueError('Latency model hash mismatch')
        model=E.M.build(arm,24101).cuda().eval()
        model.load_state_dict(torch.load(ckpath,map_location='cpu',weights_only=False)['model'])
        models[arm]=model
    records={a:{} for a in E.M.ARMS}
    for round_index in range(3):
        order=E.M.ARMS[round_index:]+E.M.ARMS[:round_index]
        for arm in order:
            model=models[arm]
            before=E.semantic_hash(model.state_dict())
            for batch in (1,128):
                with torch.no_grad():
                    for _ in range(20):model(x[:batch])
                    torch.cuda.synchronize();events=[]
                    for _ in range(100):
                        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        a.record();model(x[:batch]);b.record();events.append((a,b))
                    torch.cuda.synchronize()
                values=np.asarray([a.elapsed_time(b) for a,b in events])
                records[arm].setdefault(str(batch),[]).append(dict(round=round_index,p50_ms=float(np.median(values)),
                    p95_ms=float(np.quantile(values,.95)),throughput_samples_per_second=float(1000*batch/values.mean())))
            if E.semantic_hash(model.state_dict())!=before:raise ValueError('Benchmark changed state')
    E.write(HERE/'LATENCY.json',dict(status='COMPLETE',records=records,environment=reg['environment'],
        models='V0/S0 selected endpoint for each arm, fixed in advance of measurement; no fastest-model selection',
        scope='Full RF frontend+HStem+MHA; synthetic CUDA-resident input; no disk/host transfer time',
        synthetic_input_seed=75002,warmup=20,measurements=100,rounds=3,batches=[1,128],time=E.now()))


def archive():
    jobs=E.read(HERE/'JOBS.json');qa=E.read(HERE/'INDEPENDENT_QA.json')
    if qa['status']!='PASS':raise ValueError('Independent QA required')
    target=HERE/'SOURCE_PREDICTIONS.zip'
    if target.exists():raise FileExistsError('Keep original prediction archive')
    files={}
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for job in jobs:
            for endpoint in E.ENDPOINTS:
                p=HERE/'artifacts'/job['id']/f'{endpoint}_selection.npz'
                name=p.relative_to(HERE/'artifacts').as_posix()
                z.write(p,name);files[name]=E.sha(p)
    # Retain full scientific states/orders locally; model bytes stay outside Git.
    snapshot=HERE/'artifacts'/'SCIENTIFIC_EXECUTION.zip'
    if snapshot.exists():raise FileExistsError('Keep original scientific execution archive')
    with zipfile.ZipFile(snapshot,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for job in jobs:
            for p in sorted((HERE/'artifacts'/job['id']).rglob('*')):
                if p.is_file():z.write(p,p.relative_to(HERE/'artifacts').as_posix())
    costs={}
    for arm in E.M.ARMS:
        values=[E.read(HERE/'artifacts'/j['id']/'COMPLETE.json') for j in jobs if j['arm']==arm]
        costs[arm]=dict(fits=len(values),updates=sum(v['updates'] for v in values),epochs=sum(v['epochs'] for v in values),
            training_seconds=sum(v['costs']['training_seconds'] for v in values),
            validation_seconds=sum(v['costs']['validation_seconds'] for v in values),
            wall_seconds=sum(v['total_wall_seconds'] for v in values),
            peak_cuda_allocated_gib=max(v['peak_cuda_allocated_gib'] for v in values))
    E.write(HERE/'COSTS.json',dict(status='COMPLETE',arms=costs,scientific_fits=24,epochs=960,updates=121920,
        successful_GPU_synthetic_trajectories=4,successful_GPU_synthetic_updates=24,
        engineering_attempts=E.read(HERE/'ENGINEERING_QA.json')['prior_engineering_attempt_directories'],
        environment=E.read(HERE/'REGISTRATION.json')['environment']))
    E.write(HERE/'ARTIFACT_ARCHIVE.json',dict(status='COMPLETE',scientific_raw_IQ_in_archives=False,
        predictions=dict(path=target.name,sha256=E.sha(target),bytes=target.stat().st_size,files=files),
        local_full_execution=dict(path=snapshot.relative_to(HERE).as_posix(),sha256=E.sha(snapshot),bytes=snapshot.stat().st_size,
            scope='All scientific initial and epoch states, optimizer/scheduler/RNG, orders, checkpoints, histories and predictions; no raw IQ'),
        checkpoint_weights_committed_to_git=False))


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=E.read(HERE/'SUMMARY.json');q=E.read(HERE/'INDEPENDENT_QA.json');r=E.read(HERE/'REGISTRATION.json')
    costs=E.read(HERE/'COSTS.json');latency=E.read(HERE/'LATENCY.json')
    if q['status']!='PASS':raise ValueError('Independent verification required')
    selected=s['endpoints']['selected'];agg=selected['aggregates'];pairs=selected['comparisons'];decision=s['decision']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'periodic-span-v1'})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors=['#64748b','#d78922','#176b87','#9b5eab']
    arm_names=list(E.M.ARMS);means=[agg[a]['rx_equal']['accuracy']*100 for a in arm_names]
    ax=axes[0,0];ax.bar(arm_names,means,color=colors,width=.65)
    for i,arm in enumerate(arm_names):
        points=[agg[arm]['per_stream'][f'S{k}']['accuracy']*100 for k in range(3)]
        ax.scatter(np.array([i-.1,i,i+.1]),points,color='black',s=15,zorder=3)
        ax.text(i,means[i]+.2,f'{means[i]:.2f}',ha='center',fontsize=9)
    ax.set(ylabel='RX-equal accuracy (%)',title='A. Selected checkpoints; dots are fixed streams',ylim=(max(0,min(means)-8),min(100,max(means)+3)))
    ax=axes[0,1];comp=pairs['P2-P1'];rx=sorted(comp['per_rx'])
    delta=np.array([comp['per_rx'][a]['accuracy_delta']*100 for a in rx])
    ax.barh(rx,delta,color=['#176b87' if v>=0 else '#b43b3b' for v in delta]);ax.axvline(0,color='black',lw=.8)
    ax.axvline(-2,color='#b43b3b',ls='--',lw=.8);ax.invert_yaxis()
    ax.set(xlabel='P2 minus P1 accuracy (pp)',title='B. Each observed source-selection RX')
    with gzip.open(HERE/'ANALYSIS.json.gz','rt',encoding='utf-8') as f:full=json.load(f)
    groups=full['endpoints']['selected']['comparisons']['P2-P0']['tx_rx_stream_mean']
    heat=np.full((len(rx),10),np.nan)
    for group in groups:heat[rx.index(group['rx']),group['tx_class_index']]=group['risk_increase']*100
    ax=axes[1,0];limit=max(5,float(np.nanmax(np.abs(heat))));cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#d1d5db')
    im=ax.imshow(heat,aspect='auto',cmap=cmap,vmin=-limit,vmax=limit)
    ax.set_yticks(range(len(rx)),rx,fontsize=8);ax.set_xticks(range(10));ax.set(xlabel='TX model class index',title='C. P2-P0 error-rate increase (pp); grey=absent')
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
    ax=axes[1,1];names=['P2-P1','P2-P0','P2-N19'];x=np.arange(3)
    values=np.array([pairs[a]['rx_equal']['accuracy_delta']*100 for a in names])
    intervals=np.array([pairs[a]['paired_rx_bootstrap_95_intervals']['accuracy_delta'] for a in names])*100
    ax.errorbar(x,values,yerr=np.stack((values-intervals[:,0],intervals[:,1]-values)),fmt='o',capsize=5,color='#176b87')
    ax.axhline(0,color='black',lw=.8);ax.axhline(1,color='#64748b',ls='--',lw=.8);ax.set_xticks(x,names)
    ax.set(ylabel='RX-equal accuracy difference (pp)',title='D. Conditional RX-cluster bootstrap intervals')
    fig.suptitle('Periodic-span HStem | 24 fits | source-CV development only',fontsize=14)
    fig.savefig(HERE/'RESULTS.png',dpi=170)
    path=HERE/'RESULTS.svg';fig.savefig(path,metadata={'Date':None})
    path.write_text('\n'.join(v.rstrip() for v in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close(fig)
    lines=['# 周期跨度HStem：完整源域竞争结果','',f"**24/24科学拟合完成，判定：`{decision['status']}`。** 固定窗口方案的STOP保持不变；本结果是重复使用合法源角色的开发竞争，不是正式CRD或独立新RX确认。",'',
        '## 主要端点：checkpoint_val CE选择的checkpoint','',
        '| 臂 | RX等权Accuracy | Macro-F1 | CE | 最差绝对RX Accuracy | 参数 |','|---|---:|---:|---:|---:|---:|']
    for arm in arm_names:
        a=agg[arm];worst=a['worst_rx']
        lines.append(f"| {arm} | {a['rx_equal']['accuracy']*100:.4f}% | {a['rx_equal']['macro_f1']*100:.4f}% | {a['rx_equal']['ce']:.6f} | {worst}: {a['per_rx'][worst]['accuracy']*100:.3f}% | {r['parameters'][arm]:,} |")
    lines+=['','P0为原频域骨干；P1为普通41抽头时域分支；P2为时域dilation20；N19为固定dilation19负对照。P1与P2卷积跨度相同，P1多608参数（0.726%），并非严格等参数或等FLOPs。原骨干公共初态、分支公共层、实际包键顺序与预算均已复核；P2/N19初始全部权重相同。','',
        '| 比较 | Accuracy差 / pp | RX条件95%区间 / pp | 最差配对RX差 / pp | 最大受支持TX×RX错误率增加 / pp |',
        '|---|---:|---|---:|---:|']
    for name in ('P2-P1','P2-P0','P2-N19'):
        c=pairs[name];ci=c['paired_rx_bootstrap_95_intervals']['accuracy_delta']
        lines.append(f"| {name} | {c['rx_equal']['accuracy_delta']*100:+.4f} | [{ci[0]*100:+.4f}, {ci[1]*100:+.4f}] | {c['worst_paired_rx']['accuracy_delta']*100:+.4f} | {c['worst_supported_tx_rx_risk_increase']['risk_increase']*100:+.4f} |")
    lines+=['','![完整比较图](RESULTS.png)','','## 冻结门槛与解释','']
    for name,promotion in decision['promotion_gates'].items():
        failed=[k for k,v in promotion['checks'].items() if not v]
        lines.append(f"- {name}：{'通过全部门' if promotion['passes'] else '未通过'}；失败项：{', '.join(failed) if failed else '无'}。")
        c=pairs[name];g=c['worst_supported_tx_rx_risk_increase']
        lines.append(f"  最大受支持局部风险：RX {g['rx']}、TX类索引{g['tx_class_index']}、每流{g['packets_per_stream']}包，平均错误率增加{g['risk_increase']*100:.4f}pp；全部{promotion['supported_tx_rx_groups']}个受支持组均参与门槛，不按三流相加制造支持量。")
    lines+=['',f"20点相对19点特异性条件：{json.dumps(decision['specificity_P2_minus_N19'],ensure_ascii=False)}。N19不参与调参或候选选型。",'']
    if decision['status'].startswith('STOP'):
        lines.append('本候选未获得继续资格，依事前规则停止；不追加seed、跨度、EMA、GroupDRO或新Loss。失败不能推断周期结构不含TX信息，也不能把本轮时间前端、分支宽度、门控与优化共同作用分解成已证实的硬件机制。')
    else:
        lines.append('本候选只获得“值得另行确认”的开发资格；尚需独立协议下与EMA完整配方及既有周期相位特征公平比较，并处理新RX确认资格。本轮不自动执行正式CRD或迁移拟合。')
    lines+=['','P2对P1是受控的同时间输入比较；P2对P0同时改变前端坐标和分支，不能视为纯粹的周期连接因果效应。共同RMS归一化和时域相位乘积不代表TX/RX分离。边界mask只保证数组索引有效，不保证跨区间抽头来自同一训练字段。N19的前置局部stem使19与20跨度的有效感受野重叠，因此它是邻近跨度对照，不是严格消除周期信息的干预；特异性条件通过或失败都不能独自证明TX物理机制。','',
        '## 预定辅助端点，未用于择优','','| 端点 | P0 Accuracy | P1 Accuracy | P2 Accuracy | N19 Accuracy |','|---|---:|---:|---:|---:|']
    for endpoint in ('selected','epoch10','final40'):
        a=s['endpoints'][endpoint]['aggregates'];lines.append('| '+endpoint+' | '+' | '.join(f"{a[arm]['rx_equal']['accuracy']*100:.4f}%" for arm in arm_names)+' |')
    lines+=['','## 验证、成本与复现','',
        f"11项CPU测试通过，真实训练循环的4条合成CUDA轨迹/24次更新通过。首次工程夹具在Windows使用int32标签失败，已在科学IQ读取前修正并保留失败目录；科学拟合24次、无失败重试。独立QA {q['check_count']:,}项通过，复核全部72份端点预测、逐组风险、bootstrap、初态、40轮顺序及全部artifact摘要；QA没有读取IQ。",'',
        '科学预算共960 epoch、121,920更新。各方法训练/验证与完整wall time如下；工程验收不并入科学fit数。','',
        '| 臂 | 训练秒 | 验证秒 | 6次fit总wall秒 | batch1 p50延迟/ms | batch128 p50延迟/ms |','|---|---:|---:|---:|---:|---:|']
    for arm in arm_names:
        c=costs['arms'][arm];l=latency['records'][arm]
        lines.append(f"| {arm} | {c['training_seconds']:.1f} | {c['validation_seconds']:.1f} | {c['wall_seconds']:.1f} | {np.median([v['p50_ms'] for v in l['1']]):.4f} | {np.median([v['p50_ms'] for v in l['128']]):.4f} |")
    lines+=['','延迟使用每臂V0/S0 selected模型、固定合成CUDA驻留输入，含完整RF前端+HStem+MHA；3轮旋转顺序，每轮20次warmup及100次CUDA-event测量，batch1/128。只报告设备计算延迟，不包含磁盘与主机传输，p95和吞吐见LATENCY.json。','',
        '原源域分割及三个配对流沿用C，wd=.001由此前公开源开发结果固定；这不是未见开发集上的确认。RX bootstrap只覆盖已观察12个源留出RX，三流先平均，不把包或模型当独立样本，多比较未校正。历史held6、target8、日期3–4、Full新RX以及LoRa/ManySig均未访问。','',
        '- [冻结协议](PREREGISTRATION.zh-CN.md)、[机器登记](REGISTRATION.json)、[24次作业](JOBS.json)。',
        '- [全部端点汇总](SUMMARY.json)、[完整分组与配对分析](ANALYSIS.json.gz)、[独立复核](INDEPENDENT_QA.json)。',
        '- [72份源选择预测归档](SOURCE_PREDICTIONS.zip)、[训练封存](TRAINING_SEAL.json)、[实际初态与序列证据](RUN_EVIDENCE.json)。',
        '- [成本](COSTS.json)、[延迟](LATENCY.json)、[本地全状态归档摘要](ARTIFACT_ARCHIVE.json)。',
        '- [原固定窗口STOP证据](../periodic_hstem/AUDIT_RESULT.json)，该文件SHA与新登记逐次核对一致。','',
        '空洞卷积是已有算子，见[Yu & Koltun](https://arxiv.org/abs/1511.07122)；本轮没有把既有算子、19点对照或源开发增益包装成原创性证明。', '']
    (HERE/'RESULTS.zh-CN.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(json.dumps(dict(status=decision['status'],scientific_fits=24,QA=q['status'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['benchmark','archive','report']);a=p.parse_args()
    {'benchmark':benchmark,'archive':archive,'report':report}[a.command]()
