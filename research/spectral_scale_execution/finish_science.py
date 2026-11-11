"""Costs, complete public evidence and report after fixed sealed evaluation."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

import common as H
import executor as E

HERE,ROOT=H.HERE,H.ROOT
ARMS=('F0','F1','F2','E0')


def benchmark():
    if (HERE/'LATENCY.json').exists():raise FileExistsError('Keep original timing block')
    _,ex,_,jobs,_=E.scientific_context();seal=E.require_full_seal(jobs)
    if H.read(HERE/'INDEPENDENT_QA.json')['status']!='PASS':raise ValueError('Independent recomputation required')
    plan=H.read(HERE/'POSTPROCESSING_PLAN.json')['inference_cost'];H.A.setup(26301)
    generator=torch.Generator(device='cuda').manual_seed(plan['seed']);x=torch.randn(128,2,256,generator=generator,device='cuda')
    refs=H.read(HERE/'REFERENCE_INPUTS.json')['references']['V0/S0'];models={}
    for arm in ARMS:
        if arm in ('F0','E0'):
            item=refs[arm];path=HERE/item['copy_relative'];expected=item['sha256']
            if H.sha(path)!=expected:raise ValueError('Timing reference changed')
            state=torch.load(path,map_location='cpu',weights_only=False)['model']
            model=H.M.build('F0',24101,backbone_state=state)
        else:
            path=HERE/'artifacts/scientific'/f'V0_S0_{arm}'/'selected.pt'
            expected=seal['runs'][f'V0_S0_{arm}']['checkpoints']['selected']
            if H.sha(path)!=expected:raise ValueError('Timing endpoint changed')
            state=torch.load(path,map_location='cpu',weights_only=False)['model']
            model=H.M.build(arm,24101,26101);model.load_state_dict(state,strict=True)
        models[arm]=model.cuda().eval()
    records={a:{} for a in ARMS}
    for round_index in range(plan['rounds']):
        order=ARMS[round_index:]+ARMS[:round_index]
        for arm in order:
            model=models[arm];before=H.semantic_hash(model.state_dict())
            for batch in plan['batches']:
                with torch.no_grad():
                    for _ in range(plan['warmup_per_round']):model(x[:batch])
                    torch.cuda.synchronize();events=[]
                    for _ in range(plan['measurements_per_round']):
                        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        a.record();model(x[:batch]);b.record();events.append((a,b))
                    torch.cuda.synchronize()
                values=np.asarray([a.elapsed_time(b) for a,b in events])
                records[arm].setdefault(str(batch),[]).append(dict(round=round_index,p50_ms=float(np.median(values)),
                    p95_ms=float(np.quantile(values,.95)),mean_ms=float(values.mean()),samples_per_second=float(batch*1000/values.mean()),all_measurements_ms=values.tolist()))
            if H.semantic_hash(model.state_dict())!=before:raise ValueError('Timing changed model state')
    H.write(HERE/'LATENCY.json',dict(status='COMPLETE',records=records,environment=ex['environment'],plan=plan,optimizer_steps=0,source_IQ_reads=0,time=H.now()))
    print('Full fixed CUDA inference timing block complete',flush=True)


def archive():
    if (HERE/'ARTIFACT_ARCHIVE.json').exists():raise FileExistsError('Keep original scientific archives')
    _,ex,_,jobs,_=E.scientific_context();E.require_full_seal(jobs)
    if H.read(HERE/'INDEPENDENT_QA.json')['status']!='PASS':raise ValueError('Independent recomputation required')
    evaluation=H.read(HERE/'EVALUATION_MANIFEST.json');prediction_files={}
    with zipfile.ZipFile(HERE/'SOURCE_PREDICTIONS.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,digest in sorted(evaluation['files'].items()):
            path=HERE/'artifacts'/name
            if H.sha(path)!=digest:raise ValueError('Prediction file changed')
            z.write(path,name);prediction_files[name]=dict(bytes=path.stat().st_size,sha256=digest)
    traces={};local_files={}
    with zipfile.ZipFile(HERE/'SCIENTIFIC_TRACES.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for job in jobs:
            out=HERE/'artifacts/scientific'/job['id']
            for path in sorted(out.rglob('*')):
                if not path.is_file():continue
                name=path.relative_to(HERE/'artifacts').as_posix();item=dict(bytes=path.stat().st_size,sha256=H.sha(path));local_files[name]=item
                if path.suffix=='.pt' or (path.suffix=='.npz' and path.name.endswith('_val.npz')):continue
                z.write(path,name);traces[name]=item
    local=HERE/'artifacts/SCIENTIFIC_EXECUTION.zip'
    with zipfile.ZipFile(local,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name in sorted(local_files):z.write(HERE/'artifacts'/name,name)
        for name in sorted(prediction_files):z.write(HERE/'artifacts'/name,name)
    costs={}
    for arm in ('F1','F2'):
        values=[H.read(HERE/'artifacts/scientific'/j['id']/'COMPLETE.json') for j in jobs if j['arm']==arm]
        costs[arm]=dict(fits=6,epochs=240,updates=30480,
            training_loop_seconds=sum(v['costs']['training_seconds'] for v in values),
            validation_seconds=sum(v['costs']['validation_seconds'] for v in values),rollback_probe_seconds=sum(v['costs']['rollback_probe_seconds'] for v in values),
            per_fit_wall_seconds_sum=sum(v['total_wall_seconds'] for v in values),
            peak_cuda_allocated_gib=max(v['peak_cuda_allocated_gib'] for v in values))
    old_p=H.read(ROOT/'research/periodic_span/RUN_EVIDENCE.json');old_e=H.read(ROOT/'research/standard_method_competition_c/RUN_EVIDENCE.json')
    inherited_p=[old_p[f'{v}_{s}_P0']['COMPLETE'] for v in ('V0','V1') for s in ('S0','S1','S2')]
    inherited_e=[old_e[f'{v}_{s}_wd1e3_ERM_EMA']['COMPLETE'] for v in ('V0','V1') for s in ('S0','S1','S2')]
    inherited=dict(P0=dict(historical_fits=6,epochs=sum(v['epochs'] for v in inherited_p),updates=sum(v['updates'] for v in inherited_p),
        historical_training_seconds=sum(v['costs']['training_seconds'] for v in inherited_p),historical_wall_seconds=sum(v['total_wall_seconds'] for v in inherited_p)),
        E0=dict(historical_joint_ERM_EMA_fits=6,epochs=sum(v['epochs'] for v in inherited_e),updates=sum(v['updates'] for v in inherited_e),
            historical_training_with_EMA_seconds=sum(v['costs']['training_with_ema_seconds'] for v in inherited_e),
            historical_fit_loop_seconds=sum(v['costs']['total_fit_loop_seconds'] for v in inherited_e),
            historical_BN_calibration_seconds=sum(v['costs']['ema_calibration_seconds'] for v in inherited_e)),
        scope='Already completed historical source fits, not new fits; F1/F2 reuse the same six P0 models and their cost is not zero or duplicated per arm.')
    H.write(HERE/'COSTS.json',dict(status='COMPLETE',arms=costs,scientific_fits=12,epochs=480,updates=60960,
        invocation=H.read(HERE/'TRAINING_INVOCATION.json'),inherited_reference_training=inherited,environment=ex['environment'],
        preflight=dict(start=H.read(HERE/'SOURCE_PREFLIGHT_ATTEMPT.json')['time'],finish=H.read(HERE/'SOURCE_PREFLIGHT.json')['time']),
        synthetic_engineering_updates=dict(CPU=12,CUDA=240),science_failure_retries=0))
    def item(path,files=None):
        value=dict(path=path.relative_to(HERE).as_posix(),bytes=path.stat().st_size,sha256=H.sha(path))
        if files is not None:value['files']=files
        return value
    H.write(HERE/'ARTIFACT_ARCHIVE.json',dict(status='COMPLETE',source_IQ_included=False,source_weights_committed=False,
        predictions=item(HERE/'SOURCE_PREDICTIONS.zip',prediction_files),public_traces=item(HERE/'SCIENTIFIC_TRACES.zip',traces),
        local_full_execution=item(local),local_files=local_files))
    print('Scientific predictions, traces, costs and local full-state archive complete',flush=True)


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=H.read(HERE/'SUMMARY.json');qa=H.read(HERE/'INDEPENDENT_QA.json');costs=H.read(HERE/'COSTS.json');latency=H.read(HERE/'LATENCY.json')
    if qa['status']!='PASS':raise ValueError('Independent audit required')
    with gzip.open(HERE/'ANALYSIS.json.gz','rt',encoding='utf-8') as f:analysis=json.load(f)
    block=summary['endpoints']['selected'];aggregates=block['aggregates'];pairs=block['comparisons'];decision=summary['decision']
    colors={'F0':'#64748b','F1':'#d78922','F2':'#176b87','E0':'#9b5eab'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'spectral-scale-frozen-v1'})
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    ax=axes[0,0];means=[aggregates[a]['rx_equal']['accuracy']*100 for a in ARMS]
    ax.bar(ARMS,means,color=[colors[a] for a in ARMS],width=.65)
    for i,arm in enumerate(ARMS):
        ax.scatter([i-.1,i,i+.1],[aggregates[arm]['per_stream'][s]['accuracy']*100 for s in ('S0','S1','S2')],s=15,c='black',zorder=3)
        ax.text(i,means[i]+1,f'{means[i]:.2f}',ha='center',fontsize=9)
    ax.set(ylim=(0,100),ylabel='RX-equal accuracy (%)',title='A. Selected endpoints; dots are fixed streams')
    ax=axes[0,1];names=['F2-F1','F2-F0','F2-E0'];values=np.asarray([pairs[n]['rx_equal']['accuracy_delta']*100 for n in names])
    intervals=np.asarray([pairs[n]['paired_rx_bootstrap_95_intervals']['accuracy_delta'] for n in names])*100
    for i in range(3):
        ax.plot([i,i],intervals[i],color=colors['F2']);ax.plot([i-.06,i+.06],[intervals[i,0]]*2,color=colors['F2']);ax.plot([i-.06,i+.06],[intervals[i,1]]*2,color=colors['F2'])
    ax.scatter(range(3),values,c=colors['F2'],zorder=3);ax.set_xticks(range(3),names)
    ax.axhline(0,color='black',lw=.8);ax.axhline(1,color='#64748b',ls='--',lw=.8)
    ax.set(ylabel='Accuracy change (pp)',title='B. Conditional 95% RX bootstrap intervals')
    ax=axes[1,0];rx=sorted(pairs['F2-F1']['per_rx']);y=np.arange(12)
    for shift,name,color in ((-.18,'F2-F1','#176b87'),(.18,'F2-E0','#9b5eab')):
        ax.barh(y+shift,[pairs[name]['per_rx'][r]['accuracy_delta']*100 for r in rx],height=.34,label=name,color=color)
    ax.set_yticks(y,rx,fontsize=8);ax.invert_yaxis();ax.axvline(0,color='black',lw=.8);ax.axvline(-2,color='#b43b3b',ls='--',lw=.8)
    ax.set(xlabel='Accuracy change (pp)',title='C. Every observed source-selection RX');ax.legend(fontsize=8)
    ax=axes[1,1];heat=np.full((12,10),np.nan);support=np.zeros((12,10),dtype=int)
    groups=analysis['endpoints']['selected']['comparisons']['F2-E0']['tx_rx_stream_mean']
    for row in groups:
        i,j=rx.index(row['rx']),row['tx_class_index'];heat[i,j]=row['risk_increase']*100;support[i,j]=row['packets_per_stream']
    limit=max(5,float(np.nanmax(np.abs(heat))));cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#d1d5db')
    im=ax.imshow(heat,aspect='auto',cmap=cmap,vmin=-limit,vmax=limit);ax.set_yticks(range(12),rx,fontsize=8);ax.set_xticks(range(10))
    for i,j in zip(*np.where((support>0)&(support<50))):ax.plot(j,i,'k.',markersize=2)
    ax.set(xlabel='TX model class index',title='D. F2-E0 error increase (pp); dot: n<50')
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
    fig.suptitle('Frozen spectral-scale residual | 12 fits | source-CV development only',fontsize=14)
    fig.savefig(HERE/'RESULTS.png',dpi=170);fig.savefig(HERE/'RESULTS.svg',metadata={'Date':None});plt.close(fig)
    evidence=H.read(HERE/'RUN_EVIDENCE.json');fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for i,view in enumerate(('V0','V1')):
        for j,arm in enumerate(('F1','F2')):
            ax=axes[i,j]
            for stream in ('S0','S1','S2'):
                name=f'{view}_{stream}_{arm}';rows=evidence[name]['HISTORY'];best=evidence[name]['COMPLETE']['best_epoch']
                alpha=[0.]+[r['alpha'] for r in rows];ax.plot(range(41),alpha,label=stream)
                ax.scatter([best],[alpha[best]],s=25)
            ax.axhline(0,color='black',lw=.7);ax.set(xlabel='Epoch',ylabel='Learned alpha',title=view+' / '+arm);ax.legend(fontsize=8)
    fig.suptitle('All alpha trajectories; dots are checkpoint_val CE selections')
    fig.savefig(HERE/'ALPHA.png',dpi=170);fig.savefig(HERE/'ALPHA.svg',metadata={'Date':None});plt.close(fig)
    for name in ('RESULTS.svg','ALPHA.svg'):
        path=HERE/name;path.write_text('\n'.join(v.rstrip() for v in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    lines=['# F2 冻结频域尺度残差：完整源域结果','',
        f"**12/12 次科学拟合完成；判定：`{decision['status']}`。** 原设计 `2121255`、工程 `9fccd65` 的结构、训练配方和门槛未改。全部结果属于重复使用合法源角色的 source-CV 开发研究，不是独立确认或正式 CRD。",'',
        '## selected 主端点','',
        '| 结构 | RX 等权 Accuracy | Macro-F1 | CE | 最差 RX Accuracy | 总参数 / 本轮训练参数 |','|---|---:|---:|---:|---|---:|']
    for arm in ARMS:
        a=aggregates[arm];worst=a['worst_rx'];side=arm in ('F1','F2')
        lines.append(f"| {arm} | {a['rx_equal']['accuracy']*100:.4f}% | {a['rx_equal']['macro_f1']*100:.4f}% | {a['rx_equal']['ce']:.6f} | {worst}: {a['per_rx'][worst]['accuracy']*100:.4f}% | {85627 if side else 82426:,} / {3201 if side else 0:,} |")
    lines+=['','F0 为原六个 P0 的冻结回放，E0 为原六个 EMA 完整配方的冻结回放；两者不做本轮拟合。F1 是普通宽频域卷积控制，F2 是唯一细/宽并行候选。两者各新增 3,201 参数、最大名义频率跨度相同，拓扑、非线性和计算量不完全相同。',
        '','| 比较 | Accuracy 差 / pp | RX 条件 95% 区间 / pp | 最差配对 RX 差 / pp | 最大受支持 TX×RX 风险增加 / pp |',
        '|---|---:|---|---:|---:|']
    for name in names:
        c=pairs[name];ci=c['paired_rx_bootstrap_95_intervals']['accuracy_delta']
        lines.append(f"| {name} | {c['rx_equal']['accuracy_delta']*100:+.4f} | [{ci[0]*100:+.4f}, {ci[1]*100:+.4f}] | {c['worst_paired_rx']['accuracy_delta']*100:+.4f} | {c['worst_supported_tx_rx_risk_increase']['risk_increase']*100:+.4f} |")
    lines+=['','![完整主结果](RESULTS.png)','','## 原联合门与实际纠错','']
    for name,gate in decision['gates'].items():
        failed=[k for k,v in gate['checks'].items() if not v]
        lines.append(f"- {name}：{'全部通过' if gate['passes'] else '未通过'}；未过项：{', '.join(failed) if failed else '无'}。")
    lines+=['','三个比较均需同时满足：平均至少 +1 pp、条件区间下界 >0、两视图均正、至少 2/3 流正、最差配对 RX ≥−2 pp、每流支持≥50的全部 TX×RX 流均值风险增加≤5 pp。支持不乘以三个种子；未观察组不伪造为零风险。',
        '','| 相对参照 | 纠正错误 / 全部输入 | 新增错误 / 全部输入 | 预测不一致 |','|---|---:|---:|---:|']
    for name in names:
        c=pairs[name]['rx_equal'];lines.append(f"| {name} | {c['corrected_fraction']*100:.4f}% | {c['new_error_fraction']*100:.4f}% | {c['disagreement']*100:.4f}% |")
    lines+=['','以上为 RX 等权并先平均固定三流的比例；纠错数量不能单独替代净准确率与风险门。12 个已观察 RX 为 bootstrap cluster，10,000 次、seed24501。区间未覆盖新 RX 总体、seed 总体或历史开发选择偏差，多比较未校正。',
        '','## 所有辅助端点与固定流','',
        '| 端点 | F0 Accuracy | F1 Accuracy | F2 Accuracy | E0 Accuracy |','|---|---:|---:|---:|---:|']
    for endpoint in H.ENDPOINTS:
        a=summary['endpoints'][endpoint]['aggregates'];lines.append('| '+endpoint+' | '+' | '.join(f"{a[arm]['rx_equal']['accuracy']*100:.4f}%" for arm in ARMS)+' |')
    lines+=['','只有 selected 决定晋级；epoch10/final40 不得取代主端点，不扫描 alpha 或选择局部有效分支。',
        '','| 结构 | V0 | V1 | S0 | S1 | S2 |','|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        a=aggregates[arm];values=[a['per_view'][v]['accuracy'] for v in ('V0','V1')]+[a['per_stream'][s]['accuracy'] for s in ('S0','S1','S2')]
        lines.append('| '+arm+' | '+' | '.join(f'{v*100:.4f}%' for v in values)+' |')
    lines+=['','完整每 RX、TX×RX、日期、混淆计数、缺失组及三端点记录见 [ANALYSIS.json.gz](ANALYSIS.json.gz) 和 [SUMMARY.json](SUMMARY.json)。',
        '','## 预检、冻结与执行证据','',
        '六个 P0、六个 EMA 的真实 checkpoint_val 预检最大 logit 差全部为 0，argmax 一致；六组 F1/F2 在 alpha=0 时与 F0 位级一致。预检提交后自动进入十二次拟合；全部训练封存提交后，才读取两份合法 source_selection。评价时 F0/E0 再与历史封存预测核对，误差及身份见 [EVALUATION_MANIFEST.json](EVALUATION_MANIFEST.json)。',
        '',f"本轮 480 epoch、60,960 次更新，无科学失败重试。独立复算通过 {qa['check_count']:,} 项：逐份核验原设计/工程 SHA、492 份初始及 epoch 全状态、480 个 epoch 的冻结参数与 BN、实际配对包序、全部梯度与 alpha、epoch0–40 的 CE 最早最小选择，以及第二份 NumPy 实现的全部主统计和联合门。见 [INDEPENDENT_QA.json](INDEPENDENT_QA.json)。",'',
        'alpha 轨迹完整保留。首步侧路任务梯度为零与 coupled Adam weight decay 导致参数移动分开记录；不因 alpha 小而改初值或 LR。',
        '','![全部 alpha 轨迹](ALPHA.png)','','| 拟合 | selected epoch | selected alpha | final alpha |','|---|---:|---:|---:|']
    for name,item in evidence.items():
        best=item['COMPLETE']['best_epoch'];rows=item['HISTORY'];alpha=0. if best==0 else rows[best-1]['alpha']
        lines.append(f"| {name} | {best} | {alpha:+.8f} | {rows[-1]['alpha']:+.8f} |")
    lines+=['','## 训练与推断成本','',
        '| 侧路 | 新增拟合 / 更新 | 训练循环 / s | 验证 / s | 合成回退探针 / s | 各拟合 wall 总和 / s | 峰值已分配显存 / GiB |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm,c in costs['arms'].items():lines.append(f"| {arm} | {c['fits']} / {c['updates']:,} | {c['training_loop_seconds']:.2f} | {c['validation_seconds']:.2f} | {c['rollback_probe_seconds']:.2f} | {c['per_fit_wall_seconds_sum']:.2f} | {c['peak_cuda_allocated_gib']:.3f} |")
    lines+=['',f"整个 train-all 调用 wall 为 {costs['invocation']['elapsed_wall_seconds']:.2f} s，包含逐阶段 SHA/提交验证与日志、封存开销；训练循环计时为 wall，不能称为纯 GPU kernel 时间。显存读数是同一进程的累计峰值，未逐拟合 reset，不能比较各臂独立显存峰值。两臂复用同六个历史 P0，不把其预训练成本当零，也不在两臂间重复计作新增拟合。历史 P0 与 EMA/BN 完整配方的已用训练成本在 [COSTS.json](COSTS.json) 分列。",'',
        '| 模型 | B=1 batch p50 / ms | B=128 batch p50 / ms | B=128 samples/s |','|---|---:|---:|---:|']
    for arm in ARMS:
        r=latency['records'][arm];lines.append(f"| {arm} | {np.median([x['p50_ms'] for x in r['1']]):.4f} | {np.median([x['p50_ms'] for x in r['128']]):.4f} | {np.median([x['samples_per_second'] for x in r['128']]):,.1f} |")
    lines+=['','延迟仅取事先固定的 V0/S0 selected 模型，完整前端到分类器、合成 CUDA 常驻输入，B=1/128；四轮平衡轮转，每轮 20 warmup +100 测量。表为各轮 p50 的中位数，完整测量与 p95 见 [LATENCY.json](LATENCY.json)。不含磁盘和主机传输，不选择最快 seed。',
        '','## 解释与封存','',
        '成功只支持固定频域判别器能利用当前附加信息，仍需另行确认；失败只否决冻结骨干、3,201 参数及登记训练配方下的这个候选，不扩大为所有多尺度方法无效。参数与最大跨度匹配不能唯一归因于多尺度物理机制，本轮不声称原创性。',
        '',('F2 通过全部登记门，结论仅为值得另行确认；本次授权到此结束，不自动进行确认或下一候选。' if decision['status']=='F2_WORTH_SEPARATE_CONFIRMATION' else 'F2 未通过全部登记门，按预注册 STOP。保留完整平均收益与局部变化，不微调 kernel、alpha、Loss 或解冻骨干重跑，不自动开展下一候选。'),'',
        '未读取 target8、历史 held6、Day3–4、Full WiSig 新 RX；此前固定窗口及 P2 的 STOP 证据均未改变。',
        '', '公开 [48 份同输入预测](SOURCE_PREDICTIONS.zip)、[全部科学梯度/alpha/样本顺序轨迹](SCIENTIFIC_TRACES.zip)、[归档清单](ARTIFACT_ARCHIVE.json) 与 SHA。初始/逐 epoch 模型、optimizer/scheduler/RNG 和完整执行包留在本地忽略目录，不上传模型权重或原始 IQ。',
        '', '授权原始 JSON 的中文转写曾受 PowerShell 管道编码影响；启动封装在调用科学训练器前拒绝了这一转写。已补存 [UTF-8 授权原文](USER_AUTHORIZATION.zh-CN.md) 与 [更正说明](AUTHORIZATION_TRANSCRIPTION_NOTE.json)，保留原授权及预检 SHA。该记录问题发生时科学尝试数为 0，未重试科学拟合。',
        '']
    (HERE/'RESULTS.zh-CN.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print('Full figures and Chinese report written',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['benchmark','archive','report']);args=parser.parse_args()
    {'benchmark':benchmark,'archive':archive,'report':report}[args.command]()


if __name__=='__main__':main()
