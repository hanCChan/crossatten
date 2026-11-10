"""Derived S20 QA, random-input cost, weight description and publication report.

No optimization or new scientific candidate selection is available here.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp
from s20_core import HERE,ROOT,S19,HISTORY,UPSTREAM,read,write,sha,now,validate,committed,configure_data,official,git
from s20_run import ARMS,CONF,outdir,olddir
from s20_stats import score,decision


def require(value,why):
    if not value:raise AssertionError(why)
def same(a,b,why='',atol=1e-8):
    if isinstance(b,dict):
        require(isinstance(a,dict) and set(a)==set(b),why+' dict keys')
        for k,v in b.items():same(a[k],v,why+'.'+str(k),atol)
    elif isinstance(b,(list,tuple,np.ndarray)):
        require(len(a)==len(b),why+' length')
        for i,v in enumerate(b):same(a[i],v,why+f'[{i}]',atol)
    elif isinstance(b,(float,np.floating)):
        require(np.isfinite(a) and np.isfinite(b) and np.isclose(a,b,rtol=1e-8,atol=atol),why+' float')
    else:require(a==b,why+' value')


def verify_predictions(path,expected,names,allowed,labels=None):
    with np.load(path,allow_pickle=False) as z:
        require(set(z.files)=={'logits','y','rx','day'},'prediction schema')
        logits=z['logits'];data={k:z[k] for k in ['y','rx','day']}
        require(logits.shape==(len(data['y']),10) and np.isfinite(logits).all(),'finite ten-class logits')
        require(set(np.unique(data['day']))=={1,2},'source days only')
        require(data['rx'].dtype.kind in 'iu' and np.all((data['rx']>=0)&(data['rx']<len(names))),'RX label bounds')
        require({names[int(r)] for r in np.unique(data['rx'])}==set(allowed),'RX identities')
        if labels is not None:
            for k in data:require(np.array_equal(data[k],labels[k]),'identical labels/order '+k)
        same(expected,score(logits,data,names),'full rescoring',atol=1e-10)
        lg=logits.astype(float);y=data['y'];loss=logsumexp(lg,axis=1)-lg[np.arange(len(y)),y]
        other=lg.copy();other[np.arange(len(y)),y]=-np.inf;margin=lg[np.arange(len(y)),y]-other.max(1)
        for r in np.unique(data['rx']):
            mask=data['rx']==r;record=expected['per_rx'][names[int(r)]]
            same(record['ce'],float(loss[mask].mean()),'independent double CE',atol=2e-6)
            same(record['margin'],float(margin[mask].mean()),'independent double margin',atol=2e-6)
        return {k:v.copy() for k,v in data.items()},float(loss.mean())


def audit():
    validate();committed(HERE/'seal_pilot.json');sealed=read(HERE/'seal_pilot.json')
    require(len(sealed['checkpoints'])==8 and sealed['new_fits']==4 and sealed['reused_B0']==4,'fit budget8=4+4')
    require(sealed['all_checkpoints_complete'] and not sealed['new_ask_held_source_read'] and not sealed['real_target_access'],'seal role')
    for path,h in sealed['metadata_sha256'].items():require(sha(path)==h,'sealed metadata SHA '+path)
    require(read(HERE/'held_attempt.json')['status']=='COMPLETE','single held evaluation completed')
    data_module=configure_data();rows=[];verified=[];label_records={};official_records=read(HERE/'UPSTREAM.json')
    for rel,record in official_records['files'].items():
        contents=(UPSTREAM/rel).read_bytes();blob=hashlib.sha1(b'blob '+str(len(contents)).encode()+b'\0'+contents).hexdigest()
        require(blob==record['git_blob_sha'] and hashlib.sha256(contents).hexdigest()==record['sha256'],'exact official source blob')
    for f in range(4):
        fit,held,target=data_module.roles(f);cache,_=data_module.locations(f);manifest=read(cache/'fit_manifest.json')
        require(manifest['fit_rx']==fit and manifest['held_rx']==held,'source roles frozen')
        meta={}
        for name in ['fit_y','fit_rx','fit_day','train_ids','val_ids']:
            p=cache/(name+'.npy');require(sha(p)==manifest['sha256'][p.name],'source metadata SHA')
            meta[name]=np.load(p,allow_pickle=False)
        require(np.array_equal(np.sort(np.r_[meta['train_ids'],meta['val_ids']]),np.arange(len(meta['fit_y']))),'complete disjoint source split')
        val_labels={k:meta['fit_'+k][meta['val_ids']] for k in ['y','rx','day']}
        label_records[f]=meta
        ask=outdir(f,'ASK');cfg=read(ask/'config.json');metrics=read(ask/'metrics.json');basecfg=read(olddir(f)/'config.json')
        require(cfg['data_seed']==basecfg['data_seed'] and cfg['macro_seed']==18101 and cfg['split_seed']==16001,'fixed paired seeds/split')
        require(cfg['fit_rx']==fit and cfg['held_rx']==held,'exact ASK fit18 held6')
        require(cfg['train_packets']==len(meta['train_ids']) and cfg['val_packets']==len(meta['val_ids']),'same packets')
        for part in ['train','val']:require(cfg[part+'_indices_sha256']==hashlib.sha256(meta[part+'_ids'].tobytes()).hexdigest(),'sample index SHA')
        require(cfg['parameter_elements']==100240 and cfg['real_scalar_dof']==116369,'complex degrees reported')
        require(cfg['official_defaults']==vars(CONF) and cfg['epochs_max']==200 and cfg['batch']==128,'official defaults')
        require(metrics['status']=='COMPLETE' and read(ask/'attempt.json')['status']=='COMPLETE' and not (ask/'failure.json').exists(),'no retry/failure')
        with (ask/'history.csv').open(encoding='utf-8',newline='') as stream:history=list(csv.DictReader(stream))
        require(len(history)==metrics['epochs'] and 1<=len(history)<=200,'epoch budget')
        require([int(v['epoch']) for v in history]==list(range(1,len(history)+1)),'epoch sequence')
        best=min(history,key=lambda v:float(v['val_packet_ce']));rxbest=min(history,key=lambda v:float(v['val_rx_ce']))
        require(int(best['epoch'])==metrics['best_epoch'] and int(rxbest['epoch'])==metrics['diagnostic_rx_ce_best_epoch'],'checkpoint rule audit')
        same(float(best['val_packet_ce']),metrics['best_source_packet_ce'],'official best CE',atol=1e-12)
        require(all(len(v['order_sha256'])==64 for v in history),'sample exposure order SHA')
        require(len(metrics['k_history'])==len(history)+1,'official k history')
        selected=metrics['k_history'][metrics['best_epoch']]
        same(metrics['selected_checkpoint_cutoff'],[selected['argmax_k'],selected['soft_k'],selected['top_prob']],'selected cutoff')
        require(sha(ask/'source_val_predictions.npz')==metrics['source_predictions_sha256'],'source logits SHA')
        _,packet_ce=verify_predictions(ask/'source_val_predictions.npz',metrics['source_validation'],manifest['rx_names'],fit,val_labels)
        same(packet_ce,metrics['best_source_packet_ce'],'checkpoint full packet CE',atol=2e-6)
        old=olddir(f);base_metrics=read(old/'metrics.json');old_pred=old/'source_val_predictions.npz'
        require(sha(old_pred)==base_metrics['prediction_sha256'],'frozen B0 source logits SHA')
        verify_predictions(old_pred,base_metrics['source_validation'],manifest['rx_names'],fit,val_labels)
        baseline_held_labels=None
        for arm in ARMS:
            record=read(outdir(f,arm)/'held_metrics.json');ref=sealed['checkpoints'][f'{arm}_f{f}']
            require(sha(ref['path'])==ref['sha256'],'sealed checkpoint bytes')
            require(record['fold']==f and record['arm']==arm and record['rx_names']==sorted(fit+held),'paired role names')
            require(datetime.fromisoformat(record['time'])>=datetime.fromisoformat(sealed['time']),'held metric after seal')
            p=Path(record['prediction_path']);require(sha(p)==record['prediction_sha256'],'held logits SHA')
            labels,_=verify_predictions(p,record['held_source'],record['rx_names'],held,baseline_held_labels)
            if arm=='B0':
                baseline_held_labels=labels;previous=read(old/'held_metrics.json')
                require(record['held_source']==previous['held_source'] and record['prediction_sha256']==previous['prediction_sha256'],'B0 byte/score reuse exact')
            source_ref={(r['rx'],r['day']):r['sha256'] for r in read(old/'held_metrics.json')['held_sources']}
            require(len(record['held_sources'])==12 and {(r['rx'],r['day']):r['sha256'] for r in record['held_sources']}==source_ref,'held input hashes match frozen S19')
            rows.append(record)
        verified.append(dict(fold=f,epochs=metrics['epochs'],selected_epoch=metrics['best_epoch'],model_sha256=metrics['model_sha256'],all_logits_rescored=True))
        print('PASS S20 fold',f,'official training trace, frozen B0, all source/held logits',flush=True)
    actual=read(HERE/'decision.json');expected=decision(rows)
    for k,v in expected.items():require(actual[k]==v,'exact paired decision '+k)
    entries=[json.loads(line) for line in (HERE/'access.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    require(len(entries)==72,'exact72 scientific role access records')
    counts={kind:sum(r['kind']==kind for r in entries) for kind in set(r['kind'] for r in entries)}
    require(counts==dict(fit_cache=16,held_source_cell=48,baseline_cached_predictions=4,sealed_checkpoint=4),'access role counts')
    for r in entries:
        f=r['fold'];fit,held,target=data_module.roles(f);cache,_=data_module.locations(f)
        if r['kind']=='fit_cache':
            require(r['dataset']=='ManyRx' and r['rx']==fit and r['days']==[1,2],'fit role whitelist')
            name=Path(r['path']).name;require(name in ['fit_x.npy','fit_y.npy','fit_rx.npy','fit_day.npy'] and Path(r['path']).resolve()==(cache/name).resolve(),'fit path whitelist')
            require(r['sha256']==read(cache/'fit_manifest.json')['sha256'][name],'fit access SHA')
        else:
            require(datetime.fromisoformat(r['time'])>=datetime.fromisoformat(sealed['time']) and r['seal_sha256']==sha(HERE/'seal_pilot.json'),'all held operations after seal')
            if r['kind']=='held_source_cell':
                require(r['dataset']=='ManyRx' and r['rx'] in held and r['rx'] not in target+fit and r['day'] in [1,2],'held role whitelist')
                p=data_module.SOURCE_CELLS/'ManyRx'/f"rx_{r['rx']}_day{r['day']}.npz";require(Path(r['path']).resolve()==p.resolve(),'held path')
                expected={(v['rx'],v['day']):v['sha256'] for v in read(olddir(f)/'held_metrics.json')['held_sources']}
                require(r['sha256']==expected[r['rx'],r['day']],'held data provenance SHA')
            elif r['kind']=='baseline_cached_predictions':
                require(Path(r['path']).resolve()==(olddir(f)/'held_predictions.npz').resolve(),'B0 prediction reuse path')
            else:require(Path(r['path']).resolve()==Path(sealed['checkpoints'][f'ASK_f{f}']['path']).resolve(),'ASK weight role')
    for f in range(4):
        local=[r for r in entries if r['fold']==f]
        require(len(local)==18 and len({(r['rx'],r['day']) for r in local if r['kind']=='held_source_cell'})==12,'each held cell once')
    changed=git('diff','--name-only','9641b7b','--').splitlines()+git('ls-files','--others','--exclude-standard').splitlines()
    require(all(p.startswith('20-asknet-source-comparison/') for p in changed),'historical tree isolation')
    write(HERE/'FINAL_ACCESS_AUDIT.json',dict(time=now(),status='PASS',registration_sha256=sha(HERE/'registration.json'),audit_code_sha256=sha(__file__),
          folds=verified,new_formal_fits=4,reused_B0=4,source_access_records=len(entries),access_by_kind=counts,
          new_held_evaluation_after_seal=True,exact_decision_recomputation=True,raw_iq_opened_by_audit=False,model_forward_calls_by_audit=0,
          logits_scope='All frozen-B0 and ASK source/held logits rescored; CE/margin independently checked in float64',
          S19_archived_files_verified=len(read(S19/'MANIFEST.json')['files_sha256']),official_git_blobs_verified=3,
          real_target_access=False,ManySig_read=False,Shen_read=False,checkpoint_bytes_hashed_only=True,decision_status=actual['status']))
    print('PASS S20 audit:4 official fits, exact frozen B0, source-only inputs and all paired metrics',flush=True)


def benchmark_and_structure():
    import torch
    from s19_models import build_model
    validate();committed(HERE/'seal_pilot.json');sealed=read(HERE/'seal_pilot.json')
    if (HERE/'COSTS_AND_STRUCTURE.json').exists():raise ValueError('Retain archived benchmark; no automatic repetition')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    weight_access=[];structure=[]
    for f in range(4):
        ref=sealed['checkpoints'][f'ASK_f{f}'];require(sha(ref['path'])==ref['sha256'],'structure checkpoint seal')
        state=torch.load(ref['path'],map_location='cpu',weights_only=True)
        U=torch.linalg.qr(state['koopman.U'])[0];theta=state['koopman.theta'];K=U@torch.diag(torch.exp(1j*theta))@U.conj().T
        eye=torch.eye(127,dtype=torch.cfloat);off=K-torch.diag(torch.diag(K));prob=state['cutoff.selection_logits'].softmax(0)
        structure.append(dict(fold=f,checkpoint_sha256=ref['sha256'],selected_k=CONF.candidates[int(prob.argmax())],cutoff_probabilities=prob.tolist(),
              unitary_error_fro_relative=float(torch.linalg.norm(K.conj().T@K-eye)/np.sqrt(127)),distance_to_identity_relative=float(torch.linalg.norm(K-eye)/np.sqrt(127)),
              off_diagonal_energy_fraction=float(off.abs().square().sum()/K.abs().square().sum()),eigenphase_circular_concentration=float(torch.exp(1j*theta).mean().abs()),
              mode_count=127,patch_time_samples=64,patch_stride=32,token_count=7,token_dimension=128))
        weight_access.append(dict(kind='static_weight_description',path=ref['path'],sha256=ref['sha256'],fold=f,input_role='checkpoint only; no IQ or labels'))
    costs=[]
    for arm in ARMS:
        ref=sealed['checkpoints'][f'{arm}_f0'];require(sha(ref['path'])==ref['sha256'],'cost checkpoint seal')
        seed=read(olddir(0)/'config.json')['data_seed']
        model=build_model('B0','ManyRx',seed) if arm=='B0' else official().build_model(CONF,10)
        model.load_state_dict(torch.load(ref['path'],map_location='cpu',weights_only=True));model.cuda().eval()
        for p in model.parameters():p.requires_grad_(False)
        latency=[]
        with torch.no_grad():
            for batch in [1,128]:
                generator=torch.Generator(device='cuda').manual_seed(20205+batch);x=torch.randn(batch,2,256,device='cuda',generator=generator)
                for _ in range(10):model(x)
                torch.cuda.synchronize();samples=[]
                for _ in range(30):
                    start=time.perf_counter();model(x);torch.cuda.synchronize();samples.append(1000*(time.perf_counter()-start))
                latency.append(dict(batch=batch,median_batch_ms=float(np.median(samples)),p90_batch_ms=float(np.quantile(samples,.9)),per_packet_ms=float(np.median(samples))/batch,samples_ms=samples))
        costs.append(dict(arm=arm,parameter_elements=sum(p.numel() for p in model.parameters()),real_scalar_dof=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters()),latency=latency,
                          path='RMS->FFT five channels->frequency HStem32x64->2MHA->classifier' if arm=='B0' else 'FFT->hard learned cutoff->complexQR127+unitarySKO->IFFT->total energy norm->time patches7x128->MLP->classifier'))
        weight_access.append(dict(kind='synthetic_benchmark_checkpoint',arm=arm,fold=0,path=ref['path'],sha256=ref['sha256'],input_role='random IQ only; no scientific packet'))
        print('S20 cost',arm,'complete',flush=True)
    write(HERE/'COSTS_AND_STRUCTURE.json',dict(time=now(),status='COMPLETE',device=torch.cuda.get_device_name(),torch=torch.__version__,precision='FP32',
          warmup=10,repeats=30,batches=[1,128],timing_scope='host perf_counter and CUDA synchronize per model call; synthetic inputs generated before timing',
          QR_cached=False,model_paths_modified=False,scientific_inputs_read=False,weight_access=weight_access,costs=costs,ASK_structure=structure,
          structure_scope='Descriptive checkpoint algebra and code structure; not classification interventions or causal attribution',
          hstem_patch_scope='32 frequency-coordinate tokens x64; pre-attention receptive field32 bins/hop8; not comparable in units to ASK reconstructed-time patch64',
          training_seconds=sum(read(outdir(f,'ASK')/'metrics.json')['seconds'] for f in range(4))))


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    validate();require(read(HERE/'FINAL_ACCESS_AUDIT.json')['status']=='PASS','QA required before reporting')
    dec=read(HERE/'decision.json');cost=read(HERE/'COSTS_AND_STRUCTURE.json');reuse=read(HERE/'REUSE_AUDIT.json')
    c=dec['comparison'];rows=[]
    for f in range(4):
        for arm in ARMS:
            metrics=read(olddir(f)/'metrics.json') if arm=='B0' else read(outdir(f,arm)/'metrics.json')
            rows.append(dict(fold=f,arm=arm,source=metrics['source_validation'],held=read(outdir(f,arm)/'held_metrics.json')['held_source']))
    comparison=[]
    for row in c['folds']:
        for rx,gain in row['per_rx_pp'].items():
            left=next(v for v in rows if v['fold']==row['fold'] and v['arm']=='ASK')['held']['per_rx'][rx]
            right=next(v for v in rows if v['fold']==row['fold'] and v['arm']=='B0')['held']['per_rx'][rx]
            comparison.append(dict(fold=row['fold'],rx=rx,delta_acc_pp=gain,delta_f1_pp=100*(left['macro_f1']-right['macro_f1']),delta_ce=left['ce']-right['ce'],ASK_acc=left['acc'],B0_acc=right['acc']))
    with (HERE/'PAIRED_RECEIVERS.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(comparison[0]),lineterminator='\n');writer.writeheader();writer.writerows(comparison)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True);colors=['#2864a1','#c35b21']
    for i,arm in enumerate(ARMS):
        ys=[100*r['held']['summary']['acc'] for r in rows if r['arm']==arm]
        axes[0,0].plot(range(4),ys,'o-',label=arm,color=colors[i])
    axes[0,0].set(xticks=range(4),xlabel='Source fold',ylabel='Held-source RX-equal accuracy (%)',title='A. Identical samples; different official recipes');axes[0,0].legend()
    axes[0,1].bar(range(4),[r['delta_acc_pp'] for r in c['folds']],color=['#c35b21' if r['delta_acc_pp']<0 else '#27836b' for r in c['folds']])
    axes[0,1].axhline(0,color='gray');axes[0,1].axhline(1,color='gray',ls='--');axes[0,1].set(xticks=range(4),xlabel='Source fold',ylabel='ASK - B0 (pp)',title='B. Fixed primary paired accuracy comparison')
    names=list(c['per_rx_pp']);gains=np.array(list(c['per_rx_pp'].values()));order=np.argsort(gains)
    axes[1,0].barh(np.array(names)[order],gains[order],color='#2864a1');axes[1,0].axvline(0,color='gray');axes[1,0].set(xlabel='Unique-RX mean ASK - B0 (pp)',title='C. Repeated fold appearances averaged per RX')
    for i,row in enumerate(cost['costs']):
        axes[1,1].bar(np.arange(2)+(i-.5)*.3,[v['median_batch_ms'] for v in row['latency']],width=.3,label=row['arm'],color=colors[i])
    axes[1,1].set(xticks=[0,1],xticklabels=['batch1','batch128'],ylabel='Median synchronized GPU call (ms)',title='D. Original inference paths; random IQ');axes[1,1].legend()
    fig.suptitle('S20 | source RX dates1-2 only | 4 ASK fits + frozen HStem | '+dec['status'],fontsize=11)
    fig.savefig(HERE/'S20_OVERVIEW.png',dpi=180);plt.close(fig)
    lines=['# S20：同数据源域 HStem+MHA / ASKNet 正面对照','',f"结果分类：**{dec['status']}**。新增4次官方路径ASKNet训练，严格复用4个S19 B0；没有新模块、额外seed或真实目标CRD评估。",'',
           f"ASK−B0的四fold平均Accuracy差 **{c['mean_fold_gain_pp']:+.4f}pp**，正向{c['positive_folds']}/4，最差配对RX **{c['worst_paired_rx_pp']:+.4f}pp**。按{c['unique_rx']}个唯一RX身份等权的均值{c['unique_rx_mean_pp']:+.4f}pp，描述性条件95%区间[{c['ci95_pp'][0]:+.4f},{c['ci95_pp'][1]:+.4f}]pp。",'',
           '![S20源域比较](S20_OVERVIEW.png)','',
           '本轮结论是同数据、同评价条件下**整体模型与官方配方**的比较；不能把差距直接解释为Koopman、复数谱、截止或Patch的因果贡献。官方ASKNet最多200轮、按包等权CE选模；冻结B0最多40轮、按RX等权CE选模，数据shuffle路径也不同。差异均在读取新ASK held结果前固定。','',
           '## 首轮比较与出口','',
           '| fold | B0 Accuracy % | ASK Accuracy % | ASK−B0 pp | ΔMacro-F1 pp | 最差配对RX pp |', '|---|---:|---:|---:|---:|---:|']
    for v in c['folds']:
        f=v['fold'];b=next(r for r in rows if r['fold']==f and r['arm']=='B0')['held']['summary'];a=next(r for r in rows if r['fold']==f and r['arm']=='ASK')['held']['summary']
        lines.append(f"| {f} | {100*b['acc']:.4f} | {100*a['acc']:.4f} | {v['delta_acc_pp']:+.4f} | {v['delta_f1_pp']:+.4f} | {v['worst_paired_rx_pp']:+.4f} |")
    lines+=['',f"操作性‘明显优势’要求fold均值≥1pp、3/4正向、最差配对RX≥−2pp及唯一RX条件CI下界>0。ASK通过：{dec['ask_clear_advantage']}；HStem反方向通过：{dec['hstem_clear_advantage']}；描述性接近条件通过：{dec['practical_similarity']}。机器决定见[decision.json](decision.json)。",'',
            ('ASK达到本轮明确优势门槛，允许据此提出下一轮分离频带、复数算子、Patch及配方因素的受控问题；本轮没有开始这些新训练，也没有证明其中哪个因素负责。' if dec['ask_clear_advantage'] else
             '本轮没有建立ASK在这套源RX协议上稳定优于冻结HStem的前提。不能因此证明两方法普遍等价，或证明源协议失效；应先审视覆盖、分组及seed不确定性，而不是继续增加新模块。'),'',
            'fold等权均值和唯一RX等权均值是不同加权。fold1–3反复使用同6个held-source身份，加上fold0的另6个，总计12个唯一RX，不能当24个独立接收机。CI按同RX跨fold先平均，再bootstrap5000次、seed20101；不覆盖共享训练集和单macro-seed的不确定性。95%CI跨零不意味着等价；“接近”只有当区间完全位于±1pp且|fold均值|<1pp才成立，仍不是总体等价检验。','',
            '## 全部配对指标','', '| fold | 模型 | 角色 | Acc % | RX Macro-F1 % | RX等权CE | TX margin | 最差RX % | 最差TX % |', '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        for role in ['source','held']:
            s=r[role]['summary'];lines.append(f"| {r['fold']} | {r['arm']} | {'fit验证' if role=='source' else 'held源'} | {100*s['acc']:.4f} | {100*s['macro_f1']:.4f} | {s['ce']:.5f} | {s['margin']:.4f} | {100*s['worst_rx']:.3f} | {100*s['worst_tx']:.3f} |")
    lines+=['','逐RX配对表见[PAIRED_RECEIVERS.csv](PAIRED_RECEIVERS.csv)；各模型混淆矩阵在runs下。所有Macro-F1按声明的10TX类计算，不删除原支持缺失类。最差配对变化与绝对最差RX准确率不是同一统计量。','',
            '## 复用、配方及选模审计','',
            'S10的12个ManyRx官方重训均使用全部24个source RX，已见本轮held6且split/seed不同，因此0个可严格复用；没有读取这些权重。S19的167个归档文件逐SHA核验，B0权重和既有source/held logits严格复用，未改或续训。详见[REUSE_AUDIT.json](REUSE_AUDIT.json)。','',
            '| fold | ASK训练轮数 | 包CE选中epoch | RX等权CE诊断最优epoch | checkpoint截止k | 训练计时秒 |','|---|---:|---:|---:|---:|---:|']
    for f in range(4):
        v=read(outdir(f,'ASK')/'metrics.json');lines.append(f"| {f} | {v['epochs']} | {v['best_epoch']} | {v['diagnostic_rx_ce_best_epoch']} | {v['selected_checkpoint_cutoff'][0]} | {v['seconds']:.2f} |")
    lines+=['','RX等权CE仅作日志审计，没有据此另测或替换官方checkpoint。官方函数直接负责优化、LR、earlystop和保存；包装只记录原CPU DataLoader实际顺序及原验证forward输出。全部新模型先封存并commit，再一次性读取本轮held-source。B0的历史held结果不是全新独立证据。','',
            '## 结构描述：不等于归因','',
            '| fold | 截止k | K酉性相对Fro误差 | K距identity相对Fro | K非对角能量比例 | eigenphase圆均值长度 |','|---|---:|---:|---:|---:|---:|']
    for v in cost['ASK_structure']:
        lines.append(f"| {v['fold']} | {v['selected_k']} | {v['unitary_error_fro_relative']:.3e} | {v['distance_to_identity_relative']:.5f} | {v['off_diagonal_energy_fraction']:.6f} | {v['eigenphase_circular_concentration']:.5f} |")
    lines+=['','ASK先做中心候选硬截止，再对最大127个复数模态施加U diag(exp(iθ)) Uᴴ，之后IFFT、总能量归一化和64sample/stride32的时间Patch，产生7×128 tokens。K可把已选模态的能量组合到最大低频窗口内其他模态；这些显式交互不是只缩放实数幅度。','',
            'HStem沿完整频谱五通道（Re/Im/LogMag/邻频phase）构建32×64频率坐标tokens，再用普通MHA全局交互；它也能学习跨频率关系，不能简单说它“没有频率交互”。HStem预attention局部感受野32频点/hop8，与ASK的64个重建时域sample单位不同，不能直接比较两个数字。K的酉性、非对角能量与截止位置只是已学习权重的描述，不证明性能差由这些因素造成。','',
            '## 参数与同机实际延迟','', '| 模型 | 参数元素 | 实标量自由度 | batch1 median ms | batch128 median ms |','|---|---:|---:|---:|---:|']
    for r in cost['costs']:
        lines.append(f"| {r['arm']} | {r['parameter_elements']} | {r['real_scalar_dof']} | {r['latency'][0]['median_batch_ms']:.4f} | {r['latency'][1]['median_batch_ms']:.4f} |")
    lines+=[ '',f"设备{cost['device']}，FP32，固定同样随机IQ；每配置10次warm-up、30次逐调用CUDA同步测量。保留ASK原forward中的QR，没有预计算K或缓存QR。参数元素将一个complex数计作1，实自由度计作2，故不能把100,240元素当100,240实参数。训练4fit累计{cost['training_seconds']:.2f}s；B0本轮新增训练0。",'',
             '延迟包含Python和同步开销，是本机单次测量，不能将小幅差异解释成普遍工程优势。全部样本、路径和静态权重描述见[COSTS_AND_STRUCTURE.json](COSTS_AND_STRUCTURE.json)。','',
             '## 证据边界与复现','',
             '本轮已保存全部ASK与冻结B0逐包source/held logits引用；[FINAL_ACCESS_AUDIT.json](FINAL_ACCESS_AUDIT.json)独立重算Accuracy/F1/CE/margin/最差RX/TX、索引、SHA、选模点和结果分类，核验72条科学输入访问。审计不做forward或打开IQ；成本/静态结构另有checkpoint-only访问清单，不读科研数据。','',
             '没有访问该fold真实目标RX、日期3–4、ManySig或Shen，没有新增泛化模块。S10真实CRD ASKNet均值88.7697%与S19/S20 held-source均值不直接可比，当前结果不能宣称超过任何论文。官方不同训练上限和选模规则使架构与配方耦合，单seed和12个唯一RX仍限制研究方向判断。','',
             f"用户提出的S19补充描述已核实：{reuse['central_evidence_check']['max_in_central_two']}/40组最高权重落在中央两带，其平均权重合计{100*reuse['central_evidence_check']['mean_central_mass']:.2f}%；该描述没有用于修改本轮默认截止候选或选择超参。",'',
             '复核命令（worktree根目录，D:/miniconda/envs/crossatten4060/python.exe）：`python 20-asknet-source-comparison/s20_finalize.py audit`。会更新派生QA时间；归档记录以本次MANIFEST为准。合成检查为`python -m unittest discover -s 20-asknet-source-comparison -p "test_*.py" -v`。','',
             '原始IQ、权重、逐包预测本地保留且由SHA引用，不进Git；作者源代码只保留外部固定Git blob，不转载。代码、协议、复用拒绝理由、训练曲线/顺序SHA、日志/k trace、配置、模型SHA、配对指标、CI、成本、决策与访问审计均已归档。后续任何结构消融、seed扩展或目标测试必须另行定义，不在本轮自动启动。','',
             '入口：[设计](README.zh-CN.md)、[预注册](registration.json)、[官方来源](UPSTREAM.json)、[封存](seal_pilot.json)、[机器决定](decision.json)、[归档SHA](MANIFEST.json)。']
    (HERE/'RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    write(HERE/'STATUS.json',dict(time=now(),status=dec['status'],new_fits=4,reused_B0=4,real_target_access=False,extra_module_training=False,QA='PASS'))
    paths=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'cache' not in p.parts and p.name!='MANIFEST.json' and p.suffix not in ['.pt','.npz','.pyc']]
    write(HERE/'MANIFEST.json',dict(time=now(),files_sha256={p.relative_to(HERE).as_posix():sha(p) for p in sorted(paths)},
          checkpoint_references=read(HERE/'seal_pilot.json')['checkpoints'],local_predictions={f'{a}_f{f}':read(outdir(f,a)/'held_metrics.json')['prediction_sha256'] for f in range(4) for a in ARMS},official_source_not_redistributed=True))
    print(dec['status'],'report and evidence archive generated',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['audit','cost','report']);args=p.parse_args()
    {'audit':audit,'cost':benchmark_and_structure,'report':report}[args.phase]()
