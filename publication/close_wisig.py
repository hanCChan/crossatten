"""Archive bounded WiSig STOP outcomes and test a portable statistics audit."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
QUALITY=ROOT/'research/quality_preservation'
OUT=ROOT/'output/review'
CHECK=ROOT/'tmp/wisig_portable_check'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True);CHECK.mkdir(parents=True,exist_ok=True)
    definitions=[('P2','periodic_span','DECISION.json','RESULTS.zh-CN.md'),
        ('F2','spectral_scale_execution','DECISION.json','RESULTS.zh-CN.md'),
        ('M1','m1_mirror_identifiability','DECISION.json','RESULTS.zh-CN.md'),
        ('W1/W2','quality_preservation','W2_RESULT.json','W2_RESULTS.zh-CN.md')]
    stages=[];files=[]
    for name,folder,decision,report in definitions:
        p=ROOT/'research'/folder/decision;r=json.loads(p.read_text(encoding='utf8'))
        if not r['status'].startswith('STOP'):raise ValueError('Unexpected historical decision')
        stages.append(dict(stage=name,status=r['status'],decision=str(p.relative_to(ROOT)),sha256=sha(p),closed=True))
        if folder!='quality_preservation':
            for fn in (decision,report):files.append((p.parent/fn,'wisig_audit/prior_stops/'+name+'/'+fn))
    closure=dict(date='2026-11-13',status='CLOSED_CURRENT_WISIG_ALGORITHM_SEARCH',stages=stages,
        W1_finding_retained=True,new_neural_training_in_this_turn=0,new_IQ_reads_in_this_turn=0,
        optional_risk_coverage_not_executed=True,
        restart_requirements=['New falsifiable physically identifiable hypothesis','Available acquisition/data evidence',
            'Substantive difference from existing methods','Qualified evaluation protocol with exposure disclosed','Separate bounded preregistration before training'],
        LoRa_priority='Prepare bounded-scope manuscript without changing the verified model',
        interpretation='STOP applies to the registered implementations, not a proof that all future cross-receiver methods are impossible.')
    closurepath=ROOT/'research/WISIG_CLOSURE_2026_11_13.json'
    closurepath.write_text(json.dumps(closure,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    text='''# WiSig 正式结题与后续研究条件

2026-11-13：按用户结题判断，P2、F2、M1及W2维持原STOP，当前WiSig算法搜索正式结束。保留W1条件关联发现；不通过调整门控、阈值或Loss补救，不自动登记W3/W4。

W1对22,968源包的分析显示：同TX/RX/Day内低一致性四分位比其余包高17.5332pp错误，RX聚类95%条件区间[9.0440,26.9902]pp。W2中B2−B0仅+0.0178pp、B2−B1为+0.1606pp、B2−N0为−0.0438pp，三组区间均跨零。保留微小描述性差值，不据此声称有效的质量条件化泛化算法。

下一次启动WiSig训练前，需要新的可证伪、物理上可辨识的假设，具备实际数据证据，明确区别已有方法，并建立资格清楚且披露使用历史的评价协议。不得把旧开发接收机换名称当成新盲测。LoRa当前转入稿件准备，主模型保持冻结。

可选Risk–Coverage整理属于错误风险评估，不能替代全覆盖Accuracy；若以后确有需要，须与MSP/预测熵等零训练参照比较，并报告TX/RX覆盖公平性。本轮不额外开展该分析。

自包含审计包位于output/review/wisig_closure_audit.zip。解压后进入wisig_audit/quality_preservation，使用安装numpy的Python运行verify_results.py，可在不读取本机D盘缓存、IQ或模型权重的情况下重算175项统计/顺序/配对检查。该范围包括W1/W2统计复核及此前STOP的原始结论摘要；它不是历史训练或物理机制的独立重复。
'''
    (ROOT/'research/WISIG_CLOSURE_2026_11_13.md').write_text(text,encoding='utf8')
    # All final quality-stage files; originals and their original manifest stay unchanged.
    for p in QUALITY.iterdir():
        if p.is_file():files.append((p,'wisig_audit/quality_preservation/'+p.name))
    artifacts=[QUALITY/'artifacts'/f'{v}_packets.npz' for v in ('V0','V1')]
    artifacts+=sorted((QUALITY/'artifacts').glob('*_selection.npz'))
    if len(artifacts)!=98:raise ValueError('Expected two W1 and96 W2 prediction files')
    for p in artifacts:
        with np.load(p,allow_pickle=False) as z:
            if 'x' in z.files or any(z[k].ndim==3 and z[k].shape[1:]==(2,256) for k in z.files):raise ValueError('IQ is outside this archive scope')
        files.append((p,'wisig_audit/quality_preservation/artifacts/'+p.name))
    jobs=json.loads((QUALITY/'W2_REGISTRATION.json').read_text(encoding='utf8'))['jobs']
    for job in jobs:
        for name in ('HISTORY.json','COMPLETE.json','CONTEXT_PAIRING.npz','INITIAL.json'):
            p=QUALITY/'artifacts'/job['id']/name
            files.append((p,'wisig_audit/quality_preservation/artifacts/'+job['id']+'/'+name))
    files.append((closurepath,'wisig_audit/WISIG_CLOSURE.json'))
    readme='''# Portable WiSig closure audit

This bundle contains the full W1/W2 paired intervals, receiver/TX risk tables,
frozen selected prediction arrays, pairing/trajectory summaries, and prior P2/F2/M1
STOP summaries. No raw IQ or model/optimizer weights are included.

Requirements: Python 3.10+ and NumPy. From quality_preservation/ run:

    python verify_results.py

Expected result: PASS 175 checks. The verifier has no absolute-D-path dependency.
It recomputes statistical summaries and checks recorded orders/pairings; it does
not independently reproduce neural training, infer physical causality, or make
the development receivers new blind tests. Absolute paths in historical metadata
are retained as provenance and are not dereferenced by this verifier.

Prior STOP folders contain historical decision/report snapshots only; the full
portable statistical replay is specific to W1/W2. Do not invoke the one-shot
training scripts to inspect these results. Their original inputs are intentionally
not bundled. Every bundled input is pinned by SHA256SUMS.json.
'''
    path=OUT/'wisig_closure_audit.zip'
    manifest={dest:sha(p) for p,dest in files}
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p,dest in files:z.write(p,dest)
        z.writestr('wisig_audit/README.md',readme)
        z.writestr('wisig_audit/SHA256SUMS.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(path) as z:
        if any('..' in Path(n).parts or Path(n).is_absolute() for n in z.namelist()):raise ValueError('Unsafe archive path')
        z.extractall(CHECK)
    cwd=CHECK/'wisig_audit/quality_preservation'
    before=json.loads((CHECK/'wisig_audit/SHA256SUMS.json').read_text())
    if not all(sha(CHECK/name)==h for name,h in before.items()):raise ValueError('Archive extraction checksum mismatch')
    proc=subprocess.run([sys.executable,'verify_results.py'],cwd=cwd,text=True,encoding='utf8',stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if proc.returncode:raise RuntimeError(proc.stdout)
    qa=json.loads((cwd/'RESULTS_QA.json').read_text())
    result=dict(status='PASS',archive=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size,
        bundled_input_files=len(files),portable_verifier_stdout=proc.stdout.strip(),portable_QA_count=qa['count'],
        new_training_fits=0,new_IQ_reads=0,raw_IQ_or_weights_bundled=False)
    (ROOT/'research/WISIG_PORTABLE_AUDIT_QA.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
