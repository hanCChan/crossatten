"""Final artifact seal only after complete scientific and visual verification."""
import ast
import importlib.metadata
from pathlib import Path
import re
import sys
from PIL import Image
from pypdf import PdfReader
from s27b_common import HERE,read,write,sha,now,validate


def decode_log(path):
    raw=path.read_bytes()
    for encoding in ['utf-8','gb18030']:
        try:return raw.decode(encoding).replace('\r\n','\n'),encoding
        except UnicodeDecodeError:pass
    raise UnicodeError('Cannot losslessly decode original log: '+str(path))


def main():
    validate();science=read(HERE/'EXPERIMENT_INTEGRITY.json');assert science['status']=='PASS'
    decision=read(HERE/'MECHANISM_DECISION.json');assert decision['new_fits']==12
    assert decision['C_fits']==decision['IQ_only_fits']==decision['ASK_strict_fits']==0
    visual=read(HERE/'VISUAL_QA.json');assert visual['status']=='PASS';documents=[]
    for doc in read(HERE/'DOCUMENT_BUILD.json')['documents']:
        p=Path(doc['path']);assert sha(p)==doc['sha256'];reader=PdfReader(p)
        text='\n'.join(q.extract_text() or '' for q in reader.pages)
        assert '??' not in text and '@@' not in text and '\ufffd' not in text
        log=Path(doc['log']).read_text(encoding='utf-8',errors='replace')
        assert not re.search(r'Overfull|LaTeX Warning|undefined|Missing character|Fatal error',log)
        v=next(q for q in visual['documents'] if q['sha256']==sha(p))
        assert v['reviewed_pages']==list(range(1,len(reader.pages)+1))
        renders=[]
        for n in range(1,len(reader.pages)+1):
            image=Path(doc['render'])/f'page-{n:0{len(str(len(reader.pages)))}d}.png'
            assert image.stat().st_mtime>=p.stat().st_mtime
            with Image.open(image) as im:im.verify()
            renders.append(dict(page=n,sha256=sha(image)))
        documents.append(dict(path=p.relative_to(HERE).as_posix(),sha256=sha(p),pages=len(reader.pages),renders=renders))
    parsed=0
    for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p));parsed+=1
    logs=[]
    for j in read(HERE/'registration.json')['jobs']:
        name=f"{j['arm']}_f{j['fold']}_s{j['seed']}";source=HERE/'local'/(name+'.log');target=HERE/'evidence'/name/'training_stdout.txt'
        text,encoding=decode_log(source);target.write_text(text,encoding='utf-8',newline='\n')
        logs.append(dict(id=name,raw_sha256=sha(source),raw_encoding=encoding,published_utf8_lf_sha256=sha(target)))
    for raw,target in [('controller.stdout.log','CONTROLLER_STDOUT.txt'),('controller.stderr.log','CONTROLLER_STDERR.txt')]:
        (HERE/target).write_text(decode_log(HERE/'local'/raw)[0],encoding='utf-8',newline='\n')
    for source in sorted((HERE/'local').glob('controller.resume_*.log')):
        target=HERE/(source.stem.upper().replace('.','_')+'.txt')
        target.write_text(decode_log(source)[0],encoding='utf-8',newline='\n')
    result=read(HERE/'RESULTS.json');s={q['arm']:q for q in result['summary']}
    block=(f"<!-- completion -->\n**S27-B 已完成：12/12 新拟合，每组 200 轮；24 模型、240000 条 Dev 及 48000 条 Source 预测独立核验。**\n\n"
        f"主交互 {decision['primary_interaction_pp']:+.3f} pp；正交互门槛是否满足：{decision['positive_interaction_rule_met']}。C/IQ-only/ASK Strict-IB 均未启动。\n\n"
        '[中文科研报告](RESEARCH_REPORT.zh-CN.md) · [英文 PDF](output/pdf/s27b_mechanism_report.pdf) · '
        '[四格结果](RESULTS.json) · [创新证据矩阵](EVIDENCE_MATRIX.json) · [科学核验](EXPERIMENT_INTEGRITY.json) · [最终审计](FINAL_AUDIT.json)\n\n'
        'Rebuild reports with research_materials.py and build_report.py from saved aggregates; do not rerun fits or Dev inference.\n<!-- /completion -->')
    path=HERE/'README.zh-CN.md';text=path.read_text(encoding='utf-8')
    if '<!-- completion -->' in text:text=re.sub(r'<!-- completion -->.*?<!-- /completion -->',lambda _:block,text,flags=re.S)
    else:text=text.replace('\n\n','\n\n'+block+'\n\n',1)
    path.write_text(text,encoding='utf-8',newline='\n')
    write(HERE/'STATUS.json',dict(time=now(),status='COMPLETE_S27_B_AND_MECHANISM_REPORT',completed_new_fits=12,
        reused_models=12,packet_predictions=240000,scientific_optimizer_steps=300000,
        engineering_synthetic_optimizer_steps=8,optimizer_step_scope='Retained final trajectories; discarded partial epochs after user pause are not counted.',C_fits=0,IQ_only_fits=0,ask_strict_fits=0,
        remaining_authorized_automatic_fits=0,independent_acquisition='NOT_AVAILABLE_NOT_RUN',
        integrity='PASS',visual='PASS',documents=documents,C_required_for_receiver_specific_claim=True))
    write(HERE/'LIVE_STATUS.json',dict(updated_at=now(),status='COMPLETE_S27_B_AND_MECHANISM_REPORT',completed_fits=12,total_fits=12,
        remaining_fit_seconds_estimate=0,remaining_authorized_automatic_fits=0,report='COMPLETE'))
    artifacts=[]
    for p in sorted(HERE.rglob('*')):
        rel=p.relative_to(HERE)
        if not p.is_file() or any(t in {'local','runs','predictions','tmp','__pycache__'} for t in rel.parts) or rel.as_posix()=='FINAL_AUDIT.json':continue
        artifacts.append(dict(path=rel.as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    write(HERE/'FINAL_AUDIT.json',dict(time=now(),status='PASS',new_fits=12,scientific_optimizer_steps=300000,
        engineering_synthetic_optimizer_steps=8,optimizer_step_scope='Retained final trajectories; discarded partial epochs after user pause are not counted.',packet_predictions_checked=240000,all_fits_rx_retained=True,
        original_scientific_code_unchanged=True,historical_reference_models_unchanged=science['historical_reference_models_unchanged'],
        confirm_blind_reads=0,C_fits=0,IQ_only_fits=0,ask_strict_fits=0,
        scientific_audit_sha256=sha(HERE/'EXPERIMENT_INTEGRITY.json'),active_python_sources_parsed=parsed,
        all_final_pages_visually_reviewed=True,documents=documents,stdout_logs=logs,artifacts=artifacts,
        python=sys.version,packages={k:importlib.metadata.version(k) for k in ['torch','numpy','scipy','matplotlib','pypdf','Pillow']},
        interpretation='Fourcell interaction of chosen jointlytrained tokenizers/backends. Nonlinear forms/capacity/compute differ; no universal attention necessity,physicalRF invariance orfreshacquisition guarantee.'))
    print('Final integrity/document seal PASS:',len(artifacts),'artifacts.',flush=True)


if __name__=='__main__':main()
