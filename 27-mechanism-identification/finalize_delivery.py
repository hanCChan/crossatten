"""Final artifact seal only after complete scientific and visual verification."""
import ast
import importlib.metadata
from pathlib import Path
import re
import sys
from PIL import Image
from pypdf import PdfReader
from s27_common import HERE,read,write,sha,now,validate


def main():
    validate();science=read(HERE/'EXPERIMENT_INTEGRITY.json');assert science['status']=='PASS'
    decision=read(HERE/'MECHANISM_DECISION.json');assert decision['new_fits']==6
    assert decision['B_fits']==decision['C_fits']==decision['IQ_only_fits']==decision['ask_strict_fits']==0
    assert read(HERE/'PLAN_B.json')['automatic_fit_budget']==read(HERE/'PLAN_C.json')['automatic_fit_budget']==0
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
        name=f"LA_f{j['fold']}_s{j['seed']}";source=HERE/'local'/(name+'.log');target=HERE/'evidence'/name/'training_stdout.txt'
        target.write_text(source.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
        logs.append(dict(id=name,raw_sha256=sha(source),published_lf_sha256=sha(target)))
    for raw,target in [('controller.stdout.log','CONTROLLER_STDOUT.txt'),('controller.stderr.log','CONTROLLER_STDERR.txt')]:
        (HERE/target).write_text((HERE/'local'/raw).read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
    result=read(HERE/'RESULTS.json');s={q['arm']:q for q in result['summary']}
    block=(f"<!-- completion -->\n**S27-A 已完成：6/6 新拟合，每组 200 轮；18 模型、180,000 条 Dev 预测通过独立核验。**\n\n"
        f"HStem {s['H0']['accuracy_pct']:.3f}%，No-GELU {s['LA']['accuracy_pct']:.3f}%，Long-Patch {s['LP']['accuracy_pct']:.3f}%；"
        f"H0−LA {decision['primary_delta_pp']:+.3f} pp。B/C/IQ-only/ASK Strict-IB 均未启动。\n\n"
        '[中文科研报告](RESEARCH_REPORT.zh-CN.md) · [英文 PDF](output/pdf/s27_mechanism_report.pdf) · '
        '[机制判断](MECHANISM_DECISION.json) · [科学核验](EXPERIMENT_INTEGRITY.json) · [最终审计](FINAL_AUDIT.json)\n\n'
        '重建图文只需运行 `research_materials.py`、`build_report.py`，使用已提交聚合记录，不需要训练数据或权重。'
        '不重复启动已完成的训练或 Dev 推理。\n<!-- /completion -->')
    path=HERE/'README.zh-CN.md';text=path.read_text(encoding='utf-8')
    if '<!-- completion -->' in text:text=re.sub(r'<!-- completion -->.*?<!-- /completion -->',lambda _:block,text,flags=re.S)
    else:text=text.replace('\n\n','\n\n'+block+'\n\n',1)
    path.write_text(text,encoding='utf-8',newline='\n')
    write(HERE/'STATUS.json',dict(time=now(),status='COMPLETE_S27_A_AND_MECHANISM_REPORT',completed_new_fits=6,
        reused_models=12,packet_predictions=180000,scientific_optimizer_steps=150000,
        engineering_synthetic_optimizer_steps=8,B_fits=0,C_fits=0,IQ_only_fits=0,ask_strict_fits=0,
        remaining_authorized_automatic_fits=0,independent_acquisition='NOT_AVAILABLE_NOT_RUN',
        integrity='PASS',visual='PASS',documents=documents,next_highest_value=decision['next_highest_identification_value']))
    write(HERE/'LIVE_STATUS.json',dict(updated_at=now(),status='COMPLETE_S27_A_AND_MECHANISM_REPORT',completed_fits=6,total_fits=6,
        remaining_fit_seconds_estimate=0,remaining_authorized_automatic_fits=0,report='COMPLETE'))
    artifacts=[]
    for p in sorted(HERE.rglob('*')):
        rel=p.relative_to(HERE)
        if not p.is_file() or any(t in {'local','runs','predictions','tmp','__pycache__'} for t in rel.parts) or rel.as_posix()=='FINAL_AUDIT.json':continue
        artifacts.append(dict(path=rel.as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    write(HERE/'FINAL_AUDIT.json',dict(time=now(),status='PASS',new_fits=6,scientific_optimizer_steps=150000,
        engineering_synthetic_optimizer_steps=8,packet_predictions_checked=180000,all_fits_rx_retained=True,
        original_scientific_code_unchanged=True,historical_reference_models_unchanged=science['historical_reference_models_unchanged'],
        confirm_blind_reads=0,B_fits=0,C_fits=0,IQ_only_fits=0,ask_strict_fits=0,
        scientific_audit_sha256=sha(HERE/'EXPERIMENT_INTEGRITY.json'),active_python_sources_parsed=parsed,
        all_final_pages_visually_reviewed=True,documents=documents,stdout_logs=logs,artifacts=artifacts,
        python=sys.version,packages={k:importlib.metadata.version(k) for k in ['torch','numpy','scipy','matplotlib','pypdf','Pillow']},
        interpretation='Activation-removal effect in a fixed complete training procedure; BN/optimization/learned weights remain coupled. No physicalRF or freshacquisition guarantee.'))
    print('Final integrity/document seal PASS:',len(artifacts),'artifacts.',flush=True)


if __name__=='__main__':main()
