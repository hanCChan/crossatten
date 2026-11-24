"""Final document/provenance seal; neither A nor C needs a favorable result."""
import ast
from pathlib import Path
import re
import subprocess
import sys
from pypdf import PdfReader
from ac_common import HERE,ROOT,read,write,sha,now,validate,identity

def main():
    reg=validate();integrity=read(HERE/'EXPERIMENT_INTEGRITY.json');assert integrity['status']=='PASS'
    decision=read(HERE/'MECHANISM_DECISION.json');assert decision['scientific_fits']==18 and decision['remaining_authorized_fits']==0
    assert decision['S28B_fits']==decision['IQ_only_fits']==decision['ASK_strict_fits']==0
    visual=read(HERE/'VISUAL_QA.json');assert visual['status']=='PASS';documents=[]
    for d in read(HERE/'DOCUMENT_BUILD.json')['documents']:
        p=Path(d['path']);assert sha(p)==d['sha256'];pages=PdfReader(p).pages
        q=next(q for q in visual['documents'] if q['sha256']==sha(p));assert q['reviewed_pages']==list(range(1,len(pages)+1))
        log=Path(d['log']).read_text(encoding='utf-8',errors='replace')
        assert not re.search(r'Overfull|LaTeX Warning|undefined|Missing character|Fatal error',log)
        assert all('\ufffd' not in (page.extract_text() or '') and '??' not in (page.extract_text() or '') for page in pages)
        documents.append(dict(path=p.relative_to(HERE).as_posix(),sha256=sha(p),pages=len(pages),renders=[dict(page=n,
            sha256=sha(Path(d['render'])/f'page-{n:0{len(str(len(pages)))}d}.png')) for n in range(1,len(pages)+1)]))
    logs=[]
    for p in sorted((HERE/'local').glob('*.log')):
        raw=p.read_bytes()
        for encoding in ['utf-8','utf-16','gb18030']:
            try:text=raw.decode(encoding);break
            except UnicodeError:continue
        else:raise UnicodeError(str(p))
        target=HERE/'logs'/(p.stem+'.txt');target.parent.mkdir(exist_ok=True)
        normalized='\n'.join(line.rstrip(' \t') for line in text.replace('\r\n','\n').split('\n'))
        target.write_text(normalized,encoding='utf-8',newline='\n')
        logs.append(dict(original=p.name,original_sha256=sha(p),encoding=encoding,
            publication_normalization='UTF8, LF, trailing spaces/tabs removed; original local bytes untouched.',
            published=target.relative_to(HERE).as_posix(),sha256=sha(target)))
    for j in reg['jobs']:
        p=HERE/'runs'/identity(j);out=HERE/'evidence'/identity(j)
        for f in list(p.glob('recovery_epoch*.json'))+list(p.glob('paused.json')):(out/f.name).write_bytes(f.read_bytes())
    for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    protected=[HERE/'RESULTS.json',HERE/'EVALUATION_A.json',HERE/'EVALUATION_C.json',HERE/'access.jsonl',HERE/'seal_A.json',HERE/'seal_C.json']
    before={p.name:sha(p) for p in protected}
    guard=subprocess.run([sys.executable,str(HERE/'resume.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
    assert 'already complete' in guard.stdout.decode('utf-8')
    assert before=={p.name:sha(p) for p in protected}
    write(HERE/'COMPLETION_GUARD.json',dict(time=now(),status='PASS',output=guard.stdout.decode('utf-8').strip(),protected_sha256=before,no_new_fits_or_RF_reads=True))
    write(HERE/'STATUS.json',dict(time=now(),status='COMPLETE_18_FITS_A_C_MECHANISM_REPORT_STOP',completed_fits=18,A_completed=6,C_completed=12,
        reused_models=24,retained_optimizer_steps=390000,synthetic_engineering_updates=read(HERE/'GATE.json')['synthetic_optimizer_updates'],
        target_predictions=324000,source_predictions=79200,C_held_RX_units=2,C_geometry_vectors=72000,
        scientific_integrity='PASS',visual_QA='PASS',documents=documents,remaining_authorized_fits=0,
        S28B_fits=0,IQ_only_fits=0,ASK_strict_fits=0,new_Confirm_blind_reads=0,independent_acquisition='NOT_AVAILABLE_NOT_RUN',
        old_native_S28_precision_failure='PRESERVED',interpretation='A joint training procedure; C two historicalheldRXpilot, no population or unique physical/optimization proof.'))
    write(HERE/'LIVE_STATUS.json',dict(time=now(),status='COMPLETE_18_FITS_A_C_MECHANISM_REPORT_STOP',completed_fits=18,total_fits=18,A_completed=6,C_completed=12,remaining_authorized_fits=0))
    result=read(HERE/'RESULTS.json');effect=result['A']['primary_interaction'];pilot=result['C']['pilot']
    block=(f"<!-- completion -->\n**已完成 A6+C12，共18次200轮训练，独立核验通过，自动训练停止。**\n\n"
        f"A主交互 {effect['mean_delta_pp']:+.3f} pp，双向规则满足：{effect['two_sided_practical_rule_met']}；C两RX共同Pilot规则满足：{pilot['consistent_pilot_signal']}。\n\n"
        '[中文机制报告](RESEARCH_REPORT.zh-CN.md) · [英文PDF](output/pdf/s28_ac_mechanism_results.pdf) · [完整结果](RESULTS.json) · '
        '[独立核验](EXPERIMENT_INTEGRITY.json) · [证据矩阵](EVIDENCE_MATRIX.json) · [最终审计](FINAL_AUDIT.json)\n\n'
        'Rebuild materials and PDF from saved aggregates; do not rerun scientific fits or target evaluation.\n<!-- /completion -->')
    p=HERE/'README.zh-CN.md';text=p.read_text(encoding='utf-8')
    if '<!-- completion -->' in text:text=re.sub(r'<!-- completion -->.*?<!-- /completion -->',lambda _:block,text,flags=re.S)
    else:text=text.replace('\n\n','\n\n'+block+'\n\n',1)
    p.write_text(text,encoding='utf-8',newline='\n')
    artifacts=[]
    for p in sorted(HERE.rglob('*')):
        rel=p.relative_to(HERE)
        if not p.is_file() or any(t in ['local','runs','predictions','tmp','__pycache__'] for t in rel.parts) or p.name=='FINAL_AUDIT.json':continue
        artifacts.append(dict(path=rel.as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    write(HERE/'FINAL_AUDIT.json',dict(time=now(),status='PASS_COMPLETE_REGISTERED_18_FITS',scientific_fits=18,retained_optimizer_steps=390000,
        step_scope='Final retained trajectories; discarded partial-epoch work after pauses, if any, is not counted.',
        synthetic_engineering_updates=read(HERE/'GATE.json')['synthetic_optimizer_updates'],original_scientific_files_unchanged=True,
        original_plans_byte_exact=True,historical_models_unchanged=24,all_seeds_RX_counterexamples_retained=True,
        target_predictions_verified=324000,source_predictions_verified=79200,C_receiver_units=2,new_Confirm_blind_reads=0,
        remaining_fits=0,IO_only_repair_scientific_fits=0,IO_only_repair_receipt_sha256=sha(HERE/'TECHNICAL_REPAIR_001.json'),
        scientific_audit_sha256=sha(HERE/'EXPERIMENT_INTEGRITY.json'),logs=logs,documents=documents,artifacts=artifacts))
    print('Final A/C delivery seal PASS:',len(artifacts),'artifacts; no extra training.',flush=True)

if __name__=='__main__':main()
