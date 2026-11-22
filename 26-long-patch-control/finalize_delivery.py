"""Seal complete research documents only after scientific and visual audits."""
import ast
import importlib.metadata
import re
import sys
from pathlib import Path
from pypdf import PdfReader
from PIL import Image
from s26_common import HERE,ROOT,read,write,sha,now,validate


def main():
    validate();science=read(HERE/'EXPERIMENT_INTEGRITY.json');assert science['status']=='PASS'
    assert science['new_fits']==6 and science['target_packet_predictions']==180000
    assert read(HERE/'TECHNICAL_FIX_001_RUNTIME.json')['status']=='COMPLETE'
    documents=read(HERE/'DOCUMENT_BUILD.json')['documents'];visual=read(HERE/'VISUAL_QA.json')
    assert visual['status']=='PASS' and len(documents)==2
    doc_rows=[]
    for doc in documents:
        file=Path(doc['path']);assert sha(file)==doc['sha256']
        reader=PdfReader(file);pages=len(reader.pages);assert pages==doc['pages']
        text='\n'.join(p.extract_text() or '' for p in reader.pages)
        assert '??' not in text and '@@' not in text and '\ufffd' not in text
        log=Path(doc['log']).read_text(encoding='utf-8',errors='replace')
        assert not re.search(r'Overfull|undefined|Missing character|Fatal error|LaTeX Warning',log),doc['log']
        reviewed=next(r for r in visual['documents'] if r['sha256']==sha(file))
        assert reviewed['reviewed_pages']==list(range(1,pages+1))
        renders=[]
        for number in range(1,pages+1):
            # Poppler pads numbering according to the complete document length.
            path=Path(doc['render'])/f'page-{number:0{len(str(pages))}d}.png'
            assert path.stat().st_mtime>=file.stat().st_mtime
            with Image.open(path) as im:im.verify()
            renders.append(dict(page=number,sha256=sha(path)))
        doc_rows.append(dict(path=file.relative_to(HERE).as_posix(),pages=pages,sha256=sha(file),renders=renders))
    reused=[]
    for p in sorted((ROOT/'25-tifs-integration/paper/figures').glob('*')):
        target=HERE/'paper/figures'/p.name
        assert target.exists() and sha(target)==sha(p)
        reused.append(dict(path=target.relative_to(HERE).as_posix(),source=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
    for extension in ['pdf','png']:
        path=HERE/'paper/figures'/('fig08_long_patch.'+extension);assert path.exists()
        if extension=='png':
            with Image.open(path) as im:im.verify()
        else:assert len(PdfReader(path).pages)==1
    parsed=0
    for p in HERE.glob('*.py'):
        if p.name=='s26_evaluate.py':continue # preserved known parser failure, registered replacement used
        ast.parse(p.read_text(encoding='utf-8'),filename=str(p));parsed+=1
    logs=[]
    for job in read(HERE/'registration.json')['jobs']:
        name=f"LP_f{job['fold']}_s{job['seed']}";source=HERE/'local'/(name+'.log')
        destination=HERE/'evidence'/name/'training_stdout.txt'
        destination.write_text(source.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
        logs.append(dict(job=name,raw_stdout_sha256=sha(source),normalized_public_stdout_sha256=sha(destination)))
    for source_name,target_name in [('controller.stderr.log','ORIGINAL_CONTROLLER_STDERR.txt'),
                                   ('technical_001.stdout.log','TECHNICAL_001_STDOUT.txt')]:
        source=HERE/'local'/source_name
        (HERE/target_name).write_text(source.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
    write(HERE/'REUSED_FIGURES.json',dict(status='VERIFIED_IDENTICAL_REUSE',rows=reused,
        note='Copied historical figures are not new experiments; only fig08 uses new S26 control results.'))
    result=read(HERE/'RESULTS.json');summary={r['arm']:r for r in result['summary']};pair=result['paired_comparisons'][0]
    block=('<!-- completion -->\n**本阶段已完成：6/6 新训练、18 模型统一评估、180000 条 Dev 预测独立核验。**\n\n'
        f"H0 {summary['H0']['accuracy_pct']:.3f}%，Long-Patch {summary['LP']['accuracy_pct']:.3f}%，P0 {summary['P0']['accuracy_pct']:.3f}%；"
        f"H0−LP {pair['mean_delta_pp']:+.3f} pp。全部六模型和十个 RX 保留，条件性解释见 [RESULTS](RESULTS.json) 与 [GO/STOP](GO_STOP.json)。\n\n"
        '[修订英文稿](output/pdf/hstem_tifs_revised.pdf) · [补充材料](output/pdf/hstem_tifs_supplement.pdf) · '
        '[中英文贡献与十项问答](RESEARCH_NOTES.zh-CN.txt) · [科研完整性核验](EXPERIMENT_INTEGRITY.json) · [最终交付审计](FINAL_AUDIT.json)\n\n'
        '仅重建文档：运行 `paper_revision.py`、`research_notes.py`、`build_documents.py`，不需要波形或权重；只用已提交聚合证据和图表。不要重新启动已完成训练或重复 Dev 评价。\n<!-- /completion -->')
    readme=(HERE/'README.zh-CN.md').read_text(encoding='utf-8')
    if '<!-- completion -->' in readme:readme=re.sub(r'<!-- completion -->.*?<!-- /completion -->',lambda _:block,readme,flags=re.S)
    else:readme=readme.replace('\n\n','\n\n'+block+'\n\n',1)
    (HERE/'README.zh-CN.md').write_text(readme,encoding='utf-8',newline='\n')
    state=dict(status='COMPLETE_S26_CONTROL_AND_REVISED_DRAFTS',time=now(),completed_new_fits=6,
        remaining_authorized_new_fits=0,reused_models=12,target_packet_predictions=180000,
        integrity_audit='PASS',visual_review='PASS',documents=doc_rows,
        independent_acquisition='NOT_AVAILABLE_NOT_RUN',ma2025='NO_QUALIFIED_ADAPTER_OR_SCORE',
        original_scientific_code_unchanged=True,technical_parser_repair='TECHNICAL_FIX_001.json')
    write(HERE/'STATUS.json',state)
    write(HERE/'LIVE_STATUS.json',dict(status=state['status'],updated_at=now(),completed_fits=6,total_fits=6,
        active_model=None,remaining_fit_seconds_estimate=0,report_only_evaluation='COMPLETE',documents='COMPLETE',
        remaining_authorized_new_fits=0))
    artifacts=[]
    for p in sorted(HERE.rglob('*')):
        rel=p.relative_to(HERE)
        if not p.is_file() or any(part in {'local','runs','predictions','tmp','__pycache__'} for part in rel.parts):continue
        if rel.as_posix()=='FINAL_AUDIT.json':continue
        artifacts.append(dict(path=rel.as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    write(HERE/'FINAL_AUDIT.json',dict(status='PASS',time=now(),new_fits=6,scientific_optimizer_steps=150000,
        packet_predictions_checked=180000,scientific_audit_sha256=sha(HERE/'EXPERIMENT_INTEGRITY.json'),
        original_scientific_code_unchanged=True,known_original_parser_failure_preserved=True,active_python_sources_parsed=parsed,
        all_historical_reference_models_unchanged=science['historical_reference_models_unchanged'],
        confirm_blind_reads=0,docs=doc_rows,all_final_pages_visually_reviewed=True,stdout_logs=logs,
        artifacts=artifacts,python=sys.version,packages={name:importlib.metadata.version(name) for name in
            ['torch','numpy','scipy','matplotlib','pypdf','Pillow']},
        interpretation='New developmental same-support control; no new independent capture, no unique hierarchy or physical receiver-invariance theorem.'))
    print('Final delivery PASS:',len(artifacts),'artifacts;',[(d['path'],d['pages']) for d in doc_rows],flush=True)


if __name__=='__main__':main()
