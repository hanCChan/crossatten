"""Check final PDFs, build contact sheets, and verify manuscript-data provenance."""
from pathlib import Path
import json
import re
import subprocess
from PIL import Image,ImageDraw
from prepare_evidence import sha

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1] if HERE.parent.name=='publication' else HERE
BUILD=ROOT/'tmp/pdfs/lora_hstem'


def main():
    report=json.loads((HERE/'BUILD_REPORT.json').read_text(encoding='utf8'));checks={};page_counts={}
    for name,r in report.items():
        checks[name+'_no_build_warnings']=not r['warnings']
        text=subprocess.check_output(['pdftotext','-layout',r['pdf'],'-'],encoding='utf8')
        pages=[s for s in text.split('\f') if s.strip()]
        page_counts[name]=len(pages)
        checks[name+'_six_nonempty_pages']=len(pages)==6 and all(len(s.strip())>300 for s in pages)
        checks[name+'_no_unresolved_citations']='[?]' not in text and '??' not in text
        checks[name+'_review_author_label']='Anonymous review draft' in text
        checks[name+'_render_page_count_matches']=len(r['render_pages'])==len(pages)
        fonts=subprocess.check_output(['pdffonts',r['pdf']],encoding='utf8')
        checks[name+'_no_bitmap_fonts']='Type 3' not in fonts
        sheet=Image.new('RGB',(1560,2*720),'#dde2e6');draw=ImageDraw.Draw(sheet)
        for i,p in enumerate(r['render_pages']):
            im=Image.open(p).convert('RGB');im.thumbnail((500,675))
            x=(i%3)*520+(520-im.width)//2;y=(i//3)*720+28
            sheet.paste(im,(x,y));draw.text(((i%3)*520+16,(i//3)*720+8),f'{name} / page {i+1}',fill='#152938')
        sheet.save(BUILD/(name+'_contact.png'))
    ev=json.loads((HERE/'EVIDENCE_MANIFEST.json').read_text(encoding='utf8'))
    checks['all_frozen_source_copies_match']=all(sha(HERE/r['destination'])==r['sha256'] for r in ev['source_files'])
    checks['aggregate_evidence_checks_passed']=ev['status']=='PASS'
    checks['preparation_no_scientific_computation']=ev['new_fits']==ev['IQ_reads']==ev['checkpoint_loads']==0
    maintext=subprocess.check_output(['pdftotext',report['lora_hstem_manuscript']['pdf'],'-'],encoding='utf8')
    checks['main_contains_main_accuracy_and_parameter_budget']=all(s in maintext for s in ('97.335','133,690','571,434','2.687'))
    checks['main_discloses_TF32_and_shift_failure']=all(s in maintext for s in ('TF32','0.01957','17.775'))
    checks['main_discloses_prior_exposure']='prior exposure' in maintext.lower() and 'blind' in maintext.lower()
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,page_counts=page_counts,
        source_copies=len(ev['source_files']),new_scientific_fits=0,new_IQ_reads=0,
        visual_review='Pending assistant inspection of the latest page PNGs; automation does not establish visual correctness')
    (HERE/'DOCUMENT_QA.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(result['status'],len(checks),'checks',page_counts,flush=True)
    if not all(checks.values()):raise ValueError('Document checks failed')


if __name__=='__main__':main()
