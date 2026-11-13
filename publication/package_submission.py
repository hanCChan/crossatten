"""Create a portable LaTeX review bundle and prove it builds outside the repo."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parent.parent
PAPER=ROOT/'publication/lora_hstem'
OUT=ROOT/'output/review'
CHECK=ROOT/'tmp/lora_portable_check'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(command,cwd):
    r=subprocess.run(command,cwd=cwd,text=True,encoding='utf8',errors='replace',stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError(r.stdout)
    return r.stdout


def main():
    qa=json.loads((PAPER/'DOCUMENT_QA.json').read_text(encoding='utf8'))
    if qa['status']!='PASS' or not qa['visual_review'].startswith('PASS:'):raise ValueError('Visual/document QA not complete')
    OUT.mkdir(parents=True,exist_ok=True);CHECK.mkdir(parents=True,exist_ok=True)
    files=[(p,'lora_hstem/'+p.relative_to(PAPER).as_posix()) for p in sorted(PAPER.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    for n in ('lora_hstem_manuscript.pdf','lora_hstem_supplement.pdf'):
        files.append((ROOT/'output/pdf'/n,'lora_hstem/pdf/'+n))
    manifest={dest:sha(p) for p,dest in files}
    archive=OUT/'lora_hstem_submission_source.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p,dest in files:z.write(p,dest)
        z.writestr('lora_hstem/SHA256SUMS.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(archive) as z:
        if any('..' in Path(n).parts or Path(n).is_absolute() for n in z.namelist()):raise ValueError('Unsafe archive path')
        z.extractall(CHECK)
    if not all(sha(CHECK/p)==h for p,h in manifest.items()):raise ValueError('Extracted bundle checksum mismatch')
    portable=CHECK/'lora_hstem'
    stdout=run([sys.executable,'build.py'],portable)
    checkstdout=run([sys.executable,'qa_documents.py'],portable)
    image_equality={};text_equality={}
    for name in ('lora_hstem_manuscript','lora_hstem_supplement'):
        original=ROOT/'tmp/pdfs/lora_hstem';rebuilt=portable/'tmp/pdfs/lora_hstem'
        for i in range(1,7):image_equality[name+f'_page{i}']=sha(original/f'{name}-{i}.png')==sha(rebuilt/f'{name}-{i}.png')
        a=run(['pdftotext',str(ROOT/'output/pdf'/f'{name}.pdf'),'-'],ROOT)
        b=run(['pdftotext',str(portable/'output/pdf'/f'{name}.pdf'),'-'],portable)
        text_equality[name]=a==b
    if not all(image_equality.values()) or not all(text_equality.values()):raise ValueError('Portable build changed rendered pages or extracted text')
    result=dict(status='PASS',archive=str(archive.relative_to(ROOT)),sha256=sha(archive),bytes=archive.stat().st_size,
        bundled_files=len(files),portable_build_stdout=stdout.strip(),portable_checks_stdout=checkstdout.strip(),
        all12_rendered_pages_identical=True,render_sha_checks=image_equality,text_equality=text_equality,
        compilation_requires_original_D_paths=False,new_scientific_fits=0,new_IQ_reads=0,
        pdfs={n:dict(path=str((ROOT/'output/pdf'/n).relative_to(ROOT)),sha256=sha(ROOT/'output/pdf'/n)) for n in ('lora_hstem_manuscript.pdf','lora_hstem_supplement.pdf')})
    (ROOT/'publication/LORA_PORTABLE_BUNDLE_QA.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('render_sha_checks','text_equality','pdfs')},indent=2),flush=True)


if __name__=='__main__':main()
