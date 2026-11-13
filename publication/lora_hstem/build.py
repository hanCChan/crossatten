"""Build both review PDFs; render with Poppler for visual verification."""
from pathlib import Path
import json
import os
import re
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1] if HERE.parent.name=='publication' else HERE
BUILD=ROOT/'tmp/pdfs/lora_hstem'
OUTPUT=ROOT/'output/pdf'


def run(args,cwd,env=None):
    p=subprocess.run(args,cwd=cwd,env=env,text=True,encoding='utf8',errors='replace',stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if p.returncode:
        print(p.stdout[-9000:]);raise RuntimeError('Build failed: '+str(args))
    return p.stdout


def main():
    BUILD.mkdir(parents=True,exist_ok=True);OUTPUT.mkdir(parents=True,exist_ok=True)
    results={}
    for source,name in [('main.tex','lora_hstem_manuscript'),('supplement.tex','lora_hstem_supplement')]:
        command=['pdflatex','-interaction=nonstopmode','-halt-on-error','-file-line-error','-output-directory='+str(BUILD),'-jobname='+name,source]
        run(command,HERE)
        if source=='main.tex':
            env=os.environ.copy();env['BIBINPUTS']=str(HERE)+os.pathsep
            run(['bibtex',name],BUILD,env)
        run(command,HERE);run(command,HERE)
        log=(BUILD/(name+'.log')).read_text(encoding='utf8',errors='replace')
        warnings=re.findall(r'Overfull \\hbox[^\n]*|LaTeX Warning: [^\n]*|Package [^\n]*Warning[^\n]*',log)
        if re.search(r'undefined|multiply defined',log,re.I):raise ValueError('Unresolved references in '+name)
        pdf=BUILD/(name+'.pdf')
        run(['pdftoppm','-r','105','-png',str(pdf),str(BUILD/name)],HERE)
        shutil.copyfile(pdf,OUTPUT/pdf.name)
        results[name]=dict(pdf=str(OUTPUT/pdf.name),info=run(['pdfinfo',str(pdf)],HERE),warnings=warnings,
            render_pages=[str(p) for p in sorted(BUILD.glob(name+'-*.png'))])
        print(name,len(results[name]['render_pages']),'pages',warnings,flush=True)
    (HERE/'BUILD_REPORT.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
