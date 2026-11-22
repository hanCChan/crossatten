"""Compile/render current evidence documents; no training or data IO."""
import argparse
from pathlib import Path
import shutil
import subprocess
from pypdf import PdfReader
from s26_common import HERE,write,now,sha

LATEX='D:/bianyiqi/texlive/2026/bin/windows/latexmk.exe'
RASTER='D:/bianyiqi/texlive/2026/bin/windows/pdftoppm.exe'


def build(source,output_name,folder):
    temporary=HERE/'tmp/pdfs'/folder;temporary.mkdir(parents=True,exist_ok=True)
    command=[LATEX,'-pdf','-interaction=nonstopmode','-halt-on-error','-outdir='+temporary.as_posix(),source.name]
    r=subprocess.run(command,cwd=source.parent,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (temporary/'build_stdout.txt').write_bytes(r.stdout)
    if r.returncode:raise RuntimeError(f'LaTeX build failed: {temporary}/build_stdout.txt')
    raw=temporary/source.with_suffix('.pdf').name
    output=HERE/'output/pdf'/output_name;output.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(raw,output)
    render=temporary/'render';render.mkdir(exist_ok=True)
    subprocess.run([RASTER,'-png','-r','115',str(output),str(render/'page')],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    info=dict(path=str(output),sha256=sha(output),pages=len(PdfReader(output).pages),render=str(render),log=str(temporary/source.with_suffix('.log').name))
    print(info,flush=True);return info


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');a=parser.parse_args()
    if a.preview:
        from paper_revision import MATHEMATICS
        folder=HERE/'tmp/pdfs/math_preview';folder.mkdir(parents=True,exist_ok=True)
        source=folder/'method_preview.tex'
        source.write_text(r'\documentclass[journal]{IEEEtran}\usepackage{amsmath,amssymb,booktabs,hyperref}\begin{document}\title{Architectural Accounting: Internal Layout Preview}\author{Working note}\maketitle\section{Method definitions}'+MATHEMATICS+r'\end{document}',encoding='utf-8',newline='\n')
        r=subprocess.run([LATEX,'-pdf','-interaction=nonstopmode','-halt-on-error',source.name],cwd=folder,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (folder/'build_stdout.txt').write_bytes(r.stdout)
        assert r.returncode==0
        subprocess.run([RASTER,'-png','-r','115',str(source.with_suffix('.pdf')),str(folder/'page')],check=True)
        print('Mathematical layout preview ready; no experiment scores generated.',flush=True)
        return
    assert (HERE/'MANUSCRIPT_PROVENANCE.json').exists()
    rows=[build(HERE/'paper/main.tex','hstem_tifs_revised.pdf','main'),
          build(HERE/'paper/supplement.tex','hstem_tifs_supplement.pdf','supplement')]
    write(HERE/'DOCUMENT_BUILD.json',dict(status='COMPILED_AND_RENDERED',time=now(),documents=rows,visual_review='PENDING'))


if __name__=='__main__':main()
