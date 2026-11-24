"""Compile the independently uploadable IEEE LaTeX files and render every page."""

import re
import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent


def build(stem, filename):
    temporary = HERE / "tmp/pdfs" / stem
    temporary.mkdir(parents=True, exist_ok=True)
    command = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
               "-file-line-error", f"-outdir={temporary.as_posix()}", f"{stem}.tex"]
    result = subprocess.run(command, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    transcript = temporary / "build_stdout.txt"
    transcript.write_bytes(result.stdout)
    if result.returncode:
        raise SystemExit(f"LaTeX failed: {transcript}")
    pdf = temporary / f"{stem}.pdf"
    log = (temporary / f"{stem}.log").read_text(encoding="utf-8", errors="replace")
    if re.search(r"undefined (?:references|citations)|(?:Reference|Citation).*undefined", log):
        raise SystemExit(f"Unresolved LaTeX reference: {stem}")
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    count = int(re.search(r"(?m)^Pages:\s+(\d+)", info).group(1))
    if stem == "main" and count > 13:
        raise SystemExit(f"Manuscript has {count} pages; the first-submission limit is 13")
    output = HERE / "output/pdf" / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdf, output)
    render = temporary / "render"
    render.mkdir(exist_ok=True)
    subprocess.run(["pdftoppm", "-png", "-r", "115", str(output), str(render / "page")],
                   check=True)
    print(f"{output.name}: {count} pages", flush=True)


if __name__ == "__main__":
    build("main", "hstem_tifs_manuscript.pdf")
    build("supplement", "hstem_tifs_supplement.pdf")
