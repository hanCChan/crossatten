param([switch]$RegenerateFigures, [switch]$RenderPages)
$ErrorActionPreference = 'Stop'
$s25Root = $PSScriptRoot
$s25Python = 'D:/crossatten/.venv/Scripts/python.exe'
$s25Latex = 'D:/bianyiqi/texlive/2026/bin/windows/latexmk.exe'
$s25Raster = 'D:/bianyiqi/texlive/2026/bin/windows/pdftoppm.exe'
$s25Build = Join-Path $s25Root 'tmp/pdfs'
$s25Output = Join-Path $s25Root 'output/pdf'
New-Item -ItemType Directory -Force -Path $s25Build, $s25Output | Out-Null
if ($RegenerateFigures) {
    & $s25Python (Join-Path $s25Root 'paper_assets.py')
    if ($LASTEXITCODE -ne 0) { throw 'Figure generation failed' }
}
# Capture native stderr inside Python so harmless TeX/Perl notices cannot become
# terminating PowerShell NativeCommandError records. The process exit code rules.
$s25Compile = @'
import pathlib, subprocess, sys
exe, root, build = sys.argv[1:]
result = subprocess.run([exe, '-pdf', '-interaction=nonstopmode', '-halt-on-error',
                         '-outdir='+build, 'main.tex'], cwd=str(pathlib.Path(root)/'paper'),
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
pathlib.Path(build, 'build_stdout.txt').write_bytes(result.stdout)
print('latexmk exit:', result.returncode)
sys.exit(result.returncode)
'@
& $s25Python -c $s25Compile $s25Latex $s25Root $s25Build
if ($LASTEXITCODE -ne 0) { throw 'Manuscript compilation failed; see tmp/pdfs/build_stdout.txt' }
Copy-Item -LiteralPath (Join-Path $s25Build 'main.pdf') -Destination (Join-Path $s25Output 'hstem_tifs_draft.pdf') -Force
if ($RenderPages) {
    $s25Render = Join-Path $s25Build 'render'
    New-Item -ItemType Directory -Force -Path $s25Render | Out-Null
    & $s25Raster -png -r 115 (Join-Path $s25Output 'hstem_tifs_draft.pdf') (Join-Path $s25Render 'page')
    if ($LASTEXITCODE -ne 0) { throw 'Page rendering failed' }
}
Write-Output (Join-Path $s25Output 'hstem_tifs_draft.pdf')
