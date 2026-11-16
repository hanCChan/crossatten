$ErrorActionPreference = 'Stop'
$s24Folder = Split-Path -Parent $MyInvocation.MyCommand.Path
$s24Worktree = Split-Path -Parent (Split-Path -Parent $s24Folder)
$s24Python = 'D:/crossatten/.venv/Scripts/python.exe'
if (Test-Path -LiteralPath (Join-Path $s24Folder 'supervisor.lock')) {
    throw 'A supervisor lock exists. Check the recorded PID before restarting.'
}
$s24Process = Start-Process -FilePath $s24Python -ArgumentList @('-u', (Join-Path $s24Folder 'supervisor.py')) -WorkingDirectory $s24Worktree -WindowStyle Hidden -RedirectStandardOutput (Join-Path $s24Folder 'supervisor.stdout.log') -RedirectStandardError (Join-Path $s24Folder 'supervisor.stderr.log') -PassThru
$s24Process.Id | Set-Content -LiteralPath (Join-Path $s24Folder 'supervisor.pid') -Encoding ascii
Write-Output ('Started S24 supervisor PID ' + $s24Process.Id)
