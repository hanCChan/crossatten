$ErrorActionPreference = 'Stop'
$s24Folder = Split-Path -Parent $MyInvocation.MyCommand.Path
$s24Worktree = Split-Path -Parent (Split-Path -Parent $s24Folder)
$s24Python = 'D:/crossatten/.venv/Scripts/python.exe'
$s24Lock = Join-Path $s24Folder 'supervisor.lock'
if (Test-Path -LiteralPath $s24Lock) {
    $s24LockPid = [int](Get-Content -Raw -LiteralPath $s24Lock)
    if (Get-Process -Id $s24LockPid -ErrorAction SilentlyContinue) {
        throw 'A live supervisor owns the lock. Do not start a second GPU queue.'
    }
}
$s24Budget = Join-Path $s24Folder 'budget_update'
if (Test-Path -LiteralPath (Join-Path $s24Budget 'AMENDMENT.json')) {
    $s24Target = Join-Path $s24Budget 'controller.py'
    $s24LogFolder = $s24Budget
} else {
    $s24Target = Join-Path $s24Folder 'supervisor.py'
    $s24LogFolder = $s24Folder
}
$s24Attempt = 1 + @(Get-ChildItem -LiteralPath $s24LogFolder -Filter 'controller_stdout_attempt*.log').Count
$s24Process = Start-Process -FilePath $s24Python -ArgumentList @('-u', $s24Target) -WorkingDirectory $s24Worktree -WindowStyle Hidden -RedirectStandardOutput (Join-Path $s24LogFolder ('controller_stdout_attempt' + $s24Attempt + '.log')) -RedirectStandardError (Join-Path $s24LogFolder ('controller_stderr_attempt' + $s24Attempt + '.log')) -PassThru
$s24Process.Id | Set-Content -LiteralPath (Join-Path $s24Folder 'supervisor.pid') -Encoding ascii
Write-Output ('Started S24 supervisor PID ' + $s24Process.Id)
