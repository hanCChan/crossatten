$ErrorActionPreference = 'Stop'
$s24RepairFolder = Split-Path -Parent $MyInvocation.MyCommand.Path
$s24PhaseFolder = Split-Path -Parent $s24RepairFolder
$s24RepairWorktree = Split-Path -Parent (Split-Path -Parent $s24PhaseFolder)
$s24RepairLock = Join-Path $s24PhaseFolder 'supervisor.lock'
if (Test-Path -LiteralPath $s24RepairLock) {
    $s24RepairPid = [int](Get-Content -Raw -LiteralPath $s24RepairLock)
    if (Get-Process -Id $s24RepairPid -ErrorAction SilentlyContinue) {
        throw 'A live controller owns the lock. Do not start another queue.'
    }
}
$s24RepairAttempt = 1 + @(Get-ChildItem -LiteralPath $s24RepairFolder -Filter 'controller_stdout_attempt*.log').Count
$s24RepairTarget = Join-Path $s24RepairFolder 'technical_002.py'
$s24RepairProcess = Start-Process -FilePath 'D:/crossatten/.venv/Scripts/python.exe' -ArgumentList @('-u', $s24RepairTarget, '--resume') -WorkingDirectory $s24RepairWorktree -WindowStyle Hidden -RedirectStandardOutput (Join-Path $s24RepairFolder ('controller_stdout_attempt' + $s24RepairAttempt + '.log')) -RedirectStandardError (Join-Path $s24RepairFolder ('controller_stderr_attempt' + $s24RepairAttempt + '.log')) -PassThru
$s24RepairProcess.Id | Set-Content -LiteralPath (Join-Path $s24PhaseFolder 'supervisor.pid') -Encoding ascii
Write-Output ('Resuming S24 with NUL path manifest repair; launcher PID ' + $s24RepairProcess.Id)
