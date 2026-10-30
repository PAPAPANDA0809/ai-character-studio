$ErrorActionPreference = 'Stop'
$studioRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $studioRoot 'ComfyUI_windows_portable\python_embeded\pythonw.exe'
$runner = Join-Path $PSScriptRoot 'reelmint_supervisor.py'
if (!(Test-Path -LiteralPath $python) -or !(Test-Path -LiteralPath $runner)) {
    throw 'Reelmint background launcher or embedded Python is missing.'
}
$taskName = 'Reelmint Background Server'
$account = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute $python -Argument "-u `"$runner`"" -WorkingDirectory (Join-Path $studioRoot 'app')
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $account
$principal = New-ScheduledTaskPrincipal -UserId $account -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $taskName -Description 'Keeps the Reelmint web server running independently of ChatGPT; starts at Windows sign-in and restarts the web server if it exits.' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-Output 'Reelmint background startup enabled.'
