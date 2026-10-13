# Started by the "Sepalith\NightCheck" task at 09:05. Logs whether anything
# from the night window is still running. It never stops anything.
param(
    [Parameter(Mandatory = $true)][string]$Distro,
    [Parameter(Mandatory = $true)][string]$LinuxUser,
    [Parameter(Mandatory = $true)][string]$Repo
)
$log = Join-Path $PSScriptRoot 'night.log'
Add-Type -Namespace Sepalith -Name Power -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
# The task may have woken the PC: ES_CONTINUOUS | ES_SYSTEM_REQUIRED until the check is logged.
[void][Sepalith.Power]::SetThreadExecutionState([uint32]'0x80000001')
$output = & wsl.exe -d $Distro -u $LinuxUser -- "$Repo/scripts/night/night-runner.sh" check 2>&1
$state = if ($LASTEXITCODE -eq 0) { 'clean' } else { "NOT CLEAN (exit code $LASTEXITCODE)" }
Add-Content -Path $log -Value ("{0} check {1}: {2}" -f (Get-Date).ToString('o'), $state, ($output -join ' '))
[void][Sepalith.Power]::SetThreadExecutionState([uint32]'0x80000000')
