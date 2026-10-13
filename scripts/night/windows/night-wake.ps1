# Started by the "Sepalith\NightWake" task at 00:55. Wakes WSL, keeps the WSL
# session and the PC awake until the night runner is finished (at the latest
# 09:05), then lets Windows sleep again.
param(
    [Parameter(Mandatory = $true)][string]$Distro,
    [Parameter(Mandatory = $true)][string]$LinuxUser,
    [Parameter(Mandatory = $true)][string]$Repo
)
$ErrorActionPreference = 'Stop'
$log = Join-Path $PSScriptRoot 'night.log'
function Write-Log([string]$message) {
    Add-Content -Path $log -Value ("{0} wake {1}" -f (Get-Date).ToString('o'), $message)
}
Add-Type -Namespace Sepalith -Name Power -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no idle sleep while this script runs.
[void][Sepalith.Power]::SetThreadExecutionState([uint32]'0x80000001')
try {
    Write-Log "starting $Distro as $LinuxUser"
    $output = & wsl.exe -d $Distro -u $LinuxUser -- "$Repo/scripts/night/night-runner.sh" hold 2>&1
    Write-Log ("hold ended with exit code {0}: {1}" -f $LASTEXITCODE, ($output -join ' '))
} catch {
    Write-Log ("failed: {0}" -f $_)
    throw
} finally {
    [void][Sepalith.Power]::SetThreadExecutionState([uint32]'0x80000000')
}
