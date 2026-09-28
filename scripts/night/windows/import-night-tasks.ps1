# Import the two Sepalith night-window tasks. Run once, as administrator:
#   powershell -ExecutionPolicy Bypass -File \\wsl.localhost\Ubuntu-22.04\<repo>\scripts\night\windows\import-night-tasks.ps1
# Optional: -EnableWakeTimers sets "Allow wake timers" to Enable on AC power.
param(
    [string]$Distro = 'Ubuntu-22.04',
    [string]$LinuxUser = 'm0hawk',
    [string]$Repo = '/home/m0hawk/Documents/Sepalith',
    [switch]$EnableWakeTimers
)
$ErrorActionPreference = 'Stop'
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this script from an administrator PowerShell.'
}
$target = Join-Path $env:LOCALAPPDATA 'Sepalith\night'
New-Item -ItemType Directory -Force -Path $target | Out-Null
Copy-Item -Force -Path (Join-Path $PSScriptRoot 'night-wake.ps1'), (Join-Path $PSScriptRoot 'night-check.ps1') -Destination $target
$user = "$env:USERDOMAIN\$env:USERNAME"
foreach ($name in 'SepalithNightWake', 'SepalithNightCheck') {
    $xml = (Get-Content -Raw -Path (Join-Path $PSScriptRoot "$name.xml")).
        Replace('__USERID__', [Security.SecurityElement]::Escape($user)).
        Replace('__SCRIPTDIR__', [Security.SecurityElement]::Escape($target)).
        Replace('__DISTRO__', $Distro).Replace('__LINUXUSER__', $LinuxUser).Replace('__REPO__', $Repo)
    $task = $name -replace '^Sepalith', ''
    Register-ScheduledTask -TaskPath '\Sepalith\' -TaskName $task -Xml $xml -Force | Out-Null
    Write-Host "registered \Sepalith\$task for $user"
}
if ($EnableWakeTimers) {
    powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP RTCWAKE 1
    powercfg /setactive SCHEME_CURRENT
    Write-Host 'Allow wake timers: Enable (AC, current power plan)'
}
Write-Host ''
Write-Host 'Current "Allow wake timers" setting (AC index 1 = Enable):'
powercfg /query SCHEME_CURRENT SUB_SLEEP RTCWAKE | Select-String 'Current AC Power Setting Index'
Write-Host 'Test now with: Start-ScheduledTask -TaskPath \Sepalith\ -TaskName NightCheck; then read' (Join-Path $target 'night.log')
