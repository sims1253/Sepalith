# Started by the "Sepalith\NightCheck" task at 09:05. Logs whether anything
# from the night window is still running. It never stops anything.
param(
    [Parameter(Mandatory = $true)][string]$Distro,
    [Parameter(Mandatory = $true)][string]$LinuxUser,
    [Parameter(Mandatory = $true)][string]$Repo
)
$log = Join-Path $PSScriptRoot 'night.log'
$output = & wsl.exe -d $Distro -u $LinuxUser -- "$Repo/scripts/night/night-runner.sh" check 2>&1
$state = if ($LASTEXITCODE -eq 0) { 'clean' } else { "NOT CLEAN (exit code $LASTEXITCODE)" }
Add-Content -Path $log -Value ("{0} check {1}: {2}" -f (Get-Date).ToString('o'), $state, ($output -join ' '))
