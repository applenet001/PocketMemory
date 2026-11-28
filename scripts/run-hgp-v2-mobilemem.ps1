param(
    [Parameter(Mandatory=$true)][string]$RunRoot
)

$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$workspace = Split-Path $PSScriptRoot -Parent
$python = Join-Path $workspace '.venv/Scripts/python.exe'
$runner = Join-Path $PSScriptRoot 'run_hgp_v2_membase.py'
$runPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($RunRoot)
New-Item -ItemType Directory -Path $runPath -Force | Out-Null
$scheduleLog = Join-Path $runPath 'run.log'
$eligible = @{ 1 = $true; 2 = $true }

foreach ($stage in @('construct', 'search', 'eval')) {
    foreach ($user in @(1, 2)) {
        if (!$eligible[$user]) { continue }
        $logPath = Join-Path $runPath ('user-{0:d2}-{1}.log' -f $user, $stage)
        Add-Content -LiteralPath $scheduleLog -Encoding UTF8 "$(Get-Date -Format o) user-$user $stage started"
        $arguments = @($runner, $stage, '--user', "$user", '--run-root', $runPath, '--top-k', '5')
        $previousPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            & $python @arguments *>> $logPath
            $exitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $previousPreference
        }
        Add-Content -LiteralPath $scheduleLog -Encoding UTF8 "$(Get-Date -Format o) user-$user $stage exit=$exitCode"
        if ($exitCode -ne 0) { $eligible[$user] = $false }
    }
}

if (!$eligible[1] -or !$eligible[2]) { exit 1 }
