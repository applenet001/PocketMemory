$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot 'pocketmemory/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { $pythonPath = 'python' }
& $pythonPath (Join-Path $PSScriptRoot 'server.py') --host 0.0.0.0 @args
if ($LASTEXITCODE -ne 0) { throw "PocketMemory exited with code $LASTEXITCODE" }
