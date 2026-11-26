param(
    [int]$TopK = 5,
    [int]$Workers = 2,
    [int]$QaBatchSize = 8,
    [ValidateSet('all', 'construction', 'search', 'evaluation', 'search-and-evaluation')][string]$Stage = 'all'
)

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath "$PSScriptRoot/..").Path
$python = (Resolve-Path -LiteralPath "$workspace/pocketmemory/Scripts/python.exe").Path
$out = Join-Path $workspace 'outputs/mobilemem-qwen/hgp-original-nojudge'
New-Item -ItemType Directory -Path $out -Force | Out-Null
$configPath = Join-Path $out 'live-config.json'
$config = Get-Content -LiteralPath "$workspace/config/templates/hgp_neo4j.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$config.allow_llm_fallback = $true
$config.allow_llm_extraction = $true
$config.save_dir = $out
$config.graph_namespace = 'mobilemem-hgp3-original-nojudge'
[IO.File]::WriteAllText($configPath, ($config | ConvertTo-Json -Depth 8))
$env:HGP_DEEPSEEK_USAGE_LOG = Join-Path $out 'openai-sdk-usage-per-type.jsonl'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'

$source = Join-Path $workspace 'data/text/mobilemem_data.json'
$runner = Join-Path $PSScriptRoot 'run-mobilemem-comparison.ps1'
$argsForRun = @{
    Method = 'hgp'
    Python = $python
    DatasetPath = $source
    ConfigPath = $configPath
    UsageLogPath = (Join-Path $out 'deepseek-qa-per-type.jsonl')
    TopK = $TopK
    Workers = $Workers
    QaBatchSize = $QaBatchSize
    SkipLlmJudge = $true
}
$stages = switch ($Stage) {
    'all' { 'construction', 'search', 'evaluation' }
    'search-and-evaluation' { 'search', 'evaluation' }
    default { $Stage }
}
foreach ($runStage in $stages) {
    Write-Output "=== HGP $runStage ==="
    & $runner @argsForRun -Stage $runStage
}
