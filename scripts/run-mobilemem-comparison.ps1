param(
    [ValidateSet('raw_buffer','vector_rag','mem0','mem0g','hgp')][string]$Method='raw_buffer',
    [ValidateSet('check','construction','search','evaluation')][string]$Stage='check',
    [string]$Python='python',
    [string]$DatasetPath="$PSScriptRoot/../data/text/mobilemem_data.json",
    [string]$ConfigPath='',
    [string]$ApiConfigPath='',
    [string]$UsageLogPath='',
    [string]$QaModel='deepseek-flash',
    [string]$JudgeModel='deepseek-flash',
    [int]$TopK=30,
    [int]$Workers=1,
    [switch]$SkipLlmJudge,
    [int]$QaBatchSize=1
)
$ErrorActionPreference='Stop'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
$env:NO_PROXY='localhost,127.0.0.1,::1'
$env:no_proxy=$env:NO_PROXY
$workspace=Split-Path $PSScriptRoot -Parent
$env:MEMORY_METHOD=$Method
$env:DEEPSEEK_USAGE_LOG = if ($UsageLogPath) { $UsageLogPath } else {
    Join-Path $workspace "outputs/mobilemem-qwen/smoke20/${Method}_deepseek_usage.jsonl"
}
$localEnv=Join-Path $workspace 'config/local.env'
if (Test-Path -LiteralPath $localEnv) {
    foreach ($line in Get-Content -LiteralPath $localEnv -Encoding UTF8) {
        if ($line -match '^\s*([A-Z][A-Z0-9_]*)\s*=(.*)$') {
            $envName=$Matches[1]
            $envValue=$Matches[2].Trim().Trim('"').Trim("'")
            if ($envName -in @('DEEPSEEK_API_KEY','DEEPSEEK_BASE_URL','DEEPSEEK_MODEL','HGP_EMBEDDING_API_KEY') -and $envValue) {
                [Environment]::SetEnvironmentVariable($envName,$envValue,'Process')
            }
        }
    }
}
$repo=Join-Path $workspace 'MemBase'
$example=Join-Path $repo 'examples/evaluate_memory_systems_on_mobilemem'
$mapping=@{
    raw_buffer=@('Long-Context','long_context_config.json')
    vector_rag=@('NaiveRAG','rag_config.json')
    mem0=@('Mem0','mem0_config.json')
    mem0g=@('Mem0','mem0_graph_config.json')
    hgp=@('HGP','hgp_neo4j.json')
}
if ($Workers -lt 1 -or $TopK -lt 1 -or $QaBatchSize -lt 1) { throw 'Workers, TopK and QaBatchSize must be positive.' }
if ($Method -eq 'raw_buffer') { $TopK=1 }
if (!$ConfigPath) {
    if ($Method -in @('raw_buffer','vector_rag','mem0')) {
        $ConfigPath=Join-Path "$workspace/config/mobilemem-qwen" $mapping[$Method][1]
    } elseif ($Method -eq 'hgp') {
        $ConfigPath=Join-Path "$workspace/config/templates" $mapping[$Method][1]
    } else { $ConfigPath=Join-Path "$example/configs" $mapping[$Method][1] }
}
if (!$ApiConfigPath) { $ApiConfigPath=Join-Path $workspace 'outputs/private/deepseek-api.json' }
if ($env:DEEPSEEK_API_KEY) {
    $env:OPENAI_API_KEY=$env:DEEPSEEK_API_KEY
    New-Item -ItemType Directory -Force "$workspace/outputs/private" | Out-Null
    $apiJson=@{api_keys=@($env:DEEPSEEK_API_KEY);base_urls=@('https://api.deepseek.com/v1')} | ConvertTo-Json
    [IO.File]::WriteAllText("$workspace/outputs/private/deepseek-api.json",$apiJson)
}
$problems=@()
if (!(Test-Path -LiteralPath $DatasetPath)) { $problems+='Dataset missing.' }
if (!(Test-Path -LiteralPath $ConfigPath)) { $problems+='Method configuration missing.' }
if ($Method -eq 'hgp') {
    foreach($relative in @('scripts/run_hgp_membase.py','outputs/hgp3-train-2200/classifier/head.pt','config/local.env')) {
        if (!(Test-Path -LiteralPath "$workspace/$relative")) { $problems+="Missing HGP input: $relative" }
    }
}
$config=$null
if (Test-Path -LiteralPath $ConfigPath) {
    $config=Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($Method -eq 'mem0g' -and (!$config.graph_store_config -or !$config.graph_store_provider)) {
        $problems+='Graph configuration requires graph_store_config and graph_store_provider.'
    }
}
Write-Output "Method: $Method -> $($mapping[$Method][0]); retrieval top-k: $TopK"
Write-Output "Config: $ConfigPath"
if ($Stage -eq 'check') {
    if ($problems.Count) { $problems | ForEach-Object { Write-Output "BLOCKED: $_" } }
    else { Write-Output 'Static configuration found. Runtime, services, credentials, temporal protocol and shared QA context budget still require validation.' }
    return
}
if ($problems.Count) { throw ($problems -join [Environment]::NewLine) }
$ConfigPath=(Resolve-Path -LiteralPath $ConfigPath).Path
$DatasetPath=(Resolve-Path -LiteralPath $DatasetPath).Path
if ([IO.Path]::IsPathRooted($config.save_dir)) { $save=$config.save_dir }
else { $save=Join-Path $repo $config.save_dir }
$stage1=Join-Path $save 'MobileMem_stage_1.json'
$argsList=@()
switch($Stage) {
    construction {
        $rawData=Get-Content -LiteralPath $DatasetPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $count=$rawData.Count
        $argsList=@('memory_construction.py','--memory-type',$mapping[$Method][0],'--dataset-type','MobileMem','--dataset-path',$DatasetPath,'--config-path',$ConfigPath,'--num-workers',"$Workers",'--sample-size',"$count",'--seed','42','--token-cost-save-filename',"$save/token_cost_$Method")
    }
    default {
        if (!(Test-Path -LiteralPath $stage1)) { throw 'Run construction first.' }
        $count=@((Get-Content -LiteralPath $stage1 -Raw -Encoding UTF8 | ConvertFrom-Json).trajectories).Count
        if ($Stage -eq 'search') {
            $argsList=@('memory_search.py','--memory-type',$mapping[$Method][0],'--dataset-type','MobileMem','--dataset-path',$stage1,'--dataset-standardized','--config-path',$ConfigPath,'--num-workers',"$Workers",'--top-k',"$TopK",'--start-idx','0','--end-idx',"$count")
        } else {
            $results=Join-Path $save "${TopK}_0_${count}.json"
            if (!(Test-Path -LiteralPath $results)) { throw 'Run search first with matching top-k.' }
            $ApiConfigPath=(Resolve-Path -LiteralPath $ApiConfigPath).Path
            $argsList=@('memory_evaluation.py','--search-results-path',$results,'--dataset-type','MobileMem','--qa-model',$QaModel,'--judge-model',$JudgeModel,'--qa-batch-size',"$QaBatchSize",'--judge-batch-size','1','--api-config-path',$ApiConfigPath)
            if ($SkipLlmJudge) { $argsList += @('--metrics','f1','bleu') }
        }
    }
}
Push-Location $repo
try {
    if ($Method -eq 'hgp') {
        if ($argsList[1] -eq '--memory-type') { $argsList=@($argsList | Select-Object -Skip 3) }
        else { $argsList=@($argsList | Select-Object -Skip 1) }
        & $Python "$workspace/scripts/run_hgp_membase.py" $Stage @argsList
    } else {
        & $Python @argsList
    }
    if ($LASTEXITCODE -ne 0) { throw "MemBase $Stage failed (exit $LASTEXITCODE)." }
} finally { Pop-Location }

