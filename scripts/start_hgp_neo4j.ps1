param(
    [string]$InstallDir = "$PSScriptRoot/../data/storage/neo4j-local/neo4j-community-2026.05.0",
    [string]$JavaHome = $env:JAVA_HOME,
    [string]$EnvFile = "$PSScriptRoot/../config/local.env"
)

$ErrorActionPreference = 'Stop'
$values = @{}
foreach ($line in Get-Content -LiteralPath $EnvFile -Encoding UTF8) {
    if ($line -match '^\s*(NEO4J_URI|NEO4J_DATABASE)=(.*)$') {
        $values[$Matches[1]] = $Matches[2].Trim().Trim('"').Trim("'")
    }
}
$uri = [Uri]$values['NEO4J_URI']
if ($uri.Host -notin @('localhost','127.0.0.1') -or $uri.Scheme -notin @('bolt','neo4j')) {
    throw 'The startup helper only supports a loopback bolt:// or neo4j:// URI.'
}
$bolt = if ($uri.Port -gt 0) { $uri.Port } else { 7687 }
$database = $values['NEO4J_DATABASE']
if ($database -notmatch '^[a-z][a-z0-9.-]{2,62}$' -or $database -eq 'system') {
    throw 'Set a valid local database name in NEO4J_DATABASE.'
}

function Test-BoltPort {
    $client = [Net.Sockets.TcpClient]::new()
    try { return $client.ConnectAsync('127.0.0.1', $bolt).Wait(800) -and $client.Connected }
    catch { return $false }
    finally { $client.Dispose() }
}

if (Test-BoltPort) {
    Write-Output 'A local service is already listening on the configured Bolt port.'
    return
}
$install = (Resolve-Path -LiteralPath $InstallDir).Path
if (!$JavaHome -or !(Test-Path -LiteralPath (Join-Path $JavaHome 'bin/java.exe'))) {
    $desktopCache = Join-Path ([Environment]::GetFolderPath('UserProfile')) '.Neo4jDesktop2/Cache/runtime'
    $java = Get-ChildItem -LiteralPath $desktopCache -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like '*jre21*' } | Sort-Object Name -Descending | Select-Object -First 1
    if (!$java) { throw 'Set -JavaHome to your Java 21 installation.' }
    $JavaHome = $java.FullName
}
$conf = Join-Path $install 'conf'
$startup = Join-Path $install 'bin/neo4j.ps1'
if (!(Test-Path -LiteralPath $startup)) { throw 'Neo4j Community installation not found.' }
$runtimeConf = Join-Path $install 'hgp-conf'
if (!(Test-Path -LiteralPath $runtimeConf)) {
    New-Item -ItemType Directory -Path $runtimeConf | Out-Null
    foreach ($name in 'neo4j-admin.conf','server-logs.xml','user-logs.xml') {
        Copy-Item -LiteralPath (Join-Path $conf $name) -Destination $runtimeConf
    }
}
$config = @(
    'server.default_listen_address=127.0.0.1',
    "server.bolt.listen_address=127.0.0.1:$bolt",
    'server.http.listen_address=127.0.0.1:7474',
    "initial.dbms.default_database=$database",
    'server.memory.heap.initial_size=256m',
    'server.memory.heap.max_size=768m',
    'server.memory.pagecache.size=256m',
    'dbms.usage_report.enabled=false'
)
[IO.File]::WriteAllLines((Join-Path $runtimeConf 'neo4j.conf'), $config, [Text.Encoding]::ASCII)
$env:JAVA_HOME = $JavaHome
$env:NEO4J_HOME = $install
$env:NEO4J_CONF = $runtimeConf
$parameters = @{
    FilePath = (Get-Command powershell.exe).Source
    ArgumentList = @('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',('"' + $startup + '"'),'console')
    WorkingDirectory = $install
    WindowStyle = 'Hidden'
    PassThru = $true
    RedirectStandardOutput = (Join-Path $install 'hgp-server.stdout.log')
    RedirectStandardError = (Join-Path $install 'hgp-server.stderr.log')
}
$server = Start-Process @parameters
for ($i=0; $i -lt 25; $i++) {
    if (Test-BoltPort) { Write-Output "Local Neo4j Bolt ready (launcher PID $($server.Id))."; return }
    $server.Refresh()
    if ($server.HasExited) { throw 'Neo4j exited during startup. See hgp-server.stderr.log.' }
    Start-Sleep -Seconds 1
}
throw 'Neo4j Bolt did not open in time. See hgp-server.stdout.log and hgp-server.stderr.log.'
