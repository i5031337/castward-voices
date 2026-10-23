param([int]$Port = 8001)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$native = Join-Path $projectRoot '.cache\audio.cpp-v0.9.2\runtime\audiocpp_server.exe'
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$nativeConfig = Join-Path $projectRoot 'audiocpp-server.firered.json'
$speechConfig = Join-Path $projectRoot 'backend-firered.json'
$nativeLog = Join-Path $projectRoot '.cache\firered-native.log'
$nativeErrorLog = Join-Path $projectRoot '.cache\firered-native-error.log'
$speechLog = Join-Path $projectRoot '.cache\firered-api.log'
$speechErrorLog = Join-Path $projectRoot '.cache\firered-api-error.log'

if (-not (Test-Path -LiteralPath $native)) { throw "audio.cpp is missing: $native" }
if (-not (Test-Path -LiteralPath $python)) { throw "Python environment is missing: $python" }
if ($Port -lt 1 -or $Port -gt 65535 -or $Port -eq 7862) { throw 'Choose an API port from 1 to 65535 other than 7862.' }

function Assert-PortFree([int]$number) {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $number)
    try { $listener.Start() }
    catch { throw "Port $number is already occupied." }
    finally { $listener.Stop() }
}

Assert-PortFree 7862
Assert-PortFree $Port
$nativeProcess = $null
$speechProcess = $null
try {
    $nativeProcess = Start-Process -FilePath $native -ArgumentList @('--config', ('"' + $nativeConfig + '"'), '--no-ui') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $nativeLog -RedirectStandardError $nativeErrorLog
    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if ($nativeProcess.HasExited) { throw "audio.cpp exited. See $nativeErrorLog" }
        try {
            $health = Invoke-RestMethod -Uri 'http://127.0.0.1:7862/health' -TimeoutSec 1
            if ($health.status -eq 'ok') { $ready = $true; break }
        } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $ready) { throw "audio.cpp did not become ready. See $nativeErrorLog" }

    $speechProcess = Start-Process -FilePath $python -ArgumentList @('server.py', '--config', ('"' + $speechConfig + '"'), '--port', [string]$Port) -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $speechLog -RedirectStandardError $speechErrorLog
    Write-Host "FireRedTTS3 Voice Studio: http://127.0.0.1:$Port/"
    Write-Host 'Press Ctrl+C to stop both processes.'
    while (-not $nativeProcess.HasExited -and -not $speechProcess.HasExited) { Start-Sleep -Seconds 1 }
    if ($nativeProcess.HasExited) { throw "audio.cpp exited. See $nativeErrorLog" }
    throw "Speech API exited. See $speechErrorLog"
}
finally {
    foreach ($process in @($speechProcess, $nativeProcess)) {
        if ($null -ne $process -and -not $process.HasExited) {
            Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
            $process.WaitForExit()
        }
    }
}
