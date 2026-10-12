# HORIZON 2K26 - D07 Token-Aware Inference Scheduler
# Starts Model A, Model B, Model C, Ollama Qwen3, and Gateway.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Resolve project paths
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $scriptDir '..')).Path
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$pidFile = Join-Path $projectRoot 'run_pids.txt'

if (-not (Test-Path $venvPython)) {
    throw "Python environment not found: $venvPython"
}

# Start backends before the gateway
$services = @(
    @{ Name = 'Model A';       Port = 8001; Module = 'app.backends.model_a_main' }
    @{ Name = 'Model B';       Port = 8002; Module = 'app.backends.model_b_main' }
    @{ Name = 'Model C';       Port = 8003; Module = 'app.backends.model_c_main' }
    @{ Name = 'Ollama Qwen3';  Port = 8004; Module = 'app.backends.model_ollama' }
    @{ Name = 'Gateway';       Port = 9000; Module = 'app.gateway.main' }
)

$launched = @()

function Wait-Health {
    param(
        [string]$Name,
        [int]$Port,
        [int]$TimeoutSec = 60
    )

    $url = "http://127.0.0.1:$Port/health"
    $deadline = (Get-Date).AddSeconds($TimeoutSec)

    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-RestMethod `
                -Uri $url `
                -Method Get `
                -TimeoutSec 3

            if ($response.status -eq 'ok') {
                Write-Host "$Name is healthy on port $Port" -ForegroundColor Green
                return $true
            }
        }
        catch {
            # Retry while the service starts.
        }

        Start-Sleep -Seconds 2
    }

    Write-Host "$Name failed health check on port $Port" -ForegroundColor Red
    return $false
}

try {
    # Verify Ollama is running before starting its backend.
    try {
        $ollamaTags = $null
        try {
            $ollamaTags = Invoke-RestMethod `
                -Uri 'http://127.0.0.1:11434/api/tags' `
                -Method Get `
                -TimeoutSec 3
        }
        catch {
            Write-Host 'Ollama service is not responding on :11434. Attempting to start Ollama...' -ForegroundColor Yellow
            $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
            if ($ollamaCmd) {
                Start-Process -FilePath $ollamaCmd.Source -ArgumentList 'serve' -WindowStyle Hidden
            }
            elseif (Test-Path "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe") {
                Start-Process -FilePath "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList 'serve' -WindowStyle Hidden
            }

            $ollamaDeadline = (Get-Date).AddSeconds(12)
            while ((Get-Date) -lt $ollamaDeadline) {
                Start-Sleep -Seconds 1
                try {
                    $ollamaTags = Invoke-RestMethod `
                        -Uri 'http://127.0.0.1:11434/api/tags' `
                        -Method Get `
                        -TimeoutSec 2
                    if ($ollamaTags) { break }
                }
                catch { }
            }
        }

        if (-not $ollamaTags) {
            throw "Unable to connect to Ollama at http://127.0.0.1:11434. Start Ollama desktop or run 'ollama serve' in a terminal."
        }

        $targetModel = if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL } else { 'qwen3' }
        $modelExists = @($ollamaTags.models | Where-Object {
            $_.name -like "*$targetModel*" -or $_.model -like "*$targetModel*"
        }).Count -gt 0

        if (-not $modelExists) {
            throw "No Qwen3 model matching '$targetModel' found in Ollama. Check: ollama list"
        }

        Write-Host 'Ollama and Qwen3 model are available.' -ForegroundColor Green
    }
    catch {
        throw "Ollama is unavailable or Qwen3 is missing. Start Ollama and verify 'ollama list'. Details: $($_.Exception.Message)"
    }

    foreach ($svc in $services) {
        # Avoid accidentally starting duplicate services.
        $existing = Get-NetTCPConnection `
            -LocalPort $svc.Port `
            -State Listen `
            -ErrorAction SilentlyContinue

        if ($existing) {
            throw "Port $($svc.Port) is already in use. Stop the existing service on this port before restarting."
        }

        Write-Host "Starting $($svc.Name)..." -ForegroundColor Cyan

        $proc = Start-Process `
            -FilePath $venvPython `
            -ArgumentList @('-m', $svc.Module) `
            -WorkingDirectory $projectRoot `
            -PassThru

        $launched += $proc

        if (-not (Wait-Health -Name $svc.Name -Port $svc.Port -TimeoutSec 60)) {
            throw "$($svc.Name) did not become healthy."
        }
    }

    # Save process IDs only after all services pass health checks.
    $launched | ForEach-Object { $_.Id } |
        Set-Content -Path $pidFile -Encoding ASCII

    Write-Host ''
    Write-Host 'All services started successfully.' -ForegroundColor Green
    Write-Host 'Gateway API: http://127.0.0.1:9000/docs'
    Write-Host 'Backend status: http://127.0.0.1:9000/backends'
    Write-Host "Process IDs saved to: $pidFile"
}
catch {
    Write-Host "Startup failed: $($_.Exception.Message)" -ForegroundColor Red

    # Stop only processes started by this script.
    foreach ($proc in $launched) {
        try {
            if (-not $proc.HasExited) {
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            }
        }
        catch {
            # Continue cleanup.
        }
    }

    throw
}