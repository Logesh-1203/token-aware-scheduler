# stop_all.ps1 - Stop services started by start_all.ps1
# Usage: .\scripts\stop_all.ps1
# Reads the PID file created by start_all.ps1 and terminates those processes.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir '..')
$pidFile = Join-Path $projectRoot 'run_pids.txt'

if (-not (Test-Path $pidFile)) {
    Write-Host "PID file not found ($pidFile). No processes to stop."
    exit 0
}

$pidLines = Get-Content -Path $pidFile -ErrorAction Stop
foreach ($procId in $pidLines) {
    if ([string]::IsNullOrWhiteSpace($procId)) { continue }
    try {
        Write-Host "Stopping process tree for PID $procId..."
        & taskkill /F /T /PID $procId 2>$null
    } catch {
        Write-Warning "Failed to stop PID ${procId}: $_"
    }
}

# Also ensure no orphaned child processes remain listening on the service ports
$ports = @(8001, 8002, 8003, 8004, 9000)
foreach ($p in $ports) {
    $conns = Get-NetTCPConnection -LocalPort $p -ErrorAction SilentlyContinue
    if ($conns) {
        foreach ($conn in $conns) {
            $orphanPid = $conn.OwningProcess
            if ($orphanPid -and $orphanPid -ne 0) {
                Write-Host "Cleaning up orphaned process $orphanPid on port $p..."
                & taskkill /F /T /PID $orphanPid 2>$null
            }
        }
    }
}

# Remove the PID file after cleanup
Remove-Item -Path $pidFile -Force -ErrorAction SilentlyContinue
Write-Host "All started services have been stopped and PID file removed."
