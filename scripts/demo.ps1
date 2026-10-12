# HORIZON 2K26 · D07 — Judge Live Demonstration Script
# PowerShell script for quick jury evaluation or full multi-seed benchmark validation

param(
    [switch]$Quick,
    [switch]$Full
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "  HORIZON 2K26 · D07: Adaptive Token-Aware Inference Request Scheduler  " -ForegroundColor Cyan
Write-Host "  Hackathon Live Evaluation & Verification Pack                          " -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

# Step 1: Automated Test Suite Verification
Write-Host "`n[STEP 1/3] Executing Automated Pytest Suite..." -ForegroundColor Yellow
& $Python -m pytest -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Test suite failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "✅ All test suites PASSED (100% test pass rate)." -ForegroundColor Green

# Step 2: Benchmark Lab Execution
if ($Full) {
    Write-Host "`n[STEP 2/3] Running Full Multi-Seed Benchmark Evaluation (5 Seeds, Bootstrap 95% CIs)..." -ForegroundColor Yellow
    & $Python scripts\benchmark_runner.py --size 20 --scale 0.2 --multiseed --arrival-process poisson --workload balanced
} else {
    Write-Host "`n[STEP 2/3] Running Quick Live Evaluation Benchmark (Fast ~15s execution)..." -ForegroundColor Yellow
    & $Python scripts\benchmark_runner.py --size 15 --scale 0.1 --arrival-process poisson --workload balanced --policies "fifo,token_aware,adaptive,adaptive_pred"
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Benchmark execution failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "✅ Benchmark run complete with metrics & charts generated." -ForegroundColor Green

# Step 3: Verification & Next Steps
Write-Host "`n[STEP 3/3] Demo Verification Summary" -ForegroundColor Yellow
Write-Host "  - Results Directory:    results\" -ForegroundColor White
Write-Host "  - Documentation:        docs\JUDGE_QA.md & docs\PITCH_OUTLINE.md" -ForegroundColor White
Write-Host "  - Live Streamlit UI:    streamlit run dashboard\app.py" -ForegroundColor White

Write-Host "`n==========================================================================" -ForegroundColor Cyan
Write-Host "  Demo Completed Successfully! Ready for Jury Presentation.             " -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan
