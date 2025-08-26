# BondX Synthetic Dataset System - PowerShell Launcher

Write-Host "BondX Synthetic Dataset System" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found. Please install Python and ensure it's in your PATH." -ForegroundColor Red
    Write-Host "You can download Python from: https://python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to continue"
    exit 1
}

Write-Host "Starting system..." -ForegroundColor Green
Write-Host ""

# Generate dataset
Write-Host "1. Generating synthetic dataset..." -ForegroundColor Cyan
try {
    $result = python generate_synthetic_dataset.py 2>&1
    Write-Host $result
    Write-Host "✓ Dataset generated successfully!" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to generate dataset: $_" -ForegroundColor Red
    Read-Host "Press Enter to continue"
    exit 1
}

Write-Host ""

# Run tests
Write-Host "2. Running validation tests..." -ForegroundColor Cyan
try {
    $result = python test_dataset.py 2>&1
    Write-Host $result
    Write-Host "✓ All tests passed!" -ForegroundColor Green
} catch {
    Write-Host "❌ Tests failed: $_" -ForegroundColor Red
    Read-Host "Press Enter to continue"
    exit 1
}

Write-Host ""
Write-Host "=================================" -ForegroundColor Green
Write-Host "🎉 SYNTHETIC DATA SYSTEM COMPLETE!" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green
Write-Host ""
Write-Host "Generated files:" -ForegroundColor Yellow
Write-Host "- bondx_issuers_260.csv" -ForegroundColor White
Write-Host "- bondx_issuers_260.jsonl" -ForegroundColor White
Write-Host "- README.md" -ForegroundColor White
Write-Host ""
Write-Host "Files are ready for use in BondX development and testing!" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to continue"
