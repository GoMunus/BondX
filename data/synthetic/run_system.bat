@echo off
echo BondX Synthetic Dataset System
echo ================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Please install Python and ensure it's in your PATH.
    echo You can download Python from: https://python.org
    pause
    exit /b 1
)

echo Python found. Starting system...
echo.

REM Generate dataset
echo 1. Generating synthetic dataset...
python generate_synthetic_dataset.py
if %errorlevel% neq 0 (
    echo Failed to generate dataset.
    pause
    exit /b 1
)

echo.
echo 2. Running validation tests...
python test_dataset.py
if %errorlevel% neq 0 (
    echo Tests failed.
    pause
    exit /b 1
)

echo.
echo ================================
echo SYNTHETIC DATA SYSTEM COMPLETE!
echo ================================
echo.
echo Generated files:
echo - bondx_issuers_260.csv
echo - bondx_issuers_260.jsonl
echo - README.md
echo.
echo Files are ready for use in BondX development and testing!
echo.
pause
