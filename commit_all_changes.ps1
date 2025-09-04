# PowerShell script to commit all changes
Write-Host "Checking git status..." -ForegroundColor Green
git status

Write-Host "Adding all changes..." -ForegroundColor Green
git add -A

Write-Host "Checking what's staged..." -ForegroundColor Green
git status --short

Write-Host "Committing all changes..." -ForegroundColor Green
git commit -m "Complete BondX implementation: MLOps, AI components, trading engine, and comprehensive platform"

Write-Host "Pushing to GitHub..." -ForegroundColor Green
git push origin main

Write-Host "All changes committed and pushed successfully!" -ForegroundColor Green

