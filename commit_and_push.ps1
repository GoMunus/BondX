# PowerShell script to commit and push MLOps changes
Write-Host "Adding MLOps files to git..." -ForegroundColor Green

# Add all MLOps files
git add bondx/mlops/tracking.py
git add bondx/mlops/registry.py
git add bondx/mlops/drift.py
git add bondx/mlops/retrain.py
git add bondx/mlops/deploy.py

Write-Host "Checking git status..." -ForegroundColor Green
git status --short

Write-Host "Committing changes..." -ForegroundColor Green
git commit -m "Enhanced MLOps components: tracking, registry, drift detection, retraining, and canary deployment"

Write-Host "Pushing to GitHub..." -ForegroundColor Green
git push origin main

Write-Host "Done!" -ForegroundColor Green
