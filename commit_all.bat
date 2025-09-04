@echo off
echo Adding all changes to git...
git add -A

echo Checking git status...
git status --short

echo Committing all changes...
git commit -m "Complete BondX implementation: MLOps, AI components, trading engine, and comprehensive platform"

echo Pushing to GitHub...
git push origin main

echo All changes committed and pushed successfully!
pause

