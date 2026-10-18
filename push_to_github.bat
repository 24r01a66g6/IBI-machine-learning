@echo off
echo ===================================================
echo   Pushing Employee Attrition System to GitHub
echo   Repository: https://github.com/24r01a66g6/IBI-machine-learning
echo ===================================================
echo.

cd /d "%~dp0"

where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not in PATH!
    echo Please install Git from https://git-scm.com/download/win
    echo After installing, reopen this script.
    pause
    exit /b 1
)

echo [1/4] Initializing Git repository...
if not exist .git (
    git init -b main
)

echo [2/4] Setting remote repository...
git remote remove origin >nul 2>nul
git remote add origin https://github.com/24r01a66g6/IBI-machine-learning.git

echo [3/4] Staging and committing files...
git add .
git commit -m "feat: complete end-to-end Employee Attrition Prediction System with Streamlit dashboard, SHAP explainability, and cost analysis"

echo [4/4] Pushing to GitHub (main branch)...
git push -u origin main

echo.
echo ===================================================
echo   Done! Check your repository:
echo   https://github.com/24r01a66g6/IBI-machine-learning
echo ===================================================
pause
