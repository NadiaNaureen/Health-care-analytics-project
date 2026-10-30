@echo off
title MM-GAT-Surv Web Application
cd /d "%~dp0"
echo ========================================================
echo Starting MM-GAT-Surv Clinical AI Application...
echo ========================================================
call .venv\Scripts\activate.bat
streamlit run app.py
pause
