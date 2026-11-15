@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -m streamlit run src/dashboard/app.py
if errorlevel 1 (
  echo.
  echo Streamlit is not installed or failed to run.
  echo Install first: python -m pip install -r requirements-dashboard.txt
  pause
)
