@echo off
cd /d "%~dp0"
python -m uvicorn src.serving.search_bridge:app --host 127.0.0.1 --port 8000
pause
