@echo off
rem Starts ComfyUI (if not already running) and the AI Character Studio web app, then opens the browser.
cd /d "%~dp0"
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8188/system_stats -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 start "ComfyUI" cmd /k "%~dp0start_comfyui.bat"
start "" http://127.0.0.1:7860
cd app
"%~dp0ComfyUI_windows_portable\python_embeded\python.exe" server.py
