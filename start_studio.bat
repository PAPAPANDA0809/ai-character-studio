@echo off
rem Starts ComfyUI (if not already running) and the Reelmint web app, then opens the browser.
cd /d "%~dp0"
curl.exe -s -f -m 3 --noproxy "*" -o NUL http://127.0.0.1:8188/system_stats
if errorlevel 1 start "ComfyUI" cmd /k "%~dp0start_comfyui.bat"
start "" http://127.0.0.1:7860
cd app
"%~dp0ComfyUI_windows_portable\python_embeded\python.exe" server.py
