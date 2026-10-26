@echo off
rem Starts Reelmint through Windows Task Scheduler so it outlives this window.
cd /d "%~dp0"
curl.exe -s -f -m 3 --noproxy "*" -o NUL http://127.0.0.1:8188/system_stats
if errorlevel 1 start "ComfyUI" cmd /k "%~dp0start_comfyui.bat"
schtasks /Run /TN "Reelmint Background Server" >NUL 2>&1
if not errorlevel 1 goto open_browser
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup\enable_background.ps1"
if errorlevel 1 (
  echo Could not start Reelmint's Windows background task.
  pause
  exit /b 1
)
:open_browser
start "" http://127.0.0.1:7860
