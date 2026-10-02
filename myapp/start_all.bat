@echo off
rem Starts everything behind MyApp: ComfyUI, the Reelmint web app and the MyApp launcher page (each minimized).
set PY=C:\Users\Andy\AICharacterStudio\ComfyUI_windows_portable\python_embeded\python.exe
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8188/system_stats -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 start "ComfyUI" /min cmd /c "C:\Users\Andy\AICharacterStudio\start_comfyui.bat"
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:7860/api/status -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 start "Reelmint" /min /d "C:\Users\Andy\AICharacterStudio\app" "%PY%" server.py
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8792/ -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 start "MyApp" /min "%PY%" -m http.server 8792 --bind 127.0.0.1 --directory "C:\Users\Andy\MyApp"
