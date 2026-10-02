@echo off
rem Starts everything behind MyApp: ComfyUI, the Reelmint web app and the MyApp launcher page (each minimized).
set PY=C:\Users\Andy\AICharacterStudio\ComfyUI_windows_portable\python_embeded\python.exe
curl.exe -s -f -m 3 --noproxy "*" -o NUL http://127.0.0.1:8188/system_stats
if errorlevel 1 start "ComfyUI" /min cmd /c "C:\Users\Andy\AICharacterStudio\start_comfyui.bat"
curl.exe -s -f -m 3 --noproxy "*" -o NUL http://127.0.0.1:7860/api/status
if errorlevel 1 start "Reelmint" /min /d "C:\Users\Andy\AICharacterStudio\app" "%PY%" server.py
curl.exe -s -f -m 3 --noproxy "*" -o NUL http://127.0.0.1:8792/
if errorlevel 1 start "MyApp" /min "%PY%" -m http.server 8792 --bind 127.0.0.1 --directory "C:\Users\Andy\MyApp"
