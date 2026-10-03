@echo off
rem Starts everything behind MyApp: Ollama, ComfyUI, the Reelmint web app and the MyApp launcher page (each minimized).
rem Each piece is started only if it is not already answering. Checks use curl.exe --noproxy: a VPN's Windows
rem system proxy makes PowerShell checks fail, which once started duplicate copies.
rem Run at login by "MyApp autostart.vbs" in the Startup folder (via C:\Users\Andy\MyApp\start_all.bat).
set PY=C:\Users\Andy\AICharacterStudio\ComfyUI_windows_portable\python_embeded\python.exe

curl.exe -s -f -m 3 --noproxy "*" -o NUL http://127.0.0.1:11434/api/tags
if errorlevel 1 if exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"

curl.exe -s -f -m 3 --noproxy "*" -o NUL http://127.0.0.1:8188/system_stats
if errorlevel 1 start "ComfyUI" /min cmd /c "C:\Users\Andy\AICharacterStudio\start_comfyui.bat"

rem give ComfyUI up to ~2 minutes to come up before the app starts (the app cleans up ComfyUI on start)
for /l %%i in (1,1,24) do (
  curl.exe -s -f -m 3 --noproxy "*" -o NUL http://127.0.0.1:8188/system_stats && goto comfy_ready
  timeout /t 5 /nobreak >NUL
)
:comfy_ready

curl.exe -s -f -m 3 --noproxy "*" -o NUL http://127.0.0.1:7860/api/status
if errorlevel 1 start "Reelmint" /min /d "C:\Users\Andy\AICharacterStudio\app" "%PY%" server.py

curl.exe -s -f -m 3 --noproxy "*" -o NUL http://127.0.0.1:8792/
if errorlevel 1 start "MyApp" /min "%PY%" -m http.server 8792 --bind 127.0.0.1 --directory "C:\Users\Andy\MyApp"
