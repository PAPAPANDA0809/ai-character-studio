@echo off
rem ComfyUI tuned for an 8GB card: models stream from RAM, previews off.
cd /d "%~dp0ComfyUI_windows_portable"
.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --listen 127.0.0.1 --port 8188 --reserve-vram 0.8 --disable-auto-launch --preview-method none %*
