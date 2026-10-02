@echo off
title Reelmint model download
cd /d "%~dp0"
"%~dp0..\ComfyUI_windows_portable\python_embeded\python.exe" -u download_models.py "%~dp0extra6.txt" > "%~dp0dl6.log" 2>&1
