@echo off
title GestureDrive - Desktop Mouse Mode
cd /d "%~dp0"
echo =================================================================
echo   STARTING GESTUREDRIVE IN DESKTOP MOUSE MODE...
echo   Press [V] anytime to switch to VR Mode
echo =================================================================
echo.
python app/main.py
echo.
echo =================================================================
echo   GestureDrive process has exited.
echo =================================================================
pause
