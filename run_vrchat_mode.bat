@echo off
title GestureDrive - VRChat VR Mode (Streaming 127.0.0.1:9000)
cd /d "%~dp0"
echo =================================================================
echo   STARTING GESTUREDRIVE IN VRCHAT VR MODE...
echo   Broadcasting live OSC hand tracking to VRChat on 127.0.0.1:9000
echo =================================================================
echo.
python app/main.py --vr
echo.
echo =================================================================
echo   GestureDrive process has exited.
echo =================================================================
pause
