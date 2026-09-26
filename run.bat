@echo off
title GestureDrive Controller
cd /d %~dp0
set PYTHONDONTWRITEBYTECODE=1
echo ======================================================
echo    Starting GestureDrive Hand Gesture Controller...
echo ======================================================
python -B app\main.py
pause
