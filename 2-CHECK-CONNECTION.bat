@echo off
title New-AI-bot - check connection
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto notinstalled
".venv\Scripts\python.exe" -m tradebot check
echo.
pause
exit /b 0

:notinstalled
echo Run 1-INSTALL-AND-SETUP.bat first.
pause
exit /b 1
