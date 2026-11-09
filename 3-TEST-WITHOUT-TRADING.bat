@echo off
title New-AI-bot - TEST MODE, no orders - press Ctrl+C to stop
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto notinstalled
echo The bot studies the gold charts, then shows what it would do every 5 minutes.
echo It sends NO orders. Keep MetaTrader 5 open. Press Ctrl+C to stop.
echo.
".venv\Scripts\python.exe" -m tradebot live --dry-run
echo.
pause
exit /b 0

:notinstalled
echo Run 1-INSTALL-AND-SETUP.bat first.
pause
exit /b 1
