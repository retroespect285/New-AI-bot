@echo off
title New-AI-bot - TRADING - keep this window open - Ctrl+C to stop
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto notinstalled
echo The bot is starting. It trades by itself on the account MetaTrader 5 is logged in to.
echo Keep this window AND MetaTrader 5 open. To stop it: press Ctrl+C in this window.
echo Open trades keep their stop-loss and take-profit on the broker's server.
echo.
".venv\Scripts\python.exe" -m tradebot live %*
echo.
pause
exit /b 0

:notinstalled
echo Run 1-INSTALL-AND-SETUP.bat first.
pause
exit /b 1
