@echo off
REM Usage: scripts\backtest.bat data\XAUUSD_M5.csv   (run "python -m tradebot export" first)
cd /d "%~dp0\.."
if "%~1"=="" (".venv\Scripts\python.exe" -m tradebot backtest) else (".venv\Scripts\python.exe" -m tradebot backtest --csv %1)
pause
