@echo off
REM Usage: scripts\backtest.bat data\EURUSD_M5.csv
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
if "%~1"=="" (python -m tradebot backtest) else (python -m tradebot backtest --csv %1)
pause
