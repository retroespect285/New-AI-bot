@echo off
title New-AI-bot - install and set-up
cd /d "%~dp0"
echo.
echo ============================================================
echo   New-AI-bot - install and set-up
echo   BEFORE YOU CONTINUE: open MetaTrader 5 and log in.
echo ============================================================
echo.
set "PY="
rem Prefer a Python version the MetaTrader5 package supports (3.10 - 3.14).
for %%V in (3.12 3.13 3.11 3.14 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
  )
)
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY goto nopython
if not exist ".venv\Scripts\python.exe" (
  echo Creating a private Python environment in the .venv folder...
  %PY% -m venv .venv
)
if not exist ".venv\Scripts\python.exe" goto nopython
echo Installing the bot's components. The first time this takes a few minutes...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto pipfail
echo.
".venv\Scripts\python.exe" -m tradebot setup
echo.
pause
exit /b 0

:nopython
echo.
echo Python was not found, or the Python environment could not be created.
echo Install Python 3.12 from https://www.python.org/downloads/   (3.10 to 3.14 also work;
echo newer versions may not be supported by the MetaTrader5 package yet)
echo and tick "Add python.exe to PATH" on the first screen of the installer.
echo Then double-click this file again.
echo.
pause
exit /b 1

:pipfail
echo.
echo Installing the components failed. Check the internet connection and double-click this file again.
echo If the error mentions MetaTrader5, install Python 3.12, delete the .venv folder and try again.
echo If it keeps failing, copy the messages above and send them to your assistant.
echo.
pause
exit /b 1
