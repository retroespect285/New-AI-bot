@echo off
REM Starts the bot. Keep this window open; press Ctrl+C to stop it cleanly.
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
python -m tradebot live %*
pause
