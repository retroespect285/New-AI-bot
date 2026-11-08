@echo off
REM One-time setup on Windows: creates a virtual environment and installs the dependencies.
cd /d "%~dp0\.."
python -m venv .venv || goto :error
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :error
if not exist config.yaml copy config.example.yaml config.yaml
echo.
echo Done. Edit config.yaml, then run scripts\check.bat
goto :eof
:error
echo Setup failed. Is Python 3.10+ installed and on PATH?
exit /b 1
