@echo off
REM One-click launcher: creates the venv, installs requirements, builds the index if missing, starts app.py.
setlocal
cd /d "%~dp0"

REM 1. Create the virtual environment (first run only)
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    python -m venv .venv || (echo Failed to create venv. Is Python installed and on PATH? & pause & exit /b 1)
)

REM 2. Activate it
call ".venv\Scripts\activate.bat"

REM 3. Install requirements (re-runs only when requirements.txt changes)
fc /b requirements.txt ".venv\requirements.installed" >nul 2>&1
if errorlevel 1 (
    echo Installing requirements ...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt || (echo Failed to install requirements. & pause & exit /b 1)
    copy /y requirements.txt ".venv\requirements.installed" >nul
)

REM 4. Make sure .env exists
if not exist ".env" (
    copy .env.example .env >nul
    echo Created .env from .env.example - put your GROQ_API_KEY in it, then run this again.
    notepad .env
    pause
    exit /b 1
)

REM 5. Build the index if it hasn't been built yet
if not exist "data\vectors.npy" (
    echo Building the index ^(first run^) ...
    python ingest.py || (echo Indexing failed. & pause & exit /b 1)
)

REM 6. Start the web app
echo Starting app at http://localhost:8080 ...
python app.py
pause
