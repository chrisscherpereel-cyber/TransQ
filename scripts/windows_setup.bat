@echo off
REM ---------------------------------------------------------------------------
REM Double-click this file to set up and start Lecture Quiz Builder on Windows.
REM
REM What it does, all of it inside this folder and none of it elsewhere:
REM   1. checks Python is present
REM   2. creates a private virtual environment in .venv\
REM   3. installs the dependencies
REM   4. generates APP_SECRET into .env, once, if there isn't one
REM   5. checks whether Ollama is running and says so
REM   6. starts the app in your browser
REM
REM Safe to run again. After the first time it skips straight to starting up.
REM ---------------------------------------------------------------------------

setlocal
cd /d "%~dp0.."

echo.
echo  Lecture Quiz Builder - setup
echo  Folder: %CD%
echo.

REM --- 1. Python -------------------------------------------------------------

where python >nul 2>&1
if errorlevel 1 goto NOPYTHON

python -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)" >nul 2>&1
if errorlevel 1 goto OLDPYTHON

echo  [ok] Python found.

REM --- 2 and 3. Environment and dependencies ---------------------------------
REM A virtual environment keeps these packages inside this folder rather than
REM changing the Python Windows uses for anything else.

if not exist ".venv" (
  echo  Creating a private Python environment ^(.venv^)...
  python -m venv .venv
  if errorlevel 1 goto VENVFAIL
)

call .venv\Scripts\activate.bat

python -c "import streamlit, faster_whisper" >nul 2>&1
if errorlevel 1 (
  echo  Installing dependencies. The first run takes a few minutes...
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -r requirements.txt
  if errorlevel 1 goto PIPFAIL
  echo  [ok] Dependencies installed.
) else (
  echo  [ok] Dependencies already installed.
)

REM --- 4. APP_SECRET ---------------------------------------------------------
REM Generated once and never regenerated: changing it makes everything already
REM saved unreadable, so an existing .env is left strictly alone.

if not exist ".env" (
  echo  Creating .env with a new APP_SECRET...
  for /f "delims=" %%S in ('python -c "import secrets; print(secrets.token_urlsafe(48))"') do set "SECRET=%%S"
  >  .env echo # Created by scripts\windows_setup.bat
  >> .env echo # APP_SECRET encrypts everything this app saves.
  >> .env echo # Back it up where you keep passwords. If you lose it, your saved
  >> .env echo # lectures cannot be recovered - by anyone, ever.
  >> .env echo APP_SECRET=%SECRET%
  echo  [ok] Created .env
  echo.
  echo  Back up the APP_SECRET line in .env where you keep passwords.
  echo.
) else (
  echo  [ok] .env already exists - leaving it alone.
)

REM --- 5. Is there a model to talk to? ---------------------------------------
REM Only a courtesy check. A missing Ollama is a note, never a reason to stop.

python -c "import urllib.request,json,sys; d=json.load(urllib.request.urlopen('http://localhost:11434/v1/models', timeout=2)); print(len(d.get('data',[])))" > "%TEMP%\lqb_models.txt" 2>nul
if errorlevel 1 (
  echo  [note] No local model server detected - that is fine.
  echo         The app starts anyway and works with a hosted model.
  echo         For a local one: install Ollama from https://ollama.com/download,
  echo         then run:  ollama pull qwen2.5:14b
) else (
  set /p MODELS=<"%TEMP%\lqb_models.txt"
  echo  [ok] Ollama is running with %MODELS% model^(s^) ready.
)
del "%TEMP%\lqb_models.txt" >nul 2>&1

REM --- 6. Go -----------------------------------------------------------------

echo.
echo  Starting the app...
echo  Your browser opens at http://localhost:8501
echo.
echo  Leave this window open while you use the app.
echo  To stop it, press Control-C here, or just close the window.
echo.

streamlit run app.py

echo.
echo  The app has stopped.
pause
exit /b 0

REM --- Failure paths ---------------------------------------------------------

:NOPYTHON
echo  [X] Python is not installed.
echo.
echo      Install it from https://www.python.org/downloads/
echo      On the first screen of the installer, tick "Add python.exe to PATH"
echo      before clicking Install. That checkbox is what makes this script work.
echo.
pause
exit /b 1

:OLDPYTHON
echo  [X] Your Python is too old - this app needs 3.9 or newer.
echo      Install a current version from https://www.python.org/downloads/
echo.
pause
exit /b 1

:VENVFAIL
echo  [X] Could not create the Python environment.
echo.
pause
exit /b 1

:PIPFAIL
echo  [X] Installation failed. The output above says why.
echo.
pause
exit /b 1
