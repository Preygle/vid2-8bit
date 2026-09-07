@echo off
setlocal enabledelayedexpansion
title vid2-8bit web UI

REM ---------------------------------------------------------------------------
REM  Launches the local parameter-tweaking web UI and opens it in the browser.
REM
REM  Usage:
REM      run-webapp.bat
REM      run-webapp.bat --port 9000
REM      run-webapp.bat --no-browser
REM
REM  Everything runs on 127.0.0.1 only. Nothing is uploaded anywhere.
REM ---------------------------------------------------------------------------

cd /d "%~dp0"

echo.
echo  ================================================
echo    vid2-8bit  -  video to pixel art
echo  ================================================
echo.

REM -- 1. Find a usable Python -------------------------------------------------
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo  [ERROR] Python was not found on PATH.
    echo          Install Python 3.11+ from https://python.org and tick
    echo          "Add python.exe to PATH" during setup.
    echo.
    pause
    exit /b 1
)

for /f "delims=" %%v in ('%PY% -c "import sys;print(sys.version.split()[0])" 2^>nul') do set "PYVER=%%v"
if not defined PYVER (
    echo  [ERROR] Found Python but could not run it. Try: %PY% --version
    echo.
    pause
    exit /b 1
)
echo  [ok]    Python !PYVER!

REM -- 2. Check ffmpeg (needed for video; images still work without it) --------
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo  [WARN]  ffmpeg not found on PATH.
    echo          Still images will work; video input/output will not.
    echo          Get it from https://ffmpeg.org/download.html
) else (
    echo  [ok]    ffmpeg
)

REM -- 3. Check Python dependencies -------------------------------------------
%PY% -c "import numpy, cv2, scipy, yaml" >nul 2>&1
if errorlevel 1 (
    echo  [WARN]  Missing Python packages. Installing now...
    echo.
    %PY% -m pip install --quiet numpy opencv-python scipy PyYAML
    if errorlevel 1 (
        echo.
        echo  [ERROR] Install failed. Run this manually:
        echo          %PY% -m pip install numpy opencv-python scipy PyYAML
        echo.
        pause
        exit /b 1
    )
    echo  [ok]    Dependencies installed
) else (
    echo  [ok]    numpy, opencv, scipy, pyyaml
)

REM -- 4. Launch ---------------------------------------------------------------
REM  Run from source without needing `pip install -e .`
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"

echo.
echo  Starting server... your browser should open automatically.
echo  Leave this window open. Press Ctrl+C here to stop.
echo.

%PY% -m vid2_8bit.cli web %*

set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" (
    echo  [ERROR] The server exited with code %RC%.
    echo          Scroll up for the reason.
) else (
    echo  Server stopped.
)
echo.
pause
endlocal
