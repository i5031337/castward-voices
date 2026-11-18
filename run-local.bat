@echo off
setlocal
cd /d "%~dp0" || exit /b 1

if defined PYTHON (
    set "SPEECH_PYTHON=%PYTHON%"
) else (
    set "SPEECH_PYTHON=%CD%\.venv\Scripts\python.exe"
)
if not exist "%SPEECH_PYTHON%" (
    where "%SPEECH_PYTHON%" >nul 2>&1
    if errorlevel 1 (
        echo Python environment is missing: %SPEECH_PYTHON%. Run setup.bat first. 1>&2
        exit /b 1
    )
)

"%SPEECH_PYTHON%" "%CD%\run_local_windows.py" %*
exit /b %ERRORLEVEL%
