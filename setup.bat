@echo off
setlocal
cd /d "%~dp0" || exit /b 1

where uv >nul 2>&1
if errorlevel 1 (
    echo Install uv first: https://docs.astral.sh/uv/getting-started/installation/ 1>&2
    exit /b 1
)
if not defined UV_CACHE_DIR set "UV_CACHE_DIR=%CD%\.cache\uv"
if not defined HF_HOME set "HF_HOME=%CD%\.cache\huggingface"

uv run --no-project --python 3.11 "%CD%\setup_models.py" %*
exit /b %ERRORLEVEL%
