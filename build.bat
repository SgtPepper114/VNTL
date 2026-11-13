@echo off
echo Building VNTL...
uv sync --locked --group dev
if errorlevel 1 exit /b 1
uv run python tools/build.py
if errorlevel 1 exit /b 1
echo.
echo Done! Distributable is in dist\vntl\
pause
