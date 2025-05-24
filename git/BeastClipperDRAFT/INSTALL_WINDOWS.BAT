@echo off
title BeastClipper Ultimate v3.0 Installer
color 0A

echo ============================================================
echo            BEASTCLIPPER ULTIMATE v3.0 INSTALLER
echo ============================================================
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed!
    echo.
    echo Please install Python 3.7 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

echo [1/4] Python detected. Installing dependencies...
echo.

:: Run the Python installer
python install_beastclipper.py

if %errorlevel% neq 0 (
    echo.
    echo Installation failed! Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo          Installation Complete!
echo ============================================================
echo.
echo You can now run BeastClipper Ultimate by:
echo   - Double-clicking the desktop shortcut
echo   - Running 'python main.py' from this directory
echo.
pause
