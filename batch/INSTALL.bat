@echo off
title ORB Trading Bot - Installation
color 0B

echo.
echo ======================================================================
echo                    ORB TRADING BOT - INSTALLATION
echo ======================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed!
    echo.
    echo Please install Python from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

echo Installing required packages...
echo.

cd /d "%~dp0\.."

pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install some packages.
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo                    INSTALLATION COMPLETE!
echo ======================================================================
echo.
echo Next steps:
echo.
echo 1. Install MetaTrader 5 from: https://www.metatrader5.com/
echo.
echo 2. Create a demo account in MetaTrader 5:
echo    - Open MetaTrader 5
echo    - File ^> Open an Account
echo    - Select "MetaQuotes-Demo" server
echo    - Create demo account
echo.
echo 3. Edit the .env file with your account details:
echo    - MT5_LOGIN=your_login_number
echo    - MT5_PASSWORD=your_password
echo    - MT5_SERVER=MetaQuotes-Demo
echo.
echo 4. Run START_DEMO.bat to start trading!
echo.
echo ======================================================================
pause
