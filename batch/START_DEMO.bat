@echo off
title ORB Trading Bot - DEMO MODE
color 0A

echo.
echo ======================================================================
echo                    ORB TRADING BOT - DEMO MODE
echo ======================================================================
echo.
echo   Symbol:           AMD (only)
echo   Risk per trade:   0.5%%
echo   Max daily loss:   2%%
echo   Mode:             DEMO (real orders on demo account)
echo.
echo ======================================================================
echo.

REM Check if MetaTrader 5 is running
tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I /N "terminal64.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo [WARNING] MetaTrader 5 is not running!
    echo.
    echo Please open MetaTrader 5 and login to your demo account first.
    echo.
    echo Press any key to try anyway, or close this window...
    pause >nul
)

echo Starting bot...
echo.
echo Press Ctrl+C to stop the bot.
echo.

cd /d "%~dp0\.."
python -u scripts\run_trading.py --mode demo

echo.
echo ======================================================================
echo Bot stopped.
echo ======================================================================
pause
