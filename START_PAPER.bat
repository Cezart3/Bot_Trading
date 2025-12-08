@echo off
title ORB Trading Bot - PAPER MODE
color 0E

echo.
echo ======================================================================
echo                    ORB TRADING BOT - PAPER MODE
echo ======================================================================
echo.
echo   Symbol:           AMD (only)
echo   Risk per trade:   0.5%%
echo   Max daily loss:   2%%
echo   Mode:             PAPER (simulation - no real orders)
echo.
echo ======================================================================
echo.

REM Check if MetaTrader 5 is running
tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I /N "terminal64.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo [WARNING] MetaTrader 5 is not running!
    echo.
    echo Please open MetaTrader 5 and login first.
    echo.
    echo Press any key to try anyway, or close this window...
    pause >nul
)

echo Starting bot in PAPER mode (no real trades)...
echo.
echo Press Ctrl+C to stop the bot.
echo.

cd /d "%~dp0"
python -u run_trading.py --mode paper

echo.
echo ======================================================================
echo Bot stopped.
echo ======================================================================
pause
