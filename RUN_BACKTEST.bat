@echo off
title ORB Trading Bot - Backtest
color 0B

echo.
echo ======================================================================
echo                    ORB STRATEGY BACKTEST
echo ======================================================================
echo.
echo This will show historical performance of the strategy.
echo.
echo Options:
echo   1. Quick backtest AMD (30 days, sample data)
echo   2. Full backtest AMD (60 days, REAL Yahoo Finance data)
echo   3. Compare all stocks (AMD, NVDA, TSLA)
echo   4. Exit
echo.

set /p choice="Enter your choice (1-4): "

cd /d "%~dp0"

if "%choice%"=="1" (
    echo.
    echo Running quick backtest for AMD...
    echo.
    python run_backtest_stocks.py --symbol AMD --days 30
    goto end
)

if "%choice%"=="2" (
    echo.
    echo Downloading real data from Yahoo Finance...
    echo This may take a moment...
    echo.
    python run_backtest_stocks.py --symbol AMD --days 60 --live-data
    goto end
)

if "%choice%"=="3" (
    echo.
    echo Comparing all stocks with sample data...
    echo.
    python run_backtest_stocks.py --all-stocks --days 30
    goto end
)

if "%choice%"=="4" (
    exit
)

echo Invalid choice. Please try again.
goto end

:end
echo.
echo ======================================================================
echo Backtest complete! Check the reports in data/reports folder.
echo ======================================================================
pause
