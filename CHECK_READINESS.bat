@echo off
title ORB Trading Bot - Readiness Check
color 0B

echo.
echo ======================================================================
echo              CHECK READINESS FOR LIVE TRADING
echo ======================================================================
echo.

cd /d "%~dp0"
python -u run_trading.py --check-readiness

echo.
pause
