@echo off
title ORB Trading Bot - Stock Analysis
color 0F

echo.

cd /d "%~dp0\.."
python -u scripts\run_trading.py --analysis

echo.
pause
