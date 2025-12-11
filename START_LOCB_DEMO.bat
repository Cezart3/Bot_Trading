@echo off
echo ============================================================
echo   LOCB Trading Bot - DEMO MODE (Multi-Symbol)
echo   London Opening Candle Breakout Strategy
echo ============================================================
echo.
echo   Session: 10:00 - 13:00 Romania (08:00-11:00 UTC)
echo   Strategy: LOCB with CHoCH/iFVG/Engulfing confirmations
echo   R:R Ratio: 3:1
echo   Avg SL: ~6 pips
echo.
echo   SYMBOLS: EURUSD, GBPUSD, USDJPY, EURJPY
echo   (4 pairs = ~4x more trading opportunities!)
echo.
echo   MODE: DEMO - Real orders on demo account!
echo.
echo ============================================================
echo.

cd /d "%~dp0"
python scripts/main.py --mode demo --strategy locb

pause
