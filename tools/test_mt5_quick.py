"""Quick MT5 connection test."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MetaTrader5 as mt5
from config.settings import get_settings

s = get_settings()
print(f"Connecting to {s.mt5.server}...")

init_result = mt5.initialize(
    login=s.mt5.login,
    password=s.mt5.password,
    server=s.mt5.server,
    path=s.mt5.path,
    timeout=60000,
)
print(f"Initialize result: {init_result}")

if init_result:
    account = mt5.account_info()
    if account:
        print(f"Connected! Balance: ${account.balance:,.2f}")
        mt5.symbol_select("EURUSD", True)
        from datetime import datetime, timedelta
        rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M5, 0, 10)
        if rates is not None:
            print(f"Got {len(rates)} candles")
            print(f"Latest candle time: {datetime.fromtimestamp(rates[-1]['time'])}")
        else:
            print(f"Error getting candles: {mt5.last_error()}")
    else:
        print(f"Account info error: {mt5.last_error()}")
    mt5.shutdown()
else:
    print(f"Error: {mt5.last_error()}")
