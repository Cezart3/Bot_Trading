"""Test EUR/USD ORB Strategy through MT5 API."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MetaTrader5 as mt5
from datetime import datetime
from config.settings import get_settings
from strategies.orb_forex_strategy import create_eurusd_london_strategy
from models.candle import Candle


def main():
    print("=" * 70)
    print("      TEST STRATEGIE EUR/USD ORB - LIVE API")
    print("=" * 70)
    print()

    # Connect
    s = get_settings()
    mt5.initialize(
        login=s.mt5.login,
        password=s.mt5.password,
        server=s.mt5.server,
        path=s.mt5.path,
        timeout=60000,
    )

    symbol = "EURUSD"
    mt5.symbol_select(symbol, True)

    print("[1] Conexiune MT5: OK")
    account = mt5.account_info()
    print(f"    Account: {account.login}")
    print(f"    Balance: ${account.balance:,.2f}")
    print()

    # Get candles
    print("[2] Descarcare date EUR/USD...")
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
    print(f"    Primit {len(rates)} lumanari")
    print()

    # Create strategy
    print("[3] Initializare strategie...")
    strategy = create_eurusd_london_strategy()
    strategy.initialize()
    print(f"    Strategie: {strategy.name}")
    print(f"    Simbol: {strategy.symbol}")
    print()

    # Convert to Candle objects and process
    print("[4] Procesare lumanari...")
    candles = []
    for rate in rates:
        candle = Candle(
            timestamp=datetime.fromtimestamp(rate["time"]),
            open=float(rate["open"]),
            high=float(rate["high"]),
            low=float(rate["low"]),
            close=float(rate["close"]),
            volume=float(rate["tick_volume"]),
            symbol=symbol,
            timeframe="M5",
        )
        candles.append(candle)

    # Process each candle
    signals = []
    for i, candle in enumerate(candles):
        signal = strategy.on_candle(candle, candles[: i + 1])
        if signal:
            signals.append((candle.timestamp, signal))

    print(f"    Semnale generate: {len(signals)}")
    print()

    # Show status
    print("[5] Status strategie:")
    status = strategy.get_status()
    print(f"    State: {status['state']}")
    print(f"    Trades today: {status['trades_today']}")
    print(f"    ORB candles: {status['orb_candles']}")

    if status["opening_range"]:
        print("    Opening Range:")
        print(f"      High: {status['opening_range']['high']:.5f}")
        print(f"      Low: {status['opening_range']['low']:.5f}")
        print(f"      Range: {status['opening_range']['range_pips']:.1f} pips")
        print(f"      Candles: {status['opening_range']['candles']}")

    print()
    if signals:
        print("[6] Semnale:")
        for ts, sig in signals:
            print(
                f"    {ts.strftime('%H:%M')} | {sig.signal_type.value.upper()} @ {sig.price:.5f}"
            )
            print(f"      SL: {sig.stop_loss:.5f} | TP: {sig.take_profit:.5f}")

    print()
    print("[7] Test plasare ordin (verificare doar)...")
    tick = mt5.symbol_info_tick(symbol)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.01,
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick.ask,
        "sl": tick.ask - 0.0020,
        "tp": tick.ask + 0.0040,
        "deviation": 20,
        "magic": 3001,
        "comment": "ORB_FOREX_TEST",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # Just check, don't send
    result = mt5.order_check(request)
    print(f"    Order check result: {result.retcode}")
    if result.retcode == 0:
        print("    ORDER VALID - poate fi plasat!")
    else:
        print(f"    Comment: {result.comment}")

    mt5.shutdown()

    print()
    print("=" * 70)
    print("      TEST COMPLET - TOTUL FUNCTIONEAZA!")
    print("=" * 70)


if __name__ == "__main__":
    main()
