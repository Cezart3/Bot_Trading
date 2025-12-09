"""Simulate EUR/USD ORB Strategy for today."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MetaTrader5 as mt5
from datetime import datetime
from config.settings import get_settings
from strategies.orb_forex_strategy import create_eurusd_london_strategy
from models.candle import Candle


def main():
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

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 300)
    today = datetime.now().date()
    today_rates = [r for r in rates if datetime.fromtimestamp(r["time"]).date() == today]

    print("=" * 70)
    print("      SIMULARE STRATEGIE EUR/USD - AZI")
    print("=" * 70)
    print()
    print(f"Lumanari azi: {len(today_rates)}")
    print()

    strategy = create_eurusd_london_strategy()
    strategy.initialize()

    candles = []
    for rate in today_rates:
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

    print("Procesare lumanari de azi:")
    print("-" * 70)

    signals = []
    last_state = None

    for i, candle in enumerate(candles):
        signal = strategy.on_candle(candle, candles[: i + 1])

        status = strategy.get_status()
        ts = candle.timestamp.strftime("%H:%M")
        state = status["state"]

        # Show state changes
        if state != last_state:
            if state == "building_orb":
                print(f"[{ts}] Building Opening Range...")
            elif state == "orb_complete" and status["opening_range"]:
                or_info = status["opening_range"]
                print(
                    f"[{ts}] ORB COMPLETE | H={or_info['high']:.5f} L={or_info['low']:.5f} | "
                    f"{or_info['range_pips']:.1f} pips"
                )
            elif state == "in_trade":
                print(f"[{ts}] IN TRADE")
            last_state = state

        if signal:
            signals.append((candle.timestamp, signal))
            print(f"[{ts}] >>> SIGNAL: {signal.signal_type.value.upper()} @ {signal.price:.5f}")
            print(f"        SL: {signal.stop_loss:.5f} | TP: {signal.take_profit:.5f}")

    print()
    print("=" * 70)
    print("REZULTAT:")
    print(f"  Semnale generate: {len(signals)}")

    if signals:
        print()
        for ts, sig in signals:
            risk_pips = abs(sig.price - sig.stop_loss) / 0.0001
            reward_pips = abs(sig.take_profit - sig.price) / 0.0001
            print(f"  {sig.signal_type.value.upper()} @ {sig.price:.5f}")
            print(f"    Risk: {risk_pips:.1f} pips | Reward: {reward_pips:.1f} pips")

            # Check result
            for candle in candles:
                if candle.timestamp <= ts:
                    continue
                if sig.signal_type.value == "long":
                    if candle.low <= sig.stop_loss:
                        print(f"    Result: STOP LOSS at {candle.timestamp.strftime('%H:%M')}")
                        break
                    if candle.high >= sig.take_profit:
                        print(f"    Result: TAKE PROFIT at {candle.timestamp.strftime('%H:%M')}")
                        break
                else:  # short
                    if candle.high >= sig.stop_loss:
                        print(f"    Result: STOP LOSS at {candle.timestamp.strftime('%H:%M')}")
                        break
                    if candle.low <= sig.take_profit:
                        print(f"    Result: TAKE PROFIT at {candle.timestamp.strftime('%H:%M')}")
                        break
            else:
                # Trade still open or session ended
                last_price = candles[-1].close
                if sig.signal_type.value == "long":
                    current_pnl = (last_price - sig.price) / 0.0001
                else:
                    current_pnl = (sig.price - last_price) / 0.0001
                print(f"    Result: OPEN (current PnL: {current_pnl:+.1f} pips)")

    mt5.shutdown()


if __name__ == "__main__":
    main()
