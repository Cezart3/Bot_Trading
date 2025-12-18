"""
Script pentru plasarea unui ordin Buy Limit pe EURUSD.
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from brokers.mt5_broker import MT5Broker
from models.order import Order


def main():
    # Parametrii ordinului
    symbol = "EURUSD"
    entry_price = 1.17333
    stop_loss = 1.17303
    take_profit = 1.17378

    # Risk calculation: 0.5% din 100,000$ = 500$
    account_balance = 100_000
    risk_percent = 0.5
    risk_amount = account_balance * (risk_percent / 100)  # 500$

    # SL distance in price
    sl_distance = abs(entry_price - stop_loss)  # 0.0003

    # Pentru EURUSD: 1 lot = 100,000 units, pip value = ~10$/pip pentru 1 lot
    # SL = 3 pips (0.0003 / 0.0001 = 3 pips)
    # Risk per lot = 3 pips * 10$/pip = 30$
    # Lot size = Risk / Risk_per_lot = 500$ / 30$ = 16.67 lots

    pip_size = 0.0001
    sl_pips = sl_distance / pip_size  # 3 pips
    pip_value_per_lot = 10.0  # ~10$ per pip pentru EURUSD la 1 lot

    lot_size = risk_amount / (sl_pips * pip_value_per_lot)
    lot_size = round(lot_size, 2)  # Round to 2 decimals

    print(f"=== Order Parameters ===")
    print(f"Symbol: {symbol}")
    print(f"Type: Buy Limit")
    print(f"Entry: {entry_price}")
    print(f"Stop Loss: {stop_loss} ({sl_pips:.1f} pips)")
    print(f"Take Profit: {take_profit}")
    print(f"Risk: ${risk_amount:.2f} ({risk_percent}%)")
    print(f"Lot Size: {lot_size}")
    print()

    # Connect to MT5
    print("Connecting to MT5...")
    broker = MT5Broker(
        login=int(os.getenv("MT5_LOGIN")),
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER"),
        path=os.getenv("MT5_PATH")
    )

    if not broker.connect():
        print("ERROR: Could not connect to MT5!")
        return

    print("Connected to MT5")

    # Get current price
    bid, ask = broker.get_current_price(symbol)
    print(f"Current price - Bid: {bid}, Ask: {ask}")

    # Determine order type based on current price
    # Buy Limit: entry < current price (buy when price drops)
    # Buy Stop: entry > current price (buy when price rises)
    if ask < entry_price:
        print(f"Using BUY STOP (entry {entry_price} > current ask {ask})")
        order = Order.stop_buy(
            symbol=symbol,
            quantity=lot_size,
            stop_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment="LOCB Buy Stop"
        )
    else:
        print(f"Using BUY LIMIT (entry {entry_price} <= current ask {ask})")
        order = Order.limit_buy(
            symbol=symbol,
            quantity=lot_size,
            price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment="LOCB Buy Limit"
        )

    # Place order
    print(f"\nPlacing order...")
    order_id = broker.place_order(order)

    if order_id:
        print(f"SUCCESS! Order placed with ID: {order_id}")
    else:
        print("FAILED to place order!")

    broker.disconnect()


if __name__ == "__main__":
    main()
