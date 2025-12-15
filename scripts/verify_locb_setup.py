"""
LOCB Bot Verification Script

Verifica ca botul este configurat corect si gata de rulare.
Ruleaza acest script inainte de a porni botul pentru a te asigura ca totul functioneaza.

Usage:
    python scripts/verify_locb_setup.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

def print_header(text):
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_ok(text):
    print(f"  [OK] {text}")

def print_error(text):
    print(f"  [ERROR] {text}")

def print_info(text):
    print(f"  -> {text}")

def main():
    print_header("VERIFICARE SETUP LOCB BOT")

    errors = []

    # 1. Verifica importurile
    print("\n[1/5] Verificare importuri...")
    try:
        from config.settings import get_settings
        print_ok("config.settings importat")
    except ImportError as e:
        print_error(f"config.settings: {e}")
        errors.append("settings")

    try:
        from brokers.mt5_broker import MT5Broker
        print_ok("brokers.mt5_broker importat")
    except ImportError as e:
        print_error(f"brokers.mt5_broker: {e}")
        errors.append("mt5_broker")

    try:
        from strategies.locb_strategy import LOCBStrategy
        print_ok("strategies.locb_strategy importat")
    except ImportError as e:
        print_error(f"strategies.locb_strategy: {e}")
        errors.append("locb_strategy")

    try:
        import MetaTrader5 as mt5
        print_ok("MetaTrader5 importat")
    except ImportError as e:
        print_error(f"MetaTrader5: {e}")
        print_info("Ruleaza: pip install MetaTrader5")
        errors.append("MetaTrader5")

    if errors:
        print_header("ERORI DE IMPORT - REZOLVA MAI INTAI")
        return False

    # 2. Verifica conexiunea MT5
    print("\n[2/5] Verificare conexiune MT5...")
    settings = get_settings()

    if not mt5.initialize(
        login=settings.mt5.login,
        password=settings.mt5.password,
        server=settings.mt5.server,
        path=settings.mt5.path,
        timeout=60000,
    ):
        print_error(f"Nu ma pot conecta la MT5: {mt5.last_error()}")
        print_info("Verifica ca MT5 terminal este deschis si logat")
        return False

    print_ok("Conectat la MT5")

    account = mt5.account_info()
    if account:
        print_info(f"Account: {account.login}")
        print_info(f"Server: {account.server}")
        print_info(f"Balance: ${account.balance:,.2f}")
        print_info(f"Equity: ${account.equity:,.2f}")

    # 3. Verifica ora serverului
    print("\n[3/5] Verificare timezone server...")

    # Get server time from a tick
    tick = mt5.symbol_info_tick("EURUSD")
    if tick:
        server_time = datetime.fromtimestamp(tick.time)
        local_time = datetime.now()

        print_info(f"Ora serverului MT5: {server_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print_info(f"Ora locala: {local_time.strftime('%Y-%m-%d %H:%M:%S')}")

        diff_hours = (server_time - local_time).total_seconds() / 3600
        print_info(f"Diferenta: {diff_hours:+.1f} ore")

        # Calculate when London opens on server
        london_server_hour = 12  # Expected for Teletrade
        london_utc_hour = 8
        print()
        print_info(f"Londra deschide la 08:00 UTC = {london_server_hour}:00 ora serverului")
        print_info(f"Sesiunea se termina la 11:00 UTC = 15:00 ora serverului")

        current_server_hour = server_time.hour
        if london_server_hour <= current_server_hour < 15:
            print_ok(f"ACUM ESTE IN SESIUNE! (ora server: {current_server_hour}:00)")
        elif current_server_hour < london_server_hour:
            hours_until = london_server_hour - current_server_hour
            print_info(f"Sesiunea incepe in {hours_until} ore (la {london_server_hour}:00 server)")
        else:
            print_info(f"Sesiunea s-a terminat pentru azi (ora server: {current_server_hour}:00)")
    else:
        print_error("Nu pot obtine ora serverului")

    # 4. Verifica strategia LOCB
    print("\n[4/5] Verificare strategie LOCB...")

    strategy = LOCBStrategy(
        symbol="EURUSD",
        timeframe="M1",
        london_open_hour=12,  # Teletrade server time
        session_end_hour=15,
        risk_reward_ratio=1.5,
    )
    strategy.pip_size = 0.0001

    if strategy.initialize():
        print_ok("Strategia LOCB initializata")
        print_info(f"London open: {strategy.london_open_hour}:00 server time")
        print_info(f"Session end: {strategy.session_end_hour}:00 server time")
        print_info(f"R:R ratio: {strategy.risk_reward_ratio}:1")
    else:
        print_error("Strategia LOCB nu s-a putut initializa")
        errors.append("strategy_init")

    # 5. Verifica simbolurile
    print("\n[5/5] Verificare simboluri...")

    symbols = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY"]
    for symbol in symbols:
        if mt5.symbol_select(symbol, True):
            info = mt5.symbol_info(symbol)
            if info:
                spread = info.spread
                print_ok(f"{symbol}: spread={spread} points, min_lot={info.volume_min}")
            else:
                print_error(f"{symbol}: nu pot obtine info")
        else:
            print_error(f"{symbol}: nu este disponibil")

    # Get some candles to verify data
    print("\n  Verificare date candele...")
    candles = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M1, 0, 10)
    if candles is not None and len(candles) > 0:
        last_candle = candles[-1]
        candle_time = datetime.fromtimestamp(last_candle["time"])
        print_ok(f"Date M1 disponibile - ultima candela: {candle_time.strftime('%H:%M:%S')}")
    else:
        print_error("Nu pot obtine date M1")
        errors.append("candle_data")

    mt5.shutdown()

    # Rezultat final
    print_header("REZULTAT VERIFICARE")

    if errors:
        print_error(f"Au fost gasite {len(errors)} erori!")
        print()
        for err in errors:
            print(f"  - {err}")
        print()
        print("  Rezolva erorile inainte de a rula botul.")
        return False
    else:
        print_ok("TOATE VERIFICARILE AU TRECUT!")
        print()
        print("  Botul este gata de rulare. Foloseste comanda:")
        print()
        print("  python scripts/main.py --mode demo --strategy locb")
        print()
        print("  IMPORTANT: Porneste botul INAINTE de ora 10:00 Romania (12:00 server)")
        print("  pentru a prinde deschiderea Londrei!")
        print()
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
