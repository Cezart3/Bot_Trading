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

def print_warn(text):
    print(f"  [WARN] {text}")

def main():
    print_header("VERIFICARE SETUP LOCB BOT (DUAL SESSION)")

    errors = []

    # 1. Verifica importurile
    print("\n[1/6] Verificare importuri...")
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
        from strategies.locb_strategy import LOCBStrategy, TradingSession, SessionState
        print_ok("strategies.locb_strategy importat (cu TradingSession, SessionState)")
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
    print("\n[2/6] Verificare conexiune MT5...")
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

    # 3. Verifica ora serverului si sesiunile
    print("\n[3/6] Verificare timezone si sesiuni...")

    tick = mt5.symbol_info_tick("EURUSD")
    if tick:
        server_time = datetime.fromtimestamp(tick.time)
        local_time = datetime.now()

        print_info(f"Ora serverului MT5: {server_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print_info(f"Ora locala: {local_time.strftime('%Y-%m-%d %H:%M:%S')}")

        diff_hours = (server_time - local_time).total_seconds() / 3600
        print_info(f"Diferenta: {diff_hours:+.1f} ore")

        # Session times
        print()
        print_info("SESIUNI CONFIGURATE:")
        print_info("  LONDON: 12:00-15:00 server (08:00-11:00 UTC)")
        print_info("  NY:     18:30-21:00 server (14:30-17:00 UTC)")

        current_minutes = server_time.hour * 60 + server_time.minute

        # London: 12:00-15:00
        london_start, london_end = 12 * 60, 15 * 60
        # NY: 18:30-21:00
        ny_start, ny_end = 18 * 60 + 30, 21 * 60

        in_london = london_start <= current_minutes < london_end
        in_ny = ny_start <= current_minutes < ny_end

        print()
        if in_london:
            print_ok(f"ACUM IN SESIUNE LONDON! (server: {server_time.strftime('%H:%M')})")
        elif in_ny:
            print_ok(f"ACUM IN SESIUNE NY! (server: {server_time.strftime('%H:%M')})")
        else:
            # Calculate next session
            if current_minutes < london_start:
                mins_until = london_start - current_minutes
                print_info(f"Urmatoarea sesiune: LONDON in {mins_until // 60}h {mins_until % 60}m")
            elif current_minutes < ny_start:
                mins_until = ny_start - current_minutes
                print_info(f"Urmatoarea sesiune: NY in {mins_until // 60}h {mins_until % 60}m")
            else:
                mins_until = (24 * 60 - current_minutes) + london_start
                print_info(f"Urmatoarea sesiune: LONDON (maine) in {mins_until // 60}h {mins_until % 60}m")
    else:
        print_error("Nu pot obtine ora serverului")

    # 4. Verifica strategia LOCB
    print("\n[4/6] Verificare strategie LOCB...")

    strategy = LOCBStrategy(
        symbol="EURUSD",
        timeframe="M1",
    )
    strategy.pip_size = 0.0001

    if strategy.initialize():
        print_ok("Strategia LOCB initializata")

        # Check sessions
        london = strategy.sessions.get("london")
        ny = strategy.sessions.get("ny")

        if london and ny:
            print_ok(f"Sesiune London: {london.open_hour}:{london.open_minute:02d} - {london.end_hour}:00")
            print_ok(f"Sesiune NY: {ny.open_hour}:{ny.open_minute:02d} - {ny.end_hour}:00")
        else:
            print_error("Sesiunile nu sunt configurate corect")
            errors.append("sessions")

        print_info(f"R:R range: {strategy.min_rr_ratio}-{strategy.max_rr_ratio}:1 (dinamic)")
        print_info(f"Max trades/zi: {strategy.max_trades_per_day}")
        print_info(f"Min signal score: {strategy.min_signal_score}")
    else:
        print_error("Strategia LOCB nu s-a putut initializa")
        errors.append("strategy_init")

    # 5. Verifica simbolurile si spread
    print("\n[5/6] Verificare simboluri si spread...")

    symbols = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY"]
    for symbol in symbols:
        if mt5.symbol_select(symbol, True):
            info = mt5.symbol_info(symbol)
            tick = mt5.symbol_info_tick(symbol)
            if info and tick:
                spread_points = info.spread
                spread_price = tick.ask - tick.bid
                spread_pips = spread_price / 0.0001 if "JPY" not in symbol else spread_price / 0.01
                print_ok(f"{symbol}: spread={spread_pips:.1f} pips, min_lot={info.volume_min}")
            else:
                print_error(f"{symbol}: nu pot obtine info")
        else:
            print_error(f"{symbol}: nu este disponibil")

    # 6. Test scoring si liquidity detection
    print("\n[6/6] Verificare functii avansate...")

    try:
        from strategies.locb_strategy import SignalScore, LiquidityLevel, M1CandleData

        # Test SignalScore
        score = SignalScore()
        score.confirmation_score = 1.0
        score.sl_distance_score = 0.85
        score.breakout_strength_score = 0.75
        score.retest_quality_score = 0.7
        score.timing_score = 1.0
        total = score.calculate_total()
        print_ok(f"SignalScore functional (test score: {total:.2f})")

        # Test LiquidityLevel
        level = LiquidityLevel(
            price=1.17500,
            type="swing_high",
            strength=0.8,
            candle_time=datetime.now(),
            touches=2
        )
        print_ok(f"LiquidityLevel functional (test: {level.type} @ {level.price})")

        print_ok("Toate functiile avansate sunt functionale")
    except Exception as e:
        print_error(f"Eroare la testare functii avansate: {e}")
        errors.append("advanced_functions")

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
        print("  CARACTERISTICI NOI:")
        print("  - Dual session: LONDON (12:00-15:00) + NY (18:30-21:00)")
        print("  - Dynamic R:R bazat pe lichiditate (1.5-4:1)")
        print("  - Signal scoring pentru selectie calitativa")
        print("  - SL ajustat pentru spread")
        print("  - Max 2 trades/zi (1 per sesiune)")
        print()
        print("  IMPORTANT: Porneste botul inainte de ora 10:00 Romania (12:00 server)")
        print("  pentru sesiunea de Londra sau inainte de 16:30 Romania (18:30 server)")
        print("  pentru sesiunea de NY!")
        print()
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
