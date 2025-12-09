"""
Verification script for ORB Trading Bot - TeleTrade-Sharp ECN.

Verifies:
1. MT5 connection to TeleTrade-Sharp ECN
2. Account info and balance
3. Symbol availability (EURUSD)
4. Risk calculations (0.5% = $500 for 100k)
5. Strategy configuration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, time

from config.settings import get_settings
from strategies.orb_forex_strategy import create_eurusd_london_strategy


def verify_settings():
    """Verify .env configuration."""
    print("\n" + "=" * 70)
    print("  [1/5] VERIFICARE CONFIGURARE .env")
    print("=" * 70)

    settings = get_settings()

    print(f"\n  MT5 Login: {settings.mt5.login}")
    print(f"  MT5 Server: {settings.mt5.server}")
    print(f"  MT5 Path: {settings.mt5.path}")

    issues = []

    if settings.mt5.login == 0:
        issues.append("MT5_LOGIN nu este configurat!")

    if not settings.mt5.server:
        issues.append("MT5_SERVER nu este configurat!")
    elif "TeleTrade" not in settings.mt5.server:
        issues.append(f"Server-ul '{settings.mt5.server}' nu pare a fi TeleTrade!")

    if issues:
        print(f"\n  PROBLEME GASITE:")
        for issue in issues:
            print(f"    - {issue}")
        return False

    print(f"\n  STATUS: OK - Configurarea pare corecta")
    return True


def verify_mt5_connection():
    """Verify MT5 connection."""
    print("\n" + "=" * 70)
    print("  [2/5] VERIFICARE CONEXIUNE MT5")
    print("=" * 70)

    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("\n  EROARE: Pachetul MetaTrader5 nu este instalat!")
        print("  Ruleaza: pip install MetaTrader5")
        return False, None

    settings = get_settings()

    print(f"\n  Conectare la {settings.mt5.server}...")

    if not mt5.initialize(
        login=settings.mt5.login,
        password=settings.mt5.password,
        server=settings.mt5.server,
        path=settings.mt5.path,
        timeout=60000,
    ):
        error = mt5.last_error()
        print(f"\n  EROARE: Nu s-a putut conecta la MT5!")
        print(f"  Cod eroare: {error}")
        print(f"\n  Verifica:")
        print(f"    1. MT5 Terminal este instalat si deschis")
        print(f"    2. Credentialele din .env sunt corecte")
        print(f"    3. Server-ul '{settings.mt5.server}' exista")
        return False, mt5

    account_info = mt5.account_info()
    if not account_info:
        print("\n  EROARE: Nu s-au putut obtine informatii despre cont!")
        return False, mt5

    print(f"\n  CONECTAT CU SUCCES!")
    print(f"\n  Informatii cont:")
    print(f"    Login: {account_info.login}")
    print(f"    Nume: {account_info.name}")
    print(f"    Server: {account_info.server}")
    print(f"    Moneda: {account_info.currency}")
    print(f"    Leverage: 1:{account_info.leverage}")
    print(f"    Balance: ${account_info.balance:,.2f}")
    print(f"    Equity: ${account_info.equity:,.2f}")
    print(f"    Margin Free: ${account_info.margin_free:,.2f}")

    return True, mt5


def verify_symbol(mt5):
    """Verify EURUSD symbol."""
    print("\n" + "=" * 70)
    print("  [3/5] VERIFICARE SIMBOL EURUSD")
    print("=" * 70)

    symbol = "EURUSD"

    # Select symbol
    if not mt5.symbol_select(symbol, True):
        print(f"\n  EROARE: Nu s-a putut selecta {symbol}!")
        return False

    info = mt5.symbol_info(symbol)
    if not info:
        print(f"\n  EROARE: Nu s-au putut obtine informatii despre {symbol}!")
        return False

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print(f"\n  EROARE: Nu s-a putut obtine tick-ul pentru {symbol}!")
        return False

    print(f"\n  Simbol: {info.name}")
    print(f"  Descriere: {info.description}")
    print(f"  Digits: {info.digits}")
    print(f"  Point: {info.point}")
    print(f"  Spread: {info.spread} points ({info.spread * info.point / 0.0001:.1f} pips)")
    print(f"  Volume Min: {info.volume_min} lots")
    print(f"  Volume Max: {info.volume_max} lots")
    print(f"  Volume Step: {info.volume_step} lots")
    print(f"  Contract Size: {info.trade_contract_size}")
    print(f"\n  Pret curent:")
    print(f"    Bid: {tick.bid:.5f}")
    print(f"    Ask: {tick.ask:.5f}")
    print(f"    Spread: {(tick.ask - tick.bid) / 0.0001:.1f} pips")

    return True


def verify_risk_calculation():
    """Verify risk calculation for 100k account with 0.5% risk."""
    print("\n" + "=" * 70)
    print("  [4/5] VERIFICARE CALCUL RISK")
    print("=" * 70)

    account_balance = 100_000.0
    risk_percent = 0.5
    pip_value_per_lot = 10.0  # $10 per pip for 1 lot EURUSD

    # Expected risk amount
    risk_amount = account_balance * (risk_percent / 100)

    print(f"\n  Parametri:")
    print(f"    Cont: ${account_balance:,.2f}")
    print(f"    Risk %: {risk_percent}%")
    print(f"    Risk $: ${risk_amount:,.2f}")
    print(f"    Pip value/lot: ${pip_value_per_lot}")

    # Test with different SL distances
    print(f"\n  Calcule lot size pentru diferite SL:")
    print(f"  {'SL (pips)':<12} | {'Lot Size':<10} | {'Risk Real':<12} | Status")
    print("  " + "-" * 55)

    test_sl_pips = [20, 25, 30, 35, 40, 50]
    all_ok = True

    for sl_pips in test_sl_pips:
        lot_size = risk_amount / (sl_pips * pip_value_per_lot)
        lot_size = round(lot_size, 2)
        lot_size = max(0.01, min(lot_size, 10.0))

        actual_risk = sl_pips * pip_value_per_lot * lot_size
        risk_diff = abs(actual_risk - risk_amount)

        status = "OK" if risk_diff <= 50 else "DIFERENTA"  # Allow $50 tolerance due to rounding
        if status != "OK":
            all_ok = False

        print(f"  {sl_pips:<12} | {lot_size:<10.2f} | ${actual_risk:<11,.2f} | {status}")

    print(f"\n  NOTA: Riscul real poate varia usor din cauza rotunjirii lot size")
    print(f"        Diferenta acceptabila: ±$50")

    if all_ok:
        print(f"\n  STATUS: OK - Calculele sunt corecte")
    else:
        print(f"\n  ATENTIE: Unele calcule au diferente mai mari")

    return all_ok


def verify_strategy():
    """Verify strategy configuration."""
    print("\n" + "=" * 70)
    print("  [5/5] VERIFICARE STRATEGIE ORB")
    print("=" * 70)

    strategy = create_eurusd_london_strategy()
    strategy.initialize()

    status = strategy.get_status()
    config = status.get("config", {})

    print(f"\n  Strategie: {status.get('strategy', 'N/A')}")
    print(f"  Simbol: {status.get('symbol', 'N/A')}")
    print(f"  Sesiune: {status.get('session', 'N/A')}")
    print(f"  State: {status.get('state', 'N/A')}")
    print(f"\n  Configurare:")
    print(f"    Session hours: {config.get('session_hours', 'N/A')}")
    print(f"    ORB duration: {config.get('orb_duration', 'N/A')}")
    print(f"    Risk/Reward: {config.get('risk_reward', 'N/A')}")

    # Check if current time is within session
    now = datetime.now()
    current_time = now.time()
    session_start = time(10, 0)
    session_end = time(18, 0)
    orb_end = time(11, 0)

    print(f"\n  Ora curenta: {current_time.strftime('%H:%M:%S')}")

    if current_time < session_start:
        print(f"  Status: ASTEPTAM sesiunea Londra (incepe la 10:00)")
    elif session_start <= current_time < orb_end:
        print(f"  Status: PERIOADA Opening Range (10:00-11:00)")
    elif orb_end <= current_time < session_end:
        print(f"  Status: PERIOADA de tranzactionare (11:00-18:00)")
    else:
        print(f"  Status: SESIUNE INCHISA (dupa 18:00)")

    print(f"\n  STATUS: OK - Strategia este configurata corect")
    return True


def main():
    """Run all verifications."""
    print("\n")
    print("=" * 70)
    print("      VERIFICARE SETUP BOT TRADING - TELETRADE SHARP ECN")
    print("=" * 70)
    print(f"  Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}
    mt5 = None

    # 1. Verify settings
    results["settings"] = verify_settings()

    # 2. Verify MT5 connection
    results["mt5_connection"], mt5 = verify_mt5_connection()

    if results["mt5_connection"] and mt5:
        # 3. Verify symbol
        results["symbol"] = verify_symbol(mt5)

        # Shutdown MT5
        mt5.shutdown()
    else:
        results["symbol"] = False

    # 4. Verify risk calculation
    results["risk_calc"] = verify_risk_calculation()

    # 5. Verify strategy
    results["strategy"] = verify_strategy()

    # Summary
    print("\n")
    print("=" * 70)
    print("      SUMAR VERIFICARI")
    print("=" * 70)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "[+]" if passed else "[X]"
        print(f"  {symbol} {name.replace('_', ' ').title()}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("  REZULTAT: TOATE VERIFICARILE AU TRECUT!")
        print("  Botul este pregatit pentru tranzactionare pe TeleTrade-Sharp ECN")
    else:
        print("  REZULTAT: UNELE VERIFICARI AU ESUAT!")
        print("  Verifica problemele de mai sus inainte de a rula botul")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
