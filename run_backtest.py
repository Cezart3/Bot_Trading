"""
Run Backtest Script for ORB Strategy

Usage:
    python run_backtest.py                    # Use sample data
    python run_backtest.py --yahoo ES=F       # Download from Yahoo Finance
    python run_backtest.py --csv data.csv     # Use CSV file
"""

import argparse
from datetime import datetime, timedelta

from backtesting.engine import BacktestEngine, BacktestConfig
from backtesting.data_loader import DataLoader
from backtesting.report import BacktestReport
from strategies.orb_strategy import ORBStrategy
from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def run_backtest_sample():
    """Run backtest with sample generated data."""
    print("\n" + "=" * 70)
    print("       ORB STRATEGY BACKTEST - Sample Data (365 days)")
    print("=" * 70 + "\n")

    # Generate sample data
    loader = DataLoader()
    candles = loader.generate_sample_data(
        symbol="US500",
        timeframe="M5",
        days=90,
        start_price=5000.0,
        volatility=0.0008,
        session_start="09:30",
        session_end="16:00",
    )

    print(f"Generated {len(candles)} candles for testing\n")

    # Create strategy
    strategy = ORBStrategy(
        symbol="US500",
        timeframe="M5",
        session_start="09:30",
        session_end="16:00",
        timezone="America/New_York",
        range_minutes=5,
        breakout_buffer_pips=2.0,
        risk_reward_ratio=2.0,
        min_range_pips=3.0,
        max_range_pips=30.0,
        max_trades_per_day=1,
    )

    # Configure backtest
    config = BacktestConfig(
        initial_balance=10000.0,
        commission_per_trade=2.0,
        slippage_pips=0.5,
        pip_size=0.25,  # For US500/ES mini
        pip_value=12.50,  # $12.50 per tick per contract
        min_lot=1.0,
        max_lot=10.0,
        lot_step=1.0,
    )

    # Run backtest
    engine = BacktestEngine(config)
    result = engine.run(
        strategy=strategy,
        candles=candles,
        session_start="09:30",
        session_end="16:00",
        timezone="America/New_York",
    )

    # Generate report
    report = BacktestReport(result)
    report.print_summary()
    report.print_trades(limit=10)

    # Export results
    report.export_json()
    report.export_trades_csv()

    try:
        report.generate_html_report()
    except Exception as e:
        logger.warning(f"Could not generate HTML report: {e}")

    return result


def run_backtest_yahoo(symbol: str, days: int = 60):
    """Run backtest with Yahoo Finance data."""
    print("\n" + "=" * 70)
    print(f"       ORB STRATEGY BACKTEST - {symbol} ({days} days)")
    print("=" * 70 + "\n")

    try:
        import yfinance
    except ImportError:
        print("Installing yfinance...")
        import subprocess
        subprocess.check_call(["pip", "install", "yfinance"])
        import yfinance

    # Download data
    loader = DataLoader()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    candles = loader.download_yahoo_finance(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval="5m",
    )

    if not candles:
        print("ERROR: Could not download data. Try a different symbol.")
        print("Suggestions: ES=F (E-mini S&P), NQ=F (E-mini NASDAQ), ^GSPC (S&P 500 index)")
        return None

    print(f"Downloaded {len(candles)} candles from Yahoo Finance\n")

    # Determine pip size based on symbol
    if "ES" in symbol or "500" in symbol or "SPY" in symbol:
        pip_size = 0.25
        pip_value = 12.50
    elif "NQ" in symbol or "QQQ" in symbol:
        pip_size = 0.25
        pip_value = 5.0
    else:
        pip_size = 0.01
        pip_value = 1.0

    # Create strategy
    strategy = ORBStrategy(
        symbol=symbol,
        timeframe="M5",
        session_start="09:30",
        session_end="16:00",
        timezone="America/New_York",
        range_minutes=5,
        breakout_buffer_pips=2.0,
        risk_reward_ratio=2.0,
        min_range_pips=3.0,
        max_range_pips=50.0,
        max_trades_per_day=1,
    )

    # Configure backtest
    config = BacktestConfig(
        initial_balance=10000.0,
        commission_per_trade=2.0,
        slippage_pips=0.5,
        pip_size=pip_size,
        pip_value=pip_value,
        min_lot=1.0,
        max_lot=10.0,
        lot_step=1.0,
    )

    # Run backtest
    engine = BacktestEngine(config)
    result = engine.run(
        strategy=strategy,
        candles=candles,
        session_start="09:30",
        session_end="16:00",
        timezone="America/New_York",
    )

    # Generate report
    report = BacktestReport(result)
    report.print_summary()
    report.print_trades(limit=15)

    # Export results
    report.export_json()
    report.export_trades_csv()
    report.generate_html_report()

    # Try to plot
    try:
        report.plot_equity_curve(show=False, save=True)
    except Exception as e:
        logger.warning(f"Could not generate chart: {e}")

    return result


def run_parameter_optimization():
    """Run multiple backtests with different parameters to find optimal settings."""
    print("\n" + "=" * 70)
    print("       ORB STRATEGY PARAMETER OPTIMIZATION")
    print("=" * 70 + "\n")

    # Generate sample data
    loader = DataLoader()
    candles = loader.generate_sample_data(
        symbol="US500",
        timeframe="M5",
        days=60,
        start_price=5000.0,
        volatility=0.0008,
    )

    # Parameters to test
    range_minutes_options = [5, 10, 15]
    rr_ratio_options = [1.5, 2.0, 2.5, 3.0]
    buffer_options = [1.0, 2.0, 3.0]

    results = []

    config = BacktestConfig(
        initial_balance=10000.0,
        commission_per_trade=2.0,
        slippage_pips=0.5,
        pip_size=0.25,
        pip_value=12.50,
        min_lot=1.0,
        max_lot=10.0,
        lot_step=1.0,
    )

    total_tests = len(range_minutes_options) * len(rr_ratio_options) * len(buffer_options)
    current_test = 0

    for range_mins in range_minutes_options:
        for rr in rr_ratio_options:
            for buffer in buffer_options:
                current_test += 1
                print(f"\rTesting {current_test}/{total_tests}: Range={range_mins}min, R:R={rr}, Buffer={buffer}...", end="")

                strategy = ORBStrategy(
                    symbol="US500",
                    timeframe="M5",
                    range_minutes=range_mins,
                    risk_reward_ratio=rr,
                    breakout_buffer_pips=buffer,
                    min_range_pips=3.0,
                    max_range_pips=30.0,
                )

                engine = BacktestEngine(config)
                result = engine.run(strategy, candles)

                results.append({
                    "range_minutes": range_mins,
                    "risk_reward": rr,
                    "buffer_pips": buffer,
                    "total_trades": result.total_trades,
                    "win_rate": result.win_rate,
                    "total_pnl": result.total_pnl,
                    "profit_factor": result.profit_factor,
                    "max_drawdown": result.max_drawdown_percent,
                })

    print("\n\n" + "=" * 70)
    print("OPTIMIZATION RESULTS (Top 10 by Profit Factor)")
    print("=" * 70)

    # Sort by profit factor
    results.sort(key=lambda x: x["profit_factor"], reverse=True)

    print(f"\n{'Range':>6} | {'R:R':>5} | {'Buffer':>6} | {'Trades':>6} | {'Win%':>6} | {'PnL':>10} | {'PF':>6} | {'DD%':>6}")
    print("-" * 70)

    for r in results[:10]:
        print(
            f"{r['range_minutes']:>6} | "
            f"{r['risk_reward']:>5.1f} | "
            f"{r['buffer_pips']:>6.1f} | "
            f"{r['total_trades']:>6} | "
            f"{r['win_rate']:>5.1f}% | "
            f"${r['total_pnl']:>9,.0f} | "
            f"{r['profit_factor']:>6.2f} | "
            f"{r['max_drawdown']:>5.1f}%"
        )

    print("\n" + "=" * 70)

    # Return best result
    return results[0] if results else None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run ORB Strategy Backtest")
    parser.add_argument("--yahoo", type=str, help="Yahoo Finance symbol (e.g., ES=F)")
    parser.add_argument("--days", type=int, default=60, help="Days of data (default: 60)")
    parser.add_argument("--csv", type=str, help="Path to CSV file with historical data")
    parser.add_argument("--optimize", action="store_true", help="Run parameter optimization")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    setup_logging(level=args.log_level)

    if args.optimize:
        run_parameter_optimization()
    elif args.yahoo:
        run_backtest_yahoo(args.yahoo, args.days)
    elif args.csv:
        print("CSV backtest not yet implemented. Use --yahoo or run without args for sample data.")
    else:
        run_backtest_sample()


if __name__ == "__main__":
    main()
