#!/usr/bin/env python3
"""
Backtest script for ORB + VWAP strategy on US stocks.

Symbols: NVDA, AMD, TSLA
Timeframe: M5
Strategy: Opening Range Breakout with VWAP confirmation

Usage:
    python run_backtest_stocks.py [--symbol AMD] [--days 30] [--live-data]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime

from backtesting.engine import BacktestEngine, BacktestConfig
from backtesting.data_loader import DataLoader
from backtesting.report import BacktestReport
from strategies.orb_vwap_strategy import ORBVWAPStrategy, ORBVWAPConfig
from utils.logger import get_logger

logger = get_logger(__name__)


# Stock configurations with realistic starting prices
STOCK_CONFIGS = {
    "NVDA": {"price": 140.0, "volatility": 0.003},  # Higher volatility
    "AMD": {"price": 140.0, "volatility": 0.0025},
    "TSLA": {"price": 250.0, "volatility": 0.0035},  # Highest volatility
}


def run_stock_backtest(
    symbol: str = "AMD",
    days: int = 30,
    initial_balance: float = 10000.0,
    use_live_data: bool = False,
    use_vwap_filter: bool = True,
    risk_reward_ratio: float = 2.0,
) -> dict:
    """
    Run backtest for ORB + VWAP strategy on a stock.

    Args:
        symbol: Stock ticker
        days: Number of days to backtest
        initial_balance: Starting balance
        use_live_data: If True, download from Yahoo Finance
        use_vwap_filter: Use VWAP as trend filter
        risk_reward_ratio: Risk/Reward ratio

    Returns:
        Backtest results dictionary.
    """
    logger.info(f"Starting backtest for {symbol}")
    logger.info(f"Days: {days} | Balance: ${initial_balance:,.2f}")
    logger.info(f"VWAP Filter: {use_vwap_filter} | R:R: {risk_reward_ratio}")

    # Initialize components
    data_loader = DataLoader(data_dir="data")

    # Load data
    if use_live_data:
        logger.info(f"Downloading live data from Yahoo Finance...")
        candles = data_loader.download_stock_data(
            symbol=symbol,
            days=min(days, 60),  # Yahoo Finance limit
            interval="5m",
        )
    else:
        # Use sample data for testing
        config = STOCK_CONFIGS.get(symbol, {"price": 100.0, "volatility": 0.002})
        logger.info(f"Generating sample data (start: ${config['price']})...")
        candles = data_loader.generate_stock_sample_data(
            symbol=symbol,
            timeframe="M5",
            days=days,
            start_price=config["price"],
            volatility=config["volatility"],
        )

    if not candles:
        logger.error(f"No data loaded for {symbol}")
        return {}

    logger.info(f"Loaded {len(candles)} candles")

    # Configure strategy
    strategy_config = ORBVWAPConfig(
        use_vwap_filter=use_vwap_filter,
        risk_reward_ratio=risk_reward_ratio,
        sl_buffer_dollars=0.03,  # $0.03 buffer
        min_breakout_percent=0.1,
    )
    strategy = ORBVWAPStrategy(config=strategy_config)

    # Configure backtest engine for stocks
    engine_config = BacktestConfig(
        initial_balance=initial_balance,
        commission_per_trade=1.0,  # $1 per trade for stocks
        slippage_pips=0.01,  # $0.01 slippage for stocks
        pip_size=0.01,  # $0.01 = 1 cent
        pip_value=1.0,  # $1 per share per $0.01 move
    )
    engine = BacktestEngine(config=engine_config)

    # Run backtest
    result = engine.run(
        strategy=strategy,
        candles=candles,
        session_start="09:30",
        session_end="16:00",
        timezone="America/New_York",
    )

    # Print summary using BacktestReport
    report = BacktestReport(result, output_dir="data/reports")
    report.print_summary()

    # Calculate and print monthly stats
    if days >= 60:  # Only show monthly breakdown for longer backtests
        report.calculate_monthly_stats()
        report.print_monthly_summary()

    # Generate HTML report with monthly breakdown
    html_path = report.generate_html_report(include_monthly=(days >= 60))
    logger.info(f"Report saved to: {html_path}")

    # Export trades
    csv_path = report.export_trades_csv(
        filename=f"trades_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    logger.info(f"Trades saved to: {csv_path}")

    # Return as dict for compatibility
    return {
        "final_balance": result.final_balance,
        "total_trades": result.total_trades,
        "winning_trades": result.winning_trades,
        "losing_trades": result.losing_trades,
        "win_rate": result.win_rate,
        "avg_win": result.average_win,
        "avg_loss": result.average_loss,
        "profit_factor": result.profit_factor,
        "max_drawdown_pct": result.max_drawdown_percent,
        "sharpe_ratio": result.sharpe_ratio,
        "trades": result.trades,
    }


def run_multi_stock_backtest(
    symbols: list[str] = None,
    days: int = 30,
    initial_balance: float = 10000.0,
    use_live_data: bool = False,
):
    """
    Run backtest on multiple stocks and compare results.

    Args:
        symbols: List of stock tickers
        days: Number of days
        initial_balance: Starting balance per stock
        use_live_data: Download live data
    """
    if symbols is None:
        symbols = ["NVDA", "AMD", "TSLA"]

    print("\n" + "=" * 70)
    print("  MULTI-STOCK BACKTEST - ORB + VWAP STRATEGY")
    print("=" * 70)

    all_results = {}

    for symbol in symbols:
        print(f"\n{'-' * 50}")
        print(f"  Testing {symbol}...")
        print("-" * 50)

        results = run_stock_backtest(
            symbol=symbol,
            days=days,
            initial_balance=initial_balance,
            use_live_data=use_live_data,
        )
        all_results[symbol] = results

    # Summary comparison
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n  {'Symbol':<10} {'Trades':<10} {'Win Rate':<12} {'P&L':<15} {'Max DD':<10}")
    print("  " + "-" * 55)

    for symbol, results in all_results.items():
        if results:
            pnl = results["final_balance"] - initial_balance
            pnl_str = f"${pnl:+,.0f}"
            print(
                f"  {symbol:<10} "
                f"{results['total_trades']:<10} "
                f"{results['win_rate']:.1f}%{'':<7} "
                f"{pnl_str:<15} "
                f"{results['max_drawdown_pct']:.1f}%"
            )

    print("=" * 70 + "\n")

    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backtest ORB + VWAP strategy on US stocks"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="AMD",
        help="Stock symbol (NVDA, AMD, TSLA)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to backtest",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=10000.0,
        help="Initial account balance",
    )
    parser.add_argument(
        "--live-data",
        action="store_true",
        help="Download live data from Yahoo Finance",
    )
    parser.add_argument(
        "--all-stocks",
        action="store_true",
        help="Run backtest on all stocks (NVDA, AMD, TSLA)",
    )
    parser.add_argument(
        "--no-vwap",
        action="store_true",
        help="Disable VWAP filter",
    )
    parser.add_argument(
        "--rr",
        type=float,
        default=2.0,
        help="Risk/Reward ratio",
    )

    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  ORB + VWAP STRATEGY BACKTESTER")
    print("  Prop Trading Risk Management")
    print("=" * 50)
    print(f"  Max Daily DD: 4% | Max Account DD: 10%")
    print(f"  Risk per trade: 1% (0.5% when near limits)")
    print("=" * 50 + "\n")

    try:
        if args.all_stocks:
            run_multi_stock_backtest(
                symbols=["NVDA", "AMD", "TSLA"],
                days=args.days,
                initial_balance=args.balance,
                use_live_data=args.live_data,
            )
        else:
            run_stock_backtest(
                symbol=args.symbol.upper(),
                days=args.days,
                initial_balance=args.balance,
                use_live_data=args.live_data,
                use_vwap_filter=not args.no_vwap,
                risk_reward_ratio=args.rr,
            )

    except KeyboardInterrupt:
        print("\nBacktest cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Backtest error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
