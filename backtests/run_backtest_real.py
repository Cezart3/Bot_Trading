"""
Real Data Backtest Script for ORB Strategy

Uses real market data from Yahoo Finance and includes:
- News filter (avoids high-impact news days)
- ATR volatility filter
- EMA trend confirmation
- 90+ days of historical data

Usage:
    python run_backtest_real.py                    # Run on NVDA (default)
    python run_backtest_real.py --symbol AMD      # Run on AMD
    python run_backtest_real.py --symbol TSLA     # Run on TSLA
    python run_backtest_real.py --symbol US500    # Run on US500 (S&P 500)
    python run_backtest_real.py --all             # Run on all symbols
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime, timedelta
from typing import Optional
import json
from pathlib import Path

from backtesting.engine import BacktestEngine, BacktestConfig, BacktestResult
from backtesting.data_loader import DataLoader
from backtesting.report import BacktestReport
from strategies.orb_vwap_strategy import ORBVWAPStrategy, ORBVWAPConfig
from utils.logger import setup_logging, get_logger
from utils.news_filter import NewsFilter, NewsFilterConfig, NewsImpact

logger = get_logger(__name__)


# ============================================================================
#                          REAL DATA LOADER
# ============================================================================

def download_real_data(
    symbol: str,
    days: int = 90,
    interval: str = "5m",
) -> tuple[list, int]:
    """
    Download real market data from Yahoo Finance.

    Yahoo Finance limitations:
    - 5m data: max 60 days per request
    - Need to make multiple requests for 90+ days

    Args:
        symbol: Stock ticker (NVDA, AMD, TSLA, etc.)
        days: Total days to download (will be split into chunks)
        interval: Data interval

    Returns:
        Tuple of (candles list, trading days count)
    """
    try:
        import yfinance as yf
    except ImportError:
        print("Installing yfinance...")
        import subprocess
        subprocess.check_call(["pip", "install", "yfinance"])
        import yfinance as yf

    loader = DataLoader()
    all_candles = []

    # Yahoo Finance limits 5m data to 60 days per request
    chunk_size = 58  # Stay under limit
    chunks_needed = (days + chunk_size - 1) // chunk_size

    print(f"Downloading {days} days of {symbol} data in {chunks_needed} chunks...")

    end_date = datetime.now()

    for chunk in range(chunks_needed):
        chunk_end = end_date - timedelta(days=chunk * chunk_size)
        chunk_start = chunk_end - timedelta(days=chunk_size)

        # Don't go too far back
        if chunk_start < datetime.now() - timedelta(days=days + 5):
            chunk_start = datetime.now() - timedelta(days=days + 5)

        print(f"  Chunk {chunk + 1}/{chunks_needed}: {chunk_start.date()} to {chunk_end.date()}")

        candles = loader.download_yahoo_finance(
            symbol=symbol,
            start_date=chunk_start,
            end_date=chunk_end,
            interval=interval,
        )

        if candles:
            all_candles.extend(candles)
            print(f"    Downloaded {len(candles)} candles")

    # Remove duplicates and sort
    seen_timestamps = set()
    unique_candles = []
    for c in all_candles:
        if c.timestamp not in seen_timestamps:
            seen_timestamps.add(c.timestamp)
            unique_candles.append(c)

    unique_candles.sort(key=lambda x: x.timestamp)

    # Count unique trading days
    trading_days = len(set(c.timestamp.date() for c in unique_candles))

    print(f"Total: {len(unique_candles)} candles across {trading_days} trading days")

    return unique_candles, trading_days


# ============================================================================
#                          NEWS FILTER INTEGRATION
# ============================================================================

def filter_candles_by_news(
    candles: list,
    news_filter: NewsFilter,
) -> tuple[list, int]:
    """
    Remove candles from high-impact news days.

    Args:
        candles: List of candles
        news_filter: Configured news filter

    Returns:
        Tuple of (filtered candles, days removed)
    """
    # Get unique dates in candles
    all_dates = set(c.timestamp.date() for c in candles)

    # Find high impact dates
    high_impact_dates = set()
    for d in all_dates:
        events = news_filter.get_events_for_date(d)
        for event in events:
            if event.impact == NewsImpact.HIGH:
                high_impact_dates.add(d)
                break

    # Filter out candles from high impact days
    filtered_candles = [
        c for c in candles
        if c.timestamp.date() not in high_impact_dates
    ]

    days_removed = len(high_impact_dates)
    print(f"Removed {days_removed} high-impact news days from backtest")
    print(f"  News days removed: {sorted(high_impact_dates)[:5]}{'...' if days_removed > 5 else ''}")

    return filtered_candles, days_removed


# ============================================================================
#                          BACKTEST RUNNER
# ============================================================================

def run_real_backtest(
    symbol: str = "NVDA",
    days: int = 90,
    use_news_filter: bool = True,
    initial_balance: float = 25000.0,
    risk_per_trade: float = 1.0,
    risk_reward_ratio: float = 2.5,
) -> Optional[BacktestResult]:
    """
    Run backtest with real market data.

    Args:
        symbol: Stock symbol (NVDA, AMD, TSLA)
        days: Number of days to backtest
        use_news_filter: Whether to filter out news days
        initial_balance: Starting account balance
        risk_per_trade: Risk per trade in %
        risk_reward_ratio: Target R:R ratio

    Returns:
        BacktestResult or None if failed
    """
    print("\n" + "=" * 70)
    print(f"       ORB STRATEGY BACKTEST - {symbol} ({days} days) - REAL DATA")
    print("=" * 70 + "\n")

    # Determine the download symbol (map US500 to ^GSPC)
    download_symbol = symbol
    if symbol == "US500":
        download_symbol = "^GSPC"

    # Download real data
    candles, trading_days = download_real_data(download_symbol, days)

    if not candles:
        print(f"ERROR: Could not download data for {symbol}")
        return None

    # Apply news filter
    days_filtered = 0
    if use_news_filter:
        print("\nApplying news filter...")
        news_config = NewsFilterConfig(
            filter_high_impact=True,
            filter_medium_impact=False,
            currencies=["USD"],
        )
        news_filter = NewsFilter(news_config)
        news_filter.update_calendar()

        candles, days_filtered = filter_candles_by_news(candles, news_filter)

    # Print data summary
    print(f"\nData Summary:")
    print(f"  Symbol: {symbol}")
    print(f"  Period: {candles[0].timestamp.date()} to {candles[-1].timestamp.date()}")
    print(f"  Total candles: {len(candles)}")
    print(f"  Trading days: {trading_days - days_filtered}")
    print(f"  News days filtered: {days_filtered}")

    # Determine price characteristics for config
    avg_price = sum(c.close for c in candles) / len(candles)

    # Configure strategy
    strategy_config = ORBVWAPConfig(
        use_vwap_filter=True,
        use_ema_filter=True,
        use_atr_filter=True,
        use_time_filter=True,
        risk_reward_ratio=risk_reward_ratio,
        sl_buffer_dollars=avg_price * 0.001,  # 0.1% of price as buffer
    )

    strategy = ORBVWAPStrategy(
        config=strategy_config,
        symbol=symbol,
        timeframe="M5",
    )

    # Configure backtest
    config = BacktestConfig(
        initial_balance=initial_balance,
        commission_per_trade=1.0,  # $1 per trade
        slippage_pips=1.0,
        pip_size=0.01,  # $0.01 per pip for stocks
        pip_value=1.0,  # $1 per pip per share
        min_lot=1.0,  # Minimum 1 share
        max_lot=100.0,  # Maximum 100 shares
        lot_step=1.0,
    )

    # Run backtest
    print("\nRunning backtest...")
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
    report.print_trades(limit=20)

    # Export results
    report.export_json(filename=f"backtest_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    report.export_trades_csv(filename=f"trades_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    # Try to generate HTML report
    try:
        report.generate_html_report()
    except Exception as e:
        logger.warning(f"Could not generate HTML report: {e}")

    return result


def run_multi_symbol_backtest(
    symbols: list[str] = None,
    days: int = 90,
    use_news_filter: bool = True,
) -> dict:
    """
    Run backtest on multiple symbols and aggregate results.

    Args:
        symbols: List of symbols to test
        days: Number of days
        use_news_filter: Filter news days

    Returns:
        Dictionary with results per symbol and aggregate stats
    """
    if symbols is None:
        symbols = ["NVDA", "AMD", "TSLA", "US500"]

    print("\n" + "=" * 70)
    print("       MULTI-SYMBOL ORB STRATEGY BACKTEST")
    print("=" * 70)
    print(f"\nSymbols: {', '.join(symbols)}")
    print(f"Period: {days} days")
    print(f"News Filter: {'Enabled' if use_news_filter else 'Disabled'}")

    results = {}
    aggregate = {
        "total_trades": 0,
        "winning_trades": 0,
        "total_pnl": 0.0,
        "total_commission": 0.0,
        "symbols_profitable": 0,
    }

    for symbol in symbols:
        print(f"\n{'='*30} {symbol} {'='*30}")
        result = run_real_backtest(
            symbol=symbol,
            days=days,
            use_news_filter=use_news_filter,
        )

        if result:
            results[symbol] = {
                "total_trades": result.total_trades,
                "winning_trades": result.winning_trades,
                "win_rate": result.win_rate,
                "total_pnl": result.total_pnl,
                "profit_factor": result.profit_factor,
                "max_drawdown": result.max_drawdown_percent,
                "final_balance": result.final_balance,
            }

            aggregate["total_trades"] += result.total_trades
            aggregate["winning_trades"] += result.winning_trades
            aggregate["total_pnl"] += result.total_pnl
            aggregate["total_commission"] += result.total_commission

            if result.total_pnl > 0:
                aggregate["symbols_profitable"] += 1

    # Print aggregate summary
    print("\n" + "=" * 70)
    print("                    AGGREGATE RESULTS")
    print("=" * 70)

    if aggregate["total_trades"] > 0:
        aggregate["overall_win_rate"] = (aggregate["winning_trades"] / aggregate["total_trades"]) * 100
    else:
        aggregate["overall_win_rate"] = 0

    print(f"\n  Symbols Tested: {len(symbols)}")
    print(f"  Profitable Symbols: {aggregate['symbols_profitable']}/{len(symbols)}")
    print(f"\n  Total Trades: {aggregate['total_trades']}")
    print(f"  Winning Trades: {aggregate['winning_trades']}")
    print(f"  Overall Win Rate: {aggregate['overall_win_rate']:.1f}%")
    print(f"\n  Total P&L: ${aggregate['total_pnl']:,.2f}")
    print(f"  Total Commission: ${aggregate['total_commission']:,.2f}")
    print(f"  Net P&L: ${aggregate['total_pnl'] - aggregate['total_commission']:,.2f}")

    print("\n  Per Symbol Breakdown:")
    print("  " + "-" * 66)
    print(f"  {'Symbol':<8} | {'Trades':<6} | {'Win%':<6} | {'PnL':>12} | {'PF':<6} | {'MaxDD':<6}")
    print("  " + "-" * 66)

    for symbol, stats in results.items():
        print(
            f"  {symbol:<8} | "
            f"{stats['total_trades']:<6} | "
            f"{stats['win_rate']:<5.1f}% | "
            f"${stats['total_pnl']:>11,.0f} | "
            f"{stats['profit_factor']:<6.2f} | "
            f"{stats['max_drawdown']:<5.1f}%"
        )

    print("  " + "-" * 66)
    print("=" * 70 + "\n")

    # Save aggregate report
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "symbols": symbols,
        "days": days,
        "news_filter": use_news_filter,
        "aggregate": aggregate,
        "per_symbol": results,
    }

    report_path = Path("data/reports") / f"multi_symbol_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"Report saved to: {report_path}")

    return {"aggregate": aggregate, "per_symbol": results}


# ============================================================================
#                          MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run ORB Strategy Backtest with Real Data")
    parser.add_argument("--symbol", type=str, default="NVDA", help="Stock symbol (NVDA, AMD, TSLA, US500)")
    parser.add_argument("--days", type=int, default=90, help="Days of data (default: 90)")
    parser.add_argument("--all", action="store_true", help="Run on all symbols (NVDA, AMD, TSLA, US500)")
    parser.add_argument("--no-news-filter", action="store_true", help="Disable news filter")
    parser.add_argument("--balance", type=float, default=25000.0, help="Initial balance")
    parser.add_argument("--rr", type=float, default=2.5, help="Risk/Reward ratio")
    parser.add_argument("--log-level", default="WARNING", help="Logging level")

    args = parser.parse_args()

    setup_logging(level=args.log_level)

    use_news_filter = not args.no_news_filter

    if args.all:
        run_multi_symbol_backtest(
            symbols=["NVDA", "AMD", "TSLA", "US500"],
            days=args.days,
            use_news_filter=use_news_filter,
        )
    else:
        run_real_backtest(
            symbol=args.symbol.upper(),
            days=args.days,
            use_news_filter=use_news_filter,
            initial_balance=args.balance,
            risk_reward_ratio=args.rr,
        )


if __name__ == "__main__":
    main()