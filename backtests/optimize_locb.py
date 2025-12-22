"""
LOCB Strategy Optimizer
Grid search pentru gasirea parametrilor optimi.

Utilizare:
    python backtests/optimize_locb.py --pair eurusd --phase 1
    python backtests/optimize_locb.py --pair eurusd --phase 2 --best-ema 200 --best-disp 5
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd

# Add parent directory to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from backtesting.data_loader import DataLoader
from backtesting.engine import BacktestEngine, BacktestConfig
from models.position import Position
from strategies.locb_strategy import LOCBStrategy, LOCBConfig
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """Result from a single backtest run."""
    ema_period: int
    displacement: float
    rr_ratio: float
    max_trades_per_day: int
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    max_drawdown: float


def run_single_backtest(
    all_m1_candles: List,
    symbol: str,
    config: LOCBConfig,
    initial_balance: float = 10000.0
) -> OptimizationResult:
    """Run a single backtest with given config."""

    # Setup engine
    pip_size = 0.0001 if "JPY" not in symbol else 0.01
    engine_config = BacktestConfig(
        initial_balance=initial_balance,
        pip_size=pip_size,
        pip_value=10.0,
        commission_per_trade=1.5
    )
    engine = BacktestEngine(config=engine_config)

    # Setup strategy
    strategy = LOCBStrategy(symbol=symbol, config=config, pip_size=pip_size)
    strategy.initialize()

    # Resample data
    loader = DataLoader()
    full_h1 = loader.resample_data(all_m1_candles, "H1")
    full_m5 = loader.resample_data(all_m1_candles, "M5")

    # Run simulation
    last_date = None
    total_candles = len(all_m1_candles)
    start_idx = max(500, config.ema_period + 100)  # Ensure enough history

    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    peak_balance = initial_balance
    max_drawdown = 0.0

    for i in range(start_idx, total_candles):
        candle = all_m1_candles[i]
        ts = candle.timestamp

        # Day reset
        if last_date != ts.date():
            last_date = ts.date()
            strategy.reset()

        # Update engine
        engine._update_equity(candle)
        engine._check_sl_tp(candle)

        # Handle open positions
        for pos_bt in engine.positions:
            temp_pos = Position(
                position_id=str(pos_bt.position_id),
                symbol=pos_bt.symbol,
                entry_price=pos_bt.entry_price,
                quantity=pos_bt.quantity,
                side=pos_bt.side,
                stop_loss=pos_bt.stop_loss,
                take_profit=pos_bt.take_profit,
                opened_at=pos_bt.entry_time,
                metadata=pos_bt.metadata
            )

            exit_signal = strategy.should_exit(temp_pos, candle.close, [candle])
            pos_bt.stop_loss = temp_pos.stop_loss
            pos_bt.metadata = temp_pos.metadata

        # Generate new signals
        if not engine.positions:
            history = all_m1_candles[max(0, i - config.ema_period - 50):i + 1]
            signal = strategy.on_candle(candle, history)

            if signal and signal.is_entry:
                engine._execute_entry(signal, candle)

        # Track drawdown
        if engine.equity > peak_balance:
            peak_balance = engine.equity
        dd = (peak_balance - engine.equity) / peak_balance * 100
        if dd > max_drawdown:
            max_drawdown = dd

    # Calculate results
    for trade in engine.trades:
        if trade.pnl > 0:
            wins += 1
            gross_profit += trade.pnl
        else:
            losses += 1
            gross_loss += abs(trade.pnl)

    total_trades = len(engine.trades)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
    total_pnl = engine.balance - initial_balance

    return OptimizationResult(
        ema_period=config.ema_period,
        displacement=config.required_displacement,
        rr_ratio=config.rr_ratio,
        max_trades_per_day=config.max_trades_per_day,
        total_trades=total_trades,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_pnl=total_pnl,
        max_drawdown=max_drawdown
    )


def run_phase1_optimization(all_m1_candles: List, symbol: str) -> List[OptimizationResult]:
    """Phase 1: Find optimal EMA + Displacement."""
    print("\n" + "=" * 60)
    print("PHASE 1: EMA Period + Displacement Optimization")
    print("=" * 60)

    ema_periods = [150, 200, 250]
    displacements = [4.0, 5.0, 6.0]

    results = []
    total_tests = len(ema_periods) * len(displacements)
    current_test = 0

    for ema in ema_periods:
        for disp in displacements:
            current_test += 1
            print(f"\nTest {current_test}/{total_tests}: EMA={ema}, Disp={disp}...")

            config = LOCBConfig(
                ema_period=ema,
                required_displacement=disp,
                rr_ratio=1.8,  # Fixed for Phase 1
                max_trades_per_day=1,  # Fixed for Phase 1
                min_sl_pips=10.0,
                max_sl_pips=25.0,
                min_oc_range=3.0,
                max_oc_range=25.0,
                use_atr_filter=True,
                min_atr_pips=3.0,
                max_atr_pips=40.0
            )

            result = run_single_backtest(all_m1_candles, symbol, config)
            results.append(result)

            print(f"   Trades: {result.total_trades} | Win%: {result.win_rate:.1f}% | "
                  f"PF: {result.profit_factor:.2f} | P&L: ${result.total_pnl:.2f}")

    # Sort by profit factor
    results.sort(key=lambda x: x.profit_factor, reverse=True)

    print("\n" + "-" * 60)
    print("PHASE 1 RESULTS (sorted by Profit Factor)")
    print("-" * 60)
    print(f"{'Rank':<5} {'EMA':<6} {'Disp':<6} {'Trades':<8} {'Win%':<8} {'PF':<8} {'P&L':<12} {'MaxDD':<8}")
    print("-" * 60)

    for i, r in enumerate(results):
        print(f"{i+1:<5} {r.ema_period:<6} {r.displacement:<6.0f} {r.total_trades:<8} "
              f"{r.win_rate:<8.1f} {r.profit_factor:<8.2f} ${r.total_pnl:<11.2f} {r.max_drawdown:<8.1f}%")

    if results:
        best = results[0]
        print(f"\nBEST CONFIG: EMA={best.ema_period}, Displacement={best.displacement}")

    return results


def run_phase2_optimization(
    all_m1_candles: List,
    symbol: str,
    best_ema: int,
    best_disp: float
) -> List[OptimizationResult]:
    """Phase 2: Find optimal RR + Trades/Day with best EMA+Disp from Phase 1."""
    print("\n" + "=" * 60)
    print(f"PHASE 2: RR + Trades/Day Optimization (EMA={best_ema}, Disp={best_disp})")
    print("=" * 60)

    rr_ratios = [1.5, 1.8, 2.0]
    trades_per_day = [1, 2]

    results = []
    total_tests = len(rr_ratios) * len(trades_per_day)
    current_test = 0

    for rr in rr_ratios:
        for tpd in trades_per_day:
            current_test += 1
            print(f"\nTest {current_test}/{total_tests}: RR={rr}, Trades/Day={tpd}...")

            config = LOCBConfig(
                ema_period=best_ema,
                required_displacement=best_disp,
                rr_ratio=rr,
                max_trades_per_day=tpd,
                max_trades_per_session=1,  # Still 1 per session
                min_sl_pips=10.0,
                max_sl_pips=25.0,
                min_oc_range=3.0,
                max_oc_range=25.0,
                use_atr_filter=True,
                min_atr_pips=3.0,
                max_atr_pips=40.0
            )

            result = run_single_backtest(all_m1_candles, symbol, config)
            results.append(result)

            print(f"   Trades: {result.total_trades} | Win%: {result.win_rate:.1f}% | "
                  f"PF: {result.profit_factor:.2f} | P&L: ${result.total_pnl:.2f}")

    # Sort by profit factor
    results.sort(key=lambda x: x.profit_factor, reverse=True)

    print("\n" + "-" * 60)
    print("PHASE 2 RESULTS (sorted by Profit Factor)")
    print("-" * 60)
    print(f"{'Rank':<5} {'RR':<6} {'T/D':<6} {'Trades':<8} {'Win%':<8} {'PF':<8} {'P&L':<12} {'MaxDD':<8}")
    print("-" * 60)

    for i, r in enumerate(results):
        print(f"{i+1:<5} {r.rr_ratio:<6.1f} {r.max_trades_per_day:<6} {r.total_trades:<8} "
              f"{r.win_rate:<8.1f} {r.profit_factor:<8.2f} ${r.total_pnl:<11.2f} {r.max_drawdown:<8.1f}%")

    if results:
        best = results[0]
        print(f"\nOPTIMAL CONFIG:")
        print(f"  - EMA Period: {best_ema}")
        print(f"  - Displacement: {best_disp} pips")
        print(f"  - R:R Ratio: {best.rr_ratio}")
        print(f"  - Trades/Day: {best.max_trades_per_day}")
        print(f"\nExpected Performance:")
        print(f"  - Win Rate: {best.win_rate:.1f}%")
        print(f"  - Profit Factor: {best.profit_factor:.2f}")
        print(f"  - Total P&L: ${best.total_pnl:.2f}")
        print(f"  - Max Drawdown: {best.max_drawdown:.1f}%")

    return results


def get_historical_files(symbol_folder: str) -> List[Path]:
    """Get sorted list of historical CSV files."""
    path = Path(symbol_folder)
    if not path.exists():
        return []
    files = list(path.glob("*.csv"))
    return sorted(files, key=lambda f: f.name.upper().split('_')[-1])


def main():
    parser = argparse.ArgumentParser(description="LOCB Strategy Optimizer")
    parser.add_argument('--pair', type=str, default='eurusd', help='Currency pair')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2], help='Optimization phase')
    parser.add_argument('--best-ema', type=int, default=200, help='Best EMA from Phase 1 (for Phase 2)')
    parser.add_argument('--best-disp', type=float, default=5.0, help='Best displacement from Phase 1 (for Phase 2)')
    args = parser.parse_args()

    symbol = args.pair.upper()

    # Load data
    data_folder = Path(project_root) / "data" / "historical" / args.pair.lower()
    files = get_historical_files(str(data_folder))

    if not files:
        print(f"No data found in {data_folder}")
        print("Make sure CSV files are in: Bot_Trading/data/historical/eurusd/")
        sys.exit(1)

    print(f"Loading data for {symbol}...")
    loader = DataLoader()
    all_candles = []

    for f in files:
        print(f"  Loading {f.name}...")
        all_candles.extend(loader.load_mt5_csv(str(f), symbol=symbol))

    print(f"Total M1 candles: {len(all_candles)}")

    # Run optimization
    if args.phase == 1:
        results = run_phase1_optimization(all_candles, symbol)

        # Save results
        results_df = pd.DataFrame([{
            'ema': r.ema_period,
            'displacement': r.displacement,
            'trades': r.total_trades,
            'win_rate': r.win_rate,
            'profit_factor': r.profit_factor,
            'pnl': r.total_pnl,
            'max_dd': r.max_drawdown
        } for r in results])

        output_path = Path(project_root) / "data" / "reports" / f"optimize_locb_phase1_{args.pair}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    else:
        results = run_phase2_optimization(all_candles, symbol, args.best_ema, args.best_disp)

        # Save results
        results_df = pd.DataFrame([{
            'ema': r.ema_period,
            'displacement': r.displacement,
            'rr_ratio': r.rr_ratio,
            'trades_per_day': r.max_trades_per_day,
            'trades': r.total_trades,
            'win_rate': r.win_rate,
            'profit_factor': r.profit_factor,
            'pnl': r.total_pnl,
            'max_dd': r.max_drawdown
        } for r in results])

        output_path = Path(project_root) / "data" / "reports" / f"optimize_locb_phase2_{args.pair}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
