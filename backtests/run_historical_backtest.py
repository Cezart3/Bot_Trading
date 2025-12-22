import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import glob

# Adaugam radacina proiectului in sys.path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from backtesting.engine import BacktestEngine, BacktestConfig
from backtesting.data_loader import DataLoader
from models.position import Position
from strategies.base_strategy import SignalType
from strategies.locb_simple import LOCBSimpleStrategy, LOCBSimpleConfig
from strategies.locb_strategy import LOCBStrategy, LOCBConfig
from strategies.orb_strategy import ORBStrategy, ORBConfig
from strategies.smc_strategy_v3 import SMCStrategyV3, SMCConfigV3, InstrumentType
from utils.logger import get_logger

logger = get_logger(__name__)

class HistoricalSimulator:
    """
    Historical backtesting simulator.

    V2 Changes (2025-12-22):
    - LOCB strategies now use M15 for opening candle, M5 for FVG/breakout
    - Data is resampled from M1 to M5 for LOCB strategies
    - Default R:R is 2.5
    """
    def __init__(self, strategy_name, symbol, risk=1.0, rr_ratio=2.5, spread_points=7.0):
        self.strategy_name = strategy_name
        self.symbol = symbol.upper()
        self.risk = risk
        self.rr_ratio = rr_ratio
        self.spread_points = spread_points # 7 points = 0.7 pips
        self.loader = DataLoader()

        # Configurare engine
        if "JPY" in self.symbol:
            pip_size = 0.01
        elif any(x in self.symbol for x in ["US30", "GER40", "US500", "USTEC", "DE40"]):
            pip_size = 0.1 if "US30" in self.symbol or "DE40" in self.symbol else 0.01
            if "US500" in self.symbol: pip_size = 0.25
        else:
            pip_size = 0.0001

        self.pip_size = pip_size

        config = BacktestEngine(config=BacktestConfig(
            initial_balance=10000.0,
            pip_size=pip_size,
            pip_value=10.0 if pip_size == 0.0001 else 1.0,
            commission_per_trade=1.5
        )).config # Extract config object

        self.engine = BacktestEngine(config=config)

        # Initializare strategie
        if strategy_name == 'locb_simple':
            # LOCB Simple V2 (M15 OC, M5 FVG/Breakout) - RELAXED
            spread_buffer_pips = self.spread_points / 10.0
            locb_config = LOCBSimpleConfig(
                oc_minutes=15,                  # M15 opening candle
                rr_ratio=self.rr_ratio,         # Default 2.5
                spread_buffer_pips=spread_buffer_pips,
                min_sl_pips=5.0,                # Lower minimum SL
                max_candles_for_retest=10,      # More time for FVG retest (10 x M5 = 50 min)
                max_trades_per_session=2        # 2 trades per session
            )
            self.strategy = LOCBSimpleStrategy(symbol=self.symbol, timeframe="M5", pip_size=pip_size, config=locb_config)

        elif strategy_name == 'locb_v3':
            # LOCB V3.1 (M15 OC, M5 confirmation)
            v3_config = LOCBConfig(
                oc_minutes=15,                  # M15 opening candle
                oc_candles=3,                   # 3 x M5 = 15 min
                rr_ratio=self.rr_ratio,         # Default 2.5
                use_advanced_confirmations=True, # CHoCH, iFVG
                use_retest_logic=False,         # Skip retest for more trades
                min_oc_range=3.0,               # Adjusted for M15
                max_oc_range=45.0,              # Adjusted for M15
                min_sl_pips=8.0,                # Minimum SL for M15
                max_sl_pips=35.0,               # Maximum SL for M15
                max_trades_per_day=4
            )
            self.strategy = LOCBStrategy(symbol=self.symbol, timeframe="M5", config=v3_config, pip_size=pip_size)

        elif strategy_name == 'locb':
            # LOCB V3.1 - RELAXED for more trades (~10-15/month target)
            v3_config = LOCBConfig(
                oc_minutes=15,
                oc_candles=3,
                rr_ratio=self.rr_ratio,
                use_advanced_confirmations=True,
                use_retest_logic=False,
                require_strong_candle=True,
                min_oc_range=2.0,
                max_oc_range=50.0,
                min_sl_pips=5.0,
                max_sl_pips=40.0,
                required_displacement=3.0,
                max_trades_per_day=4,
                max_trades_per_session=2
            )
            self.strategy = LOCBStrategy(symbol=self.symbol, timeframe="M5", config=v3_config, pip_size=pip_size)

        elif strategy_name == 'orb':
            # ORB - Opening Range Breakout (Asian Range + London Breakout)
            # Based on research best practices - RELAXED for more trades
            orb_config = ORBConfig(
                range_start_hour=2,              # 02:00 Romania (00:00 GMT)
                range_end_hour=10,               # 10:00 Romania (08:00 GMT - London Open)
                trade_start_hour=10,             # Trade at London Open
                trade_end_hour=13,               # Until 13:00 Romania
                min_range_pips=8.0,              # Reduced from 10 to 8 pips
                max_range_pips=60.0,             # Increased from 50 to 60 pips
                rr_ratio=self.rr_ratio,          # Default 2.0
                min_sl_pips=6.0,                 # Reduced from 8 to 6
                max_sl_pips=50.0,                # Increased from 40 to 50
                use_adx_filter=True,             # ADX filter is IMPORTANT for profitability
                min_adx=20.0,
                max_trades_per_day=2
            )
            self.strategy = ORBStrategy(symbol=self.symbol, timeframe="M5", config=orb_config, pip_size=pip_size)

        else:
            smc_config = SMCConfigV3(
                instrument_type=InstrumentType.FOREX,
                risk_percent=risk,
                max_trades_per_day=2,
                use_strict_kill_zones=True
            )
            self.strategy = SMCStrategyV3(symbol=self.symbol, config=smc_config)

    def resample_candles(self, m1_candles, timeframe):
        if not m1_candles: return []
        return self.loader.resample_data(m1_candles, timeframe)

    def run(self, all_m1_candles):
        """
        Run backtest simulation.

        For LOCB strategies (V2):
        - Resample M1 data to M5
        - Use M5 candles for strategy execution
        - This matches the new M15 OC / M5 FVG approach
        """
        # Resample to M5 for LOCB and ORB strategies
        if self.strategy_name.startswith('locb') or self.strategy_name in ['locb', 'orb']:
            all_candles = self.resample_candles(all_m1_candles, "M5")
            print(f"[{self.symbol}] Resampled {len(all_m1_candles)} M1 candles to {len(all_candles)} M5 candles")
        else:
            all_candles = all_m1_candles

        # SMC needs HTF
        if self.strategy_name == 'smc':
            full_h4 = self.resample_candles(all_m1_candles, "H4")
            full_h1 = self.resample_candles(all_m1_candles, "H1")
            full_m5 = self.resample_candles(all_m1_candles, "M5")
            full_d1 = self.resample_candles(all_m1_candles, "D1")

        self.strategy.initialize()
        self.strategy.pip_size = self.engine.config.pip_size

        last_date = None
        total_candles = len(all_candles)
        start_idx = 100  # Buffer for indicators (reduced for M5)

        for i in range(start_idx, total_candles):
            candle = all_candles[i]
            ts = candle.timestamp

            if last_date != ts.date():
                last_date = ts.date()
                self.strategy.reset()

            if self.strategy_name == 'smc':
                # ... (SMC logic omitted for brevity as we focus on LOCB)
                pass

            self.engine._update_equity(candle)
            self.engine._check_sl_tp(candle)

            for pos_bt in self.engine.positions:
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

                exit_signal = self.strategy.should_exit(temp_pos, candle.close, [candle])
                pos_bt.stop_loss = temp_pos.stop_loss
                pos_bt.metadata = temp_pos.metadata

                if exit_signal and exit_signal.is_exit:
                     self.engine._close_position(pos_bt, candle.close, ts, exit_signal.reason)

            if not self.engine.positions:
                if self.strategy_name.startswith('locb') or self.strategy_name in ['locb', 'orb']:
                    # Use M5 candles history for LOCB and ORB
                    history = all_candles[max(0, i-100):i+1]
                    signal = self.strategy.on_candle(candle, history)
                else:
                    signal = self.strategy.on_candle(candle, [])

                if signal and signal.is_entry:
                    self.engine._execute_entry(signal, candle)

        return self.engine.trades

def get_historical_files(symbol_folder):
    path = Path(symbol_folder)
    if not path.exists(): return []
    files = list(path.glob("*.csv"))
    return sorted(files, key=lambda f: f.name.upper().split('_')[-1])

def analyze_session(trade_df):
    hours = pd.to_datetime(trade_df['entry_time']).dt.hour
    sessions = []
    for h in hours:
        if h < 13:
            sessions.append('London')
        else:
            sessions.append('NewYork')
    return sessions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Historical Backtest (ORB/LOCB)")
    parser.add_argument('--strategy', type=str, default='orb', choices=['orb', 'locb', 'locb_simple', 'locb_v3', 'smc'])
    parser.add_argument('--pair', type=str, default='all')
    parser.add_argument('--risk', type=float, default=1.0)
    parser.add_argument('--rr', type=float, default=2.0, help="Risk:Reward ratio (default 2.0 for ORB)")
    args = parser.parse_args()

    target_pairs = ['EURUSD', 'GBPUSD', 'EURGBP']
    base_hist_path = Path(project_root) / "data" / "historical"

    pairs_to_run = []
    if args.pair.lower() == 'all':
        for p in target_pairs:
            if (base_hist_path / p.lower()).exists():
                pairs_to_run.append(p)
    else:
        pairs_to_run = [p.strip().upper() for p in args.pair.split(',')]

    # Use provided R:R or test multiple scenarios
    if args.rr:
        rr_scenarios = [args.rr]
    else:
        rr_scenarios = [2.5]  # Default to 2.5 for M15/M5
    spread_points = 7.0 # 7 puncte

    print(f"="*60)
    print(f"  LOCB BACKTEST (M15 Opening Candle / M5 FVG)")
    print(f"="*60)
    print(f"Strategy: {args.strategy}")
    print(f"Pairs: {pairs_to_run}")
    print(f"R:R Scenarios: {rr_scenarios}")
    print(f"Spread Buffer: {spread_points} points (0.7 pips)")
    
    all_results = []

    # Cache loaded data to avoid reloading for each scenario
    data_cache = {}

    for pair in pairs_to_run:
        print(f"\nIncarcare date pentru {pair}...")
        data_folder = Path(project_root) / "data" / "historical" / pair.lower()
        files = get_historical_files(str(data_folder))
        loader = DataLoader()
        candles = []
        for f in files:
            candles.extend(loader.load_mt5_csv(str(f), symbol=pair))
        data_cache[pair] = candles
        print(f"Loaded {len(candles)} candles.")

    for rr in rr_scenarios:
        print(f"\n>>> RULARE SCENARIU R:R = {rr} <<<")
        
        for pair in pairs_to_run:
            candles = data_cache.get(pair)
            if not candles: continue
            
            sim = HistoricalSimulator(args.strategy, pair, args.risk, rr_ratio=rr, spread_points=spread_points)
            trades = sim.run(candles)
            
            for t in trades:
                td = t.to_dict()
                td['symbol'] = pair
                td['rr_scenario'] = rr
                all_results.append(td)
            
            # Quick summary per pair/scenario
            wins = len([t for t in trades if t.pnl > 0])
            total = len(trades)
            wr = (wins/total*100) if total > 0 else 0
            pnl = sum(t.pnl for t in trades)
            print(f"[{pair}] R:R {rr} -> Trades: {total}, WinRate: {wr:.1f}%, PnL: {pnl:.2f}$")

    # Final Report
    if not all_results:
        print("No trades generated.")
        sys.exit(0)

    df = pd.DataFrame(all_results)
    df['session'] = analyze_session(df)
    
    # Aggregation
    report = df.groupby(['symbol', 'rr_scenario', 'session']).agg({
        'pnl': 'sum',
        'trade_id': 'count',
        'result': lambda x: (x == 'win').mean() * 100
    }).rename(columns={
        'pnl': 'Profit ($)', 
        'trade_id': 'Trades',
        'result': 'Win Rate (%)'
    })
    
    print("\n" + "="*80)
    print(f"REZULTATE FINALE - {args.strategy.upper()} (M15 OC / M5 FVG)")
    print("="*80)
    print(report)
    
    # Save
    reports_dir = Path(project_root) / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = reports_dir / f"{args.strategy}_m15m5_rr{args.rr}_{timestamp}.csv"
    df.to_csv(report_file, index=False)
    print(f"\nRaport complet salvat in: {report_file}")
