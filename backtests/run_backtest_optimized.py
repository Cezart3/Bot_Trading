import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Setup Paths
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from backtesting.engine import BacktestEngine, BacktestConfig
from backtesting.data_loader import DataLoader
from strategies.orb_optimized import ORBOptimizedStrategy, ORBOptimizedConfig, SessionConfig
from strategies.orb_strategy import ORBStrategy, ORBConfig # Import Original Strategy
from models.position import Position
from utils.logger import get_logger

logger = get_logger(__name__)

class HybridBacktestRunner:
    def __init__(self, symbol, risk=0.65, account_size=100000.0):
        self.symbol = symbol.upper()
        self.risk = risk
        self.account_size = account_size
        self.loader = DataLoader()
        
        # Prop Rules
        self.max_daily_dd = 4.0
        self.max_total_dd = 8.0
        
        # Pip Config
        is_jpy = "JPY" in self.symbol
        self.pip_size = 0.01 if is_jpy else 0.0001
        self.pip_value = 6.80 if is_jpy else 10.0
        
        config = BacktestConfig(
            initial_balance=self.account_size,
            pip_size=self.pip_size,
            pip_value=self.pip_value,
            commission_per_trade=3.0
        )
        self.engine = BacktestEngine(config=config)
        
        # --- HYBRID STRATEGY SELECTION ---
        
        if self.symbol == "EURUSD":
            # STRATEGY A: THE "GOLD MINE" (Dual Session Optimized)
            london = SessionConfig("London", 0, 8, 13, min_range_pips=5, max_range_pips=100)
            ny = SessionConfig("NewYork", 12, 14, 20, min_range_pips=5, max_range_pips=80)
            
            orb_config = ORBOptimizedConfig(
                risk_percent=risk,
                rr_ratio=2.0,
                use_breakeven=True,
                be_trigger_rr=1.0, 
                min_sl_pips=10.0,
                max_sl_pips=40.0,
                use_ema_filter=False, 
                ema_period=200,
                sessions=[london, ny]
            )
            self.strategy = ORBOptimizedStrategy(self.symbol, "M15", orb_config, pip_size=self.pip_size)
            self.strategy_type = "OPTIMIZED"
            
        else:
            # STRATEGY B: THE "OLD RELIABLE" (Original ORB Logic)
            # Replicating the exact config from run_historical_backtest.py
            original_config = ORBConfig(
                rr_ratio=2.0,
                use_adx_filter=True,
                min_adx=20.0,
                range_start_hour=0,
                range_end_hour=8,
                trade_start_hour=8,
                trade_end_hour=18,
                max_trades_per_day=2,
                use_breakeven=True,
                be_trigger_rr=1.0,
                min_sl_pips=10.0,
                max_range_pips=150.0
            )
            
            # Specific Original Tweaks
            if self.symbol == "USDJPY":
                original_config.use_breakeven = True
            elif self.symbol == "GBPJPY":
                original_config.max_range_pips = 250.0
                original_config.min_sl_pips = 15.0
            elif self.symbol == "EURJPY":
                original_config.be_trigger_rr = 1.2
                original_config.max_range_pips = 200.0
                original_config.min_sl_pips = 12.0

            self.strategy = ORBStrategy(self.symbol, "M15", original_config, pip_size=self.pip_size)
            self.strategy_type = "ORIGINAL"

    def run(self, candles):
        # Ensure M15 data
        m15_candles = self.loader.resample_data(candles, "M15")
        
        self.strategy.initialize()
        
        high_water_mark = self.engine.balance
        daily_start_equity = self.engine.balance
        last_date = None
        
        fail_reason = None
        
        for i in range(100, len(m15_candles)):
            candle = m15_candles[i]
            ts = candle.timestamp
            
            if last_date != ts.date():
                last_date = ts.date()
                daily_start_equity = self.engine.equity
                if hasattr(self.strategy, 'reset'): self.strategy.reset() # Reset for Original
                # Optimized resets internally
            
            if self.engine.equity > high_water_mark: high_water_mark = self.engine.equity
            
            dd_total = (high_water_mark - self.engine.equity) / high_water_mark * 100
            dd_daily = (daily_start_equity - self.engine.equity) / daily_start_equity * 100
            
            if dd_total >= self.max_total_dd:
                fail_reason = f"Total DD {dd_total:.1f}%"
                break
            if dd_daily >= self.max_daily_dd:
                fail_reason = f"Daily DD {dd_daily:.1f}%"
                break
                
            self.engine._update_equity(candle)
            self.engine._check_sl_tp(candle)
            
            for p in self.engine.positions:
                pos_obj = Position(
                    str(p.position_id), p.symbol, p.entry_price, p.quantity, 
                    p.side, p.stop_loss, p.take_profit, p.entry_time, p.metadata
                )
                self.strategy.should_exit(pos_obj, candle.close, [candle])
                p.stop_loss = pos_obj.stop_loss
                p.metadata = pos_obj.metadata

            if not self.engine.positions:
                # Handle different signature of on_candle if necessary, but usually consistent
                history = m15_candles[max(0, i-300):i+1]
                signal = self.strategy.on_candle(candle, history)
                
                if signal:
                    balance = self.engine.balance
                    risk_amt = balance * (self.risk / 100)
                    sl_pips = abs(signal.price - signal.stop_loss) / self.pip_size
                    if sl_pips > 0:
                        lot = round(risk_amt / (sl_pips * self.engine.config.pip_value), 2)
                        lot = max(0.01, min(lot, 100.0))
                        self.engine._execute_entry(signal, candle)
                        if self.engine.positions:
                            self.engine.positions[-1].quantity = lot

        return self.engine.trades, fail_reason

def get_historical_files(symbol_folder):
    path = Path(symbol_folder)
    return sorted(list(path.glob("*.csv")), key=lambda f: f.name.upper().split('_')[-1]) if path.exists() else []

if __name__ == "__main__":
    pairs = ['USDJPY', 'EURUSD', 'GBPJPY', 'EURJPY']
    base_path = Path(project_root) / "data" / "historical"
    
    print("="*60)
    print("HYBRID PORTFOLIO BACKTEST")
    print("EURUSD: Optimized Dual Session | OTHERS: Original Asian Session")
    print("="*60)
    
    summary = []
    
    for pair in pairs:
        files = get_historical_files(str(base_path / pair.lower()))
        if not files: continue
        
        loader = DataLoader()
        all_candles = []
        for f in files: all_candles.extend(loader.load_mt5_csv(str(f), symbol=pair))
        
        if not all_candles: continue
        
        print(f"Running {pair}...", end=" ")
        
        # FINAL RISK SETTINGS (Based on High-Yield Simulation)
        if pair == "EURUSD":
            pair_risk = 0.60 # Yields $30,250.91 Profit
        elif pair == "USDJPY":
            pair_risk = 0.70 # Yields $34,203.65 Profit
        else:
            pair_risk = 0.65 # Stable for JPY pairs

        runner = HybridBacktestRunner(pair, risk=pair_risk)
        print(f"[{runner.strategy_type} | Risk: {pair_risk}%]")
        
        trades, fail = runner.run(all_candles)
        
        pnl = sum(t.pnl for t in trades)
        wr = (len([t for t in trades if t.pnl > 0]) / len(trades) * 100) if trades else 0
        
        max_dd = 0
        peak = 100000
        for t, eq in runner.engine.equity_curve:
            if eq > peak: peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd: max_dd = dd
        
        summary.append({
            "Symbol": pair,
            "Type": runner.strategy_type,
            "Trades": len(trades),
            "WinRate": f"{wr:.1f}%",
            "Profit": f"${pnl:.2f}",
            "MaxDD": f"{max_dd:.1f}%",
            "Status": "PASS" if not fail else "FAIL"
        })
        
    print("\nFINAL RESULTS:")
    print(pd.DataFrame(summary).to_string(index=False))
