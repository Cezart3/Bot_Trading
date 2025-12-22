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
from strategies.orb_strategy import ORBStrategy, ORBConfig
from utils.logger import get_logger

logger = get_logger(__name__)

class PropTradingSimulator:
    def __init__(self, strategy_name, symbol, risk=0.5, rr_ratio=2.0, account_size=100000.0):
        self.strategy_name = strategy_name
        self.symbol = symbol.upper()
        self.risk = risk
        self.rr_ratio = rr_ratio
        self.account_size = account_size
        self.loader = DataLoader()
        
        # Prop Rules
        self.max_daily_dd_percent = 4.0 
        self.max_total_dd_percent = 8.0
        
        # Pip Size & Value
        if "JPY" in self.symbol:
            self.pip_size = 0.01
            self.pip_value = 6.80 
        elif "XAU" in self.symbol:
            self.pip_size = 0.01 
            self.pip_value = 1.0 
        else:
            self.pip_size = 0.0001
            self.pip_value = 10.0 
            
        config = BacktestConfig(
            initial_balance=self.account_size,
            pip_size=self.pip_size,
            pip_value=self.pip_value, 
            commission_per_trade=3.0 
        )
        self.engine = BacktestEngine(config=config)
        
        # --- PER-PAIR CONFIGURATION ---
        orb_config = ORBConfig(
            rr_ratio=self.rr_ratio,
            use_adx_filter=True,
            min_adx=20.0,
            range_start_hour=0,
            range_end_hour=8,
            trade_start_hour=8,
            trade_end_hour=18, # Coverage for NY
            max_trades_per_day=2,
            use_breakeven=True,
            be_trigger_rr=1.0,
            min_sl_pips=10.0,
            max_range_pips=150.0
        )

        if self.symbol == "USDJPY":
            # Working perfect
            orb_config.use_breakeven = True
            orb_config.be_trigger_rr = 1.0
        elif self.symbol == "EURUSD":
            # Filtered but no BE to allow breathing
            orb_config.use_breakeven = False
            orb_config.use_ema_filter = True
            orb_config.require_momentum = True
            orb_config.momentum_ratio = 0.6
            orb_config.breakout_buffer_pips = 2.0
            orb_config.min_sl_pips = 8.0
        elif self.symbol == "GBPUSD":
            # Restore Old Star settings
            orb_config.use_breakeven = False
            orb_config.min_sl_pips = 8.0
            orb_config.max_range_pips = 60.0
        elif self.symbol == "XAUUSD":
            # Fix Gold
            orb_config.use_breakeven = True
            orb_config.be_trigger_rr = 1.5
            orb_config.max_range_pips = 2500.0
            orb_config.min_sl_pips = 50.0
            orb_config.max_sl_pips = 300.0
        elif self.symbol == "GBPJPY":
            orb_config.max_range_pips = 250.0
            orb_config.min_sl_pips = 15.0
        elif self.symbol == "EURJPY":
            # EURJPY optimization
            orb_config.use_breakeven = True
            orb_config.be_trigger_rr = 1.2
            orb_config.max_range_pips = 200.0
            orb_config.min_sl_pips = 12.0

        self.strategy = ORBStrategy(symbol=self.symbol, timeframe="M15", config=orb_config, pip_size=self.pip_size)

    def resample_candles(self, m1_candles, timeframe):
        if not m1_candles: return []
        return self.loader.resample_data(m1_candles, timeframe)

    def run(self, all_m1_candles):
        m15_candles = self.resample_candles(all_m1_candles, "M15")
        self.strategy.initialize()
        
        last_date = None
        current_daily_start_equity = self.engine.balance
        high_water_mark = self.engine.balance
        failed_reason = None
        fail_date = None
        
        for i in range(100, len(m15_candles)):
            candle = m15_candles[i]
            ts = candle.timestamp
            
            if last_date != ts.date():
                last_date = ts.date()
                current_daily_start_equity = self.engine.equity
                self.strategy.reset()

            if self.engine.equity > high_water_mark:
                high_water_mark = self.engine.equity
            
            total_dd = (high_water_mark - self.engine.equity) / high_water_mark * 100
            if total_dd >= self.max_total_dd_percent:
                failed_reason = f"MAX TOTAL DD HIT: {total_dd:.2f}%"
                fail_date = ts
                break 

            daily_dd = (current_daily_start_equity - self.engine.equity) / current_daily_start_equity * 100
            if daily_dd >= self.max_daily_dd_percent:
                failed_reason = f"MAX DAILY DD HIT: {daily_dd:.2f}%"
                fail_date = ts
                break

            self.engine._update_equity(candle)
            self.engine._check_sl_tp(candle)
            
            for pos_bt in self.engine.positions:
                temp_pos = Position(
                    position_id=str(pos_bt.position_id), symbol=pos_bt.symbol,
                    entry_price=pos_bt.entry_price, quantity=pos_bt.quantity,
                    side=pos_bt.side, stop_loss=pos_bt.stop_loss,
                    take_profit=pos_bt.take_profit, opened_at=pos_bt.entry_time,
                    metadata=pos_bt.metadata
                )
                self.strategy.should_exit(temp_pos, candle.close, [candle])
                pos_bt.stop_loss = temp_pos.stop_loss
                pos_bt.metadata = temp_pos.metadata

            if not self.engine.positions:
                history = m15_candles[max(0, i-100):i+1] 
                signal = self.strategy.on_candle(candle, history) 

                if signal and signal.is_entry:
                    risk_amount = self.engine.balance * (self.risk / 100.0)
                    sl_pips = abs(signal.price - signal.stop_loss) / self.pip_size
                    if sl_pips > 0:
                        lot_size = round(risk_amount / (sl_pips * self.engine.config.pip_value), 2)
                        lot_size = max(0.01, min(lot_size, 100.0))
                        self.engine._execute_entry(signal, candle)
                        if self.engine.positions:
                            self.engine.positions[-1].quantity = lot_size

        return self.engine.trades, failed_reason, fail_date

def get_historical_files(symbol_folder):
    path = Path(symbol_folder)
    return sorted(list(path.glob("*.csv")), key=lambda f: f.name.upper().split('_')[-1]) if path.exists() else []

def generate_monthly_stats(trades):
    if not trades: return pd.DataFrame()
    df = pd.DataFrame([t.to_dict() for t in trades])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['month_year'] = df['exit_time'].dt.strftime('%Y-%m')
    return df.groupby('month_year').agg({'trade_id': 'count', 'pnl': 'sum', 'result': lambda x: (x == 'win').mean() * 100}).rename(columns={'trade_id': 'Trades', 'pnl': 'PnL ($)', 'result': 'Win Rate'})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='orb')
    args = parser.parse_args()

    target_pairs = ['GBPJPY', 'USDJPY', 'EURJPY', 'EURUSD', 'XAUUSD', 'GBPUSD']
    base_hist_path = Path(project_root) / "data" / "historical"
    pairs_to_run = [p for p in target_pairs if (base_hist_path / p.lower()).exists()]

    account_size = 100000.0
    risk_per_trade = 0.5
    
    print(f"\n" + "="*80)
    print(f"PROP TRADING SIMULATION | Account: ${account_size:,.0f} | Risk: {risk_per_trade}%")
    print(f"Strategy: ORB-Multi-Session | Pairs: {pairs_to_run}")
    print("="*80)
    
    results_summary = []

    for pair in pairs_to_run:
        print(f"\n>>> PROCESSING {pair} <<<")
        files = get_historical_files(str(Path(project_root) / "data" / "historical" / pair.lower()))
        loader = DataLoader()
        candles = []
        for f in files: candles.extend(loader.load_mt5_csv(str(f), symbol=pair))
        
        if not candles: continue
            
        sim = PropTradingSimulator(args.strategy, pair, risk=risk_per_trade, rr_ratio=2.0, account_size=account_size)
        trades, fail_reason, fail_date = sim.run(candles)
        
        total_pnl = sum(t.pnl for t in trades)
        wr = (len([t for t in trades if t.pnl > 0]) / len(trades) * 100) if trades else 0
        
        max_dd = 0
        peak = account_size
        for t, eq in sim.engine.equity_curve:
            if eq > peak: peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd: max_dd = dd
            
        results_summary.append({"Symbol": pair, "Status": "PASSED" if not fail_reason else "FAILED", "Fail Reason": fail_reason if fail_reason else "-", "Trades": len(trades), "Win Rate": f"{wr:.1f}%", "Profit": f"${total_pnl:.2f}", "Max DD": f"{max_dd:.2f}%"})
        print(f"[RESULT {pair}] {'PASSED' if not fail_reason else 'FAILED'} | Profit: ${total_pnl:.2f} | Max DD: {max_dd:.2f}%")

    print("\n" + "="*80)
    print("FINAL PROP REPORT (SUMMARY)")
    print("="*80)
    print(pd.DataFrame(results_summary).to_string(index=False))