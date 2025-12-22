"""
Multi-Instrument Backtest for SMC V4 Strategy.
"""

import sys
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.data_loader import DataLoader
from models.candle import Candle
from strategies.smc_strategy_v3 import SMCStrategyV3, SMCConfigV3, InstrumentType
from strategies.base_strategy import SignalType  # Added for CLOSE_POSITION
from scripts.run_smc_v3 import SYMBOL_CONFIGS
from utils.logger import get_logger, setup_logging # Import setup_logging

logger = get_logger(__name__)

@dataclass
class InstrumentConfig:
    """Configuration for each instrument's market characteristics."""
    symbol: str
    instrument_type: InstrumentType
    pip_size: float
    pip_value: float
    typical_spread: float

MARKET_DATA = {
    "EURUSD": InstrumentConfig("EURUSD", InstrumentType.FOREX, 0.0001, 10.0, 0.8),
    "GBPUSD": InstrumentConfig("GBPUSD", InstrumentType.FOREX, 0.0001, 10.0, 1.0),
    "AUDUSD": InstrumentConfig("AUDUSD", InstrumentType.FOREX, 0.0001, 10.0, 0.9),
    "US30": InstrumentConfig("US30", InstrumentType.INDEX, 1.0, 1.0, 2.0),
    "GER30": InstrumentConfig("GER30", InstrumentType.INDEX, 0.1, 1.0, 1.2),
}

@dataclass
class Trade:
    """Represents a completed trade."""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    pnl: float
    pnl_pips: float
    result: str
    exit_reason: str
    session: str

@dataclass
class BacktestResult:
    """Results from backtesting one instrument."""
    symbol: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    total_pnl_pips: float
    avg_win: float
    avg_loss: float
    avg_rr: float
    max_drawdown: float
    max_consecutive_losses: int
    trades: List[Trade] = field(default_factory=list)

def create_strategy_for_symbol(symbol: str, risk_percent: float = 0.5) -> SMCStrategyV3:
    """Creates an SMCStrategyV3 instance for a given symbol with its specific configuration."""
    config_data = SYMBOL_CONFIGS.get(symbol, SYMBOL_CONFIGS.get("GBPUSD", {})).copy()

    # Create a clean config dict by mapping keys from SYMBOL_CONFIGS to SMCConfigV3 fields
    clean_config = {}
    
    # Direct mapping
    direct_mapping_keys = [
        'min_sl_pips', 'max_sl_pips', 'poi_min_score', 
        'adx_trending', 'min_rr', 'ob_min_impulse_atr'
    ]
    for key in direct_mapping_keys:
        if key in config_data:
            clean_config[key] = config_data[key]

    # Renamed keys
    if 'type' in config_data:
        clean_config['instrument_type'] = config_data['type']
    if 'require_sweep' in config_data:
        clean_config['require_sweep_for_low_score'] = config_data['require_sweep']

    # Create config object and set runtime risk
    config = SMCConfigV3(**clean_config)
    config.risk_percent = risk_percent

    return SMCStrategyV3(symbol=symbol, timeframe="M5", config=config)

class SMCBacktestEngine:
    def __init__(self, symbol: str, market_config: InstrumentConfig, initial_balance: float = 10000.0, risk_per_trade: float = 0.5):
        self.symbol = symbol
        self.market_config = market_config
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.balance = initial_balance
        self.equity_curve = [initial_balance]
        self.trades: List[Trade] = []
        self.strategy = create_strategy_for_symbol(symbol, risk_per_trade)

    def run_backtest(self, candle_data: dict) -> BacktestResult:
        self.strategy.initialize()
        position = None
        position_risk_amount = 0
        peak_balance = self.initial_balance
        max_drawdown = 0
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        m1_candles = candle_data.get("m1", [])
        if not m1_candles:
            print(f"Error: No M1 candles available for {self.symbol}. Cannot run backtest.")
            return BacktestResult(self.symbol, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # Return empty result

        m5_candles = candle_data.get("m5", [])
        if not m5_candles:
            print(f"Error: No M5 candles available for {self.symbol}. Cannot run backtest.")
            return BacktestResult(self.symbol, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # Return empty result

        h1_candles = candle_data.get("h1", [])
        h4_candles = candle_data.get("h4", [])
        daily_candles = candle_data.get("d1", [])

        m5_idx, h1_idx, h4_idx, daily_idx = 0, 0, 0, 0

        # Iterate through M1 candles as the primary processing unit
        for i, candle in enumerate(m1_candles):
            # Skip initial candles for indicator calculations
            if i < 200: continue # Needs at least 200 candles for initial calculations (e.g., ADX)

            # Advance indices for higher timeframes based on current M1 candle's timestamp
            while m5_idx < len(m5_candles) and m5_candles[m5_idx].timestamp <= candle.timestamp: m5_idx += 1
            while h1_idx < len(h1_candles) and h1_candles[h1_idx].timestamp <= candle.timestamp: h1_idx += 1
            while h4_idx < len(h4_candles) and h4_candles[h4_idx].timestamp <= candle.timestamp: h4_idx += 1
            while daily_idx < len(daily_candles) and daily_candles[daily_idx].timestamp.date() < candle.timestamp.date(): daily_idx += 1

            self.strategy.set_candles(
                m1_candles=m1_candles[:i + 1], # Pass current and past M1 candles
                m5_candles=m5_candles[:m5_idx],
                h1_candles=h1_candles[:h1_idx],
                h4_candles=h4_candles[:h4_idx],
                daily_candles=daily_candles[:daily_idx]
            )
            self.strategy.update_spread(self.market_config.typical_spread * self.market_config.pip_size)

            if position:
                # Calculate pnl for potential exit (required for strategy exits and consistent tracking)
                pnl_on_exit = 0.0
                if position['direction'] == "long":
                    # PnL in USD based on pip value, lot size implied by risk amount
                    pnl_on_exit = (candle.close - position['entry']) / self.market_config.pip_size * self.market_config.pip_value * (position_risk_amount / abs(position['entry'] - position['sl']) / self.market_config.pip_value * self.market_config.pip_size)
                else: # Short
                    pnl_on_exit = (position['entry'] - candle.close) / self.market_config.pip_size * self.market_config.pip_value * (position_risk_amount / abs(position['entry'] - position['sl']) / self.market_config.pip_value * self.market_config.pip_size)

                # Check for strategy-based exit
                exit_signal = self.strategy.should_exit(position, candle.close, m1_candles[:i + 1])
                if exit_signal and exit_signal.signal_type == SignalType.CLOSE_POSITION:
                    self._close_position(candle.timestamp, exit_signal.price, pnl_on_exit, position, exit_signal.reason)
                    position = None
                    if pnl_on_exit < 0: consecutive_losses += 1
                    else: consecutive_losses = 0

                # Check SL/TP from the broker perspective (hit during current candle)
                elif position['direction'] == "long":
                    if candle.low <= position['sl']: # Use candle.low for SL hit
                        pnl = -position_risk_amount
                        self._close_position(candle.timestamp, position['sl'], pnl, position, "sl_hit")
                        position = None; consecutive_losses += 1
                    elif candle.high >= position['tp']: # Use candle.high for TP hit
                        pnl = position_risk_amount * abs(position['tp'] - position['entry']) / abs(position['entry'] - position['sl'])
                        self._close_position(candle.timestamp, position['tp'], pnl, position, "tp_hit")
                        position = None; consecutive_losses = 0
                else: # Short
                    if candle.high >= position['sl']: # Use candle.high for SL hit
                        pnl = -position_risk_amount
                        self._close_position(candle.timestamp, position['sl'], pnl, position, "sl_hit")
                        position = None; consecutive_losses += 1
                    elif candle.low <= position['tp']: # Use candle.low for TP hit
                        pnl = position_risk_amount * abs(position['entry'] - position['tp']) / abs(position['sl'] - position['entry'])
                        self._close_position(candle.timestamp, position['tp'], pnl, position, "tp_hit")
                        position = None; consecutive_losses = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

            if not position:
                # Pass the current M1 candle, and all M1 candles up to this point
                signal = self.strategy.on_candle(candle, m1_candles[:i + 1])
                if signal and signal.is_entry:
                    position_risk_amount = self.balance * (self.risk_per_trade / 100)
                    position = {'entry': signal.price, 'sl': signal.stop_loss, 'tp': signal.take_profit, 'direction': "long" if signal.is_long else "short", 'entry_time': candle.timestamp, 'session': signal.metadata.get("session", ""), 'pnl': 0}
            
            # Update equity curve at each step
            current_equity = self.balance
            if position:
                if position['direction'] == "long":
                    current_equity += (candle.close - position['entry']) / self.market_config.pip_size * self.market_config.pip_value * (position_risk_amount / abs(position['entry'] - position['sl']) / self.market_config.pip_value * self.market_config.pip_size)
                else:
                    current_equity += (position['entry'] - candle.close) / self.market_config.pip_size * self.market_config.pip_value * (position_risk_amount / abs(position['entry'] - position['sl']) / self.market_config.pip_value * self.market_config.pip_size)
            
            self.equity_curve.append(current_equity)

            # Update drawdown
            peak_balance = max(peak_balance, current_equity)
            max_drawdown = max(max_drawdown, (peak_balance - current_equity) / peak_balance * 100)


        return self._calculate_results(max_drawdown, max_consecutive_losses)

    def _close_position(self, exit_time: datetime, exit_price: float, pnl: float, trade_data: dict, exit_reason: str):
        self.balance += pnl
        pnl_pips = (exit_price - trade_data['entry']) / self.market_config.pip_size
        if trade_data['direction'] == 'short': pnl_pips *= -1
        
        self.trades.append(Trade(entry_time=trade_data['entry_time'], exit_time=exit_time, direction=trade_data['direction'], entry_price=trade_data['entry'], exit_price=exit_price, sl=trade_data['sl'], tp=trade_data['tp'], pnl=pnl, pnl_pips=pnl_pips, result="win" if pnl > 0 else "loss", exit_reason=exit_reason, session=trade_data['session']))
        self.strategy.on_position_closed(None, pnl)

    def _calculate_results(self, max_drawdown, max_consecutive_losses):
        wins = sum(1 for t in self.trades if t.pnl > 0)
        losses = len(self.trades) - wins
        total_wins = sum(t.pnl for t in self.trades if t.pnl > 0)
        total_losses = abs(sum(t.pnl for t in self.trades if t.pnl <= 0))
        
        return BacktestResult(
            symbol=self.symbol, total_trades=len(self.trades), wins=wins, losses=losses,
            win_rate=(wins / len(self.trades) * 100 if self.trades else 0),
            profit_factor=(total_wins / total_losses if total_losses > 0 else 0),
            total_pnl=self.balance - self.initial_balance, total_pnl_pips=sum(t.pnl_pips for t in self.trades),
            avg_win=(total_wins / wins if wins > 0 else 0), avg_loss=(total_losses / losses if losses > 0 else 0),
            avg_rr=((total_wins / wins) / (total_losses / losses) if wins > 0 and losses > 0 else 0),
            max_drawdown=max_drawdown, max_consecutive_losses=max_consecutive_losses, trades=self.trades
        )

def run_multi_instrument_backtest(symbols: List[str], months: int, initial_balance: float, risk_per_trade: float, source: str):
    if not symbols: symbols = ["GBPUSD", "AUDUSD", "EURUSD", "US30"]
    results = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)

    print(f"\nRunning backtest from {start_date.date()} to {end_date.date()} using {source} data...")
    data_loader = DataLoader()

    for symbol in symbols:
        market_config = MARKET_DATA.get(symbol)
        if not market_config:
            print(f"Unknown symbol: {symbol}, skipping...")
            continue
        
        print(f"\n--- Testing: {symbol} ---")
        if source == 'yfinance':
            candle_data = data_loader.fetch_and_resample_data(symbol, start_date, end_date)
            logger.debug(f"[{symbol}] Data fetched: m1_len={len(candle_data.get('m1', []))}, m5_len={len(candle_data.get('m5', []))}")
            # Add this check here:
            if not candle_data or not candle_data.get('m1') or not candle_data.get('m5') or len(candle_data['m1']) == 0:
                print(f"Could not fetch sufficient data for {symbol}. Skipping.")
                continue
        else:
            print("Simulated data is no longer supported. Please use --source yfinance.")
            return {}

        engine = SMCBacktestEngine(symbol, market_config, initial_balance, risk_per_trade)
        result = engine.run_backtest(candle_data)
        results[symbol] = result
        print(f"Total Trades: {result.total_trades} | Win Rate: {result.win_rate:.1f}% | P&L: ${result.total_pnl:,.2f}")

    print("\n" + "="*50 + "\nSUMMARY\n" + "="*50)
    print(f"{'Symbol':<10} {'Trades':<8} {'Win%':<8} {'PF':<8} {'P&L':<12}")
    for symbol, result in results.items():
        print(f"{symbol:<10} {result.total_trades:<8} {result.win_rate:<8.1f} {result.profit_factor:<8.2f} ${result.total_pnl:<11,.2f}")
    print(f"TOTAL P&L: ${sum(r.total_pnl for r in results.values()):,.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMC V4 Multi-Instrument Backtest")
    parser.add_argument("--symbols", nargs="+", help="Symbols to test")
    parser.add_argument("--months", type=int, default=2, help="Number of months to backtest")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance")
    parser.add_argument("--risk", type=float, default=0.5, help="Risk percent per trade")
    parser.add_argument("--source", type=str, default="yfinance", choices=['yfinance', 'simulate'], help="Data source")
    args = parser.parse_args()

    setup_logging(level="DEBUG") # Add this line

    run_multi_instrument_backtest(args.symbols, args.months, args.balance, args.risk, args.source)
