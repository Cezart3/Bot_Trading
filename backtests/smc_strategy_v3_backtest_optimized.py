"""
SMC V3 Backtest with REAL MT5 Data.
Temporary script for debugging performance issues.
"""

import sys
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import MetaTrader5 as mt5

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.candle import Candle
from strategies.smc_strategy_v3 import SMCStrategyV3, SMCConfigV3, InstrumentType
from scripts.run_smc_v3 import SYMBOL_CONFIGS
from utils.logger import setup_logging, get_logger

# Ensure logger is set up for DEBUG level for this script
setup_logging(level="DEBUG") 
logger = get_logger(__name__)


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
    sl_pips: float
    lot_size: float
    risk_amount: float  # Exact $ risked
    pnl: float
    pnl_pips: float
    pnl_percent: float  # % of account
    result: str
    exit_reason: str


@dataclass
class BacktestResult:
    """Results from backtesting."""
    symbol: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    total_pnl_percent: float
    avg_win_percent: float
    avg_loss_percent: float
    max_drawdown_percent: float
    max_consecutive_losses: int
    trades: List[Trade] = field(default_factory=list)


class RealDataBacktester:
    """Backtester using real MT5 data with correct position sizing."""

    # Pip values per standard lot (100,000 units for forex)
    PIP_VALUES = {
        "EURUSD": 10.0,
        "GBPUSD": 10.0,
        "AUDUSD": 10.0,
        "EURGBP": 10.0,  # Approximately, depends on GBP/USD
        "US30": 1.0,     # $1 per point per contract
        "NDX100": 1.0,
        "GER30": 1.0,
    }

    PIP_SIZES = {
        "EURUSD": 0.0001,
        "GBPUSD": 0.0001,
        "AUDUSD": 0.0001,
        "EURGBP": 0.0001,
        "US30": 1.0,
        "NDX100": 0.1,
        "GER30": 0.1,
    }

    def __init__(
        self,
        symbol: str,
        initial_balance: float = 10000.0,
        risk_percent: float = 0.5,  # 0.5% risk per trade
    ):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.risk_percent = risk_percent
        self.balance = initial_balance
        self.trades: List[Trade] = []

        self.pip_size = self.PIP_SIZES.get(symbol, 0.0001)
        self.pip_value = self.PIP_VALUES.get(symbol, 10.0)

        # Determine instrument type
        instrument_type_map = {
            "US30": InstrumentType.INDEX,
            "NDX100": InstrumentType.INDEX,
            "GER30": InstrumentType.INDEX,
            "USTEC": InstrumentType.INDEX,
            "NAS100": InstrumentType.INDEX,
        }
        self.instrument_type = instrument_type_map.get(symbol, InstrumentType.FOREX)

    def _create_strategy_for_symbol(self) -> SMCStrategyV3:
        """Create SMC V3 strategy using the exact same config as the live bot."""
        config_data = SYMBOL_CONFIGS.get(self.symbol)
        if not config_data:
            raise ValueError(f"No configuration found for symbol {self.symbol} in SYMBOL_CONFIGS")

        # Get per-symbol settings from the imported config
        poi_min_score = config_data.get("poi_min_score", 2.0)
        require_sweep = config_data.get("require_sweep", False)
        adx_trending = config_data.get("adx_trending", 22.0)
        min_rr = config_data.get("min_rr", 1.5)
        ob_min_impulse_atr = config_data.get("ob_min_impulse_atr", 0.8)

        # Base config using live settings
        base_config = {
            "instrument_type": self.instrument_type,
            "min_sl_pips": config_data.get("min_sl_pips", 10),
            "max_sl_pips": config_data.get("max_sl_pips", 30),
            "risk_percent": self.risk_percent,
            "max_trades_per_day": 100,  # No trade limit in backtest
            "use_partial_tp": False, # Disable partials for simpler backtest logic
            "use_time_exit": False, # Disable time exit
            "use_equity_curve": False, # Disable equity curve
            "poi_min_score": poi_min_score,
            "require_sweep_for_low_score": require_sweep,
            "adx_trending": adx_trending,
            "min_rr": min_rr,
            "ob_min_impulse_atr": ob_min_impulse_atr,
            "adx_weak_trend": adx_trending - 5,
        }

        logger.debug(f"[{self.symbol}] Backtest Strategy config: POI={poi_min_score}, Sweep={require_sweep}, ADX={adx_trending}, RR={min_rr}, ImpulseATR={ob_min_impulse_atr}")

        config = SMCConfigV3(**base_config)
        strategy = SMCStrategyV3(symbol=self.symbol, config=config)
        return strategy

    def calculate_lot_size(self, sl_pips: float) -> float:
        """
        Calculate lot size for EXACT risk amount.

        Formula:
        Risk Amount = Balance * Risk% / 100
        Lot Size = Risk Amount / (SL_pips * Pip_Value_per_lot)

        Example with 0.5% risk, $10,000 balance, 15 pip SL:
        Risk Amount = $10,000 * 0.5% = $50
        Lot Size = $50 / (15 * $10) = 0.333 lots
        Loss at SL = 15 pips * 0.333 lots * $10 = $50 (exactly 0.5%)
        """
        if sl_pips <= 0:
            return 0.01  # Minimum

        risk_amount = self.balance * self.risk_percent / 100
        lot_size = risk_amount / (sl_pips * self.pip_value)

        # Apply broker limits
        lot_size = max(0.01, min(lot_size, 10.0))  # Min 0.01, max 10 lots
        lot_size = round(lot_size, 2)  # Round to 2 decimals

        return lot_size

    def calculate_pnl(self, pnl_pips: float, lot_size: float) -> float:
        """Calculate P&L from pips and lot size."""
        return pnl_pips * lot_size * self.pip_value

    def fetch_mt5_candles(
        self,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Candle]:
        """Fetch real candles from MT5."""
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

        mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M5)

        rates = mt5.copy_rates_range(self.symbol, mt5_tf, start_date, end_date)

        if rates is None or len(rates) == 0:
            print(f"  [!] No data for {self.symbol} {timeframe}")
            return []

        candles = []
        for rate in rates:
            candles.append(Candle(
                timestamp=datetime.fromtimestamp(rate['time']),
                open=rate['open'],
                high=rate['high'],
                low=rate['low'],
                close=rate['close'],
                volume=rate['tick_volume']
            ))

        return candles

    def run_backtest(self, days: int = 60) -> Optional[BacktestResult]:
        """Run backtest on real MT5 data."""

        # Initialize MT5
        if not mt5.initialize():
            print("Failed to initialize MT5")
            return None

        # Check symbol
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"Symbol {self.symbol} not found")
            mt5.shutdown()
            return None

        if not symbol_info.visible:
            mt5.symbol_select(self.symbol, True)

        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        logger.debug(f"  [{self.symbol}] Fetching real data: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # Fetch all timeframes - strategy needs M1 primarily
        m1_candles = self.fetch_mt5_candles("M1", start_date, end_date) # NEW
        m5_candles = self.fetch_mt5_candles("M5", start_date, end_date)
        h1_candles = self.fetch_mt5_candles("H1", start_date, end_date)
        h4_candles = self.fetch_mt5_candles("H4", start_date, end_date)
        daily_candles = self.fetch_mt5_candles("D1", start_date - timedelta(days=30), end_date)

        mt5.shutdown()

        if len(m1_candles) < 500:
            logger.warning(f"  [!] Insufficient data: {len(m1_candles)} M1 candles")
            return None

        logger.debug(f"  M1: {len(m1_candles)}, M5: {len(m5_candles)}, H1: {len(h1_candles)}, H4: {len(h4_candles)}, D1: {len(daily_candles)}")

        # Create strategy using the live bot's configuration
        try:
            strategy = self._create_strategy_for_symbol()
            strategy.initialize()
        except ValueError as e:
            logger.error(f"  [!] Error creating strategy for {self.symbol}: {e}")
            return None

        # Backtest state
        self.balance = self.initial_balance
        self.trades = []
        position = None
        peak_balance = self.initial_balance
        max_drawdown = 0
        consecutive_losses = 0
        max_consecutive_losses = 0

        m5_index = 0
        h1_index = 0
        h4_index = 0
        daily_index = 0

        logger.debug(f"  [{self.symbol}] Starting candle processing loop...")
        import time
        start_loop_time = time.time()

        for i, m1_candle in enumerate(m1_candles):
            if i < 200: continue # Skip initial candles to build history

            current_timestamp = m1_candle.timestamp
            
            # Efficiently find the latest corresponding candles for other timeframes
            while m5_index < len(m5_candles) -1 and m5_candles[m5_index + 1].timestamp <= current_timestamp:
                m5_index += 1
            while h1_index < len(h1_candles) - 1 and h1_candles[h1_index + 1].timestamp <= current_timestamp:
                h1_index += 1
            while h4_index < len(h4_candles) - 1 and h4_candles[h4_index + 1].timestamp <= current_timestamp:
                h4_index += 1
            while daily_index < len(daily_candles) -1 and daily_candles[daily_index + 1].timestamp <= current_timestamp:
                daily_index += 1

            # Set candles on strategy
            strategy.set_candles(
                h4_candles=h4_candles[:h4_index+1],
                h1_candles=h1_candles[:h1_index+1],
                m5_candles=m5_candles[:m5_index+1],
                m1_candles=m1_candles[:i+1], # NEW: Pass M1 candles
                daily_candles=daily_candles[:daily_index+1]
            )

            # Update spread (estimate)
            strategy.update_spread(self.pip_size * 1.0)  # 1 pip spread

            # Check existing position
            if position:
                if position["direction"] == "long":
                    # Check SL
                    if m1_candle.low <= position["sl"]:
                        pnl_pips = (position["sl"] - position["entry"]) / self.pip_size
                        pnl = self.calculate_pnl(pnl_pips, position["lot_size"])
                        pnl_percent = (pnl / self.balance) * 100

                        self.balance += pnl
                        self.trades.append(Trade(
                            entry_time=position["entry_time"],
                            exit_time=m1_candle.timestamp,
                            direction="long",
                            entry_price=position["entry"],
                            exit_price=position["sl"],
                            sl=position["sl"],
                            tp=position["tp"],
                            sl_pips=position["sl_pips"],
                            lot_size=position["lot_size"],
                            risk_amount=position["risk_amount"],
                            pnl=pnl,
                            pnl_pips=pnl_pips,
                            pnl_percent=pnl_percent,
                            result="loss",
                            exit_reason="sl_hit"
                        ))
                        position = None
                        consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                        continue

                    # Check TP
                    if m1_candle.high >= position["tp"]:
                        pnl_pips = (position["tp"] - position["entry"]) / self.pip_size
                        pnl = self.calculate_pnl(pnl_pips, position["lot_size"])
                        pnl_percent = (pnl / self.balance) * 100

                        self.balance += pnl
                        self.trades.append(Trade(
                            entry_time=position["entry_time"],
                            exit_time=m1_candle.timestamp,
                            direction="long",
                            entry_price=position["entry"],
                            exit_price=position["tp"],
                            sl=position["sl"],
                            tp=position["tp"],
                            sl_pips=position["sl_pips"],
                            lot_size=position["lot_size"],
                            risk_amount=position["risk_amount"],
                            pnl=pnl,
                            pnl_pips=pnl_pips,
                            pnl_percent=pnl_percent,
                            result="win",
                            exit_reason="tp_hit"
                        ))
                        position = None
                        consecutive_losses = 0
                        continue

                else:  # Short
                    # Check SL
                    if m1_candle.high >= position["sl"]:
                        pnl_pips = (position["entry"] - position["sl"]) / self.pip_size
                        pnl = self.calculate_pnl(pnl_pips, position["lot_size"])
                        pnl_percent = (pnl / self.balance) * 100

                        self.balance += pnl
                        self.trades.append(Trade(
                            entry_time=position["entry_time"],
                            exit_time=m1_candle.timestamp,
                            direction="short",
                            entry_price=position["entry"],
                            exit_price=position["sl"],
                            sl=position["sl"],
                            tp=position["tp"],
                            sl_pips=position["sl_pips"],
                            lot_size=position["lot_size"],
                            risk_amount=position["risk_amount"],
                            pnl=pnl,
                            pnl_pips=pnl_pips,
                            pnl_percent=pnl_percent,
                            result="loss",
                            exit_reason="sl_hit"
                        ))
                        position = None
                        consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                        continue

                    # Check TP
                    if m1_candle.low <= position["tp"]:
                        pnl_pips = (position["entry"] - position["tp"]) / self.pip_size
                        pnl = self.calculate_pnl(pnl_pips, position["lot_size"])
                        pnl_percent = (pnl / self.balance) * 100

                        self.balance += pnl
                        self.trades.append(Trade(
                            entry_time=position["entry_time"],
                            exit_time=m1_candle.timestamp,
                            direction="short",
                            entry_price=position["entry"],
                            exit_price=position["tp"],
                            sl=position["sl"],
                            tp=position["tp"],
                            sl_pips=position["sl_pips"],
                            lot_size=position["lot_size"],
                            risk_amount=position["risk_amount"],
                            pnl=pnl,
                            pnl_pips=pnl_pips,
                            pnl_percent=pnl_percent,
                            result="win",
                            exit_reason="tp_hit"
                        ))
                        position = None
                        consecutive_losses = 0
                        continue

            # Look for new signals
            if not position:
                signal = strategy.on_candle(m1_candle, m1_candles[:i+1])

                if signal and signal.stop_loss and signal.take_profit:
                    is_long = "LONG" in str(signal.signal_type)

                    # Calculate SL distance in pips
                    sl_pips = abs(signal.price - signal.stop_loss) / self.pip_size

                    # Calculate lot size for EXACT risk
                    risk_amount = self.balance * self.risk_percent / 100
                    lot_size = self.calculate_lot_size(sl_pips)

                    position = {
                        "direction": "long" if is_long else "short",
                        "entry": signal.price,
                        "sl": signal.stop_loss,
                        "tp": signal.take_profit,
                        "sl_pips": sl_pips,
                        "lot_size": lot_size,
                        "risk_amount": risk_amount,
                        "entry_time": m1_candle.timestamp,
                    }

            # Track drawdown
            if self.balance > peak_balance:
                peak_balance = self.balance
            drawdown = (peak_balance - self.balance) / peak_balance * 100
            max_drawdown = max(max_drawdown, drawdown)

        logger.debug(f"  [{self.symbol}] Candle processing finished in {time.time() - start_loop_time:.2f} seconds.")

        # Calculate results
        if not self.trades:
            return BacktestResult(
                symbol=self.symbol,
                total_trades=0,
                wins=0,
                losses=0,
                win_rate=0,
                profit_factor=0,
                total_pnl=0,
                total_pnl_percent=0,
                avg_win_percent=0,
                avg_loss_percent=0,
                max_drawdown_percent=max_drawdown,
                max_consecutive_losses=0,
                trades=[]
            )

        wins = [t for t in self.trades if t.result == "win"]
        losses = [t for t in self.trades if t.result == "loss"]

        total_win_pnl = sum(t.pnl for t in wins)
        total_loss_pnl = abs(sum(t.pnl for t in losses))

        return BacktestResult(
            symbol=self.symbol,
            total_trades=len(self.trades),
            wins=len(wins),
            losses=len(losses),
            win_rate=len(wins) / len(self.trades) * 100,
            profit_factor=total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else 0,
            total_pnl=self.balance - self.initial_balance,
            total_pnl_percent=(self.balance - self.initial_balance) / self.initial_balance * 100,
            avg_win_percent=sum(t.pnl_percent for t in wins) / len(wins) if wins else 0,
            avg_loss_percent=sum(t.pnl_percent for t in losses) / len(losses) if losses else 0,
            max_drawdown_percent=max_drawdown,
            max_consecutive_losses=max_consecutive_losses,
            trades=self.trades
        )


def run_real_backtest(
    symbols: List[str] = None,
    days: int = 60,
    balance: float = 10000.0,
    risk: float = 0.5
):
    """Run backtest with real MT5 data."""

    if symbols is None:
        symbols = ["EURUSD", "GBPUSD", "AUDUSD", "US30", "NDX100", "USTEC", "NAS100", "GER30"]

    logger.debug("=" * 70)
    logger.debug("SMC V3 BACKTEST - REAL MT5 DATA")
    logger.debug("=" * 70)
    logger.debug(f"Period: Last {days} days")
    logger.debug(f"Initial Balance: ${balance:,.2f}")
    logger.debug(f"Risk per Trade: {risk}% = ${balance * risk / 100:.2f}")
    logger.debug(f"Symbols: {', '.join(symbols)}")
    logger.debug("=" * 70)
    logger.debug("")

    all_results = {}

    for symbol in symbols:
        logger.debug(f"\n{'='*50}")
        logger.debug(f"Testing: {symbol}")
        logger.debug(f"{ '='*50}")

        backtester = RealDataBacktester(
            symbol=symbol,
            initial_balance=balance,
            risk_percent=risk
        )

        result = backtester.run_backtest(days=days)

        if result:
            all_results[symbol] = result

            logger.debug(f"\n--- {symbol} Results ---")
            logger.debug(f"Total Trades: {result.total_trades}")
            logger.debug(f"Wins: {result.wins} | Losses: {result.losses}")
            logger.debug(f"Win Rate: {result.win_rate:.1f}%")
            logger.debug(f"Profit Factor: {result.profit_factor:.2f}")
            logger.debug(f"Total P&L: ${result.total_pnl:,.2f} ({result.total_pnl_percent:+.2f}%)")
            logger.debug(f"Avg Win: {result.avg_win_percent:+.2f}% | Avg Loss: {result.avg_loss_percent:.2f}%")
            logger.debug(f"Max Drawdown: {result.max_drawdown_percent:.2f}%")
            logger.debug(f"Max Consecutive Losses: {result.max_consecutive_losses}")

            # Show individual trades
            if result.trades:
                logger.debug(f"\nLast 5 trades:")
                for t in result.trades[-5:]:
                    logger.debug(f"  {t.entry_time.strftime('%m/%d %H:%M')} {t.direction.upper():5} "
                          f"SL:{t.sl_pips:.1f}p Lot:{t.lot_size:.2f} "
                          f"Risk:${t.risk_amount:.0f} PnL:${t.pnl:+.2f} ({t.pnl_percent:+.2f}%) [{t.result}]")

    # Summary
    logger.debug("\n")
    logger.debug("=" * 70)
    logger.debug("SUMMARY - ALL INSTRUMENTS (REAL DATA)")
    logger.debug("=" * 70)
    logger.debug(f"{ 'Symbol':<10} {'Trades':<8} {'Win%':<8} {'PF':<8} {'P&L':<12} {'MaxDD':<8}")
    logger.debug("-" * 70)

    total_pnl = 0
    total_trades = 0

    for symbol, result in all_results.items():
        logger.debug(f"{symbol:<10} {result.total_trades:<8} {result.win_rate:<8.1f} "
              f"{result.profit_factor:<8.2f} ${result.total_pnl:<11,.2f} {result.max_drawdown_percent:<8.2f}%")
        total_pnl += result.total_pnl
        total_trades += result.total_trades

    logger.debug("-" * 70)
    logger.debug(f"{ 'TOTAL':<10} {total_trades:<8} {'-':<8} {'-':<8} ${total_pnl:<11,.2f}")
    logger.debug("=" * 70)

    # Verification
    logger.debug("\n" + "=" * 70)
    logger.debug("RISK VERIFICATION")
    logger.debug("=" * 70)

    all_losses = []
    for result in all_results.values():
        for t in result.trades:
            if t.result == "loss":
                all_losses.append(t)

    if all_losses:
        avg_loss_pct = sum(t.pnl_percent for t in all_losses) / len(all_losses)
        max_loss_pct = min(t.pnl_percent for t in all_losses)
        logger.debug(f"Expected Loss per Trade: -{risk}%")
        logger.debug(f"Actual Avg Loss: {avg_loss_pct:.2f}%")
        logger.debug(f"Actual Max Loss: {max_loss_pct:.2f}%")

        if abs(avg_loss_pct) <= risk * 1.1:  # Within 10% tolerance
            logger.debug("[OK] RISK IS CORRECT!")
        else:
            logger.debug("[ERROR] WARNING: RISK CALCULATION MAY BE OFF!")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SMC V3 Backtest with Real MT5 Data")
    parser.add_argument("--symbols", nargs="+", default=None,
                       help="Symbols to test")
    parser.add_argument("--days", type=int, default=60,
                       help="Days of history (default: 60)")
    parser.add_argument("--balance", type=float, default=10000.0,
                       help="Initial balance (default: 10000)")
    parser.add_argument("--risk", type=float, default=0.5,
                       help="Risk percent per trade (default: 0.5)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level for output")


    args = parser.parse_args()
    setup_logging(level=args.log_level)


    results = run_real_backtest(
        symbols=args.symbols,
        days=args.days,
        balance=args.balance,
        risk=args.risk
    )
