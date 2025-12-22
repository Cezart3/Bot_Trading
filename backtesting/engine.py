"""Backtesting engine - simulates trading on historical data."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import pytz

from models.candle import Candle
from models.order import Order, OrderSide, OrderStatus, OrderType
from models.position import Position, PositionStatus
from models.trade import Trade
from strategies.base_strategy import BaseStrategy, StrategySignal, SignalType
from utils.logger import get_logger
from utils.time_utils import is_trading_hours

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Backtesting configuration."""

    initial_balance: float = 10000.0
    commission_per_trade: float = 2.0  # Per side
    slippage_pips: float = 0.5
    pip_size: float = 0.01  # For indices like US500
    pip_value: float = 1.0  # $1 per pip per contract
    min_lot: float = 0.1
    max_lot: float = 10.0
    lot_step: float = 0.1


@dataclass
class BacktestPosition:
    """Position tracking during backtest."""

    symbol: str
    side: str  # "long" or "short"
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_id: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    trades: list[Trade] = field(default_factory=list)
    initial_balance: float = 0.0
    final_balance: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_commission: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    average_rr: float = 0.0
    sharpe_ratio: float = 0.0
    equity_curve: list = field(default_factory=list)
    daily_returns: list = field(default_factory=list)


class BacktestEngine:
    """
    Backtesting engine that simulates trading on historical data.

    Features:
    - Realistic order execution with slippage
    - Commission calculation
    - Stop loss and take profit handling
    - Equity curve tracking
    - Performance metrics calculation
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """Initialize backtest engine."""
        self.config = config or BacktestConfig()
        self._reset()

    def _reset(self):
        """Reset engine state."""
        self.balance = self.config.initial_balance
        self.equity = self.config.initial_balance
        self.peak_equity = self.config.initial_balance
        self.positions: list[BacktestPosition] = []
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.position_counter = 0

    def run(
        self,
        strategy: BaseStrategy,
        candles: list[Candle],
        session_start: str = "09:30",
        session_end: str = "16:00",
        timezone: str = "America/New_York",
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            strategy: Trading strategy to test
            candles: Historical candle data
            session_start: Trading session start time
            session_end: Trading session end time
            timezone: Timezone for session times

        Returns:
            BacktestResult with all metrics
        """
        logger.info(f"Starting backtest with {len(candles)} candles")
        logger.info(f"Initial balance: ${self.config.initial_balance:,.2f}")

        self._reset()
        strategy.initialize()
        strategy.pip_size = self.config.pip_size

        tz = pytz.timezone(timezone)
        last_date = None

        for i, candle in enumerate(candles):
            # Get historical candles for strategy
            lookback = min(i + 1, 100)
            historical_candles = candles[max(0, i - lookback + 1):i + 1]

            # Check for new trading day
            candle_date = candle.timestamp.date()
            if last_date != candle_date:
                last_date = candle_date
                strategy.reset()  # Reset strategy for new day
                logger.debug(f"New trading day: {candle_date}")

            # Skip if outside trading hours
            if not is_trading_hours(session_start, session_end, timezone, candle.timestamp):
                continue

            # Update equity curve
            self._update_equity(candle)

            # Check for SL/TP hits on existing positions
            self._check_sl_tp(candle)

            # Check for time-based exit (end of session)
            self._check_session_exit(candle, session_end, timezone)

            # Get strategy signal (only if no open positions)
            if not self.positions:
                signal = strategy.on_candle(candle, historical_candles)

                if signal and signal.is_entry:
                    self._execute_entry(signal, candle)

        # Close any remaining positions at last price
        if self.positions and candles:
            last_candle = candles[-1]
            for pos in self.positions[:]:
                self._close_position(pos, last_candle.close, last_candle.timestamp, "backtest_end")

        # Calculate results
        result = self._calculate_results()

        logger.info("=" * 60)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total Trades: {result.total_trades}")
        logger.info(f"Win Rate: {result.win_rate:.1f}%")
        logger.info(f"Total PnL: ${result.total_pnl:,.2f}")
        logger.info(f"Final Balance: ${result.final_balance:,.2f}")
        logger.info(f"Max Drawdown: {result.max_drawdown_percent:.1f}%")
        logger.info(f"Profit Factor: {result.profit_factor:.2f}")
        logger.info("=" * 60)

        return result

    def _execute_entry(self, signal: StrategySignal, candle: Candle) -> None:
        """Execute entry order."""
        # Apply slippage
        slippage = self.config.slippage_pips * self.config.pip_size

        if signal.is_long:
            entry_price = candle.close + slippage  # Buy at ask (higher)
            side = "long"
        else:
            entry_price = candle.close - slippage  # Sell at bid (lower)
            side = "short"

        # Calculate position size (simple: risk 1% of balance)
        risk_percent = 1.0
        risk_amount = self.balance * (risk_percent / 100)

        if signal.stop_loss:
            sl_distance = abs(entry_price - signal.stop_loss)
            sl_pips = sl_distance / self.config.pip_size

            if sl_pips > 0:
                quantity = risk_amount / (sl_pips * self.config.pip_value)
            else:
                quantity = self.config.min_lot
        else:
            quantity = self.config.min_lot

        # Clamp to limits
        quantity = max(self.config.min_lot, min(self.config.max_lot, quantity))
        quantity = round(quantity / self.config.lot_step) * self.config.lot_step

        # Create position
        self.position_counter += 1
        position = BacktestPosition(
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=candle.timestamp,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            position_id=self.position_counter,
            metadata=signal.metadata.copy() if signal.metadata else {},
        )

        self.positions.append(position)

        # Deduct commission
        self.balance -= self.config.commission_per_trade

        sl_str = f"{signal.stop_loss:.2f}" if signal.stop_loss else "None"
        tp_str = f"{signal.take_profit:.2f}" if signal.take_profit else "None"
        logger.debug(
            f"ENTRY: {side.upper()} {quantity} @ {entry_price:.2f} | "
            f"SL: {sl_str} | TP: {tp_str}"
        )

    def _check_sl_tp(self, candle: Candle) -> None:
        """Check if stop loss or take profit is hit."""
        for pos in self.positions[:]:  # Copy list to allow removal
            if pos.side == "long":
                # Check SL (price goes below)
                if pos.stop_loss and candle.low <= pos.stop_loss:
                    exit_price = pos.stop_loss - (self.config.slippage_pips * self.config.pip_size)
                    self._close_position(pos, exit_price, candle.timestamp, "sl")
                    continue

                # Check TP (price goes above)
                if pos.take_profit and candle.high >= pos.take_profit:
                    exit_price = pos.take_profit - (self.config.slippage_pips * self.config.pip_size)
                    self._close_position(pos, exit_price, candle.timestamp, "tp")

            else:  # Short position
                # Check SL (price goes above)
                if pos.stop_loss and candle.high >= pos.stop_loss:
                    exit_price = pos.stop_loss + (self.config.slippage_pips * self.config.pip_size)
                    self._close_position(pos, exit_price, candle.timestamp, "sl")
                    continue

                # Check TP (price goes below)
                if pos.take_profit and candle.low <= pos.take_profit:
                    exit_price = pos.take_profit + (self.config.slippage_pips * self.config.pip_size)
                    self._close_position(pos, exit_price, candle.timestamp, "tp")

    def _check_session_exit(self, candle: Candle, session_end: str, timezone: str) -> None:
        """Close positions near end of session."""
        tz = pytz.timezone(timezone)

        if candle.timestamp.tzinfo is None:
            candle_time = tz.localize(candle.timestamp)
        else:
            candle_time = candle.timestamp.astimezone(tz)

        end_parts = session_end.split(":")
        session_end_time = candle_time.replace(
            hour=int(end_parts[0]),
            minute=int(end_parts[1]),
            second=0,
        )

        # Close 10 minutes before session end
        close_time = session_end_time - timedelta(minutes=10)

        if candle_time >= close_time:
            for pos in self.positions[:]:
                self._close_position(pos, candle.close, candle.timestamp, "session_end")

    def _close_position(
        self,
        position: BacktestPosition,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
    ) -> None:
        """Close a position and record the trade."""
        # Calculate PnL
        if position.side == "long":
            pnl_points = exit_price - position.entry_price
        else:
            pnl_points = position.entry_price - exit_price

        pnl_pips = pnl_points / self.config.pip_size
        pnl = pnl_pips * self.config.pip_value * position.quantity

        # Deduct commission
        commission = self.config.commission_per_trade
        net_pnl = pnl - commission

        # Update balance
        self.balance += net_pnl

        # Create trade record
        trade = Trade(
            symbol=position.symbol,
            side=position.side,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=exit_time,
            pnl=pnl,
            commission=commission * 2,  # Entry + exit
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            exit_reason=exit_reason,
            strategy_name="ORB",
        )

        self.trades.append(trade)
        self.positions.remove(position)

        logger.debug(
            f"EXIT ({exit_reason}): {position.side.upper()} @ {exit_price:.2f} | "
            f"PnL: ${pnl:.2f} | Balance: ${self.balance:.2f}"
        )

    def _update_equity(self, candle: Candle) -> None:
        """Update equity including unrealized PnL."""
        unrealized_pnl = 0.0

        for pos in self.positions:
            if pos.side == "long":
                points = candle.close - pos.entry_price
            else:
                points = pos.entry_price - candle.close

            pips = points / self.config.pip_size
            unrealized_pnl += pips * self.config.pip_value * pos.quantity

        self.equity = self.balance + unrealized_pnl
        self.equity_curve.append((candle.timestamp, self.equity))

        # Track peak for drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest performance metrics."""
        result = BacktestResult()
        result.trades = self.trades
        result.initial_balance = self.config.initial_balance
        result.final_balance = self.balance
        result.total_trades = len(self.trades)
        result.equity_curve = self.equity_curve

        if not self.trades:
            return result

        # Win/Loss stats
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]

        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = (len(wins) / len(self.trades)) * 100 if self.trades else 0

        # PnL stats
        result.total_pnl = sum(t.pnl for t in self.trades)
        result.total_commission = sum(t.commission for t in self.trades)

        if wins:
            result.average_win = sum(t.pnl for t in wins) / len(wins)
        if losses:
            result.average_loss = abs(sum(t.pnl for t in losses) / len(losses))

        # Profit factor
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Average R:R
        rr_values = [t.r_multiple for t in self.trades if t.r_multiple is not None]
        result.average_rr = sum(rr_values) / len(rr_values) if rr_values else 0

        # Drawdown calculation
        peak = self.config.initial_balance
        max_dd = 0
        max_dd_pct = 0

        for _, equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_pct = (dd / peak) * 100 if peak > 0 else 0

            if dd > max_dd:
                max_dd = dd
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        result.max_drawdown = max_dd
        result.max_drawdown_percent = max_dd_pct

        # Daily returns for Sharpe ratio
        if len(self.equity_curve) > 1:
            import numpy as np

            equities = [e[1] for e in self.equity_curve]
            returns = np.diff(equities) / equities[:-1]
            result.daily_returns = returns.tolist()

            if len(returns) > 0 and np.std(returns) > 0:
                result.sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)

        return result
