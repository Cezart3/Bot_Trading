"""
Risk management utilities for Prop Trading.

Designed for funded prop accounts with strict drawdown limits:
- Max Daily Drawdown: 4%
- Max Account Drawdown: 10%
- Dynamic risk scaling based on drawdown proximity
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

from models.position import Position
from models.trade import Trade
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DailyStats:
    """Daily trading statistics."""

    date: date
    starting_balance: float
    current_balance: float
    trades_count: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_balance: float = 0.0

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.trades_count == 0:
            return 0.0
        return (self.wins / self.trades_count) * 100

    @property
    def daily_return(self) -> float:
        """Calculate daily return percentage."""
        if self.starting_balance == 0:
            return 0.0
        return ((self.current_balance - self.starting_balance) / self.starting_balance) * 100

    @property
    def daily_loss_percent(self) -> float:
        """Calculate current daily loss percentage."""
        if self.starting_balance == 0:
            return 0.0
        loss = self.starting_balance - self.current_balance
        if loss <= 0:
            return 0.0
        return (loss / self.starting_balance) * 100


@dataclass
class PropAccountLimits:
    """Prop trading account limits."""

    max_daily_drawdown: float = 4.0  # 4% max daily loss - STOP trading
    max_account_drawdown: float = 10.0  # 10% max account drawdown - STOP trading
    warning_drawdown: float = 2.0  # 2% daily DD - reduce risk to 0.5%
    account_warning_drawdown: float = 5.0  # 5% account DD - reduce risk to 0.5%
    normal_risk_per_trade: float = 1.0  # 1% risk normally
    reduced_risk_per_trade: float = 0.5  # 0.5% when in drawdown
    max_positions: int = 1  # Only 1 position at a time for this strategy


class RiskManager:
    """
    Risk management system for Prop Trading accounts.

    Features:
    - Strict daily and account drawdown limits
    - Dynamic risk scaling (reduces risk when approaching limits)
    - Position sizing based on risk percentage
    - Real-time drawdown tracking
    """

    def __init__(
        self,
        initial_account_balance: float = 0.0,
        limits: Optional[PropAccountLimits] = None,
    ):
        """
        Initialize risk manager for prop trading.

        Args:
            initial_account_balance: Starting account balance (for drawdown calculation)
            limits: Prop account limits configuration
        """
        self.limits = limits or PropAccountLimits()
        self.initial_account_balance = initial_account_balance

        self._daily_stats: Optional[DailyStats] = None
        self._trades_today: list[Trade] = []
        self._account_peak: float = initial_account_balance
        self._current_account_drawdown: float = 0.0

    def initialize_daily_stats(self, balance: float) -> None:
        """
        Initialize daily statistics.

        Args:
            balance: Current account balance.
        """
        today = date.today()
        if self._daily_stats is None or self._daily_stats.date != today:
            self._daily_stats = DailyStats(
                date=today,
                starting_balance=balance,
                current_balance=balance,
                peak_balance=balance,
            )
            self._trades_today = []

            # Set initial account balance if not set
            if self.initial_account_balance == 0:
                self.initial_account_balance = balance
                self._account_peak = balance

            logger.info(f"Daily stats initialized | Balance: ${balance:,.2f}")
            logger.info(f"Limits: Daily DD {self.limits.max_daily_drawdown}% | Account DD {self.limits.max_account_drawdown}%")

    def update_balance(self, balance: float) -> None:
        """
        Update current balance and track drawdown.

        Args:
            balance: Current account balance.
        """
        if self._daily_stats is None:
            self.initialize_daily_stats(balance)
            return

        self._daily_stats.current_balance = balance

        # Track daily peak
        if balance > self._daily_stats.peak_balance:
            self._daily_stats.peak_balance = balance

        # Track account peak
        if balance > self._account_peak:
            self._account_peak = balance

        # Calculate drawdowns
        if self._daily_stats.peak_balance > 0:
            daily_dd = ((self._daily_stats.peak_balance - balance) / self._daily_stats.peak_balance) * 100
            if daily_dd > self._daily_stats.max_drawdown:
                self._daily_stats.max_drawdown = daily_dd

        if self.initial_account_balance > 0:
            self._current_account_drawdown = ((self.initial_account_balance - balance) / self.initial_account_balance) * 100

    def get_current_risk_percent(self) -> float:
        """
        Get current risk percentage based on drawdown proximity.

        Returns reduced risk when:
        - Daily drawdown >= warning_drawdown (default 2%)
        - Account drawdown >= account_warning_drawdown (default 5%)
        """
        if self._daily_stats is None:
            return self.limits.normal_risk_per_trade

        daily_loss = self._daily_stats.daily_loss_percent
        account_dd = max(0, self._current_account_drawdown)

        # Reduce risk if daily drawdown >= warning level (2%)
        if daily_loss >= self.limits.warning_drawdown:
            logger.warning(f"Daily DD {daily_loss:.1f}% >= {self.limits.warning_drawdown}% -> Risk reduced to {self.limits.reduced_risk_per_trade}%")
            return self.limits.reduced_risk_per_trade

        # Reduce risk if account drawdown >= account warning level (5%)
        if account_dd >= self.limits.account_warning_drawdown:
            logger.warning(f"Account DD {account_dd:.1f}% >= {self.limits.account_warning_drawdown}% -> Risk reduced to {self.limits.reduced_risk_per_trade}%")
            return self.limits.reduced_risk_per_trade

        return self.limits.normal_risk_per_trade

    def record_trade(self, trade: Trade) -> None:
        """
        Record a completed trade.

        Args:
            trade: Completed trade object.
        """
        if self._daily_stats is None:
            return

        self._trades_today.append(trade)
        self._daily_stats.trades_count += 1
        self._daily_stats.total_pnl += trade.pnl

        if trade.is_win:
            self._daily_stats.wins += 1
        elif trade.is_loss:
            self._daily_stats.losses += 1

        logger.info(
            f"Trade recorded: {trade.side} {trade.symbol} | "
            f"PnL: ${trade.pnl:+,.2f} | Daily PnL: ${self._daily_stats.total_pnl:+,.2f}"
        )

    def can_trade(self, current_positions: int = 0) -> tuple[bool, str]:
        """
        Check if trading is allowed based on prop trading rules.

        Args:
            current_positions: Number of currently open positions.

        Returns:
            Tuple of (can_trade, reason).
        """
        if self._daily_stats is None:
            return (False, "Daily stats not initialized")

        # Check max positions
        if current_positions >= self.limits.max_positions:
            return (False, f"Max positions reached ({self.limits.max_positions})")

        # Check daily loss limit
        daily_loss = self._daily_stats.daily_loss_percent
        if daily_loss >= self.limits.max_daily_drawdown:
            return (
                False,
                f"DAILY LIMIT BREACHED: {daily_loss:.2f}% >= {self.limits.max_daily_drawdown}%",
            )

        # Check account drawdown limit
        if self._current_account_drawdown >= self.limits.max_account_drawdown:
            return (
                False,
                f"ACCOUNT LIMIT BREACHED: {self._current_account_drawdown:.2f}% >= {self.limits.max_account_drawdown}%",
            )

        # Warning if approaching limits
        if daily_loss >= self.limits.max_daily_drawdown * 0.8:
            logger.warning(f"WARNING: Approaching daily limit ({daily_loss:.1f}%/{self.limits.max_daily_drawdown}%)")

        return (True, "OK")

    def calculate_position_size(
        self,
        account_balance: float,
        stop_loss_distance: float,
        price: float,
        contract_size: float = 1.0,
        min_qty: float = 1.0,
        max_qty: float = 1000.0,
        qty_step: float = 1.0,
    ) -> float:
        """
        Calculate position size for stocks based on risk percentage.

        Args:
            account_balance: Current account balance.
            stop_loss_distance: Stop loss distance in price (not pips).
            price: Current stock price.
            contract_size: Contract multiplier (usually 1 for stocks).
            min_qty: Minimum quantity.
            max_qty: Maximum quantity.
            qty_step: Quantity increment.

        Returns:
            Calculated position size (number of shares).
        """
        if stop_loss_distance <= 0:
            logger.warning("Invalid stop loss distance, using minimum quantity")
            return min_qty

        # Get current risk percentage (may be reduced if near limits)
        risk_percent = self.get_current_risk_percent()

        # Calculate risk amount in dollars
        risk_amount = account_balance * (risk_percent / 100)

        # Calculate position size
        # Risk = Shares * SL_Distance
        # Shares = Risk / SL_Distance
        shares = risk_amount / stop_loss_distance

        # Round to quantity step
        shares = round(shares / qty_step) * qty_step

        # Clamp to min/max
        shares = max(min_qty, min(max_qty, shares))

        logger.debug(
            f"Position size: {shares:.0f} shares | "
            f"Risk: ${risk_amount:.2f} ({risk_percent}%) | SL distance: ${stop_loss_distance:.2f}"
        )

        return shares

    def calculate_forex_lot_size(
        self,
        account_balance: float,
        stop_loss_pips: float,
        pip_value_per_lot: float = 10.0,
        min_lot: float = 0.01,
        max_lot: float = 10.0,
        lot_step: float = 0.01,
        risk_percent_override: float = None,
    ) -> tuple[float, float]:
        """
        Calculate lot size for FOREX based on EXACT risk percentage.

        CRITICAL: This ensures loss at SL = exactly risk_percent of account.

        Formula:
        Risk Amount = Balance * Risk% / 100
        Lot Size = Risk Amount / (SL_pips * Pip_Value_per_lot)

        Example with 0.5% risk, $10,000 balance, 15 pip SL, $10 pip value:
        Risk Amount = $10,000 * 0.5% = $50
        Lot Size = $50 / (15 * $10) = 0.333 lots
        Loss at SL = 15 pips * 0.333 * $10 = $50 (exactly 0.5%)

        Args:
            account_balance: Current account balance in USD.
            stop_loss_pips: Stop loss distance in PIPS (not price!).
            pip_value_per_lot: Value of 1 pip per 1.0 lot (usually $10 for major pairs).
            min_lot: Minimum lot size (usually 0.01).
            max_lot: Maximum lot size.
            lot_step: Lot size increment (usually 0.01).
            risk_percent_override: Override risk percentage (if None, uses current risk).

        Returns:
            Tuple of (lot_size, risk_amount_usd).
        """
        if stop_loss_pips <= 0:
            logger.warning("Invalid stop loss pips, using minimum lot")
            return (min_lot, 0)

        # Get current risk percentage (may be reduced if near limits)
        if risk_percent_override is not None:
            risk_percent = risk_percent_override
        else:
            risk_percent = self.get_current_risk_percent()

        # Calculate risk amount in dollars
        risk_amount = account_balance * (risk_percent / 100)

        # Calculate lot size
        # Lot Size = Risk Amount / (SL_pips * Pip_Value)
        lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)

        # Round to lot step
        lot_size = round(lot_size / lot_step) * lot_step

        # Clamp to min/max
        lot_size = max(min_lot, min(max_lot, lot_size))

        # Recalculate actual risk based on clamped lot size
        actual_risk = lot_size * stop_loss_pips * pip_value_per_lot

        logger.info(
            f"FOREX Lot Size: {lot_size:.2f} lots | "
            f"SL: {stop_loss_pips:.1f} pips | "
            f"Risk: ${actual_risk:.2f} ({risk_percent}% of ${account_balance:,.0f})"
        )

        return (lot_size, actual_risk)

    def calculate_stop_loss_price(
        self,
        entry_price: float,
        is_long: bool,
        buffer_amount: float = 0.03,  # $0.03 default buffer
    ) -> float:
        """
        Calculate stop loss price with buffer.

        Args:
            entry_price: Entry price.
            is_long: True for long position.
            buffer_amount: Buffer amount in dollars.

        Returns:
            Stop loss price.
        """
        if is_long:
            return entry_price - buffer_amount
        else:
            return entry_price + buffer_amount

    def calculate_take_profit_price(
        self,
        entry_price: float,
        stop_loss: float,
        is_long: bool,
        risk_reward_ratio: float = 2.0,
    ) -> float:
        """
        Calculate take profit based on risk/reward ratio.

        Args:
            entry_price: Entry price.
            stop_loss: Stop loss price.
            is_long: True for long position.
            risk_reward_ratio: R:R ratio (default 2:1).

        Returns:
            Take profit price.
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio

        if is_long:
            return entry_price + reward
        else:
            return entry_price - reward

    def get_daily_stats(self) -> Optional[DailyStats]:
        """Get current daily statistics."""
        return self._daily_stats

    def get_risk_status(self) -> dict:
        """
        Get comprehensive risk status for prop trading.

        Returns:
            Dictionary with all risk metrics.
        """
        if self._daily_stats is None:
            return {"status": "not_initialized"}

        can_trade, reason = self.can_trade()
        current_risk = self.get_current_risk_percent()

        return {
            "status": "active" if can_trade else "STOPPED",
            "reason": reason,
            "can_trade": can_trade,

            # Daily metrics
            "daily_pnl": self._daily_stats.total_pnl,
            "daily_return_percent": self._daily_stats.daily_return,
            "daily_loss_percent": self._daily_stats.daily_loss_percent,
            "daily_limit": self.limits.max_daily_drawdown,
            "daily_limit_remaining": self.limits.max_daily_drawdown - self._daily_stats.daily_loss_percent,

            # Account metrics
            "account_drawdown_percent": self._current_account_drawdown,
            "account_limit": self.limits.max_account_drawdown,
            "account_limit_remaining": self.limits.max_account_drawdown - self._current_account_drawdown,

            # Risk settings
            "current_risk_percent": current_risk,
            "normal_risk": self.limits.normal_risk_per_trade,
            "reduced_risk": self.limits.reduced_risk_per_trade,
            "risk_reduced": current_risk < self.limits.normal_risk_per_trade,

            # Trade stats
            "trades_today": self._daily_stats.trades_count,
            "win_rate": self._daily_stats.win_rate,
        }

    def print_status(self) -> None:
        """Print formatted risk status to console."""
        status = self.get_risk_status()

        if status["status"] == "not_initialized":
            print("Risk Manager not initialized")
            return

        print("\n" + "=" * 50)
        print("        PROP TRADING RISK STATUS")
        print("=" * 50)

        # Status indicator
        if status["can_trade"]:
            print(f"  Status: ACTIVE")
        else:
            print(f"  Status: STOPPED - {status['reason']}")

        print(f"\n  Daily Drawdown:   {status['daily_loss_percent']:>6.2f}% / {status['daily_limit']}%")
        print(f"  Account Drawdown: {status['account_drawdown_percent']:>6.2f}% / {status['account_limit']}%")
        print(f"  Current Risk:     {status['current_risk_percent']:>6.2f}%", end="")
        if status["risk_reduced"]:
            print(" (REDUCED)")
        else:
            print()

        print(f"\n  Daily PnL: ${status['daily_pnl']:>+10,.2f}")
        print(f"  Trades:    {status['trades_today']:>10}")
        print(f"  Win Rate:  {status['win_rate']:>10.1f}%")
        print("=" * 50 + "\n")
