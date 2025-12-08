"""
Prop Trading Rules & Readiness Check System

This module enforces strict prop trading rules on BOTH demo and live accounts.
The goal is to treat demo EXACTLY like live so you know when you're ready.

Based on backtesting analysis:
- AMD is the ONLY recommended symbol (Profit Factor 1.46, Max DD 7.4%)
- NVDA: EXCLUDED (PF 0.64, negative returns)
- TSLA: EXCLUDED (PF 1.06, marginal performance)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional
from enum import Enum

from utils.logger import get_logger

logger = get_logger(__name__)


class ReadinessLevel(Enum):
    """Bot readiness levels for live trading."""
    NOT_READY = "not_ready"
    TESTING = "testing"
    VALIDATED = "validated"
    READY_FOR_LIVE = "ready_for_live"


@dataclass
class TradingSession:
    """Record of a single trading session."""
    date: date
    symbol: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    max_drawdown: float = 0.0
    rules_followed: bool = True
    violations: list[str] = field(default_factory=list)


@dataclass
class PropTradingConfig:
    """
    Strict prop trading configuration.
    SAME rules for demo and live - no exceptions.
    """
    # SYMBOL RESTRICTION - Based on backtesting analysis
    allowed_symbol: str = "AMD"  # ONLY AMD - best performer

    # RISK LIMITS (Prop trading standard)
    max_risk_per_trade: float = 0.5  # 0.5% per trade (conservative)
    max_daily_loss: float = 2.0  # 2% max daily loss (stop trading)
    max_account_drawdown: float = 5.0  # 5% max total drawdown

    # POSITION LIMITS
    max_positions: int = 1  # Only 1 position at a time
    max_trades_per_day: int = 2  # Max 2 trades per day

    # CONSECUTIVE LOSS RULES
    max_consecutive_losses: int = 2  # Stop after 2 consecutive losses
    cooldown_after_losses: int = 1  # Skip 1 day after hitting loss limit

    # TIME RULES
    no_trading_monday: bool = False  # Optional: avoid Mondays
    no_trading_friday: bool = False  # Optional: avoid Fridays
    stop_trading_after_time: str = "15:30"  # No new trades after 15:30

    # NEWS FILTER
    skip_news_days: bool = True  # ALWAYS skip high-impact news days

    # READINESS CRITERIA (to go live)
    min_demo_days: int = 10  # Minimum 10 trading days on demo
    min_demo_trades: int = 15  # Minimum 15 trades on demo
    min_win_rate: float = 35.0  # Minimum 35% win rate
    min_profit_factor: float = 1.2  # Minimum 1.2 profit factor
    max_allowed_drawdown: float = 6.0  # Max 6% drawdown during demo
    required_profitable_weeks: int = 2  # At least 2 profitable weeks


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: date
    starting_balance: float
    current_balance: float
    trades_taken: int = 0
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0
    daily_pnl: float = 0.0
    daily_drawdown: float = 0.0
    rules_violated: list[str] = field(default_factory=list)
    can_trade: bool = True


class PropTradingTracker:
    """
    Tracks trading performance and enforces prop trading rules.

    CRITICAL: This treats demo and live IDENTICALLY.
    If you can't follow rules on demo, you're not ready for live.
    """

    def __init__(
        self,
        config: Optional[PropTradingConfig] = None,
        data_path: str = "data/prop_trading_stats.json",
    ):
        """Initialize prop trading tracker."""
        self.config = config or PropTradingConfig()
        self.data_path = Path(data_path)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)

        # Current state
        self.sessions: list[TradingSession] = []
        self.daily_stats: Optional[DailyStats] = None
        self.account_high_water_mark: float = 0.0
        self.account_drawdown: float = 0.0
        self.consecutive_loss_days: int = 0
        self.is_in_cooldown: bool = False
        self.cooldown_until: Optional[date] = None

        # Load history
        self._load_history()

    def _load_history(self) -> None:
        """Load trading history from file."""
        if not self.data_path.exists():
            return

        try:
            with open(self.data_path) as f:
                data = json.load(f)

            self.account_high_water_mark = data.get("high_water_mark", 0)
            self.consecutive_loss_days = data.get("consecutive_loss_days", 0)

            if data.get("cooldown_until"):
                self.cooldown_until = date.fromisoformat(data["cooldown_until"])
                self.is_in_cooldown = self.cooldown_until >= date.today()

            for session_data in data.get("sessions", []):
                session = TradingSession(
                    date=date.fromisoformat(session_data["date"]),
                    symbol=session_data["symbol"],
                    trades=session_data.get("trades", 0),
                    wins=session_data.get("wins", 0),
                    losses=session_data.get("losses", 0),
                    pnl=session_data.get("pnl", 0),
                    rules_followed=session_data.get("rules_followed", True),
                )
                self.sessions.append(session)

            logger.info(f"Loaded {len(self.sessions)} trading sessions from history")

        except Exception as e:
            logger.warning(f"Failed to load history: {e}")

    def _save_history(self) -> None:
        """Save trading history to file."""
        try:
            data = {
                "high_water_mark": self.account_high_water_mark,
                "consecutive_loss_days": self.consecutive_loss_days,
                "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
                "sessions": [
                    {
                        "date": s.date.isoformat(),
                        "symbol": s.symbol,
                        "trades": s.trades,
                        "wins": s.wins,
                        "losses": s.losses,
                        "pnl": s.pnl,
                        "rules_followed": s.rules_followed,
                        "violations": s.violations,
                    }
                    for s in self.sessions[-100:]  # Keep last 100 sessions
                ],
                "last_updated": datetime.now().isoformat(),
            }

            with open(self.data_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def start_day(self, balance: float) -> DailyStats:
        """
        Start a new trading day.

        Args:
            balance: Current account balance

        Returns:
            DailyStats with can_trade status
        """
        today = date.today()

        self.daily_stats = DailyStats(
            date=today,
            starting_balance=balance,
            current_balance=balance,
        )

        # Update high water mark
        if balance > self.account_high_water_mark:
            self.account_high_water_mark = balance

        # Calculate account drawdown
        if self.account_high_water_mark > 0:
            self.account_drawdown = ((self.account_high_water_mark - balance) /
                                      self.account_high_water_mark) * 100

        # Check if we can trade today
        violations = []
        can_trade = True

        # Rule 1: Check cooldown period
        if self.is_in_cooldown and self.cooldown_until and today <= self.cooldown_until:
            violations.append(f"In cooldown until {self.cooldown_until}")
            can_trade = False
        else:
            self.is_in_cooldown = False

        # Rule 2: Check account drawdown
        if self.account_drawdown >= self.config.max_account_drawdown:
            violations.append(f"Account drawdown {self.account_drawdown:.1f}% exceeds limit {self.config.max_account_drawdown}%")
            can_trade = False

        # Rule 3: Check day of week
        weekday = today.weekday()
        if weekday == 0 and self.config.no_trading_monday:
            violations.append("No trading on Mondays")
            can_trade = False
        if weekday == 4 and self.config.no_trading_friday:
            violations.append("No trading on Fridays")
            can_trade = False

        self.daily_stats.rules_violated = violations
        self.daily_stats.can_trade = can_trade

        return self.daily_stats

    def check_can_open_trade(
        self,
        symbol: str,
        current_time: Optional[datetime] = None,
    ) -> tuple[bool, str]:
        """
        Check if we can open a new trade.

        Args:
            symbol: Symbol to trade
            current_time: Current time (default: now)

        Returns:
            Tuple of (can_trade, reason)
        """
        if self.daily_stats is None:
            return False, "Day not started - call start_day() first"

        if not self.daily_stats.can_trade:
            return False, f"Trading disabled: {', '.join(self.daily_stats.rules_violated)}"

        current_time = current_time or datetime.now()

        # Rule: Symbol restriction
        if symbol != self.config.allowed_symbol:
            return False, f"Only {self.config.allowed_symbol} is allowed (based on backtesting)"

        # Rule: Max trades per day
        if self.daily_stats.trades_taken >= self.config.max_trades_per_day:
            return False, f"Max {self.config.max_trades_per_day} trades per day reached"

        # Rule: Consecutive losses
        if self.daily_stats.consecutive_losses >= self.config.max_consecutive_losses:
            return False, f"Max {self.config.max_consecutive_losses} consecutive losses reached - stop trading"

        # Rule: Daily loss limit
        daily_loss_pct = (self.daily_stats.daily_pnl / self.daily_stats.starting_balance) * 100
        if daily_loss_pct <= -self.config.max_daily_loss:
            return False, f"Daily loss limit {self.config.max_daily_loss}% reached"

        # Rule: Time restriction
        stop_time = datetime.strptime(self.config.stop_trading_after_time, "%H:%M").time()
        if current_time.time() > stop_time:
            return False, f"No new trades after {self.config.stop_trading_after_time}"

        return True, "OK"

    def record_trade(
        self,
        symbol: str,
        pnl: float,
        is_win: bool,
    ) -> None:
        """
        Record a completed trade.

        Args:
            symbol: Traded symbol
            pnl: Profit/Loss
            is_win: Whether trade was profitable
        """
        if self.daily_stats is None:
            return

        self.daily_stats.trades_taken += 1
        self.daily_stats.daily_pnl += pnl
        self.daily_stats.current_balance += pnl

        if is_win:
            self.daily_stats.wins += 1
            self.daily_stats.consecutive_losses = 0
        else:
            self.daily_stats.losses += 1
            self.daily_stats.consecutive_losses += 1

        # Update drawdown
        if self.daily_stats.daily_pnl < 0:
            self.daily_stats.daily_drawdown = abs(self.daily_stats.daily_pnl /
                                                   self.daily_stats.starting_balance) * 100

        logger.info(
            f"Trade recorded: {symbol} | PnL: ${pnl:.2f} | "
            f"Daily PnL: ${self.daily_stats.daily_pnl:.2f} | "
            f"Consecutive losses: {self.daily_stats.consecutive_losses}"
        )

    def end_day(self) -> TradingSession:
        """
        End the trading day and save statistics.

        Returns:
            TradingSession summary
        """
        if self.daily_stats is None:
            raise ValueError("No daily stats - day not started")

        # Check if rules were followed
        rules_followed = len(self.daily_stats.rules_violated) == 0

        # Check for additional violations
        violations = list(self.daily_stats.rules_violated)

        if self.daily_stats.daily_drawdown > self.config.max_daily_loss:
            violations.append(f"Daily loss exceeded: {self.daily_stats.daily_drawdown:.1f}%")
            rules_followed = False

        # Create session record
        session = TradingSession(
            date=self.daily_stats.date,
            symbol=self.config.allowed_symbol,
            trades=self.daily_stats.trades_taken,
            wins=self.daily_stats.wins,
            losses=self.daily_stats.losses,
            pnl=self.daily_stats.daily_pnl,
            max_drawdown=self.daily_stats.daily_drawdown,
            rules_followed=rules_followed,
            violations=violations,
        )

        self.sessions.append(session)

        # Update consecutive loss days
        if self.daily_stats.daily_pnl < 0:
            self.consecutive_loss_days += 1
        else:
            self.consecutive_loss_days = 0

        # Check if we need cooldown
        if self.daily_stats.consecutive_losses >= self.config.max_consecutive_losses:
            self.is_in_cooldown = True
            self.cooldown_until = date.today() + timedelta(days=self.config.cooldown_after_losses)
            logger.warning(f"Entering cooldown until {self.cooldown_until}")

        # Save history
        self._save_history()

        return session

    def get_readiness_status(self) -> dict:
        """
        Check if the bot is ready for live trading.

        Returns:
            Dictionary with readiness status and details
        """
        if len(self.sessions) == 0:
            return {
                "level": ReadinessLevel.NOT_READY.value,
                "message": "No trading history - start demo testing first",
                "criteria": {},
                "ready": False,
            }

        # Calculate statistics from sessions
        total_trades = sum(s.trades for s in self.sessions)
        total_wins = sum(s.wins for s in self.sessions)
        total_pnl = sum(s.pnl for s in self.sessions)
        total_losses_amount = sum(s.pnl for s in self.sessions if s.pnl < 0)
        total_wins_amount = sum(s.pnl for s in self.sessions if s.pnl > 0)

        trading_days = len(self.sessions)
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        profit_factor = (total_wins_amount / abs(total_losses_amount)) if total_losses_amount != 0 else 0

        # Calculate max drawdown from sessions
        max_dd = max((s.max_drawdown for s in self.sessions), default=0)

        # Calculate profitable weeks
        weeks = {}
        for session in self.sessions:
            week_num = session.date.isocalendar()[1]
            year = session.date.year
            key = f"{year}-W{week_num}"
            weeks[key] = weeks.get(key, 0) + session.pnl
        profitable_weeks = sum(1 for pnl in weeks.values() if pnl > 0)

        # Count rule violations
        violations = sum(1 for s in self.sessions if not s.rules_followed)

        # Check each criterion
        criteria = {
            "min_demo_days": {
                "required": self.config.min_demo_days,
                "actual": trading_days,
                "passed": trading_days >= self.config.min_demo_days,
            },
            "min_demo_trades": {
                "required": self.config.min_demo_trades,
                "actual": total_trades,
                "passed": total_trades >= self.config.min_demo_trades,
            },
            "min_win_rate": {
                "required": f"{self.config.min_win_rate}%",
                "actual": f"{win_rate:.1f}%",
                "passed": win_rate >= self.config.min_win_rate,
            },
            "min_profit_factor": {
                "required": self.config.min_profit_factor,
                "actual": round(profit_factor, 2),
                "passed": profit_factor >= self.config.min_profit_factor,
            },
            "max_drawdown": {
                "required": f"<{self.config.max_allowed_drawdown}%",
                "actual": f"{max_dd:.1f}%",
                "passed": max_dd < self.config.max_allowed_drawdown,
            },
            "profitable_weeks": {
                "required": self.config.required_profitable_weeks,
                "actual": profitable_weeks,
                "passed": profitable_weeks >= self.config.required_profitable_weeks,
            },
            "rules_followed": {
                "required": "100%",
                "actual": f"{((len(self.sessions) - violations) / len(self.sessions) * 100):.0f}%",
                "passed": violations == 0,
            },
        }

        # Determine readiness level
        passed_criteria = sum(1 for c in criteria.values() if c["passed"])
        total_criteria = len(criteria)

        if passed_criteria == total_criteria:
            level = ReadinessLevel.READY_FOR_LIVE
            message = "All criteria passed - READY FOR LIVE TRADING!"
        elif passed_criteria >= total_criteria - 1:
            level = ReadinessLevel.VALIDATED
            message = "Almost ready - 1 criterion remaining"
        elif trading_days >= 5:
            level = ReadinessLevel.TESTING
            message = f"Testing in progress - {passed_criteria}/{total_criteria} criteria passed"
        else:
            level = ReadinessLevel.NOT_READY
            message = "Not enough trading history - continue demo testing"

        return {
            "level": level.value,
            "message": message,
            "ready": level == ReadinessLevel.READY_FOR_LIVE,
            "criteria": criteria,
            "summary": {
                "trading_days": trading_days,
                "total_trades": total_trades,
                "win_rate": f"{win_rate:.1f}%",
                "profit_factor": round(profit_factor, 2),
                "total_pnl": round(total_pnl, 2),
                "max_drawdown": f"{max_dd:.1f}%",
                "profitable_weeks": profitable_weeks,
                "rule_violations": violations,
            },
        }

    def print_readiness_report(self) -> None:
        """Print detailed readiness report."""
        status = self.get_readiness_status()

        print("\n" + "=" * 70)
        print("           PROP TRADING READINESS REPORT")
        print("=" * 70)

        # Status header
        level = status["level"].upper().replace("_", " ")
        print(f"\n  STATUS: {level}")
        print(f"  {status['message']}")

        if status.get("summary"):
            print("\n  TRADING SUMMARY:")
            print("  " + "-" * 40)
            summary = status["summary"]
            print(f"  Trading Days:     {summary['trading_days']}")
            print(f"  Total Trades:     {summary['total_trades']}")
            print(f"  Win Rate:         {summary['win_rate']}")
            print(f"  Profit Factor:    {summary['profit_factor']}")
            print(f"  Total P&L:        ${summary['total_pnl']:,.2f}")
            print(f"  Max Drawdown:     {summary['max_drawdown']}")
            print(f"  Profitable Weeks: {summary['profitable_weeks']}")
            print(f"  Rule Violations:  {summary['rule_violations']}")

        print("\n  CRITERIA CHECKLIST:")
        print("  " + "-" * 60)

        for name, criterion in status.get("criteria", {}).items():
            status_icon = "[X]" if criterion["passed"] else "[ ]"
            name_formatted = name.replace("_", " ").title()
            print(
                f"  {status_icon} {name_formatted:<20} "
                f"Required: {criterion['required']:<10} "
                f"Actual: {criterion['actual']}"
            )

        print("  " + "-" * 60)

        if status["ready"]:
            print("\n  >>> READY FOR LIVE TRADING <<<")
            print("  You have demonstrated consistent, rule-following trading.")
            print("  Start with SMALL position sizes on live account.")
        else:
            print("\n  Continue demo trading until all criteria are met.")
            print("  Focus on following rules consistently.")

        print("=" * 70 + "\n")


def get_analysis_report() -> str:
    """
    Generate analysis report explaining why AMD is the only recommended symbol.
    """
    return """
================================================================================
                    STOCK ANALYSIS REPORT
                Based on 39 Trading Days Backtesting
================================================================================

WINNER: AMD (Advanced Micro Devices)
---------------------------------------
  Profit:         +$2,539.95 (+9.9%)
  Win Rate:       38.9%
  Profit Factor:  1.46 (excellent - above 1.3 is good)
  Max Drawdown:   7.4% (within prop limits)
  Avg Win:        $576.00
  Avg Loss:       $251.09
  Win/Loss Ratio: 2.29:1

WHY AMD?
  1. Highest profit factor (1.46) - strategy works best here
  2. Lowest max drawdown (7.4%) - safest for prop trading
  3. Best risk-adjusted returns
  4. Consistent price action during ORB timeframe
  5. Good liquidity and tight spreads

---------------------------------------

EXCLUDED: TSLA (Tesla)
---------------------------------------
  Profit:         +$467.47 (+1.6%)
  Win Rate:       38.9%
  Profit Factor:  1.06 (marginal - barely profitable)
  Max Drawdown:   8.8%

WHY EXCLUDED?
  - Profit factor too close to 1.0 (break-even)
  - Higher drawdown than AMD
  - Commission and slippage can easily turn it negative
  - Too volatile for consistent strategy performance

---------------------------------------

EXCLUDED: NVDA (NVIDIA)
---------------------------------------
  Profit:         -$2,345.79 (-9.7%)
  Win Rate:       24.3% (very low)
  Profit Factor:  0.64 (losing strategy)
  Max Drawdown:   10.5% (exceeds prop limits)

WHY EXCLUDED?
  - LOSING MONEY - profit factor below 1.0
  - Would fail prop account rules (>10% drawdown)
  - Too volatile during earnings/AI hype periods
  - ORB strategy doesn't work well on this stock

================================================================================
                         RECOMMENDATION
================================================================================

TRADE ONLY AMD with these settings:
  - Risk per trade: 0.5% (conservative)
  - Max daily loss: 2%
  - Max trades per day: 2
  - Stop after 2 consecutive losses
  - Always use news filter

Expected monthly performance on AMD:
  - Trades: ~15-20
  - Win Rate: 35-40%
  - Expected Return: 3-5%
  - Max Expected Drawdown: 8%

================================================================================
"""


# Quick access function
def create_prop_tracker() -> PropTradingTracker:
    """Create a prop trading tracker with default settings."""
    config = PropTradingConfig(
        allowed_symbol="AMD",
        max_risk_per_trade=0.5,
        max_daily_loss=2.0,
        max_account_drawdown=5.0,
        max_trades_per_day=2,
        max_consecutive_losses=2,
    )
    return PropTradingTracker(config)


if __name__ == "__main__":
    # Print analysis report
    print(get_analysis_report())

    # Create tracker and show readiness
    tracker = create_prop_tracker()
    tracker.print_readiness_report()
