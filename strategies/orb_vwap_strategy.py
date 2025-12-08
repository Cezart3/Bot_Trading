"""
ORB + VWAP Strategy with M1 Confirmation - OPTIMIZED VERSION.

Strategy Logic:
1. Mark first 5-minute candle at NY market open (Opening Range)
2. Wait for next M5 candle to close above/below opening candle
3. Use VWAP as trend filter (above VWAP for longs, below for shorts)
4. Use EMA 20/50 as additional trend confirmation
5. Filter by ATR for volatility (avoid extreme volatility days)
6. Time filter: Best hours 9:35-11:30 and 14:00-15:30
7. Enter on breakout candle close
8. SL: Buffer below/above breakout candle
9. TP: 2.5:1 Risk/Reward ratio

Designed for US stocks: NVDA, AMD, TSLA
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Optional

from models.candle import Candle
from models.order import OrderSide
from strategies.base_strategy import BaseStrategy, StrategySignal, SignalType
from utils.indicators import VWAP, ATR, EMA
from utils.logger import get_logger

logger = get_logger(__name__)


class BreakoutState(Enum):
    """State machine for breakout detection."""
    WAITING_FOR_OPEN = "waiting_for_open"
    OPENING_RANGE_SET = "opening_range_set"
    BREAKOUT_CONFIRMED = "breakout_confirmed"
    WAITING_M1_REJECTION = "waiting_m1_rejection"
    TRADE_TAKEN = "trade_taken"


@dataclass
class OpeningRange:
    """Opening range data."""
    high: float
    low: float
    open_price: float
    close_price: float
    timestamp: datetime


@dataclass
class BreakoutInfo:
    """Breakout candle information."""
    direction: OrderSide
    candle_high: float
    candle_low: float
    close_price: float
    timestamp: datetime


@dataclass
class ORBVWAPConfig:
    """Configuration for ORB + VWAP strategy - OPTIMIZED."""

    # Session times (NY time)
    session_start: time = field(default_factory=lambda: time(9, 30))
    session_end: time = field(default_factory=lambda: time(16, 0))

    # Opening range settings
    opening_range_minutes: int = 5  # First candle duration

    # Entry settings
    use_vwap_filter: bool = True  # Require VWAP confirmation
    use_ema_filter: bool = True  # Require EMA trend confirmation
    use_atr_filter: bool = True  # Filter extreme volatility
    use_time_filter: bool = True  # Only trade during optimal hours
    min_breakout_percent: float = 0.1  # Min % above/below OR for valid breakout

    # EMA settings
    ema_fast_period: int = 20
    ema_slow_period: int = 50

    # ATR volatility filter
    atr_period: int = 14
    atr_min_multiplier: float = 0.5  # Min ATR for trade (avoid low volatility)
    atr_max_multiplier: float = 3.0  # Max ATR (avoid extreme volatility)

    # Time filter - optimal trading windows
    morning_window_start: time = field(default_factory=lambda: time(9, 35))
    morning_window_end: time = field(default_factory=lambda: time(11, 30))
    afternoon_window_start: time = field(default_factory=lambda: time(14, 0))
    afternoon_window_end: time = field(default_factory=lambda: time(15, 30))

    # Risk management - OPTIMIZED
    sl_buffer_dollars: float = 0.05  # $0.05 buffer for SL (slightly wider)
    risk_reward_ratio: float = 2.5  # 2.5:1 R:R for better expectancy

    # M1 rejection settings (for future multi-timeframe)
    rejection_lookback: int = 3
    min_rejection_wick_ratio: float = 0.5


class ORBVWAPStrategy(BaseStrategy):
    """
    Opening Range Breakout + VWAP Strategy with M1 Confirmation.

    Multi-timeframe strategy:
    - M5: Identify opening range and breakout
    - M1: Confirm entry with rejection pattern
    - VWAP: Trend filter
    """

    def __init__(
        self,
        config: Optional[ORBVWAPConfig] = None,
        symbol: str = "NVDA",
        timeframe: str = "M5",
    ):
        """
        Initialize ORB + VWAP strategy.

        Args:
            config: Strategy configuration.
            symbol: Trading symbol.
            timeframe: Candle timeframe.
        """
        super().__init__(
            name="ORB_VWAP_OPT",
            symbol=symbol,
            timeframe=timeframe,
            magic_number=2001,
        )
        self.config = config or ORBVWAPConfig()
        self.pip_size = 0.01  # For stocks

        # Indicators
        self.vwap = VWAP()
        self.atr = ATR(period=self.config.atr_period)
        self.ema_fast = EMA(period=self.config.ema_fast_period)
        self.ema_slow = EMA(period=self.config.ema_slow_period)

        # ATR baseline for volatility filter
        self._atr_baseline: Optional[float] = None
        self._atr_values: list[float] = []

        # State tracking
        self._state = BreakoutState.WAITING_FOR_OPEN
        self._opening_range: Optional[OpeningRange] = None
        self._breakout_info: Optional[BreakoutInfo] = None
        self._current_date: Optional[str] = None

        # M5 candle buffer (for breakout detection)
        self._m5_candles: list[Candle] = []

        # M1 candle buffer (for rejection detection)
        self._m1_candles: list[Candle] = []

        # Track if trade taken today
        self._trade_taken_today = False

        # Statistics for filtering
        self._trades_filtered_by_ema = 0
        self._trades_filtered_by_atr = 0
        self._trades_filtered_by_time = 0

    def initialize(self) -> bool:
        """Initialize strategy."""
        logger.info(f"Initializing {self.name} strategy for {self.symbol}")
        self._reset_daily_state()
        return True

    def reset(self) -> None:
        """Reset strategy state for a new trading day/session."""
        super().reset()
        self._reset_daily_state()

    def _reset_daily_state(self, session_date: Optional[datetime] = None) -> None:
        """Reset state for new trading day."""
        self._state = BreakoutState.WAITING_FOR_OPEN
        self._opening_range = None
        self._breakout_info = None
        self._m5_candles = []
        self._m1_candles = []
        self._trade_taken_today = False
        self.vwap.reset()
        # Don't reset ATR and EMA - they need history
        # self.atr.reset()
        # self.ema_fast.reset()
        # self.ema_slow.reset()

        date_str = session_date.strftime("%Y-%m-%d") if session_date else "new"
        logger.debug(f"Daily state reset for {date_str}")

    def _is_session_open(self, timestamp: datetime) -> bool:
        """Check if market session is open."""
        current_time = timestamp.time()
        return self.config.session_start <= current_time <= self.config.session_end

    def _is_opening_candle(self, candle: Candle) -> bool:
        """Check if this is the first candle of the session."""
        candle_time = candle.timestamp.time()
        session_start = self.config.session_start

        # Allow 5 minutes window for first candle
        end_window = time(
            session_start.hour,
            session_start.minute + self.config.opening_range_minutes
        )

        return session_start <= candle_time < end_window

    def _check_m5_breakout(self, candle: Candle) -> Optional[OrderSide]:
        """
        Check if M5 candle confirms breakout.

        Args:
            candle: M5 candle.

        Returns:
            OrderSide if breakout confirmed, None otherwise.
        """
        if self._opening_range is None:
            return None

        or_high = self._opening_range.high
        or_low = self._opening_range.low
        or_range = or_high - or_low

        min_breakout = or_range * (self.config.min_breakout_percent / 100)

        # Bullish breakout: candle closes above OR high
        if candle.close > or_high + min_breakout:
            logger.info(
                f"BULLISH breakout confirmed | Close: {candle.close:.2f} > OR High: {or_high:.2f}"
            )
            return OrderSide.BUY

        # Bearish breakout: candle closes below OR low
        if candle.close < or_low - min_breakout:
            logger.info(
                f"BEARISH breakout confirmed | Close: {candle.close:.2f} < OR Low: {or_low:.2f}"
            )
            return OrderSide.SELL

        return None

    def _check_m1_rejection(self, candles: list[Candle], direction: OrderSide) -> bool:
        """
        Check for M1 rejection pattern at opening range level.

        For LONG: Look for bullish rejection (long lower wick) at OR high
        For SHORT: Look for bearish rejection (long upper wick) at OR low

        Args:
            candles: Recent M1 candles.
            direction: Expected trade direction.

        Returns:
            True if valid rejection pattern found.
        """
        if len(candles) < 2 or self._opening_range is None:
            return False

        last_candle = candles[-1]
        body = abs(last_candle.close - last_candle.open)

        if direction == OrderSide.BUY:
            # Bullish rejection at OR high level
            or_high = self._opening_range.high

            # Price should be near OR high
            if last_candle.low > or_high + 0.50:  # Too far above
                return False
            if last_candle.high < or_high - 0.50:  # Below level
                return False

            # Look for bullish candle with rejection
            lower_wick = min(last_candle.open, last_candle.close) - last_candle.low

            # Rejection: lower wick should be significant
            if body > 0:
                wick_ratio = lower_wick / body
                if wick_ratio >= self.config.min_rejection_wick_ratio:
                    # Bullish close
                    if last_candle.close > last_candle.open:
                        logger.info(
                            f"M1 BULLISH rejection at OR high {or_high:.2f} | "
                            f"Wick ratio: {wick_ratio:.2f}"
                        )
                        return True

        elif direction == OrderSide.SELL:
            # Bearish rejection at OR low level
            or_low = self._opening_range.low

            # Price should be near OR low
            if last_candle.high < or_low - 0.50:  # Too far below
                return False
            if last_candle.low > or_low + 0.50:  # Above level
                return False

            # Look for bearish candle with rejection
            upper_wick = last_candle.high - max(last_candle.open, last_candle.close)

            # Rejection: upper wick should be significant
            if body > 0:
                wick_ratio = upper_wick / body
                if wick_ratio >= self.config.min_rejection_wick_ratio:
                    # Bearish close
                    if last_candle.close < last_candle.open:
                        logger.info(
                            f"M1 BEARISH rejection at OR low {or_low:.2f} | "
                            f"Wick ratio: {wick_ratio:.2f}"
                        )
                        return True

        return False

    def _check_vwap_filter(self, price: float, direction: OrderSide) -> bool:
        """
        Check VWAP filter.

        Args:
            price: Current price.
            direction: Trade direction.

        Returns:
            True if VWAP confirms direction.
        """
        if not self.config.use_vwap_filter:
            return True

        vwap_value = self.vwap.vwap
        if vwap_value == 0:
            return True  # No VWAP data yet

        if direction == OrderSide.BUY:
            result = price > vwap_value
            if not result:
                logger.debug(f"VWAP filter rejected LONG | Price {price:.2f} < VWAP {vwap_value:.2f}")
            return result

        elif direction == OrderSide.SELL:
            result = price < vwap_value
            if not result:
                logger.debug(f"VWAP filter rejected SHORT | Price {price:.2f} > VWAP {vwap_value:.2f}")
            return result

        return True

    def _check_ema_filter(self, price: float, direction: OrderSide) -> bool:
        """
        Check EMA trend filter.

        For LONG: Price > EMA20 > EMA50 (uptrend)
        For SHORT: Price < EMA20 < EMA50 (downtrend)

        Args:
            price: Current price.
            direction: Trade direction.

        Returns:
            True if EMA confirms trend direction.
        """
        if not self.config.use_ema_filter:
            return True

        ema_fast = self.ema_fast.value
        ema_slow = self.ema_slow.value

        if ema_fast is None or ema_slow is None:
            return True  # Not enough data yet

        if direction == OrderSide.BUY:
            # Uptrend: Price > EMA20 > EMA50
            result = price > ema_fast and ema_fast > ema_slow
            if not result:
                self._trades_filtered_by_ema += 1
                logger.debug(
                    f"EMA filter rejected LONG | Price {price:.2f} | "
                    f"EMA20 {ema_fast:.2f} | EMA50 {ema_slow:.2f}"
                )
            return result

        elif direction == OrderSide.SELL:
            # Downtrend: Price < EMA20 < EMA50
            result = price < ema_fast and ema_fast < ema_slow
            if not result:
                self._trades_filtered_by_ema += 1
                logger.debug(
                    f"EMA filter rejected SHORT | Price {price:.2f} | "
                    f"EMA20 {ema_fast:.2f} | EMA50 {ema_slow:.2f}"
                )
            return result

        return True

    def _check_atr_filter(self, candle: Candle) -> bool:
        """
        Check ATR volatility filter.

        Avoid trading during:
        - Very low volatility (no momentum)
        - Extreme volatility (unpredictable)

        Args:
            candle: Current candle.

        Returns:
            True if volatility is within acceptable range.
        """
        if not self.config.use_atr_filter:
            return True

        current_atr = self.atr.value
        if current_atr is None:
            return True  # Not enough data

        # Store ATR values to calculate baseline
        self._atr_values.append(current_atr)
        if len(self._atr_values) > 50:
            self._atr_values.pop(0)

        # Calculate baseline ATR (average of last 50 values)
        if len(self._atr_values) >= 14:
            self._atr_baseline = sum(self._atr_values) / len(self._atr_values)
        else:
            return True  # Not enough data for baseline

        # Check if current ATR is within acceptable range
        min_atr = self._atr_baseline * self.config.atr_min_multiplier
        max_atr = self._atr_baseline * self.config.atr_max_multiplier

        if current_atr < min_atr:
            self._trades_filtered_by_atr += 1
            logger.debug(
                f"ATR filter rejected: Low volatility | "
                f"ATR {current_atr:.4f} < Min {min_atr:.4f}"
            )
            return False

        if current_atr > max_atr:
            self._trades_filtered_by_atr += 1
            logger.debug(
                f"ATR filter rejected: High volatility | "
                f"ATR {current_atr:.4f} > Max {max_atr:.4f}"
            )
            return False

        return True

    def _check_time_filter(self, timestamp: datetime) -> bool:
        """
        Check if current time is within optimal trading windows.

        Best windows:
        - Morning: 9:35 - 11:30 (opening momentum)
        - Afternoon: 14:00 - 15:30 (before close momentum)

        Avoid:
        - 11:30 - 14:00 (lunch lull)
        - 15:30 - 16:00 (closing volatility)

        Args:
            timestamp: Current timestamp.

        Returns:
            True if within optimal trading window.
        """
        if not self.config.use_time_filter:
            return True

        current_time = timestamp.time()

        # Morning window
        in_morning = (
            self.config.morning_window_start <= current_time <= self.config.morning_window_end
        )

        # Afternoon window
        in_afternoon = (
            self.config.afternoon_window_start <= current_time <= self.config.afternoon_window_end
        )

        if not (in_morning or in_afternoon):
            self._trades_filtered_by_time += 1
            logger.debug(f"Time filter rejected: {current_time} not in optimal windows")
            return False

        return True

    def process_m5_candle(self, candle: Candle) -> Optional[StrategySignal]:
        """
        Process M5 candle for opening range and breakout detection.

        Args:
            candle: M5 timeframe candle.

        Returns:
            StrategySignal if breakout confirmed and M1 entry awaited.
        """
        # Check for new day
        candle_date = candle.timestamp.strftime("%Y-%m-%d")
        if self._current_date != candle_date:
            self._reset_daily_state(candle.timestamp)
            self._current_date = candle_date

        # Update VWAP
        self.vwap.update(candle)
        self.atr.update(candle)

        # Skip if outside session
        if not self._is_session_open(candle.timestamp):
            return None

        # Skip if trade already taken today
        if self._trade_taken_today:
            return None

        # Store M5 candle
        self._m5_candles.append(candle)
        if len(self._m5_candles) > 20:
            self._m5_candles.pop(0)

        # State machine
        if self._state == BreakoutState.WAITING_FOR_OPEN:
            # Look for opening range candle
            if self._is_opening_candle(candle):
                self._opening_range = OpeningRange(
                    high=candle.high,
                    low=candle.low,
                    open_price=candle.open,
                    close_price=candle.close,
                    timestamp=candle.timestamp,
                )
                self._state = BreakoutState.OPENING_RANGE_SET

                logger.info(
                    f"Opening Range SET | High: {candle.high:.2f} | "
                    f"Low: {candle.low:.2f} | Range: {candle.high - candle.low:.2f}"
                )

        elif self._state == BreakoutState.OPENING_RANGE_SET:
            # Wait for breakout confirmation
            breakout_direction = self._check_m5_breakout(candle)

            if breakout_direction:
                # Check VWAP filter
                if not self._check_vwap_filter(candle.close, breakout_direction):
                    logger.info("Breakout rejected by VWAP filter")
                    return None

                self._breakout_info = BreakoutInfo(
                    direction=breakout_direction,
                    candle_high=candle.high,
                    candle_low=candle.low,
                    close_price=candle.close,
                    timestamp=candle.timestamp,
                )
                self._state = BreakoutState.WAITING_M1_REJECTION

                logger.info(
                    f"Breakout confirmed - waiting for M1 rejection | "
                    f"Direction: {breakout_direction.value}"
                )

        return None

    def process_m1_candle(self, candle: Candle) -> Optional[StrategySignal]:
        """
        Process M1 candle for rejection entry.

        Args:
            candle: M1 timeframe candle.

        Returns:
            Entry signal if rejection pattern confirmed.
        """
        if self._state != BreakoutState.WAITING_M1_REJECTION:
            return None

        if self._breakout_info is None or self._opening_range is None:
            return None

        # Store M1 candle
        self._m1_candles.append(candle)
        if len(self._m1_candles) > 10:
            self._m1_candles.pop(0)

        # Check for rejection pattern
        direction = self._breakout_info.direction

        if self._check_m1_rejection(self._m1_candles, direction):
            # Generate entry signal
            if direction == OrderSide.BUY:
                entry_price = candle.close
                stop_loss = self._breakout_info.candle_low - self.config.sl_buffer_dollars
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * self.config.risk_reward_ratio)

                signal = StrategySignal(
                    signal_type=SignalType.LONG,
                    symbol=self.symbol,
                    price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    timestamp=candle.timestamp,
                    reason=f"ORB breakout above {self._opening_range.high:.2f} with M1 rejection",
                    metadata={
                        "strategy": "ORB_VWAP",
                        "or_high": self._opening_range.high,
                        "or_low": self._opening_range.low,
                        "vwap": self.vwap.vwap,
                        "breakout_candle_low": self._breakout_info.candle_low,
                        "risk_dollars": risk,
                    },
                )

            else:  # SELL
                entry_price = candle.close
                stop_loss = self._breakout_info.candle_high + self.config.sl_buffer_dollars
                risk = stop_loss - entry_price
                take_profit = entry_price - (risk * self.config.risk_reward_ratio)

                signal = StrategySignal(
                    signal_type=SignalType.SHORT,
                    symbol=self.symbol,
                    price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    timestamp=candle.timestamp,
                    reason=f"ORB breakout below {self._opening_range.low:.2f} with M1 rejection",
                    metadata={
                        "strategy": "ORB_VWAP",
                        "or_high": self._opening_range.high,
                        "or_low": self._opening_range.low,
                        "vwap": self.vwap.vwap,
                        "breakout_candle_high": self._breakout_info.candle_high,
                        "risk_dollars": risk,
                    },
                )

            self._state = BreakoutState.TRADE_TAKEN
            self._trade_taken_today = True

            sl_str = f"{signal.stop_loss:.2f}" if signal.stop_loss else "None"
            tp_str = f"{signal.take_profit:.2f}" if signal.take_profit else "None"

            logger.info(
                f"ENTRY SIGNAL: {signal.signal_type.value} @ {signal.price:.2f} | "
                f"SL: {sl_str} | TP: {tp_str} | VWAP: {self.vwap.vwap:.2f}"
            )

            self._last_signal = signal
            return signal

        return None

    def on_candle(
        self,
        candle: Candle,
        candles: list[Candle],
    ) -> Optional[StrategySignal]:
        """
        Process candle - OPTIMIZED VERSION with all filters.

        Filters applied:
        1. VWAP trend filter
        2. EMA 20/50 trend confirmation
        3. ATR volatility filter
        4. Time window filter

        Args:
            candle: Latest candle.
            candles: Historical candles including the latest.

        Returns:
            Trading signal or None.
        """
        # Check for new day
        candle_date = candle.timestamp.strftime("%Y-%m-%d")
        if self._current_date != candle_date:
            self._reset_daily_state(candle.timestamp)
            self._current_date = candle_date

        # Update ALL indicators
        self.vwap.update(candle)
        self.atr.update(candle)
        self.ema_fast.update(candle.close)
        self.ema_slow.update(candle.close)

        # Skip if outside session
        if not self._is_session_open(candle.timestamp):
            return None

        # Skip if trade already taken today
        if self._trade_taken_today:
            return None

        # Store candle
        self._m5_candles.append(candle)
        if len(self._m5_candles) > 20:
            self._m5_candles.pop(0)

        # State machine
        if self._state == BreakoutState.WAITING_FOR_OPEN:
            # Look for opening range candle
            if self._is_opening_candle(candle):
                self._opening_range = OpeningRange(
                    high=candle.high,
                    low=candle.low,
                    open_price=candle.open,
                    close_price=candle.close,
                    timestamp=candle.timestamp,
                )
                self._state = BreakoutState.OPENING_RANGE_SET

                logger.info(
                    f"Opening Range SET | High: {candle.high:.2f} | "
                    f"Low: {candle.low:.2f} | Range: {candle.high - candle.low:.2f}"
                )

        elif self._state == BreakoutState.OPENING_RANGE_SET:
            # Wait for breakout confirmation and generate signal directly
            breakout_direction = self._check_m5_breakout(candle)

            if breakout_direction:
                # ===== APPLY ALL FILTERS =====

                # 1. Time filter - only trade during optimal hours
                if not self._check_time_filter(candle.timestamp):
                    logger.debug("Breakout rejected by TIME filter")
                    return None

                # 2. VWAP filter - trade with trend
                if not self._check_vwap_filter(candle.close, breakout_direction):
                    logger.debug("Breakout rejected by VWAP filter")
                    return None

                # 3. EMA filter - confirm trend direction
                if not self._check_ema_filter(candle.close, breakout_direction):
                    logger.debug("Breakout rejected by EMA filter")
                    return None

                # 4. ATR filter - avoid extreme volatility
                if not self._check_atr_filter(candle):
                    logger.debug("Breakout rejected by ATR filter")
                    return None

                # ===== ALL FILTERS PASSED - GENERATE SIGNAL =====

                # Generate entry signal on breakout
                if breakout_direction == OrderSide.BUY:
                    entry_price = candle.close
                    stop_loss = candle.low - self.config.sl_buffer_dollars
                    risk = entry_price - stop_loss
                    take_profit = entry_price + (risk * self.config.risk_reward_ratio)

                    signal = StrategySignal(
                        signal_type=SignalType.LONG,
                        symbol=self.symbol,
                        price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        timestamp=candle.timestamp,
                        reason=f"ORB LONG | OR High: {self._opening_range.high:.2f} | EMA+VWAP confirmed",
                        metadata={
                            "or_high": self._opening_range.high,
                            "or_low": self._opening_range.low,
                            "vwap": self.vwap.vwap,
                            "ema_fast": self.ema_fast.value,
                            "ema_slow": self.ema_slow.value,
                            "atr": self.atr.value,
                            "risk": risk,
                            "rr_ratio": self.config.risk_reward_ratio,
                        },
                    )
                else:  # SHORT
                    entry_price = candle.close
                    stop_loss = candle.high + self.config.sl_buffer_dollars
                    risk = stop_loss - entry_price
                    take_profit = entry_price - (risk * self.config.risk_reward_ratio)

                    signal = StrategySignal(
                        signal_type=SignalType.SHORT,
                        symbol=self.symbol,
                        price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        timestamp=candle.timestamp,
                        reason=f"ORB SHORT | OR Low: {self._opening_range.low:.2f} | EMA+VWAP confirmed",
                        metadata={
                            "or_high": self._opening_range.high,
                            "or_low": self._opening_range.low,
                            "vwap": self.vwap.vwap,
                            "ema_fast": self.ema_fast.value,
                            "ema_slow": self.ema_slow.value,
                            "atr": self.atr.value,
                            "risk": risk,
                            "rr_ratio": self.config.risk_reward_ratio,
                        },
                    )

                self._state = BreakoutState.TRADE_TAKEN
                self._trade_taken_today = True

                sl_str = f"{signal.stop_loss:.2f}" if signal.stop_loss else "None"
                tp_str = f"{signal.take_profit:.2f}" if signal.take_profit else "None"

                logger.info(
                    f"SIGNAL: {signal.signal_type.value} @ {signal.price:.2f} | "
                    f"SL: {sl_str} | TP: {tp_str} | VWAP: {self.vwap.vwap:.2f}"
                )

                self._last_signal = signal
                return signal

        return None

    def on_tick(
        self,
        bid: float,
        ask: float,
        timestamp: datetime,
    ) -> Optional[StrategySignal]:
        """
        Process a tick update.

        Not used for this candle-based strategy.

        Args:
            bid: Current bid price.
            ask: Current ask price.
            timestamp: Tick timestamp.

        Returns:
            None - this strategy only uses candles.
        """
        return None

    def get_status(self) -> dict:
        """Get current strategy status including filter statistics."""
        or_info = None
        if self._opening_range:
            or_info = {
                "high": self._opening_range.high,
                "low": self._opening_range.low,
                "range": self._opening_range.high - self._opening_range.low,
            }

        breakout_info = None
        if self._breakout_info:
            breakout_info = {
                "direction": self._breakout_info.direction.value,
                "candle_high": self._breakout_info.candle_high,
                "candle_low": self._breakout_info.candle_low,
            }

        return {
            "strategy": self.name,
            "version": "OPTIMIZED",
            "state": self._state.value,
            "opening_range": or_info,
            "breakout": breakout_info,
            # Indicators
            "vwap": self.vwap.vwap,
            "vwap_upper": self.vwap.upper_band,
            "vwap_lower": self.vwap.lower_band,
            "ema_fast": self.ema_fast.value,
            "ema_slow": self.ema_slow.value,
            "atr": self.atr.value,
            "atr_baseline": self._atr_baseline,
            # Status
            "trade_taken_today": self._trade_taken_today,
            "m5_candles": len(self._m5_candles),
            # Filter statistics
            "filters": {
                "ema_filtered": self._trades_filtered_by_ema,
                "atr_filtered": self._trades_filtered_by_atr,
                "time_filtered": self._trades_filtered_by_time,
            },
            # Configuration
            "config": {
                "risk_reward_ratio": self.config.risk_reward_ratio,
                "use_vwap_filter": self.config.use_vwap_filter,
                "use_ema_filter": self.config.use_ema_filter,
                "use_atr_filter": self.config.use_atr_filter,
                "use_time_filter": self.config.use_time_filter,
            },
        }
