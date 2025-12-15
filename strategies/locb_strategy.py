"""
London Opening Candle Breakout (LOCB) Strategy Implementation.

Strategy Rules:
1. First M5 candle at London open (08:00 UTC) - mark HIGH/LOW
2. Second M5 candle closes above HIGH (bullish) or below LOW (bearish) -> direction confirmed
3. Switch to M1, wait for retest of HIGH (for buy) or LOW (for sell)
4. Wait for confirmation: CHoCH / Inversed FVG / Engulfing
5. Entry with SL below/above M5 opening candle extreme
6. TP: Configurable R:R (default 3:1)

Confirmations:
- CHoCH (Change of Character): Break of recent swing high/low
- iFVG (Inversed Fair Value Gap): FVG that gets violated
- Engulfing: Bullish/Bearish engulfing pattern
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List
import pytz

from models.candle import Candle
from models.position import Position
from strategies.base_strategy import BaseStrategy, SignalType, StrategySignal
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class M1CandleData:
    """M1 candle data for confirmation patterns."""
    time: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass
class FVG:
    """Fair Value Gap."""
    type: str  # "bullish" or "bearish"
    high: float
    low: float
    candle_time: datetime


@dataclass
class LOCBState:
    """State tracking for LOCB strategy."""

    # Opening candle info
    opening_candle_high: Optional[float] = None
    opening_candle_low: Optional[float] = None
    opening_candle_time: Optional[datetime] = None
    opening_candle_valid: bool = False

    # Direction and phases
    direction: Optional[str] = None  # "long" or "short"
    breakout_confirmed: bool = False
    retest_level: Optional[float] = None
    retest_found: bool = False
    confirmation_found: bool = False
    confirmation_type: Optional[str] = None
    confirmation_candle: Optional[M1CandleData] = None

    # Counters
    candles_since_breakout: int = 0
    candles_since_retest: int = 0

    # M1 candle buffers
    opening_period_candles: List[M1CandleData] = field(default_factory=list)  # First 5 min
    breakout_period_candles: List[M1CandleData] = field(default_factory=list)  # Second 5 min
    m1_candles: List[M1CandleData] = field(default_factory=list)  # For pattern detection

    # Session tracking
    session_date: Optional[datetime] = None
    trades_today: int = 0
    max_trades_per_day: int = 1

    # Entry info
    entry_price: Optional[float] = None
    entry_triggered: bool = False


class LOCBStrategy(BaseStrategy):
    """
    London Opening Candle Breakout Strategy.

    This strategy trades breakouts from the first 5-minute candle of London session,
    with pullback entry after confirmation patterns.

    Optimized for shorter trades with tighter stop losses.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "M1",  # Primary timeframe for entries
        magic_number: int = 12346,
        # Session settings (server time - Teletrade uses UTC+4 equivalent)
        # London opens at 08:00 UTC = 12:00 server time
        london_open_hour: int = 12,  # 08:00 UTC = 12:00 server
        session_end_hour: int = 15,  # 11:00 UTC = 15:00 server
        timezone: str = "Etc/GMT-4",  # MT5 server timezone (Teletrade)
        # Strategy settings
        risk_reward_ratio: float = 1.5,
        sl_buffer_pips: float = 10.0,
        min_range_pips: float = 2.0,
        max_range_pips: float = 30.0,
        # Timing parameters (in M1 candles)
        max_retest_candles: int = 30,
        max_confirm_candles: int = 20,
        retest_tolerance_pips: float = 3.0,
        # Trade management
        max_trades_per_day: int = 1,
        close_before_session_end_minutes: int = 10,
    ):
        """
        Initialize LOCB strategy.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: Primary timeframe for entries (M1 recommended)
            magic_number: Unique identifier for orders``
            london_open_hour: Hour of London open in server time
            session_end_hour: Hour to stop trading in server time
            timezone: Server timezone
            risk_reward_ratio: Take profit / stop loss ratio
            sl_buffer_pips: Extra pips below/above SL level
            min_range_pips: Minimum opening candle range
            max_range_pips: Maximum opening candle range
            max_retest_candles: Max M1 candles to wait for retest
            max_confirm_candles: Max M1 candles after retest to wait for confirmation
            retest_tolerance_pips: How close price must get to level for retest
            max_trades_per_day: Maximum trades per session
            close_before_session_end_minutes: Close positions X minutes before session end
        """
        super().__init__(
            name="LOCB",
            symbol=symbol,
            timeframe=timeframe,
            magic_number=magic_number,
        )

        # Session settings
        self.london_open_hour = london_open_hour
        self.session_end_hour = session_end_hour
        self.timezone = timezone
        self.tz = pytz.timezone(timezone)

        # Strategy settings
        self.risk_reward_ratio = risk_reward_ratio
        self.sl_buffer_pips = sl_buffer_pips
        self.min_range_pips = min_range_pips
        self.max_range_pips = max_range_pips

        # Timing parameters
        self.max_retest_candles = max_retest_candles
        self.max_confirm_candles = max_confirm_candles
        self.retest_tolerance_pips = retest_tolerance_pips

        # Trade management
        self.max_trades_per_day = max_trades_per_day
        self.close_before_session_end_minutes = close_before_session_end_minutes

        # Pip size (will be set based on symbol)
        self.pip_size = 0.0001  # Default for forex

        # State
        self._state = LOCBState(max_trades_per_day=self.max_trades_per_day)

    def initialize(self) -> bool:
        """Initialize strategy."""
        logger.info(f"Initializing LOCB strategy for {self.symbol}")
        logger.info(f"London open: {self.london_open_hour}:00 server time")
        logger.info(f"Session end: {self.session_end_hour}:00 server time")
        logger.info(f"R:R ratio: {self.risk_reward_ratio}:1")
        self._reset_daily_state()
        return True

    def _reset_daily_state(self, session_date=None) -> None:
        """Reset state for a new trading day."""
        self._state = LOCBState(max_trades_per_day=self.max_trades_per_day)
        self._state.session_date = session_date or datetime.now(self.tz).date()
        logger.info("LOCB state reset for new session")

    def _check_new_session(self, timestamp: datetime) -> None:
        """Check if it's a new trading session and reset if needed.

        Uses timestamp directly as it comes from MT5 in server time.
        """
        current_date = timestamp.date()

        if self._state.session_date != current_date:
            self._reset_daily_state(current_date)

    def _is_in_opening_range_period(self, candle: Candle) -> bool:
        """Check if we're in the first 5 minutes (opening candle period).

        Uses candle timestamp directly as it comes from MT5 in server time.
        """
        ts = candle.timestamp
        # Candle timestamps from MT5 are already in server time
        return ts.hour == self.london_open_hour and ts.minute < 5

    def _is_in_breakout_check_period(self, candle: Candle) -> bool:
        """Check if we're in the second 5-minute period (breakout check).

        Uses candle timestamp directly as it comes from MT5 in server time.
        """
        ts = candle.timestamp
        return ts.hour == self.london_open_hour and 5 <= ts.minute < 10

    def _is_past_breakout_period(self, candle: Candle) -> bool:
        """Check if we're past the first 10 minutes (retest/confirm phase).

        Uses candle timestamp directly as it comes from MT5 in server time.
        """
        ts = candle.timestamp
        return (ts.hour == self.london_open_hour and ts.minute >= 10) or ts.hour > self.london_open_hour

    def _detect_choch(self, direction: str, lookback: int = 8) -> bool:
        """
        Detect Change of Character (CHoCH).

        For bullish CHoCH: price breaks above recent swing high
        For bearish CHoCH: price breaks below recent swing low
        """
        candles = self._state.m1_candles
        if len(candles) < lookback + 2:
            return False

        recent = candles[-(lookback + 1):-1]
        current = candles[-1]

        if direction == "bullish":
            swing_high = max(c.high for c in recent)
            return current.close > swing_high
        else:  # bearish
            swing_low = min(c.low for c in recent)
            return current.close < swing_low

    def _find_fvgs(self) -> List[FVG]:
        """Find Fair Value Gaps in recent M1 candles."""
        candles = self._state.m1_candles
        fvgs = []

        if len(candles) < 3:
            return fvgs

        for i in range(2, len(candles)):
            c0 = candles[i - 2]
            c2 = candles[i]

            # Bullish FVG
            if c2.low > c0.high:
                fvgs.append(FVG(
                    type="bullish",
                    high=c2.low,
                    low=c0.high,
                    candle_time=c2.time
                ))

            # Bearish FVG
            if c2.high < c0.low:
                fvgs.append(FVG(
                    type="bearish",
                    high=c0.low,
                    low=c2.high,
                    candle_time=c2.time
                ))

        return fvgs

    def _detect_ifvg(self, direction: str, lookback: int = 15) -> bool:
        """
        Detect Inversed Fair Value Gap (iFVG).

        iFVG = FVG that gets violated (price closes through it)
        """
        candles = self._state.m1_candles
        if len(candles) < lookback:
            return False

        # Find FVGs in recent candles (excluding current)
        recent_candles = candles[-lookback:-1]
        temp_state_candles = self._state.m1_candles
        self._state.m1_candles = recent_candles
        fvgs = self._find_fvgs()
        self._state.m1_candles = temp_state_candles

        if not fvgs:
            return False

        current = candles[-1]

        for fvg in fvgs:
            if direction == "bullish" and fvg.type == "bearish":
                if current.close > fvg.high:
                    return True
            elif direction == "bearish" and fvg.type == "bullish":
                if current.close < fvg.low:
                    return True

        return False

    def _detect_engulfing(self, direction: str) -> bool:
        """Detect Engulfing pattern."""
        candles = self._state.m1_candles
        if len(candles) < 2:
            return False

        prev = candles[-2]
        curr = candles[-1]

        prev_body_high = max(prev.open, prev.close)
        prev_body_low = min(prev.open, prev.close)
        curr_body_high = max(curr.open, curr.close)
        curr_body_low = min(curr.open, curr.close)

        if direction == "bullish":
            prev_bearish = prev.close < prev.open
            curr_bullish = curr.close > curr.open
            engulfs = curr_body_low <= prev_body_low and curr_body_high >= prev_body_high
            return prev_bearish and curr_bullish and engulfs
        else:  # bearish
            prev_bullish = prev.close > prev.open
            curr_bearish = curr.close < curr.open
            engulfs = curr_body_low <= prev_body_low and curr_body_high >= prev_body_high
            return prev_bullish and curr_bearish and engulfs

    def _check_retest(self, candle: Candle) -> bool:
        """Check if price has retested the level."""
        if self._state.retest_level is None:
            return False

        tolerance = self.retest_tolerance_pips * self.pip_size
        level = self._state.retest_level

        if self._state.direction == "long":
            # For buy: price should touch or approach level from above
            return candle.low <= level + tolerance and candle.close > level
        else:  # short
            # For sell: price should touch or approach level from below
            return candle.high >= level - tolerance and candle.close < level

    def _check_confirmation(self) -> tuple[bool, str]:
        """Check for any confirmation pattern."""
        direction = "bullish" if self._state.direction == "long" else "bearish"

        # Check CHoCH first (strongest signal)
        if self._detect_choch(direction, lookback=8):
            return True, "CHoCH"

        # Check iFVG
        if self._detect_ifvg(direction, lookback=15):
            return True, "iFVG"

        # Check Engulfing
        if self._detect_engulfing(direction):
            return True, "Engulfing"

        return False, ""

    def on_candle(
        self,
        candle: Candle,
        candles: list[Candle],
    ) -> Optional[StrategySignal]:
        """Process new candle and check for signals."""
        if not self._enabled:
            return None

        # Check for new session
        self._check_new_session(candle.timestamp)

        # Check if max trades reached
        if self._state.trades_today >= self._state.max_trades_per_day:
            return None

        # Check if entry already triggered today
        if self._state.entry_triggered:
            return None

        # Check if within trading hours (using candle timestamp which is in MT5 server time)
        ts = candle.timestamp
        if ts.hour < self.london_open_hour or ts.hour >= self.session_end_hour:
            return None

        # Create M1 candle data
        m1_candle = M1CandleData(
            time=candle.timestamp,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close
        )

        # Phase 0: Accumulate opening period candles (first 5 minutes)
        if self._is_in_opening_range_period(candle):
            if len(self._state.opening_period_candles) == 0:
                logger.info(f"LOCB [{self.symbol}] Opening range period started at {ts.strftime('%H:%M:%S')}")
            self._state.opening_period_candles.append(m1_candle)
            logger.debug(f"LOCB [{self.symbol}] Accumulated candle {len(self._state.opening_period_candles)}/5 in opening range")
            return None

        # Phase 0.5: Calculate opening range from accumulated candles
        if not self._state.opening_candle_valid and self._state.opening_period_candles:
            # Calculate high/low from all candles in opening period
            oc_high = max(c.high for c in self._state.opening_period_candles)
            oc_low = min(c.low for c in self._state.opening_period_candles)
            oc_range = oc_high - oc_low
            oc_range_pips = oc_range / self.pip_size

            if self.min_range_pips <= oc_range_pips <= self.max_range_pips:
                self._state.opening_candle_high = oc_high
                self._state.opening_candle_low = oc_low
                self._state.opening_candle_time = self._state.opening_period_candles[0].time
                self._state.opening_candle_valid = True
                logger.info(
                    f"LOCB Opening range (5 min): High={oc_high:.5f}, Low={oc_low:.5f}, "
                    f"Range={oc_range_pips:.1f} pips"
                )
            else:
                logger.debug(f"Opening range {oc_range_pips:.1f} pips out of bounds [{self.min_range_pips}-{self.max_range_pips}]")
                self._state.entry_triggered = True  # Skip this day
                return None

        # Phase 1: Check for breakout during second 5-minute period
        if self._is_in_breakout_check_period(candle) and self._state.opening_candle_valid and not self._state.breakout_confirmed:
            self._state.breakout_period_candles.append(m1_candle)

            oc_high = self._state.opening_candle_high
            oc_low = self._state.opening_candle_low

            # Check if any candle in breakout period closes above/below range
            if candle.close > oc_high:
                self._state.direction = "long"
                self._state.retest_level = oc_high
                self._state.breakout_confirmed = True
                logger.info(f"LOCB LONG breakout confirmed: close {candle.close:.5f} > high {oc_high:.5f}")
            elif candle.close < oc_low:
                self._state.direction = "short"
                self._state.retest_level = oc_low
                self._state.breakout_confirmed = True
                logger.info(f"LOCB SHORT breakout confirmed: close {candle.close:.5f} < low {oc_low:.5f}")
            return None

        # If breakout period ended without breakout, skip day
        if self._is_past_breakout_period(candle) and self._state.opening_candle_valid and not self._state.breakout_confirmed:
            if not self._state.entry_triggered:
                logger.debug("No breakout during second period, skipping day")
                self._state.entry_triggered = True
            return None

        # Phase 2 & 3: Look for retest and confirmation (on M1 candles)
        if self._state.breakout_confirmed and not self._state.entry_triggered:
            # Add candle to M1 buffer (m1_candle already created above)
            self._state.m1_candles.append(m1_candle)
            self._state.candles_since_breakout += 1

            # Phase 2: Look for retest
            if not self._state.retest_found:
                if self._state.candles_since_breakout > self.max_retest_candles:
                    logger.debug(f"Retest timeout after {self._state.candles_since_breakout} candles")
                    self._state.entry_triggered = True  # Skip this day
                    return None

                if self._check_retest(candle):
                    self._state.retest_found = True
                    self._state.candles_since_retest = 0
                    logger.info(f"LOCB Retest found at {candle.timestamp}")
                return None

            # Phase 3: Look for confirmation after retest
            self._state.candles_since_retest += 1

            if self._state.candles_since_retest > self.max_confirm_candles:
                logger.debug(f"Confirmation timeout after {self._state.candles_since_retest} candles")
                self._state.entry_triggered = True  # Skip this day
                return None

            confirmed, confirm_type = self._check_confirmation()

            if confirmed:
                self._state.confirmation_found = True
                self._state.confirmation_type = confirm_type
                self._state.confirmation_candle = m1_candle
                self._state.entry_triggered = True

                # Generate entry signal
                entry_price = candle.close

                if self._state.direction == "long":
                    sl = self._state.confirmation_candle.low - (self.sl_buffer_pips * self.pip_size)
                    risk = entry_price - sl
                    tp = entry_price + (risk * self.risk_reward_ratio)
                    signal_type = SignalType.LONG
                else:  # short
                    sl = self._state.confirmation_candle.high + (self.sl_buffer_pips * self.pip_size)
                    risk = sl - entry_price
                    tp = entry_price - (risk * self.risk_reward_ratio)
                    signal_type = SignalType.SHORT

                sl_pips = abs(entry_price - sl) / self.pip_size

                logger.info(
                    f"LOCB {self._state.direction.upper()} entry: {entry_price:.5f}, "
                    f"SL: {sl:.5f} ({sl_pips:.1f}p), TP: {tp:.5f}, "
                    f"Confirmation: {confirm_type}"
                )

                signal = StrategySignal(
                    signal_type=signal_type,
                    symbol=self.symbol,
                    price=entry_price,
                    stop_loss=sl,
                    take_profit=tp,
                    confidence=0.8,
                    reason=f"LOCB {self._state.direction.upper()} - {confirm_type} confirmation",
                    metadata={
                        "opening_candle_high": self._state.opening_candle_high,
                        "opening_candle_low": self._state.opening_candle_low,
                        "confirmation_type": confirm_type,
                        "sl_pips": sl_pips,
                        "candles_to_entry": self._state.candles_since_breakout,
                    },
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
        """Process tick update."""
        # For now, we only use candle-based signals
        return None

    def should_exit(
        self,
        position: Position,
        current_price: float,
        candles: Optional[list[Candle]] = None,
    ) -> Optional[StrategySignal]:
        """Check if position should be exited.

        Uses candle timestamps (MT5 server time) for accurate session timing.
        """
        # Use candle timestamp for server time, fallback to local time
        if candles and len(candles) > 0:
            now = candles[-1].timestamp
        else:
            now = datetime.now(self.tz)

        # Check time-based exit
        session_end = now.replace(
            hour=self.session_end_hour,
            minute=0,
            second=0
        )
        close_time = session_end - timedelta(minutes=self.close_before_session_end_minutes)

        if now >= close_time:
            logger.info(f"Session ending soon (server time: {now.strftime('%H:%M')}), closing position")
            exit_type = (
                SignalType.EXIT_LONG if position.is_long else SignalType.EXIT_SHORT
            )
            return StrategySignal(
                signal_type=exit_type,
                symbol=self.symbol,
                price=current_price,
                reason="Session end - time-based exit",
            )

        return None

    def on_position_opened(self, position: Position) -> None:
        """Called when position is opened."""
        self._state.trades_today += 1
        self._state.entry_price = position.entry_price
        logger.info(
            f"LOCB position opened: {position.side} {position.quantity} @ {position.entry_price}"
        )

    def on_position_closed(self, position: Position, pnl: float) -> None:
        """Called when position is closed."""
        logger.info(
            f"LOCB position closed: {position.side} @ {position.exit_price}, PnL: {pnl:.2f}"
        )

    def calculate_stop_loss(
        self,
        entry_price: float,
        is_long: bool,
        candles: Optional[list[Candle]] = None,
    ) -> Optional[float]:
        """Calculate stop loss based on confirmation candle."""
        if self._state.confirmation_candle is None:
            return None

        buffer = self.sl_buffer_pips * self.pip_size

        if is_long:
            return self._state.confirmation_candle.low - buffer
        else:
            return self._state.confirmation_candle.high + buffer

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        is_long: bool,
    ) -> Optional[float]:
        """Calculate take profit based on risk/reward ratio."""
        risk = abs(entry_price - stop_loss)
        reward = risk * self.risk_reward_ratio

        if is_long:
            return entry_price + reward
        else:
            return entry_price - reward

    def get_status(self) -> dict:
        """Get strategy status."""
        base_status = super().get_status()
        base_status.update({
            "opening_candle_high": self._state.opening_candle_high,
            "opening_candle_low": self._state.opening_candle_low,
            "opening_candle_valid": self._state.opening_candle_valid,
            "direction": self._state.direction,
            "breakout_confirmed": self._state.breakout_confirmed,
            "retest_found": self._state.retest_found,
            "confirmation_found": self._state.confirmation_found,
            "confirmation_type": self._state.confirmation_type,
            "entry_triggered": self._state.entry_triggered,
            "trades_today": self._state.trades_today,
            "max_trades_per_day": self._state.max_trades_per_day,
            "london_open_hour": self.london_open_hour,
            "session_end_hour": self.session_end_hour,
            "risk_reward_ratio": self.risk_reward_ratio,
        })
        return base_status

    def reset(self) -> None:
        """Reset strategy for new session."""
        super().reset()
        self._reset_daily_state()