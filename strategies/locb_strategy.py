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
from utils.news_filter import NewsFilter, NewsFilterConfig, NewsImpact

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
class LiquidityLevel:
    """Represents a liquidity level (swing high/low, equal highs/lows)."""
    price: float
    type: str  # "swing_high", "swing_low", "equal_highs", "equal_lows"
    strength: float  # 0-1, higher = more significant
    candle_time: datetime
    touches: int = 1  # Number of times price touched this level


@dataclass
class TradingSession:
    """Configuration for a trading session."""
    name: str  # "london" or "ny"
    open_hour: int  # Hour when session opens (server time)
    open_minute: int  # Minute when session opens
    end_hour: int  # Hour when to stop trading (server time)
    end_minute: int = 0


@dataclass
class SessionState:
    """State for a single trading session."""
    name: str
    # Opening candle info
    opening_candle_high: Optional[float] = None
    opening_candle_low: Optional[float] = None
    opening_candle_time: Optional[datetime] = None
    opening_candle_valid: bool = False

    # Direction and phases
    direction: Optional[str] = None
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
    opening_period_candles: List[M1CandleData] = field(default_factory=list)
    breakout_period_candles: List[M1CandleData] = field(default_factory=list)
    m1_candles: List[M1CandleData] = field(default_factory=list)

    # Breakout candle - the candle that closed beyond the opening range
    # SL should be placed beyond this candle (below low for LONG, above high for SHORT)
    breakout_candle: Optional[M1CandleData] = None

    # Entry tracking
    entry_price: Optional[float] = None
    entry_triggered: bool = False
    trade_taken: bool = False  # True if a trade was executed this session


@dataclass
class SignalScore:
    """Score components for signal quality evaluation."""

    total: float = 0.0
    confirmation_score: float = 0.0  # CHoCH > iFVG > Engulfing
    sl_distance_score: float = 0.0   # Tighter SL = better (but not too tight)
    breakout_strength_score: float = 0.0  # How strong was the breakout
    retest_quality_score: float = 0.0  # How clean was the retest
    timing_score: float = 0.0  # Earlier in session = better

    def calculate_total(self) -> float:
        """Calculate weighted total score."""
        self.total = (
            self.confirmation_score * 0.30 +      # 30% - confirmation type
            self.sl_distance_score * 0.25 +       # 25% - risk/reward potential
            self.breakout_strength_score * 0.20 + # 20% - breakout momentum
            self.retest_quality_score * 0.15 +    # 15% - clean retest
            self.timing_score * 0.10              # 10% - timing in session
        )
        return self.total


@dataclass
class LOCBState:
    """State tracking for LOCB strategy with multi-session support."""

    # Day tracking
    current_date: Optional[datetime] = None
    trades_today: int = 0
    max_trades_per_day: int = 2  # 1 per session (London + NY)

    # Session states - keyed by session name
    session_states: dict = field(default_factory=dict)

    # Current active session
    active_session: Optional[str] = None

    # Quality metrics (for current session)
    breakout_candle_body_percent: float = 0.0
    retest_wick_touch: bool = False

    def get_session_state(self, session_name: str) -> Optional[SessionState]:
        """Get state for a specific session."""
        return self.session_states.get(session_name)

    def reset_session(self, session_name: str) -> SessionState:
        """Reset state for a specific session."""
        state = SessionState(name=session_name)
        self.session_states[session_name] = state
        return state

    def reset_day(self) -> None:
        """Reset all state for a new trading day."""
        self.trades_today = 0
        self.session_states = {}
        self.active_session = None


class LOCBStrategy(BaseStrategy):
    """
    London & NY Opening Candle Breakout Strategy.

    This strategy trades breakouts from the first 5-minute candle of London and NY sessions,
    with pullback entry after confirmation patterns (CHoCH, iFVG, Engulfing).

    Features:
    - Multi-session support (London + NY)
    - Dynamic TP based on liquidity levels
    - Spread-aware SL calculation
    - Signal quality scoring
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "M1",
        magic_number: int = 12346,
        timezone: str = "Etc/GMT-4",  # MT5 server timezone (Teletrade = UTC+4)
        # Session settings (all times in SERVER TIME)
        # London: 08:00 UTC = 12:00 server, ends 11:00 UTC = 15:00 server
        # NY: 09:30 EST = 14:30 UTC = 18:30 server, ends 12:00 EST = 17:00 UTC = 21:00 server
        london_open_hour: int = 12,
        london_open_minute: int = 0,
        london_end_hour: int = 15,
        ny_open_hour: int = 18,
        ny_open_minute: int = 30,
        ny_end_hour: int = 21,
        # Strategy settings
        sl_buffer_pips: float = 1.0,  # Buffer beyond confirmation level
        min_range_pips: float = 2.0,
        max_range_pips: float = 30.0,
        # Dynamic R:R settings
        min_rr_ratio: float = 1.5,
        max_rr_ratio: float = 4.0,
        fallback_rr_ratio: float = 2.5,
        # Timing parameters (in M1 candles)
        max_retest_candles: int = 30,
        max_confirm_candles: int = 20,
        retest_tolerance_pips: float = 3.0,
        # Liquidity detection
        swing_lookback: int = 50,
        equal_level_tolerance_pips: float = 2.0,
        # Trade management
        max_trades_per_day: int = 2,  # 1 per session (London + NY)
        close_before_session_end_minutes: int = 10,
        # News filter
        use_news_filter: bool = True,  # Filter out high-impact news days
    ):
        """
        Initialize LOCB strategy with dual session support.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: Primary timeframe for entries (M1 recommended)
            magic_number: Unique identifier for orders
            timezone: MT5 server timezone
            london_open_hour/minute: London session open time (server time)
            london_end_hour: London session end time (server time)
            ny_open_hour/minute: NY session open time (server time)
            ny_end_hour: NY session end time (server time)
            sl_buffer_pips: Extra buffer for SL (spread is added separately)
            min_range_pips: Minimum opening candle range to trade
            max_range_pips: Maximum opening candle range to trade
            min_rr_ratio: Minimum R:R to accept trade
            max_rr_ratio: Maximum R:R (cap for liquidity targets)
            fallback_rr_ratio: R:R when no liquidity target found
            max_retest_candles: Max M1 candles to wait for retest
            max_confirm_candles: Max M1 candles after retest for confirmation
            retest_tolerance_pips: Tolerance for retest detection
            swing_lookback: Candles to analyze for liquidity detection
            equal_level_tolerance_pips: Tolerance for equal highs/lows detection
            max_trades_per_day: Max trades per day (1 per session)
            close_before_session_end_minutes: Close positions X min before session end
            use_news_filter: Filter out high-impact news days from Forex Factory
        """
        super().__init__(
            name="LOCB",
            symbol=symbol,
            timeframe=timeframe,
            magic_number=magic_number,
        )

        self.timezone = timezone
        self.tz = pytz.timezone(timezone)

        # Define trading sessions
        self.sessions = {
            "london": TradingSession(
                name="london",
                open_hour=london_open_hour,
                open_minute=london_open_minute,
                end_hour=london_end_hour,
            ),
            "ny": TradingSession(
                name="ny",
                open_hour=ny_open_hour,
                open_minute=ny_open_minute,
                end_hour=ny_end_hour,
            ),
        }

        # Strategy settings
        self.sl_buffer_pips = sl_buffer_pips
        self.min_range_pips = min_range_pips
        self.max_range_pips = max_range_pips

        # Dynamic R:R settings
        self.min_rr_ratio = min_rr_ratio
        self.max_rr_ratio = max_rr_ratio
        self.fallback_rr_ratio = fallback_rr_ratio

        # Timing parameters
        self.max_retest_candles = max_retest_candles
        self.max_confirm_candles = max_confirm_candles
        self.retest_tolerance_pips = retest_tolerance_pips

        # Liquidity detection
        self.swing_lookback = swing_lookback
        self.equal_level_tolerance_pips = equal_level_tolerance_pips

        # Trade management
        self.max_trades_per_day = max_trades_per_day
        self.close_before_session_end_minutes = close_before_session_end_minutes

        # Signal quality thresholds
        self.min_signal_score = 0.60
        self.min_sl_pips = 3.0
        self.max_sl_pips = 15.0

        # Spread tracking (will be updated from broker)
        self.current_spread: float = 0.0

        # Pip size (will be set based on symbol)
        self.pip_size = 0.0001  # Default for forex

        # State
        self._state = LOCBState(max_trades_per_day=self.max_trades_per_day)

        # News filter - extract currencies from symbol (e.g., "EURUSD" -> ["EUR", "USD"])
        self.use_news_filter = use_news_filter
        self.news_filter: Optional[NewsFilter] = None
        if self.use_news_filter:
            # Extract currencies from forex pair
            currencies = self._extract_currencies_from_symbol(symbol)
            config = NewsFilterConfig(
                filter_high_impact=True,
                filter_medium_impact=False,
                filter_entire_day=True,  # Avoid entire day with high-impact news
                currencies=currencies,
            )
            self.news_filter = NewsFilter(config)
            logger.info(f"News filter enabled for {symbol}, monitoring currencies: {currencies}")

    def _extract_currencies_from_symbol(self, symbol: str) -> list[str]:
        """Extract currency codes from a forex pair symbol."""
        symbol = symbol.upper().replace(".", "")
        # Common forex pairs are 6 characters: EURUSD, GBPUSD, etc.
        if len(symbol) >= 6:
            base = symbol[:3]  # First 3 chars = base currency
            quote = symbol[3:6]  # Next 3 chars = quote currency
            return [base, quote]
        return ["USD"]  # Default fallback

    def initialize(self) -> bool:
        """Initialize strategy."""
        logger.info(f"Initializing LOCB strategy for {self.symbol}")
        london = self.sessions["london"]
        ny = self.sessions["ny"]
        logger.info(f"London session: {london.open_hour}:{london.open_minute:02d} - {london.end_hour}:00 server time")
        logger.info(f"NY session: {ny.open_hour}:{ny.open_minute:02d} - {ny.end_hour}:00 server time")
        logger.info(f"R:R range: {self.min_rr_ratio}-{self.max_rr_ratio}:1 (dynamic)")
        logger.info(f"Max trades per day: {self.max_trades_per_day}")

        # Initialize news filter calendar
        if self.news_filter:
            logger.info("Updating economic calendar from Forex Factory...")
            if self.news_filter.update_calendar():
                high_impact_dates = self.news_filter.get_high_impact_dates(30)
                logger.info(f"News filter active: {len(high_impact_dates)} high-impact days in next 30 days")
                if self.news_filter.is_high_impact_day():
                    logger.warning("⚠️ TODAY has HIGH IMPACT NEWS - Trading may be blocked!")
            else:
                logger.warning("Failed to update news calendar - using cached/hardcoded data")

        self._reset_daily_state()
        return True

    def update_spread(self, spread: float) -> None:
        """
        Update current spread from broker.

        Args:
            spread: Current spread in price (e.g., 0.00010 for 1 pip)
        """
        self.current_spread = spread

    def _reset_daily_state(self, current_date=None) -> None:
        """Reset state for a new trading day."""
        self._state = LOCBState(max_trades_per_day=self.max_trades_per_day)
        self._state.current_date = current_date or datetime.now(self.tz).date()
        logger.info("LOCB state reset for new day")

    def _check_new_day(self, timestamp: datetime) -> None:
        """Check if it's a new trading day and reset if needed."""
        current_date = timestamp.date()
        if self._state.current_date != current_date:
            self._reset_daily_state(current_date)

    def _get_active_session(self, timestamp: datetime) -> Optional[TradingSession]:
        """
        Determine which session is currently active based on timestamp.

        Args:
            timestamp: Current candle timestamp (server time)

        Returns:
            Active TradingSession or None if outside all sessions
        """
        hour = timestamp.hour
        minute = timestamp.minute
        time_value = hour * 60 + minute

        for session in self.sessions.values():
            session_start = session.open_hour * 60 + session.open_minute
            session_end = session.end_hour * 60 + session.end_minute

            if session_start <= time_value < session_end:
                return session

        return None

    def _is_in_opening_range_period(self, candle: Candle, session: TradingSession) -> bool:
        """Check if we're in the first 5 minutes of a session (opening range period)."""
        ts = candle.timestamp
        time_value = ts.hour * 60 + ts.minute
        session_start = session.open_hour * 60 + session.open_minute

        return session_start <= time_value < session_start + 5

    def _is_in_breakout_check_period(self, candle: Candle, session: TradingSession) -> bool:
        """Check if we're in the second 5-minute period (breakout check)."""
        ts = candle.timestamp
        time_value = ts.hour * 60 + ts.minute
        session_start = session.open_hour * 60 + session.open_minute

        return session_start + 5 <= time_value < session_start + 10

    def _is_past_breakout_period(self, candle: Candle, session: TradingSession) -> bool:
        """Check if we're past the first 10 minutes (retest/confirm phase)."""
        ts = candle.timestamp
        time_value = ts.hour * 60 + ts.minute
        session_start = session.open_hour * 60 + session.open_minute

        return time_value >= session_start + 10

    def _get_session_state(self, session: TradingSession) -> SessionState:
        """Get or create state for a specific session."""
        state = self._state.get_session_state(session.name)
        if state is None:
            state = self._state.reset_session(session.name)
            logger.info(f"Initialized state for {session.name.upper()} session")
        return state

    def _calculate_sl_with_spread(self, base_sl: float, is_long: bool) -> float:
        """
        Calculate SL including spread protection.

        For BUY: SL is below entry, spread is added to increase distance
        For SELL: SL is above entry, spread is added to increase distance

        Args:
            base_sl: The base SL price without spread
            is_long: True for long position

        Returns:
            Adjusted SL price including spread
        """
        spread_pips = self.current_spread / self.pip_size
        spread_buffer = self.current_spread  # Add full spread as buffer

        if is_long:
            # For long: SL is below, move it further down
            adjusted_sl = base_sl - spread_buffer
        else:
            # For short: SL is above, move it further up
            adjusted_sl = base_sl + spread_buffer

        logger.debug(
            f"SL adjusted for spread: {base_sl:.5f} -> {adjusted_sl:.5f} "
            f"(spread: {spread_pips:.1f} pips)"
        )
        return adjusted_sl

    def _detect_choch(self, session_state: SessionState, direction: str, lookback: int = 8) -> tuple[bool, Optional[float]]:
        """
        Detect Change of Character (CHoCH).

        For bullish CHoCH: price breaks above recent swing high
        For bearish CHoCH: price breaks below recent swing low

        Args:
            session_state: Current session state containing M1 candles
            direction: "bullish" or "bearish"
            lookback: Number of candles to look back

        Returns:
            Tuple of (choch_detected, sl_level)
            - sl_level: For bullish = swing low before the break
                        For bearish = swing high before the break
        """
        candles = session_state.m1_candles
        if len(candles) < lookback + 2:
            return False, None

        recent = candles[-(lookback + 1):-1]
        current = candles[-1]

        if direction == "bullish":
            swing_high = max(c.high for c in recent)
            if current.close > swing_high:
                # SL should be below the swing low that preceded the CHoCH
                swing_low = min(c.low for c in recent)
                return True, swing_low
            return False, None
        else:  # bearish
            swing_low = min(c.low for c in recent)
            if current.close < swing_low:
                # SL should be above the swing high that preceded the CHoCH
                swing_high = max(c.high for c in recent)
                return True, swing_high
            return False, None

    def _find_fvgs(self, candles: List[M1CandleData]) -> List[FVG]:
        """Find Fair Value Gaps in candle data."""
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

    def _detect_ifvg(self, session_state: SessionState, direction: str, lookback: int = 15) -> bool:
        """
        Detect Inversed Fair Value Gap (iFVG).

        iFVG = FVG that gets violated (price closes through it)
        """
        candles = session_state.m1_candles
        if len(candles) < lookback:
            return False

        # Find FVGs in recent candles (excluding current)
        recent_candles = candles[-lookback:-1]
        fvgs = self._find_fvgs(recent_candles)

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

    def _detect_engulfing(self, session_state: SessionState, direction: str) -> bool:
        """Detect Engulfing pattern."""
        candles = session_state.m1_candles
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

    def _check_retest(self, session_state: SessionState, candle: Candle) -> bool:
        """Check if price has retested the level."""
        if session_state.retest_level is None:
            return False

        tolerance = self.retest_tolerance_pips * self.pip_size
        level = session_state.retest_level

        if session_state.direction == "long":
            # For buy: price should touch or approach level from above
            return candle.low <= level + tolerance and candle.close > level
        else:  # short
            # For sell: price should touch or approach level from below
            return candle.high >= level - tolerance and candle.close < level

    def _check_confirmation(self, session_state: SessionState) -> tuple[bool, str, Optional[float]]:
        """Check for any confirmation pattern.

        Args:
            session_state: Current session state

        Returns:
            Tuple of (confirmed, confirmation_type, sl_level)
            - sl_level: The proper SL level based on the pattern (or None to use default)
        """
        direction = "bullish" if session_state.direction == "long" else "bearish"

        # Check CHoCH first (strongest signal) - returns proper SL level
        choch_detected, choch_sl = self._detect_choch(session_state, direction, lookback=8)
        if choch_detected:
            return True, "CHoCH", choch_sl

        # Check iFVG - use default SL (confirmation candle extreme)
        if self._detect_ifvg(session_state, direction, lookback=15):
            return True, "iFVG", None

        # Check Engulfing - use default SL (confirmation candle extreme)
        if self._detect_engulfing(session_state, direction):
            return True, "Engulfing", None

        return False, "", None

    def _find_swing_points(self, candles: List[M1CandleData], lookback: int = None) -> List[LiquidityLevel]:
        """
        Find swing highs and lows in price data.

        A swing high is a high with lower highs on both sides.
        A swing low is a low with higher lows on both sides.

        Args:
            candles: List of M1 candle data
            lookback: Number of candles to analyze (default: self.swing_lookback)

        Returns:
            List of LiquidityLevel objects
        """
        if lookback is None:
            lookback = self.swing_lookback

        if len(candles) < 5:
            return []

        levels = []
        data = candles[-lookback:] if len(candles) > lookback else candles

        # Find swing points (need at least 2 candles on each side)
        for i in range(2, len(data) - 2):
            # Swing High: higher than 2 candles before and after
            if (data[i].high > data[i-1].high and data[i].high > data[i-2].high and
                data[i].high > data[i+1].high and data[i].high > data[i+2].high):

                # Calculate strength based on how prominent the swing is
                avg_surrounding = (data[i-1].high + data[i-2].high + data[i+1].high + data[i+2].high) / 4
                prominence = (data[i].high - avg_surrounding) / self.pip_size
                strength = min(1.0, prominence / 10)  # Normalize to 0-1

                levels.append(LiquidityLevel(
                    price=data[i].high,
                    type="swing_high",
                    strength=strength,
                    candle_time=data[i].time,
                    touches=1
                ))

            # Swing Low: lower than 2 candles before and after
            if (data[i].low < data[i-1].low and data[i].low < data[i-2].low and
                data[i].low < data[i+1].low and data[i].low < data[i+2].low):

                avg_surrounding = (data[i-1].low + data[i-2].low + data[i+1].low + data[i+2].low) / 4
                prominence = (avg_surrounding - data[i].low) / self.pip_size
                strength = min(1.0, prominence / 10)

                levels.append(LiquidityLevel(
                    price=data[i].low,
                    type="swing_low",
                    strength=strength,
                    candle_time=data[i].time,
                    touches=1
                ))

        return levels

    def _find_equal_levels(self, candles: List[M1CandleData], lookback: int = None) -> List[LiquidityLevel]:
        """
        Find equal highs and equal lows (double/triple tops/bottoms).

        These are strong liquidity zones where stops accumulate.

        Args:
            candles: List of M1 candle data
            lookback: Number of candles to analyze

        Returns:
            List of LiquidityLevel objects for equal highs/lows
        """
        if lookback is None:
            lookback = self.swing_lookback

        if len(candles) < 10:
            return []

        levels = []
        data = candles[-lookback:] if len(candles) > lookback else candles
        tolerance = self.equal_level_tolerance_pips * self.pip_size

        # Group highs and lows that are within tolerance
        highs = [(c.high, c.time) for c in data]
        lows = [(c.low, c.time) for c in data]

        # Find clusters of equal highs
        processed_highs = set()
        for i, (h1, t1) in enumerate(highs):
            if i in processed_highs:
                continue

            cluster = [(h1, t1)]
            for j, (h2, t2) in enumerate(highs[i+1:], i+1):
                if j in processed_highs:
                    continue
                if abs(h1 - h2) <= tolerance:
                    cluster.append((h2, t2))
                    processed_highs.add(j)

            if len(cluster) >= 2:  # At least double top
                avg_price = sum(h for h, _ in cluster) / len(cluster)
                strength = min(1.0, len(cluster) * 0.4)  # More touches = stronger
                levels.append(LiquidityLevel(
                    price=avg_price,
                    type="equal_highs",
                    strength=strength,
                    candle_time=cluster[-1][1],
                    touches=len(cluster)
                ))
                processed_highs.add(i)

        # Find clusters of equal lows
        processed_lows = set()
        for i, (l1, t1) in enumerate(lows):
            if i in processed_lows:
                continue

            cluster = [(l1, t1)]
            for j, (l2, t2) in enumerate(lows[i+1:], i+1):
                if j in processed_lows:
                    continue
                if abs(l1 - l2) <= tolerance:
                    cluster.append((l2, t2))
                    processed_lows.add(j)

            if len(cluster) >= 2:  # At least double bottom
                avg_price = sum(l for l, _ in cluster) / len(cluster)
                strength = min(1.0, len(cluster) * 0.4)
                levels.append(LiquidityLevel(
                    price=avg_price,
                    type="equal_lows",
                    strength=strength,
                    candle_time=cluster[-1][1],
                    touches=len(cluster)
                ))
                processed_lows.add(i)

        return levels

    def _find_liquidity_target(
        self,
        entry_price: float,
        sl_price: float,
        is_long: bool,
        candles: List[M1CandleData],
    ) -> tuple[Optional[float], float, str]:
        """
        Find the best liquidity target for take profit.

        Looks for swing highs/lows and equal highs/lows above (for long) or
        below (for short) the entry price.

        Args:
            entry_price: The entry price
            sl_price: Stop loss price (to calculate R:R)
            is_long: True for long position
            candles: Historical candle data for analysis

        Returns:
            Tuple of (tp_price, actual_rr, target_type)
            - tp_price: The calculated TP price (None if no valid target)
            - actual_rr: The R:R ratio achieved
            - target_type: Description of the liquidity target
        """
        risk = abs(entry_price - sl_price)

        # Find all liquidity levels
        swing_levels = self._find_swing_points(candles)
        equal_levels = self._find_equal_levels(candles)
        all_levels = swing_levels + equal_levels

        if not all_levels:
            # No liquidity found, use fallback R:R
            if is_long:
                tp = entry_price + (risk * self.fallback_rr_ratio)
            else:
                tp = entry_price - (risk * self.fallback_rr_ratio)
            return tp, self.fallback_rr_ratio, "fallback_rr"

        # Filter levels based on direction
        if is_long:
            # For long: look for liquidity ABOVE entry (swing highs, equal highs)
            valid_targets = [
                lvl for lvl in all_levels
                if lvl.price > entry_price and lvl.type in ("swing_high", "equal_highs")
            ]
            # Sort by price (ascending - nearest first)
            valid_targets.sort(key=lambda x: x.price)
        else:
            # For short: look for liquidity BELOW entry (swing lows, equal lows)
            valid_targets = [
                lvl for lvl in all_levels
                if lvl.price < entry_price and lvl.type in ("swing_low", "equal_lows")
            ]
            # Sort by price (descending - nearest first)
            valid_targets.sort(key=lambda x: x.price, reverse=True)

        if not valid_targets:
            # No valid liquidity targets, use fallback
            if is_long:
                tp = entry_price + (risk * self.fallback_rr_ratio)
            else:
                tp = entry_price - (risk * self.fallback_rr_ratio)
            return tp, self.fallback_rr_ratio, "fallback_rr"

        # Find best target that gives at least min_rr_ratio
        best_target = None
        best_rr = 0
        best_type = ""

        for target in valid_targets:
            if is_long:
                potential_reward = target.price - entry_price
            else:
                potential_reward = entry_price - target.price

            rr = potential_reward / risk if risk > 0 else 0

            # Check if R:R is within acceptable range
            if self.min_rr_ratio <= rr <= self.max_rr_ratio:
                # Prefer stronger liquidity levels
                if best_target is None or (rr >= best_rr * 0.9 and target.strength > best_target.strength):
                    best_target = target
                    best_rr = rr
                    best_type = f"{target.type}({target.touches}x)" if target.touches > 1 else target.type

        if best_target:
            # Place TP slightly before the liquidity level (to ensure fill)
            buffer = 1 * self.pip_size  # 1 pip buffer
            if is_long:
                tp = best_target.price - buffer
            else:
                tp = best_target.price + buffer

            logger.info(
                f"Liquidity target found: {best_type} @ {best_target.price:.5f}, "
                f"R:R = {best_rr:.2f}, strength = {best_target.strength:.2f}"
            )
            return tp, best_rr, best_type

        # No target within acceptable R:R range
        # Check if nearest target is too close (R:R < min)
        nearest = valid_targets[0]
        if is_long:
            nearest_rr = (nearest.price - entry_price) / risk
        else:
            nearest_rr = (entry_price - nearest.price) / risk

        if nearest_rr < self.min_rr_ratio:
            logger.info(
                f"Nearest liquidity {nearest.type} @ {nearest.price:.5f} gives R:R {nearest_rr:.2f} < {self.min_rr_ratio}"
            )
            return None, nearest_rr, "insufficient_rr"

        # Target is beyond max_rr, cap it
        if is_long:
            tp = entry_price + (risk * self.max_rr_ratio)
        else:
            tp = entry_price - (risk * self.max_rr_ratio)

        return tp, self.max_rr_ratio, "capped_rr"

    def _calculate_signal_score(
        self,
        session_state: SessionState,
        session: TradingSession,
        entry_price: float,
        sl: float,
        confirm_type: str,
        candle: Candle,
    ) -> SignalScore:
        """
        Calculate quality score for a trading signal.

        Higher score = better quality setup.

        Args:
            session_state: Current session state
            session: Current trading session
            entry_price: Proposed entry price
            sl: Stop loss price
            confirm_type: Type of confirmation (CHoCH, iFVG, Engulfing)
            candle: Current candle

        Returns:
            SignalScore with component scores and total
        """
        score = SignalScore()

        # 1. Confirmation type score (CHoCH is strongest)
        confirmation_scores = {
            "CHoCH": 1.0,      # Best - structural change
            "iFVG": 0.75,     # Good - imbalance fill
            "Engulfing": 0.5,  # Okay - basic pattern
        }
        score.confirmation_score = confirmation_scores.get(confirm_type, 0.3)

        # 2. SL distance score (prefer 5-10 pips, penalize extremes)
        sl_pips = abs(entry_price - sl) / self.pip_size
        if sl_pips < self.min_sl_pips:
            # Too tight - high chance of stop hunt
            score.sl_distance_score = 0.3
        elif sl_pips <= 6:
            # Ideal range - tight but safe
            score.sl_distance_score = 1.0
        elif sl_pips <= 10:
            # Good range
            score.sl_distance_score = 0.85
        elif sl_pips <= self.max_sl_pips:
            # Acceptable but wider
            score.sl_distance_score = 0.6
        else:
            # Too wide - poor R:R potential
            score.sl_distance_score = 0.2

        # 3. Breakout strength score
        if session_state.breakout_period_candles:
            breakout_candles = session_state.breakout_period_candles
            # Check if breakout candle had strong body (>60% of range)
            for bc in breakout_candles:
                body = abs(bc.close - bc.open)
                total_range = bc.high - bc.low
                if total_range > 0:
                    body_percent = body / total_range
                    if body_percent > 0.7:
                        score.breakout_strength_score = 1.0
                        break
                    elif body_percent > 0.5:
                        score.breakout_strength_score = 0.75
                        break
            if score.breakout_strength_score == 0:
                score.breakout_strength_score = 0.5  # Default if no strong candle

        # 4. Retest quality score
        if session_state.retest_level:
            level = session_state.retest_level

            # Check how precisely the retest touched the level
            if session_state.direction == "long":
                # For long: best if low touched level exactly
                touch_distance = abs(candle.low - level) / self.pip_size
            else:
                touch_distance = abs(candle.high - level) / self.pip_size

            if touch_distance < 1:
                score.retest_quality_score = 1.0  # Perfect touch
            elif touch_distance < 2:
                score.retest_quality_score = 0.85
            elif touch_distance < 3:
                score.retest_quality_score = 0.7
            else:
                score.retest_quality_score = 0.5

        # 5. Timing score (earlier in session = more time for trade to work)
        ts = candle.timestamp
        time_value = ts.hour * 60 + ts.minute
        session_start = session.open_hour * 60 + session.open_minute
        session_end = session.end_hour * 60 + session.end_minute
        minutes_into_session = time_value - session_start

        if minutes_into_session < 30:
            score.timing_score = 1.0  # First 30 min - excellent
        elif minutes_into_session < 60:
            score.timing_score = 0.85  # First hour - good
        elif minutes_into_session < 90:
            score.timing_score = 0.65  # 1-1.5 hours - okay
        else:
            score.timing_score = 0.4  # Later - less time for TP

        # Calculate total
        score.calculate_total()

        logger.debug(
            f"Signal score: {score.total:.2f} | "
            f"Confirm: {score.confirmation_score:.2f}, SL: {score.sl_distance_score:.2f}, "
            f"Breakout: {score.breakout_strength_score:.2f}, Retest: {score.retest_quality_score:.2f}, "
            f"Timing: {score.timing_score:.2f}"
        )

        return score

    def on_candle(
        self,
        candle: Candle,
        candles: list[Candle],
    ) -> Optional[StrategySignal]:
        """
        Process new candle and check for signals.

        Supports multiple sessions (London and NY) with independent state tracking.
        """
        if not self._enabled:
            return None

        ts = candle.timestamp

        # Check for new day
        self._check_new_day(ts)

        # Check news filter - skip high impact news days
        if self.news_filter and not self.news_filter.is_safe_to_trade(ts, self.symbol):
            # Log only once per day
            if not hasattr(self, '_news_blocked_logged') or self._news_blocked_logged != ts.date():
                logger.warning(f"LOCB [{self.symbol}] 🚫 HIGH IMPACT NEWS DAY - Skipping trading")
                events = self.news_filter.get_events_for_date(ts.date())
                high_events = [e for e in events if e.impact == NewsImpact.HIGH]
                for event in high_events[:3]:  # Show first 3 events
                    logger.warning(f"  📰 {event.currency}: {event.event}")
                self._news_blocked_logged = ts.date()
            return None

        # Check if max trades reached for today
        if self._state.trades_today >= self._state.max_trades_per_day:
            return None

        # Determine which session we're in
        session = self._get_active_session(ts)
        if session is None:
            return None  # Outside all trading sessions

        # Get or create session state
        session_state = self._get_session_state(session)

        # Skip if this session already had a trade or entry was triggered
        if session_state.trade_taken or session_state.entry_triggered:
            return None

        # Create M1 candle data
        m1_candle = M1CandleData(
            time=candle.timestamp,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close
        )

        # Phase 0: Accumulate opening period candles (first 5 minutes of session)
        if self._is_in_opening_range_period(candle, session):
            if len(session_state.opening_period_candles) == 0:
                logger.info(f"LOCB [{self.symbol}] {session.name.upper()} session - Opening range started at {ts.strftime('%H:%M:%S')}")
            session_state.opening_period_candles.append(m1_candle)
            logger.debug(f"LOCB [{self.symbol}] Accumulated candle {len(session_state.opening_period_candles)}/5 in opening range")
            return None

        # Phase 0.5: Calculate opening range from accumulated candles
        if not session_state.opening_candle_valid and session_state.opening_period_candles:
            oc_high = max(c.high for c in session_state.opening_period_candles)
            oc_low = min(c.low for c in session_state.opening_period_candles)
            oc_range = oc_high - oc_low
            oc_range_pips = oc_range / self.pip_size

            if self.min_range_pips <= oc_range_pips <= self.max_range_pips:
                session_state.opening_candle_high = oc_high
                session_state.opening_candle_low = oc_low
                session_state.opening_candle_time = session_state.opening_period_candles[0].time
                session_state.opening_candle_valid = True
                logger.info(
                    f"LOCB [{self.symbol}] {session.name.upper()} Opening range: "
                    f"High={oc_high:.5f}, Low={oc_low:.5f}, Range={oc_range_pips:.1f} pips"
                )
            else:
                logger.debug(f"Opening range {oc_range_pips:.1f} pips out of bounds [{self.min_range_pips}-{self.max_range_pips}]")
                session_state.entry_triggered = True
                return None

        # Phase 1: Check for breakout during second 5-minute period
        if self._is_in_breakout_check_period(candle, session) and session_state.opening_candle_valid and not session_state.breakout_confirmed:
            session_state.breakout_period_candles.append(m1_candle)

            oc_high = session_state.opening_candle_high
            oc_low = session_state.opening_candle_low

            if candle.close > oc_high:
                session_state.direction = "long"
                session_state.retest_level = oc_high
                session_state.breakout_confirmed = True
                logger.info(f"LOCB [{self.symbol}] {session.name.upper()} LONG breakout: {candle.close:.5f} > {oc_high:.5f}")
            elif candle.close < oc_low:
                session_state.direction = "short"
                session_state.retest_level = oc_low
                session_state.breakout_confirmed = True
                logger.info(f"LOCB [{self.symbol}] {session.name.upper()} SHORT breakout: {candle.close:.5f} < {oc_low:.5f}")
            return None

        # If breakout period ended without breakout, skip session
        if self._is_past_breakout_period(candle, session) and session_state.opening_candle_valid and not session_state.breakout_confirmed:
            if not session_state.entry_triggered:
                logger.debug(f"No breakout during {session.name.upper()} session, skipping")
                session_state.entry_triggered = True
            return None

        # Phase 2 & 3: Look for retest and confirmation
        if session_state.breakout_confirmed and not session_state.entry_triggered:
            session_state.m1_candles.append(m1_candle)
            session_state.candles_since_breakout += 1

            # Phase 2: Look for retest
            if not session_state.retest_found:
                if session_state.candles_since_breakout > self.max_retest_candles:
                    logger.debug(f"Retest timeout after {session_state.candles_since_breakout} candles")
                    session_state.entry_triggered = True
                    return None

                if self._check_retest(session_state, candle):
                    session_state.retest_found = True
                    session_state.candles_since_retest = 0
                    logger.info(f"LOCB [{self.symbol}] {session.name.upper()} Retest found at {ts}")
                return None

            # Phase 3: Look for confirmation after retest
            session_state.candles_since_retest += 1

            if session_state.candles_since_retest > self.max_confirm_candles:
                logger.debug(f"Confirmation timeout after {session_state.candles_since_retest} candles")
                session_state.entry_triggered = True
                return None

            confirmed, confirm_type, pattern_sl = self._check_confirmation(session_state)

            if confirmed:
                session_state.confirmation_found = True
                session_state.confirmation_type = confirm_type
                session_state.confirmation_candle = m1_candle

                # Generate entry signal
                entry_price = candle.close
                is_long = session_state.direction == "long"

                # Calculate base SL
                if is_long:
                    if pattern_sl is not None:
                        base_sl = pattern_sl - (self.sl_buffer_pips * self.pip_size)
                    else:
                        base_sl = session_state.confirmation_candle.low - (self.sl_buffer_pips * self.pip_size)
                    signal_type = SignalType.LONG
                else:
                    if pattern_sl is not None:
                        base_sl = pattern_sl + (self.sl_buffer_pips * self.pip_size)
                    else:
                        base_sl = session_state.confirmation_candle.high + (self.sl_buffer_pips * self.pip_size)
                    signal_type = SignalType.SHORT

                # Apply spread to SL
                sl = self._calculate_sl_with_spread(base_sl, is_long)
                sl_pips = abs(entry_price - sl) / self.pip_size

                # Find dynamic TP based on liquidity levels
                tp, actual_rr, liquidity_target = self._find_liquidity_target(
                    entry_price=entry_price,
                    sl_price=sl,
                    is_long=is_long,
                    candles=session_state.m1_candles
                )

                # Check if R:R is acceptable
                if tp is None or actual_rr < self.min_rr_ratio:
                    logger.info(
                        f"LOCB [{self.symbol}] {session.name.upper()} Signal REJECTED - R:R {actual_rr:.2f} < {self.min_rr_ratio} | "
                        f"Target: {liquidity_target}"
                    )
                    session_state.entry_triggered = True
                    return None

                tp_pips = abs(entry_price - tp) / self.pip_size

                # Calculate signal quality score
                signal_score = self._calculate_signal_score(
                    session_state, session, entry_price, sl, confirm_type, candle
                )

                # Check if signal meets minimum quality threshold
                if signal_score.total < self.min_signal_score:
                    logger.info(
                        f"LOCB [{self.symbol}] {session.name.upper()} Signal REJECTED - score {signal_score.total:.2f} < {self.min_signal_score}"
                    )
                    session_state.entry_triggered = True
                    return None

                # Mark session and day as having a trade
                session_state.entry_triggered = True
                session_state.trade_taken = True
                self._state.trades_today += 1

                spread_pips = self.current_spread / self.pip_size

                logger.info(
                    f"LOCB [{self.symbol}] {session.name.upper()} {session_state.direction.upper()} SIGNAL | "
                    f"Score: {signal_score.total:.2f} | Entry: {entry_price:.5f}, "
                    f"SL: {sl:.5f} ({sl_pips:.1f}p, spread: {spread_pips:.1f}p), TP: {tp:.5f} ({tp_pips:.1f}p), "
                    f"R:R: {actual_rr:.2f} | Target: {liquidity_target} | Confirm: {confirm_type}"
                )

                signal = StrategySignal(
                    signal_type=signal_type,
                    symbol=self.symbol,
                    price=entry_price,
                    stop_loss=sl,
                    take_profit=tp,
                    confidence=signal_score.total,
                    reason=f"LOCB {session.name.upper()} {session_state.direction.upper()} - {confirm_type} (R:R {actual_rr:.1f})",
                    metadata={
                        "session": session.name,
                        "opening_candle_high": session_state.opening_candle_high,
                        "opening_candle_low": session_state.opening_candle_low,
                        "confirmation_type": confirm_type,
                        "sl_pips": sl_pips,
                        "tp_pips": tp_pips,
                        "spread_pips": spread_pips,
                        "actual_rr": actual_rr,
                        "liquidity_target": liquidity_target,
                        "candles_to_entry": session_state.candles_since_breakout,
                        "signal_score": signal_score.total,
                        "score_details": {
                            "confirmation": signal_score.confirmation_score,
                            "sl_distance": signal_score.sl_distance_score,
                            "breakout_strength": signal_score.breakout_strength_score,
                            "retest_quality": signal_score.retest_quality_score,
                            "timing": signal_score.timing_score,
                        },
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

        # Check time-based exit - use active session's end hour
        active_session = self._state.active_session
        if active_session and active_session in self.sessions:
            session_end_hour = self.sessions[active_session].end_hour
        else:
            # Fallback to London if no active session
            session_end_hour = self.sessions["london"].end_hour

        session_end = now.replace(
            hour=session_end_hour,
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

        # Update active session state if available
        active_session = self._state.active_session
        if active_session:
            session_state = self._state.get_session_state(active_session)
            if session_state:
                session_state.entry_price = position.entry_price
                session_state.trade_taken = True

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
        # Get confirmation candle from active session state
        active_session = self._state.active_session
        if not active_session:
            return None

        session_state = self._state.get_session_state(active_session)
        if not session_state or session_state.confirmation_candle is None:
            return None

        buffer = self.sl_buffer_pips * self.pip_size

        if is_long:
            return session_state.confirmation_candle.low - buffer
        else:
            return session_state.confirmation_candle.high + buffer

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        is_long: bool,
    ) -> Optional[float]:
        """Calculate take profit based on fallback risk/reward ratio."""
        risk = abs(entry_price - stop_loss)
        reward = risk * self.fallback_rr_ratio

        if is_long:
            return entry_price + reward
        else:
            return entry_price - reward

    def get_status(self) -> dict:
        """Get strategy status."""
        base_status = super().get_status()

        # Get active session state if available
        active_session = self._state.active_session
        session_state = self._state.get_session_state(active_session) if active_session else None

        base_status.update({
            "active_session": active_session,
            "trades_today": self._state.trades_today,
            "max_trades_per_day": self._state.max_trades_per_day,
            "min_rr_ratio": self.min_rr_ratio,
            "max_rr_ratio": self.max_rr_ratio,
            "sessions": {
                name: {
                    "open": f"{s.open_hour}:{s.open_minute:02d}",
                    "end": f"{s.end_hour}:00"
                }
                for name, s in self.sessions.items()
            },
        })

        # Add session-specific state if available
        if session_state:
            base_status.update({
                "opening_candle_high": session_state.opening_candle_high,
                "opening_candle_low": session_state.opening_candle_low,
                "opening_candle_valid": session_state.opening_candle_valid,
                "direction": session_state.direction,
                "breakout_confirmed": session_state.breakout_confirmed,
                "retest_found": session_state.retest_found,
                "confirmation_found": session_state.confirmation_found,
                "confirmation_type": session_state.confirmation_type,
                "entry_triggered": session_state.entry_triggered,
            })

        return base_status

    def reset(self) -> None:
        """Reset strategy for new session."""
        super().reset()
        self._reset_daily_state()