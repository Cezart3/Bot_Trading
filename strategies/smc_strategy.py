"""
Smart Money Concepts (SMC) Strategy Implementation.

A sophisticated multi-timeframe strategy based on institutional trading concepts:
- Market Structure (BOS, CHoCH, Swing Points)
- Order Blocks (OB)
- Fair Value Gaps (FVG)
- Liquidity Levels (Swing Highs/Lows, Equal Highs/Lows)
- Premium/Discount Zones

Timeframe Usage:
- H4: Bias determination (trend direction)
- H1: Structure analysis, POI identification
- M5: Entry precision with CHoCH confirmation

Trading Sessions:
- London: 07:00-11:00 UTC
- New York: 13:00-17:00 UTC
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from enum import Enum
import pytz

from models.candle import Candle
from models.position import Position
from strategies.base_strategy import BaseStrategy, SignalType, StrategySignal
from utils.logger import get_logger
from utils.news_filter import NewsFilter, NewsFilterConfig, NewsImpact

logger = get_logger(__name__)


# ==================== Enums and Data Classes ====================

class MarketBias(Enum):
    """Market bias direction."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class StructureType(Enum):
    """Type of market structure break."""
    BOS = "bos"  # Break of Structure - trend continuation
    CHOCH = "choch"  # Change of Character - potential reversal


class POIType(Enum):
    """Point of Interest type."""
    BULLISH_OB = "bullish_ob"
    BEARISH_OB = "bearish_ob"
    BULLISH_FVG = "bullish_fvg"
    BEARISH_FVG = "bearish_fvg"


@dataclass
class SwingPoint:
    """Represents a swing high or low."""
    price: float
    timestamp: datetime
    is_high: bool  # True for swing high, False for swing low
    strength: int = 1  # Number of candles on each side that confirm this swing


@dataclass
class StructureBreak:
    """Represents a break of structure (BOS or CHoCH)."""
    type: StructureType
    direction: str  # "bullish" or "bearish"
    break_price: float
    timestamp: datetime
    swing_broken: SwingPoint


@dataclass
class OrderBlock:
    """Represents an Order Block zone."""
    type: POIType
    high: float
    low: float
    timestamp: datetime
    is_valid: bool = True  # Becomes False if price returns and mitigates it
    mitigated: bool = False
    impulse_strength: float = 0.0  # How strong was the move away


@dataclass
class FairValueGap:
    """Represents a Fair Value Gap (imbalance)."""
    type: POIType
    high: float
    low: float
    timestamp: datetime
    is_valid: bool = True  # Becomes False if filled
    fill_percentage: float = 0.0


@dataclass
class LiquidityLevel:
    """Represents a liquidity level (stops accumulation zone)."""
    price: float
    type: str  # "swing_high", "swing_low", "equal_highs", "equal_lows"
    strength: float  # 0-1, higher = more significant
    timestamp: datetime
    touches: int = 1
    swept: bool = False


@dataclass
class POI:
    """Point of Interest - confluence zone for entries."""
    price_high: float
    price_low: float
    type: str  # Description of what created this POI
    score: int  # 0-5 quality score
    timestamp: datetime
    order_block: Optional[OrderBlock] = None
    fvg: Optional[FairValueGap] = None
    liquidity_nearby: Optional[LiquidityLevel] = None
    is_premium: bool = False  # True if in premium zone
    is_discount: bool = False  # True if in discount zone
    is_fresh: bool = True  # Not yet tested


@dataclass
class SMCState:
    """State tracking for SMC strategy."""
    # Bias
    h4_bias: MarketBias = MarketBias.NEUTRAL
    h1_bias: MarketBias = MarketBias.NEUTRAL

    # Structure tracking
    h1_swing_highs: List[SwingPoint] = field(default_factory=list)
    h1_swing_lows: List[SwingPoint] = field(default_factory=list)
    h1_structure_breaks: List[StructureBreak] = field(default_factory=list)

    # POIs
    active_pois: List[POI] = field(default_factory=list)

    # Order Blocks and FVGs
    order_blocks: List[OrderBlock] = field(default_factory=list)
    fvgs: List[FairValueGap] = field(default_factory=list)

    # Liquidity
    liquidity_levels: List[LiquidityLevel] = field(default_factory=list)

    # Entry tracking
    waiting_for_entry: bool = False
    active_poi: Optional[POI] = None
    m5_structure: List[SwingPoint] = field(default_factory=list)
    entry_choch_detected: bool = False

    # Session tracking
    current_date: Optional[datetime] = None
    trades_today: int = 0
    last_trade_session: Optional[str] = None

    # Premium/Discount reference
    swing_range_high: Optional[float] = None
    swing_range_low: Optional[float] = None


@dataclass
class SignalScore:
    """Score components for signal quality."""
    total: float = 0.0
    poi_score: float = 0.0  # POI quality (max 5)
    confluence_score: float = 0.0  # OB + FVG + liquidity
    structure_score: float = 0.0  # Clean structure
    timing_score: float = 0.0  # Session timing
    rr_score: float = 0.0  # Risk/Reward potential

    def calculate_total(self) -> float:
        """Calculate weighted total score (0-1 range)."""
        # Normalize POI score from 0-5 to 0-1
        normalized_poi = self.poi_score / 5.0

        self.total = (
            normalized_poi * 0.30 +
            self.confluence_score * 0.25 +
            self.structure_score * 0.20 +
            self.timing_score * 0.15 +
            self.rr_score * 0.10
        )
        return self.total


# ==================== Configuration ====================

@dataclass
class SMCConfig:
    """Configuration for SMC strategy."""
    # Timeframes
    tf_bias: str = "H4"
    tf_structure: str = "H1"
    tf_entry: str = "M5"

    # Structure Detection
    swing_lookback: int = 20  # Candles for swing detection
    bos_min_move_atr: float = 2.0  # Minimum move in ATR for valid BOS
    ob_min_impulse_atr: float = 2.0  # Minimum impulse for valid OB
    fvg_min_gap_atr: float = 0.5  # Minimum gap size for valid FVG
    poi_min_score: int = 3  # Minimum POI score to consider (0-5)

    # Entry
    entry_mode: str = "conservative"  # "aggressive" or "conservative"
    choch_lookback: int = 8  # Candles for CHoCH detection on M5

    # Risk Management
    risk_percent: float = 1.0
    max_sl_atr: float = 3.0  # Maximum SL in ATR
    min_rr: float = 1.5  # Minimum Risk/Reward

    # Take Profits
    tp1_rr: float = 1.5
    tp1_percent: int = 40
    tp2_rr: float = 2.5
    tp2_percent: int = 30
    tp3_rr: float = 4.0
    tp3_percent: int = 30

    # Sessions (UTC)
    london_start_hour: int = 7
    london_start_minute: int = 0
    london_end_hour: int = 11
    london_end_minute: int = 0
    ny_start_hour: int = 13
    ny_start_minute: int = 0
    ny_end_hour: int = 17
    ny_end_minute: int = 0

    # Filters
    news_buffer_hours: int = 2  # Hours before red news to stop trading
    max_spread_multiplier: float = 2.0
    volatility_min_atr: float = 0.7
    volatility_max_atr: float = 2.5

    # Trade limits
    max_trades_per_day: int = 3
    max_trades_per_session: int = 2
    max_concurrent_trades: int = 2


class SMCStrategy(BaseStrategy):
    """
    Smart Money Concepts Trading Strategy.

    Multi-timeframe analysis strategy that identifies institutional order flow
    through market structure, order blocks, and liquidity concepts.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "M5",
        magic_number: int = 12350,
        timezone: str = "UTC",
        config: Optional[SMCConfig] = None,
        use_news_filter: bool = True,
    ):
        """
        Initialize SMC strategy.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: Primary timeframe for entries (M5 recommended)
            magic_number: Unique identifier for orders
            timezone: Trading timezone
            config: Strategy configuration
            use_news_filter: Enable news filter with 2-hour buffer
        """
        super().__init__(
            name="SMC",
            symbol=symbol,
            timeframe=timeframe,
            magic_number=magic_number,
        )

        self.config = config or SMCConfig()
        self.timezone = timezone
        self.tz = pytz.timezone(timezone) if timezone != "UTC" else pytz.UTC

        # State
        self._state = SMCState()

        # Pip size (will be set based on symbol)
        self.pip_size = 0.0001

        # ATR values for each timeframe
        self.atr_h4: float = 0.0
        self.atr_h1: float = 0.0
        self.atr_m5: float = 0.0

        # Spread tracking
        self.current_spread: float = 0.0
        self.avg_spread: float = 0.0

        # Candle buffers for multi-timeframe analysis
        self.h4_candles: List[Candle] = []
        self.h1_candles: List[Candle] = []
        self.m5_candles: List[Candle] = []

        # News filter setup
        self.use_news_filter = use_news_filter
        self.news_filter: Optional[NewsFilter] = None
        if self.use_news_filter:
            currencies = self._extract_currencies(symbol)
            filter_config = NewsFilterConfig(
                filter_high_impact=True,
                filter_medium_impact=False,
                filter_entire_day=False,  # Only filter around news time
                buffer_before_minutes=self.config.news_buffer_hours * 60,  # 2 hours
                buffer_after_minutes=30,  # 30 min after
                currencies=currencies,
            )
            self.news_filter = NewsFilter(filter_config)
            logger.info(f"SMC [{symbol}] News filter: {self.config.news_buffer_hours}h buffer before red news")

    def _extract_currencies(self, symbol: str) -> List[str]:
        """Extract currency codes from forex pair."""
        symbol = symbol.upper().replace(".", "")
        if len(symbol) >= 6:
            return [symbol[:3], symbol[3:6]]
        return ["USD"]

    def initialize(self) -> bool:
        """Initialize strategy."""
        logger.info(f"Initializing SMC strategy for {self.symbol}")
        logger.info(f"Timeframes: Bias={self.config.tf_bias}, Structure={self.config.tf_structure}, Entry={self.config.tf_entry}")
        logger.info(f"Min POI score: {self.config.poi_min_score}, Min R:R: {self.config.min_rr}")
        logger.info(f"Sessions: London {self.config.london_start_hour}:00-{self.config.london_end_hour}:00, NY {self.config.ny_start_hour}:00-{self.config.ny_end_hour}:00 UTC")

        if self.news_filter:
            self.news_filter.update_calendar()
            logger.info(f"News filter active with {self.config.news_buffer_hours}h buffer before red news")

        self._reset_daily_state()
        return True

    def _reset_daily_state(self, current_date: Optional[datetime] = None) -> None:
        """Reset state for a new trading day."""
        self._state = SMCState()
        self._state.current_date = current_date or datetime.now(self.tz).date()
        logger.info("SMC state reset for new day")

    def update_spread(self, spread: float) -> None:
        """Update current spread."""
        self.current_spread = spread
        # Update average spread (simple moving average)
        if self.avg_spread == 0:
            self.avg_spread = spread
        else:
            self.avg_spread = self.avg_spread * 0.95 + spread * 0.05

    def set_candles(self, h4_candles: List[Candle], h1_candles: List[Candle], m5_candles: List[Candle]) -> None:
        """
        Set candle data for all timeframes.

        This method should be called by the bot before on_candle() to provide
        multi-timeframe data.
        """
        self.h4_candles = h4_candles
        self.h1_candles = h1_candles
        self.m5_candles = m5_candles

        # Calculate ATR for each timeframe
        if len(h4_candles) >= 14:
            self.atr_h4 = self._calculate_atr(h4_candles, 14)
        if len(h1_candles) >= 14:
            self.atr_h1 = self._calculate_atr(h1_candles, 14)
        if len(m5_candles) >= 14:
            self.atr_m5 = self._calculate_atr(m5_candles, 14)

    # ==================== ATR Calculation ====================

    def _calculate_atr(self, candles: List[Candle], period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(candles) < period + 1:
            return 0.0

        tr_values = []
        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i-1].close

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_values.append(tr)

        if len(tr_values) < period:
            return sum(tr_values) / len(tr_values) if tr_values else 0.0

        return sum(tr_values[-period:]) / period

    # ==================== Session Detection ====================

    def _get_current_session(self, timestamp: datetime) -> Optional[str]:
        """Determine which session is active."""
        # Convert to UTC if needed
        if timestamp.tzinfo is None:
            ts_utc = timestamp
        else:
            ts_utc = timestamp.astimezone(pytz.UTC)

        hour = ts_utc.hour
        minute = ts_utc.minute
        time_value = hour * 60 + minute

        # London session
        london_start = self.config.london_start_hour * 60 + self.config.london_start_minute
        london_end = self.config.london_end_hour * 60 + self.config.london_end_minute
        if london_start <= time_value < london_end:
            return "london"

        # NY session
        ny_start = self.config.ny_start_hour * 60 + self.config.ny_start_minute
        ny_end = self.config.ny_end_hour * 60 + self.config.ny_end_minute
        if ny_start <= time_value < ny_end:
            return "ny"

        return None

    def _is_in_killzone(self, timestamp: datetime) -> bool:
        """Check if we're in a valid trading session."""
        return self._get_current_session(timestamp) is not None

    # ==================== Swing Point Detection ====================

    def _find_swing_points(self, candles: List[Candle], lookback: int = 5) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """
        Find swing highs and lows in candle data.

        A swing high has lower highs on both sides.
        A swing low has higher lows on both sides.
        """
        swing_highs = []
        swing_lows = []

        if len(candles) < lookback * 2 + 1:
            return swing_highs, swing_lows

        for i in range(lookback, len(candles) - lookback):
            candle = candles[i]

            # Check for swing high
            is_swing_high = True
            for j in range(1, lookback + 1):
                if candles[i - j].high >= candle.high or candles[i + j].high >= candle.high:
                    is_swing_high = False
                    break

            if is_swing_high:
                swing_highs.append(SwingPoint(
                    price=candle.high,
                    timestamp=candle.timestamp,
                    is_high=True,
                    strength=lookback
                ))

            # Check for swing low
            is_swing_low = True
            for j in range(1, lookback + 1):
                if candles[i - j].low <= candle.low or candles[i + j].low <= candle.low:
                    is_swing_low = False
                    break

            if is_swing_low:
                swing_lows.append(SwingPoint(
                    price=candle.low,
                    timestamp=candle.timestamp,
                    is_high=False,
                    strength=lookback
                ))

        return swing_highs, swing_lows

    # ==================== Market Structure Analysis ====================

    def _analyze_market_structure(self, candles: List[Candle], atr: float) -> Tuple[MarketBias, List[StructureBreak]]:
        """
        Analyze market structure to determine bias and identify structure breaks.

        Returns:
            Tuple of (bias, list of structure breaks)
        """
        if len(candles) < 30 or atr == 0:
            return MarketBias.NEUTRAL, []

        swing_highs, swing_lows = self._find_swing_points(candles, lookback=3)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return MarketBias.NEUTRAL, []

        structure_breaks = []

        # Analyze recent swings for BOS/CHoCH
        recent_highs = sorted(swing_highs, key=lambda x: x.timestamp)[-5:]
        recent_lows = sorted(swing_lows, key=lambda x: x.timestamp)[-5:]

        # Count higher highs/higher lows vs lower highs/lower lows
        hh_count = 0
        hl_count = 0
        lh_count = 0
        ll_count = 0

        for i in range(1, len(recent_highs)):
            if recent_highs[i].price > recent_highs[i-1].price:
                hh_count += 1
            else:
                lh_count += 1

        for i in range(1, len(recent_lows)):
            if recent_lows[i].price > recent_lows[i-1].price:
                hl_count += 1
            else:
                ll_count += 1

        # Determine bias
        bullish_score = hh_count + hl_count
        bearish_score = lh_count + ll_count

        if bullish_score > bearish_score + 1:
            bias = MarketBias.BULLISH
        elif bearish_score > bullish_score + 1:
            bias = MarketBias.BEARISH
        else:
            bias = MarketBias.NEUTRAL

        # Detect structure breaks on most recent candles
        current_price = candles[-1].close

        # Check for BOS (break in direction of trend)
        if bias == MarketBias.BULLISH and recent_highs:
            last_high = recent_highs[-1]
            if current_price > last_high.price:
                move_size = (current_price - last_high.price) / atr
                if move_size >= self.config.bos_min_move_atr:
                    structure_breaks.append(StructureBreak(
                        type=StructureType.BOS,
                        direction="bullish",
                        break_price=current_price,
                        timestamp=candles[-1].timestamp,
                        swing_broken=last_high
                    ))

        elif bias == MarketBias.BEARISH and recent_lows:
            last_low = recent_lows[-1]
            if current_price < last_low.price:
                move_size = (last_low.price - current_price) / atr
                if move_size >= self.config.bos_min_move_atr:
                    structure_breaks.append(StructureBreak(
                        type=StructureType.BOS,
                        direction="bearish",
                        break_price=current_price,
                        timestamp=candles[-1].timestamp,
                        swing_broken=last_low
                    ))

        # Check for CHoCH (break against trend - potential reversal)
        if bias == MarketBias.BULLISH and recent_lows:
            last_low = recent_lows[-1]
            if current_price < last_low.price:
                move_size = (last_low.price - current_price) / atr
                if move_size >= self.config.bos_min_move_atr:
                    structure_breaks.append(StructureBreak(
                        type=StructureType.CHOCH,
                        direction="bearish",
                        break_price=current_price,
                        timestamp=candles[-1].timestamp,
                        swing_broken=last_low
                    ))

        elif bias == MarketBias.BEARISH and recent_highs:
            last_high = recent_highs[-1]
            if current_price > last_high.price:
                move_size = (current_price - last_high.price) / atr
                if move_size >= self.config.bos_min_move_atr:
                    structure_breaks.append(StructureBreak(
                        type=StructureType.CHOCH,
                        direction="bullish",
                        break_price=current_price,
                        timestamp=candles[-1].timestamp,
                        swing_broken=last_high
                    ))

        return bias, structure_breaks

    # ==================== Order Block Detection ====================

    def _find_order_blocks(self, candles: List[Candle], atr: float) -> List[OrderBlock]:
        """
        Find valid Order Blocks in candle data.

        Bullish OB: Last bearish candle before strong bullish move
        Bearish OB: Last bullish candle before strong bearish move
        """
        order_blocks = []

        if len(candles) < 10 or atr == 0:
            return order_blocks

        min_impulse = self.config.ob_min_impulse_atr * atr

        for i in range(2, len(candles) - 1):
            current = candles[i]
            next_candle = candles[i + 1]

            # Check for bullish OB (bearish candle before bullish impulse)
            if current.close < current.open:  # Bearish candle
                # Check impulse move up
                impulse_move = next_candle.close - current.low
                if impulse_move >= min_impulse and next_candle.close > next_candle.open:
                    # Verify it created structure break
                    recent_highs = [c.high for c in candles[max(0, i-10):i]]
                    if recent_highs and next_candle.close > max(recent_highs):
                        order_blocks.append(OrderBlock(
                            type=POIType.BULLISH_OB,
                            high=current.high,
                            low=current.low,
                            timestamp=current.timestamp,
                            impulse_strength=impulse_move / atr
                        ))

            # Check for bearish OB (bullish candle before bearish impulse)
            elif current.close > current.open:  # Bullish candle
                # Check impulse move down
                impulse_move = current.high - next_candle.close
                if impulse_move >= min_impulse and next_candle.close < next_candle.open:
                    # Verify it created structure break
                    recent_lows = [c.low for c in candles[max(0, i-10):i]]
                    if recent_lows and next_candle.close < min(recent_lows):
                        order_blocks.append(OrderBlock(
                            type=POIType.BEARISH_OB,
                            high=current.high,
                            low=current.low,
                            timestamp=current.timestamp,
                            impulse_strength=impulse_move / atr
                        ))

        return order_blocks

    # ==================== Fair Value Gap Detection ====================

    def _find_fvgs(self, candles: List[Candle], atr: float) -> List[FairValueGap]:
        """
        Find Fair Value Gaps (imbalances) in candle data.

        FVG is identified by 3 consecutive candles where there's no overlap
        between candle 1's body and candle 3's body.
        """
        fvgs = []

        if len(candles) < 3 or atr == 0:
            return fvgs

        min_gap = self.config.fvg_min_gap_atr * atr

        for i in range(2, len(candles)):
            c1 = candles[i - 2]
            c2 = candles[i - 1]
            c3 = candles[i]

            # Bullish FVG: gap between c1 high and c3 low
            if c3.low > c1.high:
                gap_size = c3.low - c1.high
                if gap_size >= min_gap:
                    fvgs.append(FairValueGap(
                        type=POIType.BULLISH_FVG,
                        high=c3.low,
                        low=c1.high,
                        timestamp=c2.timestamp
                    ))

            # Bearish FVG: gap between c1 low and c3 high
            elif c3.high < c1.low:
                gap_size = c1.low - c3.high
                if gap_size >= min_gap:
                    fvgs.append(FairValueGap(
                        type=POIType.BEARISH_FVG,
                        high=c1.low,
                        low=c3.high,
                        timestamp=c2.timestamp
                    ))

        return fvgs

    # ==================== Liquidity Detection ====================

    def _find_liquidity_levels(self, candles: List[Candle]) -> List[LiquidityLevel]:
        """
        Find liquidity levels (where stops likely accumulate).

        - Swing highs/lows
        - Equal highs/lows (double tops/bottoms)
        """
        levels = []

        if len(candles) < 20:
            return levels

        swing_highs, swing_lows = self._find_swing_points(candles, lookback=3)

        # Add swing points as liquidity levels
        for sh in swing_highs:
            levels.append(LiquidityLevel(
                price=sh.price,
                type="swing_high",
                strength=0.7,
                timestamp=sh.timestamp
            ))

        for sl in swing_lows:
            levels.append(LiquidityLevel(
                price=sl.price,
                type="swing_low",
                strength=0.7,
                timestamp=sl.timestamp
            ))

        # Find equal highs/lows
        tolerance = 2 * self.pip_size

        # Group equal highs
        for i, sh1 in enumerate(swing_highs):
            for sh2 in swing_highs[i+1:]:
                if abs(sh1.price - sh2.price) <= tolerance:
                    avg_price = (sh1.price + sh2.price) / 2
                    # Check if already added
                    existing = [l for l in levels if l.type == "equal_highs" and abs(l.price - avg_price) <= tolerance]
                    if existing:
                        existing[0].touches += 1
                        existing[0].strength = min(1.0, existing[0].strength + 0.2)
                    else:
                        levels.append(LiquidityLevel(
                            price=avg_price,
                            type="equal_highs",
                            strength=0.85,
                            timestamp=sh2.timestamp,
                            touches=2
                        ))

        # Group equal lows
        for i, sl1 in enumerate(swing_lows):
            for sl2 in swing_lows[i+1:]:
                if abs(sl1.price - sl2.price) <= tolerance:
                    avg_price = (sl1.price + sl2.price) / 2
                    existing = [l for l in levels if l.type == "equal_lows" and abs(l.price - avg_price) <= tolerance]
                    if existing:
                        existing[0].touches += 1
                        existing[0].strength = min(1.0, existing[0].strength + 0.2)
                    else:
                        levels.append(LiquidityLevel(
                            price=avg_price,
                            type="equal_lows",
                            strength=0.85,
                            timestamp=sl2.timestamp,
                            touches=2
                        ))

        return levels

    # ==================== POI Identification ====================

    def _identify_pois(
        self,
        bias: MarketBias,
        order_blocks: List[OrderBlock],
        fvgs: List[FairValueGap],
        liquidity_levels: List[LiquidityLevel],
        current_price: float,
        swing_range: Tuple[float, float]
    ) -> List[POI]:
        """
        Identify Points of Interest based on confluence.

        Score system:
        - Order Block valid: +1
        - FVG overlaps with OB: +1
        - Liquidity nearby: +1
        - In correct zone (discount for longs, premium for shorts): +1
        - Fresh (untested): +1
        """
        pois = []

        swing_high, swing_low = swing_range
        if swing_high == swing_low:
            return pois

        # Calculate 50% level (equilibrium)
        equilibrium = (swing_high + swing_low) / 2

        # For bullish bias, look for bullish POIs in discount zone
        if bias == MarketBias.BULLISH:
            for ob in order_blocks:
                if ob.type != POIType.BULLISH_OB or ob.mitigated:
                    continue

                score = 1  # Base score for valid OB

                # Check if in discount zone
                ob_mid = (ob.high + ob.low) / 2
                is_discount = ob_mid < equilibrium
                if is_discount:
                    score += 1

                # Check for FVG overlap
                overlapping_fvg = None
                for fvg in fvgs:
                    if fvg.type == POIType.BULLISH_FVG and fvg.is_valid:
                        if fvg.low <= ob.high and fvg.high >= ob.low:
                            score += 1
                            overlapping_fvg = fvg
                            break

                # Check for nearby liquidity
                nearby_liq = None
                for liq in liquidity_levels:
                    if liq.type in ["swing_low", "equal_lows"]:
                        if abs(liq.price - ob.low) <= 5 * self.pip_size:
                            score += 1
                            nearby_liq = liq
                            break

                # Fresh (not tested yet)
                if ob.is_valid:
                    score += 1

                if score >= self.config.poi_min_score:
                    pois.append(POI(
                        price_high=ob.high,
                        price_low=ob.low,
                        type=f"Bullish OB (score: {score})",
                        score=score,
                        timestamp=ob.timestamp,
                        order_block=ob,
                        fvg=overlapping_fvg,
                        liquidity_nearby=nearby_liq,
                        is_discount=is_discount
                    ))

        # For bearish bias, look for bearish POIs in premium zone
        elif bias == MarketBias.BEARISH:
            for ob in order_blocks:
                if ob.type != POIType.BEARISH_OB or ob.mitigated:
                    continue

                score = 1

                ob_mid = (ob.high + ob.low) / 2
                is_premium = ob_mid > equilibrium
                if is_premium:
                    score += 1

                overlapping_fvg = None
                for fvg in fvgs:
                    if fvg.type == POIType.BEARISH_FVG and fvg.is_valid:
                        if fvg.low <= ob.high and fvg.high >= ob.low:
                            score += 1
                            overlapping_fvg = fvg
                            break

                nearby_liq = None
                for liq in liquidity_levels:
                    if liq.type in ["swing_high", "equal_highs"]:
                        if abs(liq.price - ob.high) <= 5 * self.pip_size:
                            score += 1
                            nearby_liq = liq
                            break

                if ob.is_valid:
                    score += 1

                if score >= self.config.poi_min_score:
                    pois.append(POI(
                        price_high=ob.high,
                        price_low=ob.low,
                        type=f"Bearish OB (score: {score})",
                        score=score,
                        timestamp=ob.timestamp,
                        order_block=ob,
                        fvg=overlapping_fvg,
                        liquidity_nearby=nearby_liq,
                        is_premium=is_premium
                    ))

        # Sort by score (highest first)
        pois.sort(key=lambda x: x.score, reverse=True)

        return pois

    # ==================== Entry Confirmation (M5 CHoCH) ====================

    def _detect_m5_choch(self, candles: List[Candle], direction: str) -> Tuple[bool, Optional[float]]:
        """
        Detect Change of Character on M5 for entry confirmation.

        For LONG: Look for break of recent Lower High
        For SHORT: Look for break of recent Higher Low

        Returns:
            Tuple of (choch_detected, sl_level)
        """
        if len(candles) < self.config.choch_lookback + 2:
            return False, None

        recent = candles[-(self.config.choch_lookback + 1):-1]
        current = candles[-1]

        if direction == "long":
            # Find swing points in bearish structure
            swing_high = max(c.high for c in recent)
            swing_low = min(c.low for c in recent)

            # CHoCH = price closes above recent swing high
            if current.close > swing_high:
                return True, swing_low

        elif direction == "short":
            swing_high = max(c.high for c in recent)
            swing_low = min(c.low for c in recent)

            # CHoCH = price closes below recent swing low
            if current.close < swing_low:
                return True, swing_high

        return False, None

    # ==================== Volatility Filter ====================

    def _check_volatility_filter(self) -> bool:
        """Check if volatility is within acceptable range."""
        if self.atr_h1 == 0:
            return True  # Can't check without ATR

        # Compare current ATR to average (approximated)
        # For now, just check if ATR is reasonable
        atr_pips = self.atr_h1 / self.pip_size

        # Typical forex pairs have 5-50 pip ATR on H1
        if atr_pips < 3 or atr_pips > 100:
            logger.debug(f"Volatility filter: ATR {atr_pips:.1f} pips out of range")
            return False

        return True

    # ==================== Spread Filter ====================

    def _check_spread_filter(self) -> bool:
        """Check if spread is acceptable."""
        if self.current_spread == 0 or self.avg_spread == 0:
            return True

        if self.current_spread > self.avg_spread * self.config.max_spread_multiplier:
            logger.debug(f"Spread filter: {self.current_spread/self.pip_size:.1f} pips > {self.config.max_spread_multiplier}x avg")
            return False

        return True

    # ==================== News Filter ====================

    def _check_news_filter(self, timestamp: datetime) -> bool:
        """
        Check if it's safe to trade considering news events.

        Blocks trading:
        - 2 hours before red news affecting the pair
        - 30 minutes after red news
        """
        if not self.news_filter:
            return True

        return self.news_filter.is_safe_to_trade(timestamp, self.symbol)

    # ==================== Calculate TP Based on Liquidity ====================

    def _calculate_tp(
        self,
        entry_price: float,
        sl_price: float,
        is_long: bool,
        liquidity_levels: List[LiquidityLevel]
    ) -> Tuple[float, float, str]:
        """
        Calculate Take Profit based on liquidity levels.

        Returns:
            Tuple of (tp_price, actual_rr, target_description)
        """
        risk = abs(entry_price - sl_price)

        if risk == 0:
            return entry_price, 0, "invalid"

        # Filter relevant liquidity levels
        if is_long:
            targets = [l for l in liquidity_levels
                      if l.price > entry_price and l.type in ("swing_high", "equal_highs")]
            targets.sort(key=lambda x: x.price)
        else:
            targets = [l for l in liquidity_levels
                      if l.price < entry_price and l.type in ("swing_low", "equal_lows")]
            targets.sort(key=lambda x: x.price, reverse=True)

        # Find best target within acceptable R:R
        for target in targets:
            if is_long:
                potential_rr = (target.price - entry_price) / risk
            else:
                potential_rr = (entry_price - target.price) / risk

            if self.config.min_rr <= potential_rr <= self.config.tp3_rr:
                # Add small buffer before liquidity
                buffer = 1 * self.pip_size
                if is_long:
                    tp = target.price - buffer
                else:
                    tp = target.price + buffer

                return tp, potential_rr, f"{target.type}({target.touches}x)"

        # Fallback to fixed R:R
        if is_long:
            tp = entry_price + (risk * self.config.tp2_rr)
        else:
            tp = entry_price - (risk * self.config.tp2_rr)

        return tp, self.config.tp2_rr, "fallback_rr"

    # ==================== Calculate Signal Score ====================

    def _calculate_signal_score(
        self,
        poi: POI,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        timestamp: datetime
    ) -> SignalScore:
        """Calculate quality score for the trading signal."""
        score = SignalScore()

        # POI score (0-5)
        score.poi_score = poi.score

        # Confluence score (0-1)
        confluence = 0.0
        if poi.order_block:
            confluence += 0.4
        if poi.fvg:
            confluence += 0.3
        if poi.liquidity_nearby:
            confluence += 0.3
        score.confluence_score = confluence

        # Structure score (based on impulse strength)
        if poi.order_block:
            score.structure_score = min(1.0, poi.order_block.impulse_strength / 3.0)
        else:
            score.structure_score = 0.5

        # Timing score (early in session is better)
        session = self._get_current_session(timestamp)
        if session:
            if session == "london":
                session_start = self.config.london_start_hour
            else:
                session_start = self.config.ny_start_hour

            minutes_into_session = (timestamp.hour - session_start) * 60 + timestamp.minute
            if minutes_into_session < 60:
                score.timing_score = 1.0
            elif minutes_into_session < 120:
                score.timing_score = 0.7
            else:
                score.timing_score = 0.4

        # R:R score
        risk = abs(entry_price - sl_price)
        reward = abs(tp_price - entry_price)
        rr = reward / risk if risk > 0 else 0

        if rr >= 3:
            score.rr_score = 1.0
        elif rr >= 2:
            score.rr_score = 0.8
        elif rr >= 1.5:
            score.rr_score = 0.6
        else:
            score.rr_score = 0.3

        score.calculate_total()
        return score

    # ==================== Main Entry Point ====================

    def on_candle(
        self,
        candle: Candle,
        candles: list[Candle],
    ) -> Optional[StrategySignal]:
        """
        Process new M5 candle and check for trading signals.

        The multi-timeframe candles should be set via set_candles() before calling this.
        """
        if not self._enabled:
            return None

        ts = candle.timestamp

        # Check for new day
        current_date = ts.date()
        if self._state.current_date != current_date:
            self._reset_daily_state(current_date)

        # === FILTER 1: Session Check ===
        session = self._get_current_session(ts)
        if not session:
            return None  # Outside trading sessions

        # === FILTER 2: News Filter (2h before red news) ===
        if not self._check_news_filter(ts):
            if not hasattr(self, '_news_blocked_logged') or self._news_blocked_logged != ts.date():
                logger.warning(f"SMC [{self.symbol}] News filter: Trade blocked (red news within {self.config.news_buffer_hours}h)")
                self._news_blocked_logged = ts.date()
            return None

        # === FILTER 3: Max trades check ===
        if self._state.trades_today >= self.config.max_trades_per_day:
            return None

        # === FILTER 4: Volatility ===
        if not self._check_volatility_filter():
            return None

        # === FILTER 5: Spread ===
        if not self._check_spread_filter():
            return None

        # === STEP 1: Analyze H4 for overall bias ===
        if len(self.h4_candles) >= 30:
            self._state.h4_bias, _ = self._analyze_market_structure(self.h4_candles, self.atr_h4)

        # === STEP 2: Analyze H1 for structure and POIs ===
        if len(self.h1_candles) < 50:
            return None

        h1_bias, h1_breaks = self._analyze_market_structure(self.h1_candles, self.atr_h1)
        self._state.h1_bias = h1_bias
        self._state.h1_structure_breaks = h1_breaks

        # Determine trading bias (H4 and H1 should agree)
        if self._state.h4_bias == MarketBias.NEUTRAL or self._state.h1_bias == MarketBias.NEUTRAL:
            logger.debug(f"SMC [{self.symbol}] No clear bias: H4={self._state.h4_bias.value}, H1={h1_bias.value}")
            return None

        if self._state.h4_bias != self._state.h1_bias:
            logger.debug(f"SMC [{self.symbol}] Bias conflict: H4={self._state.h4_bias.value}, H1={h1_bias.value}")
            return None

        trading_bias = self._state.h4_bias

        # === STEP 3: Find Order Blocks and FVGs on H1 ===
        order_blocks = self._find_order_blocks(self.h1_candles, self.atr_h1)
        fvgs = self._find_fvgs(self.h1_candles, self.atr_h1)
        liquidity_levels = self._find_liquidity_levels(self.h1_candles)

        self._state.order_blocks = order_blocks
        self._state.fvgs = fvgs
        self._state.liquidity_levels = liquidity_levels

        # Calculate swing range for premium/discount
        h1_highs, h1_lows = self._find_swing_points(self.h1_candles[-50:], lookback=3)
        if h1_highs and h1_lows:
            swing_range = (max(h.price for h in h1_highs[-3:]), min(l.price for l in h1_lows[-3:]))
        else:
            swing_range = (candle.high, candle.low)

        # === STEP 4: Identify POIs ===
        current_price = candle.close
        pois = self._identify_pois(
            bias=trading_bias,
            order_blocks=order_blocks,
            fvgs=fvgs,
            liquidity_levels=liquidity_levels,
            current_price=current_price,
            swing_range=swing_range
        )

        self._state.active_pois = pois

        if not pois:
            logger.debug(f"SMC [{self.symbol}] No valid POIs found")
            return None

        # === STEP 5: Check if price is in POI zone ===
        active_poi = None
        for poi in pois:
            if poi.price_low <= current_price <= poi.price_high:
                active_poi = poi
                logger.info(f"SMC [{self.symbol}] Price in POI: {poi.type} @ {poi.price_low:.5f}-{poi.price_high:.5f}")
                break

        if not active_poi:
            # Check if approaching POI
            for poi in pois:
                distance_pips = abs(current_price - (poi.price_high + poi.price_low) / 2) / self.pip_size
                if distance_pips < 10:
                    logger.debug(f"SMC [{self.symbol}] Approaching POI: {poi.type} ({distance_pips:.1f} pips away)")
            return None

        self._state.active_poi = active_poi

        # === STEP 6: Wait for M5 CHoCH confirmation ===
        direction = "long" if trading_bias == MarketBias.BULLISH else "short"
        choch_detected, sl_level = self._detect_m5_choch(self.m5_candles, direction)

        if not choch_detected:
            logger.debug(f"SMC [{self.symbol}] In POI, waiting for M5 CHoCH confirmation")
            return None

        logger.info(f"SMC [{self.symbol}] M5 CHoCH detected! Direction: {direction}")

        # === STEP 7: Calculate Entry, SL, TP ===
        entry_price = current_price
        is_long = direction == "long"

        # SL calculation
        buffer = 0.5 * self.atr_m5  # Buffer beyond CHoCH level
        spread_buffer = self.current_spread

        if is_long:
            base_sl = sl_level - buffer
            sl = base_sl - spread_buffer
            signal_type = SignalType.LONG
        else:
            base_sl = sl_level + buffer
            sl = base_sl + spread_buffer
            signal_type = SignalType.SHORT

        # Check SL distance
        sl_pips = abs(entry_price - sl) / self.pip_size
        max_sl_pips = self.config.max_sl_atr * (self.atr_h1 / self.pip_size)

        if sl_pips > max_sl_pips:
            logger.info(f"SMC [{self.symbol}] SL too wide: {sl_pips:.1f} pips > {max_sl_pips:.1f} max")
            return None

        # TP calculation
        tp, actual_rr, target_type = self._calculate_tp(entry_price, sl, is_long, liquidity_levels)

        if actual_rr < self.config.min_rr:
            logger.info(f"SMC [{self.symbol}] R:R too low: {actual_rr:.2f} < {self.config.min_rr}")
            return None

        # === STEP 8: Calculate Signal Score ===
        signal_score = self._calculate_signal_score(active_poi, entry_price, sl, tp, ts)

        min_score = 0.5  # Minimum acceptable score
        if signal_score.total < min_score:
            logger.info(f"SMC [{self.symbol}] Signal score too low: {signal_score.total:.2f} < {min_score}")
            return None

        # === STEP 9: Generate Signal ===
        self._state.trades_today += 1
        self._state.last_trade_session = session

        tp_pips = abs(tp - entry_price) / self.pip_size
        spread_pips = self.current_spread / self.pip_size

        logger.info(
            f"SMC [{self.symbol}] {direction.upper()} SIGNAL | "
            f"Score: {signal_score.total:.2f} | POI: {active_poi.score}/5 | "
            f"Entry: {entry_price:.5f}, SL: {sl:.5f} ({sl_pips:.1f}p), TP: {tp:.5f} ({tp_pips:.1f}p) | "
            f"R:R: {actual_rr:.2f} | Target: {target_type} | "
            f"Bias: H4={self._state.h4_bias.value}, H1={h1_bias.value}"
        )

        signal = StrategySignal(
            signal_type=signal_type,
            symbol=self.symbol,
            price=entry_price,
            stop_loss=sl,
            take_profit=tp,
            confidence=signal_score.total,
            reason=f"SMC {direction.upper()} - {active_poi.type} (R:R {actual_rr:.1f})",
            metadata={
                "session": session,
                "h4_bias": self._state.h4_bias.value,
                "h1_bias": h1_bias.value,
                "poi_score": active_poi.score,
                "poi_type": active_poi.type,
                "sl_pips": sl_pips,
                "tp_pips": tp_pips,
                "spread_pips": spread_pips,
                "actual_rr": actual_rr,
                "target_type": target_type,
                "signal_score": signal_score.total,
                "score_details": {
                    "poi": signal_score.poi_score,
                    "confluence": signal_score.confluence_score,
                    "structure": signal_score.structure_score,
                    "timing": signal_score.timing_score,
                    "rr": signal_score.rr_score,
                },
                "order_block": {
                    "high": active_poi.order_block.high if active_poi.order_block else None,
                    "low": active_poi.order_block.low if active_poi.order_block else None,
                } if active_poi.order_block else None,
            },
        )

        self._last_signal = signal
        return signal

    def on_tick(
        self,
        bid: float,
        ask: float,
        timestamp: datetime,
    ) -> Optional[StrategySignal]:
        """Process tick update (not used for SMC)."""
        return None

    def should_exit(
        self,
        position: Position,
        current_price: float,
        candles: Optional[list[Candle]] = None,
    ) -> Optional[StrategySignal]:
        """Check if position should be exited."""
        if not candles:
            return None

        ts = candles[-1].timestamp
        session = self._get_current_session(ts)

        # Exit before session end
        if session:
            if session == "london":
                end_hour = self.config.london_end_hour
            else:
                end_hour = self.config.ny_end_hour

            minutes_to_end = (end_hour * 60) - (ts.hour * 60 + ts.minute)
            if minutes_to_end <= 10:
                logger.info(f"SMC [{self.symbol}] Session ending, closing position")
                return StrategySignal(
                    signal_type=SignalType.EXIT_LONG if position.is_long else SignalType.EXIT_SHORT,
                    symbol=self.symbol,
                    price=current_price,
                    reason="Session end exit"
                )

        return None

    def get_status(self) -> dict:
        """Get strategy status."""
        base_status = super().get_status()

        base_status.update({
            "h4_bias": self._state.h4_bias.value,
            "h1_bias": self._state.h1_bias.value,
            "trades_today": self._state.trades_today,
            "max_trades_per_day": self.config.max_trades_per_day,
            "active_pois": len(self._state.active_pois),
            "order_blocks": len(self._state.order_blocks),
            "fvgs": len(self._state.fvgs),
            "atr_h1": self.atr_h1,
            "atr_m5": self.atr_m5,
        })

        return base_status

    def reset(self) -> None:
        """Reset strategy for new session."""
        super().reset()
        self._reset_daily_state()
