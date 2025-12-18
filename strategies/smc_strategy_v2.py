"""
Smart Money Concepts (SMC) Strategy V2 - Optimized for Profitability.

Key Improvements over V1:
1. EMA 200 trend filter on H1 for direction bias
2. Liquidity sweep detection - wait for stops to be taken before entry
3. More flexible bias requirements (H1 only when H4 is neutral)
4. Lower POI score requirement with better confluence detection
5. Multiple confirmation types (CHoCH, Engulfing, Rejection, Break & Retest)
6. Extended session times for more opportunities
7. Dynamic R:R based on market structure
8. Improved Order Block and FVG detection
9. Momentum filter using price action

Timeframe Usage:
- H4: Overall trend context (optional agreement)
- H1: Primary structure analysis, POI identification, EMA 200 trend
- M15: Entry timeframe for confirmations
- M5: Precision entries

Trading Sessions (Extended):
- London: 07:00-12:00 UTC (5 hours)
- New York: 12:00-17:00 UTC (5 hours, overlaps with London close)
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

logger = get_logger(__name__)


# ==================== Enums and Data Classes ====================

class MarketBias(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class StructureType(Enum):
    BOS = "bos"
    CHOCH = "choch"


class POIType(Enum):
    BULLISH_OB = "bullish_ob"
    BEARISH_OB = "bearish_ob"
    BULLISH_FVG = "bullish_fvg"
    BEARISH_FVG = "bearish_fvg"
    BULLISH_BREAKER = "bullish_breaker"
    BEARISH_BREAKER = "bearish_breaker"


class ConfirmationType(Enum):
    CHOCH = "choch"
    ENGULFING = "engulfing"
    REJECTION = "rejection"
    BREAK_RETEST = "break_retest"
    LIQUIDITY_SWEEP = "liquidity_sweep"


@dataclass
class SwingPoint:
    price: float
    timestamp: datetime
    is_high: bool
    strength: int = 1
    swept: bool = False  # True if liquidity was taken


@dataclass
class StructureBreak:
    type: StructureType
    direction: str
    break_price: float
    timestamp: datetime
    swing_broken: SwingPoint


@dataclass
class OrderBlock:
    type: POIType
    high: float
    low: float
    timestamp: datetime
    is_valid: bool = True
    mitigated: bool = False
    impulse_strength: float = 0.0
    candle_body_high: float = 0.0
    candle_body_low: float = 0.0


@dataclass
class FairValueGap:
    type: POIType
    high: float
    low: float
    timestamp: datetime
    is_valid: bool = True
    fill_percentage: float = 0.0
    size_atr: float = 0.0


@dataclass
class LiquidityLevel:
    price: float
    type: str
    strength: float
    timestamp: datetime
    touches: int = 1
    swept: bool = False
    sweep_time: Optional[datetime] = None


@dataclass
class POI:
    price_high: float
    price_low: float
    type: str
    score: float  # Changed to float for more granular scoring
    timestamp: datetime
    order_block: Optional[OrderBlock] = None
    fvg: Optional[FairValueGap] = None
    liquidity_nearby: Optional[LiquidityLevel] = None
    is_premium: bool = False
    is_discount: bool = False
    is_fresh: bool = True
    has_liquidity_sweep: bool = False  # Key for SMC


@dataclass
class Confirmation:
    type: ConfirmationType
    timestamp: datetime
    entry_price: float
    sl_price: float
    strength: float  # 0-1


@dataclass
class SMCStateV2:
    """State tracking for SMC V2 strategy."""
    h4_bias: MarketBias = MarketBias.NEUTRAL
    h1_bias: MarketBias = MarketBias.NEUTRAL
    ema_trend: MarketBias = MarketBias.NEUTRAL  # EMA 200 based trend

    h1_swing_highs: List[SwingPoint] = field(default_factory=list)
    h1_swing_lows: List[SwingPoint] = field(default_factory=list)
    h1_structure_breaks: List[StructureBreak] = field(default_factory=list)

    active_pois: List[POI] = field(default_factory=list)
    order_blocks: List[OrderBlock] = field(default_factory=list)
    fvgs: List[FairValueGap] = field(default_factory=list)
    liquidity_levels: List[LiquidityLevel] = field(default_factory=list)

    # Entry tracking
    waiting_for_entry: bool = False
    active_poi: Optional[POI] = None
    pending_confirmation: Optional[Confirmation] = None

    # Session tracking
    current_date: Optional[datetime] = None
    trades_today: int = 0
    last_trade_session: Optional[str] = None
    consecutive_losses: int = 0

    # Premium/Discount
    swing_range_high: Optional[float] = None
    swing_range_low: Optional[float] = None

    # Recent liquidity sweeps
    recent_sweeps: List[LiquidityLevel] = field(default_factory=list)


@dataclass
class SMCConfigV2:
    """Optimized configuration for SMC V2 strategy."""
    # Timeframes
    tf_bias: str = "H4"
    tf_structure: str = "H1"
    tf_entry: str = "M15"
    tf_precision: str = "M5"

    # Trend Filter
    ema_period: int = 200
    use_ema_filter: bool = True
    require_h4_agreement: bool = False  # More flexible

    # Structure Detection - More sensitive
    swing_lookback: int = 3  # Reduced for more swings
    swing_lookback_h1: int = 5
    bos_min_move_atr: float = 0.5  # Reduced from 2.0
    ob_min_impulse_atr: float = 1.0  # Reduced from 2.0
    fvg_min_gap_atr: float = 0.3  # Reduced from 0.5

    # POI Scoring - More granular
    poi_min_score: float = 1.5  # Reduced from 3
    poi_ob_score: float = 1.0
    poi_fvg_overlap_score: float = 0.5
    poi_liquidity_score: float = 0.5
    poi_zone_score: float = 0.5  # Premium/Discount
    poi_fresh_score: float = 0.3
    poi_sweep_score: float = 1.0  # Bonus for liquidity sweep

    # Entry Confirmations
    choch_lookback: int = 12  # Increased from 8
    require_liquidity_sweep: bool = False  # Recommended but not required
    confirmation_types: List[str] = field(default_factory=lambda: ["choch", "engulfing", "rejection"])

    # Risk Management
    risk_percent: float = 1.0
    max_sl_pips: float = 30.0  # Fixed max SL in pips
    min_sl_pips: float = 8.0   # Min SL for proper risk
    min_rr: float = 1.5
    target_rr: float = 2.0
    max_rr: float = 4.0

    # Take Profits - Partial exits
    use_partial_tp: bool = True
    tp1_rr: float = 1.0
    tp1_percent: int = 50
    tp2_rr: float = 2.0
    tp2_percent: int = 30
    tp3_rr: float = 3.0
    tp3_percent: int = 20

    # Sessions (Extended for more opportunities)
    london_start_hour: int = 7
    london_start_minute: int = 0
    london_end_hour: int = 12  # Extended
    london_end_minute: int = 0
    ny_start_hour: int = 12  # Overlap with London
    ny_start_minute: int = 0
    ny_end_hour: int = 17
    ny_end_minute: int = 0

    # Filters
    max_spread_pips: float = 2.0  # Fixed max spread
    max_daily_loss_percent: float = 3.0

    # Trade limits
    max_trades_per_day: int = 4  # Increased
    max_trades_per_session: int = 2
    max_consecutive_losses: int = 3  # Stop after 3 losses


class SMCStrategyV2(BaseStrategy):
    """
    Smart Money Concepts Trading Strategy V2 - Optimized.

    Key features:
    - EMA 200 trend filter
    - Liquidity sweep detection
    - Multiple confirmation types
    - Extended sessions
    - Flexible bias requirements
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "M15",
        magic_number: int = 12360,
        timezone: str = "UTC",
        config: Optional[SMCConfigV2] = None,
    ):
        super().__init__(
            name="SMC_V2",
            symbol=symbol,
            timeframe=timeframe,
            magic_number=magic_number,
        )

        self.config = config or SMCConfigV2()
        self.timezone = timezone
        self.tz = pytz.timezone(timezone) if timezone != "UTC" else pytz.UTC

        self._state = SMCStateV2()

        # Pip size based on symbol
        self.pip_size = self._get_pip_size(symbol)

        # ATR values
        self.atr_h4: float = 0.0
        self.atr_h1: float = 0.0
        self.atr_m15: float = 0.0
        self.atr_m5: float = 0.0

        # EMA 200 value
        self.ema_200: float = 0.0

        # Spread tracking
        self.current_spread: float = 0.0
        self.avg_spread: float = 0.0

        # Candle buffers
        self.h4_candles: List[Candle] = []
        self.h1_candles: List[Candle] = []
        self.m15_candles: List[Candle] = []
        self.m5_candles: List[Candle] = []

    def _get_pip_size(self, symbol: str) -> float:
        """Get pip size for symbol."""
        symbol = symbol.upper()
        if "JPY" in symbol:
            return 0.01
        elif symbol in ["XAUUSD", "GOLD"]:
            return 0.1
        elif symbol in ["US500", "SP500", "SPX500"]:
            return 0.1
        elif symbol in ["US30", "DJ30", "DOW"]:
            return 1.0
        elif symbol in ["NAS100", "USTEC", "NDX"]:
            return 0.1
        elif symbol in ["GER40", "DE40", "DAX"]:
            return 0.1
        return 0.0001  # Default for forex

    def initialize(self) -> bool:
        """Initialize strategy."""
        logger.info(f"Initializing SMC V2 strategy for {self.symbol}")
        logger.info(f"Pip size: {self.pip_size}")
        logger.info(f"EMA filter: {'ON' if self.config.use_ema_filter else 'OFF'}")
        logger.info(f"Min POI score: {self.config.poi_min_score}")
        logger.info(f"Min R:R: {self.config.min_rr}")
        logger.info(f"Sessions: London 07:00-12:00, NY 12:00-17:00 UTC")
        self._reset_daily_state()
        return True

    def _reset_daily_state(self, current_date: Optional[datetime] = None) -> None:
        """Reset state for new trading day."""
        # Preserve consecutive losses across days
        consecutive_losses = self._state.consecutive_losses if hasattr(self._state, 'consecutive_losses') else 0
        self._state = SMCStateV2()
        self._state.current_date = current_date
        self._state.consecutive_losses = consecutive_losses

    def update_spread(self, spread: float) -> None:
        """Update current spread."""
        self.current_spread = spread
        if self.avg_spread == 0:
            self.avg_spread = spread
        else:
            self.avg_spread = self.avg_spread * 0.95 + spread * 0.05

    def set_candles(
        self,
        h4_candles: List[Candle],
        h1_candles: List[Candle],
        m5_candles: List[Candle],
        m15_candles: Optional[List[Candle]] = None
    ) -> None:
        """Set candle data for all timeframes."""
        self.h4_candles = h4_candles
        self.h1_candles = h1_candles
        self.m5_candles = m5_candles
        self.m15_candles = m15_candles if m15_candles else []

        # Calculate ATR
        if len(h4_candles) >= 14:
            self.atr_h4 = self._calculate_atr(h4_candles, 14)
        if len(h1_candles) >= 14:
            self.atr_h1 = self._calculate_atr(h1_candles, 14)
        if len(m5_candles) >= 14:
            self.atr_m5 = self._calculate_atr(m5_candles, 14)
        if m15_candles and len(m15_candles) >= 14:
            self.atr_m15 = self._calculate_atr(m15_candles, 14)

        # Calculate EMA 200 on H1
        if len(h1_candles) >= self.config.ema_period:
            self.ema_200 = self._calculate_ema(h1_candles, self.config.ema_period)

    def _calculate_atr(self, candles: List[Candle], period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(candles) < period + 1:
            return 0.0

        tr_values = []
        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i-1].close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)

        if len(tr_values) < period:
            return sum(tr_values) / len(tr_values) if tr_values else 0.0
        return sum(tr_values[-period:]) / period

    def _calculate_ema(self, candles: List[Candle], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(candles) < period:
            return 0.0

        multiplier = 2 / (period + 1)
        closes = [c.close for c in candles]

        # Start with SMA
        ema = sum(closes[:period]) / period

        # Calculate EMA
        for close in closes[period:]:
            ema = (close - ema) * multiplier + ema

        return ema

    def _get_current_session(self, timestamp: datetime) -> Optional[str]:
        """Determine which session is active."""
        if timestamp.tzinfo is None:
            ts_utc = timestamp
        else:
            ts_utc = timestamp.astimezone(pytz.UTC)

        time_value = ts_utc.hour * 60 + ts_utc.minute

        london_start = self.config.london_start_hour * 60
        london_end = self.config.london_end_hour * 60
        ny_start = self.config.ny_start_hour * 60
        ny_end = self.config.ny_end_hour * 60

        if london_start <= time_value < london_end:
            return "london"
        if ny_start <= time_value < ny_end:
            return "ny"
        return None

    def _find_swing_points(
        self,
        candles: List[Candle],
        lookback: int = 3
    ) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """Find swing highs and lows."""
        swing_highs = []
        swing_lows = []

        if len(candles) < lookback * 2 + 1:
            return swing_highs, swing_lows

        for i in range(lookback, len(candles) - lookback):
            candle = candles[i]

            # Swing high
            is_swing_high = all(
                candles[i - j].high < candle.high and candles[i + j].high < candle.high
                for j in range(1, lookback + 1)
            )
            if is_swing_high:
                swing_highs.append(SwingPoint(
                    price=candle.high,
                    timestamp=candle.timestamp,
                    is_high=True,
                    strength=lookback
                ))

            # Swing low
            is_swing_low = all(
                candles[i - j].low > candle.low and candles[i + j].low > candle.low
                for j in range(1, lookback + 1)
            )
            if is_swing_low:
                swing_lows.append(SwingPoint(
                    price=candle.low,
                    timestamp=candle.timestamp,
                    is_high=False,
                    strength=lookback
                ))

        return swing_highs, swing_lows

    def _detect_liquidity_sweep(
        self,
        candles: List[Candle],
        swing_points: List[SwingPoint],
        is_high: bool
    ) -> List[LiquidityLevel]:
        """Detect liquidity sweeps - price takes out stops and returns."""
        sweeps = []

        if len(candles) < 5 or not swing_points:
            return sweeps

        recent_candles = candles[-20:]  # Check last 20 candles

        for swing in swing_points[-5:]:  # Check last 5 swing points
            for i, candle in enumerate(recent_candles):
                if is_high:
                    # Look for sweep of swing high (wick above, close below)
                    if candle.high > swing.price and candle.close < swing.price:
                        # Verify price moved away after sweep
                        if i < len(recent_candles) - 1:
                            next_candle = recent_candles[i + 1]
                            if next_candle.close < candle.low:
                                sweeps.append(LiquidityLevel(
                                    price=swing.price,
                                    type="sweep_high",
                                    strength=0.9,
                                    timestamp=candle.timestamp,
                                    swept=True,
                                    sweep_time=candle.timestamp
                                ))
                else:
                    # Look for sweep of swing low
                    if candle.low < swing.price and candle.close > swing.price:
                        if i < len(recent_candles) - 1:
                            next_candle = recent_candles[i + 1]
                            if next_candle.close > candle.high:
                                sweeps.append(LiquidityLevel(
                                    price=swing.price,
                                    type="sweep_low",
                                    strength=0.9,
                                    timestamp=candle.timestamp,
                                    swept=True,
                                    sweep_time=candle.timestamp
                                ))

        return sweeps

    def _analyze_market_structure(
        self,
        candles: List[Candle],
        atr: float
    ) -> Tuple[MarketBias, List[StructureBreak]]:
        """Analyze market structure with improved detection."""
        if len(candles) < 30 or atr == 0:
            return MarketBias.NEUTRAL, []

        swing_highs, swing_lows = self._find_swing_points(
            candles,
            lookback=self.config.swing_lookback_h1
        )

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return MarketBias.NEUTRAL, []

        structure_breaks = []
        recent_highs = sorted(swing_highs, key=lambda x: x.timestamp)[-5:]
        recent_lows = sorted(swing_lows, key=lambda x: x.timestamp)[-5:]

        # Count HH/HL vs LH/LL
        hh_count = sum(1 for i in range(1, len(recent_highs))
                       if recent_highs[i].price > recent_highs[i-1].price)
        hl_count = sum(1 for i in range(1, len(recent_lows))
                       if recent_lows[i].price > recent_lows[i-1].price)
        lh_count = sum(1 for i in range(1, len(recent_highs))
                       if recent_highs[i].price < recent_highs[i-1].price)
        ll_count = sum(1 for i in range(1, len(recent_lows))
                       if recent_lows[i].price < recent_lows[i-1].price)

        bullish_score = hh_count + hl_count
        bearish_score = lh_count + ll_count

        # More sensitive bias detection
        if bullish_score >= bearish_score + 1:
            bias = MarketBias.BULLISH
        elif bearish_score >= bullish_score + 1:
            bias = MarketBias.BEARISH
        else:
            bias = MarketBias.NEUTRAL

        # Detect structure breaks
        current_price = candles[-1].close
        min_move = self.config.bos_min_move_atr * atr

        if recent_highs and current_price > recent_highs[-1].price:
            if current_price - recent_highs[-1].price >= min_move:
                break_type = StructureType.BOS if bias == MarketBias.BULLISH else StructureType.CHOCH
                structure_breaks.append(StructureBreak(
                    type=break_type,
                    direction="bullish",
                    break_price=current_price,
                    timestamp=candles[-1].timestamp,
                    swing_broken=recent_highs[-1]
                ))

        if recent_lows and current_price < recent_lows[-1].price:
            if recent_lows[-1].price - current_price >= min_move:
                break_type = StructureType.BOS if bias == MarketBias.BEARISH else StructureType.CHOCH
                structure_breaks.append(StructureBreak(
                    type=break_type,
                    direction="bearish",
                    break_price=current_price,
                    timestamp=candles[-1].timestamp,
                    swing_broken=recent_lows[-1]
                ))

        return bias, structure_breaks

    def _find_order_blocks(
        self,
        candles: List[Candle],
        atr: float
    ) -> List[OrderBlock]:
        """Find Order Blocks with improved detection."""
        order_blocks = []

        if len(candles) < 10 or atr == 0:
            return order_blocks

        min_impulse = self.config.ob_min_impulse_atr * atr

        for i in range(2, len(candles) - 2):
            curr = candles[i]
            prev = candles[i - 1]
            next1 = candles[i + 1]
            next2 = candles[i + 2] if i + 2 < len(candles) else next1

            # Bullish OB: Bearish candle followed by strong bullish move
            if curr.close < curr.open:  # Bearish candle
                # Check for impulse move
                impulse = max(next1.close, next2.close) - curr.low
                if impulse >= min_impulse:
                    # Check it broke structure
                    recent_highs = [c.high for c in candles[max(0, i-10):i]]
                    if recent_highs and max(next1.close, next2.close) > max(recent_highs) * 0.999:
                        order_blocks.append(OrderBlock(
                            type=POIType.BULLISH_OB,
                            high=curr.high,
                            low=curr.low,
                            timestamp=curr.timestamp,
                            impulse_strength=impulse / atr,
                            candle_body_high=max(curr.open, curr.close),
                            candle_body_low=min(curr.open, curr.close)
                        ))

            # Bearish OB
            elif curr.close > curr.open:  # Bullish candle
                impulse = curr.high - min(next1.close, next2.close)
                if impulse >= min_impulse:
                    recent_lows = [c.low for c in candles[max(0, i-10):i]]
                    if recent_lows and min(next1.close, next2.close) < min(recent_lows) * 1.001:
                        order_blocks.append(OrderBlock(
                            type=POIType.BEARISH_OB,
                            high=curr.high,
                            low=curr.low,
                            timestamp=curr.timestamp,
                            impulse_strength=impulse / atr,
                            candle_body_high=max(curr.open, curr.close),
                            candle_body_low=min(curr.open, curr.close)
                        ))

        return order_blocks

    def _find_fvgs(self, candles: List[Candle], atr: float) -> List[FairValueGap]:
        """Find Fair Value Gaps."""
        fvgs = []

        if len(candles) < 3 or atr == 0:
            return fvgs

        min_gap = self.config.fvg_min_gap_atr * atr

        for i in range(2, len(candles)):
            c1 = candles[i - 2]
            c2 = candles[i - 1]
            c3 = candles[i]

            # Bullish FVG
            if c3.low > c1.high:
                gap_size = c3.low - c1.high
                if gap_size >= min_gap:
                    fvgs.append(FairValueGap(
                        type=POIType.BULLISH_FVG,
                        high=c3.low,
                        low=c1.high,
                        timestamp=c2.timestamp,
                        size_atr=gap_size / atr
                    ))

            # Bearish FVG
            elif c3.high < c1.low:
                gap_size = c1.low - c3.high
                if gap_size >= min_gap:
                    fvgs.append(FairValueGap(
                        type=POIType.BEARISH_FVG,
                        high=c1.low,
                        low=c3.high,
                        timestamp=c2.timestamp,
                        size_atr=gap_size / atr
                    ))

        return fvgs

    def _find_liquidity_levels(self, candles: List[Candle]) -> List[LiquidityLevel]:
        """Find liquidity levels."""
        levels = []

        if len(candles) < 20:
            return levels

        swing_highs, swing_lows = self._find_swing_points(candles, lookback=3)

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
        tolerance = 3 * self.pip_size

        for i, sh1 in enumerate(swing_highs):
            for sh2 in swing_highs[i+1:]:
                if abs(sh1.price - sh2.price) <= tolerance:
                    avg_price = (sh1.price + sh2.price) / 2
                    existing = [l for l in levels if l.type == "equal_highs" and abs(l.price - avg_price) <= tolerance]
                    if existing:
                        existing[0].touches += 1
                        existing[0].strength = min(1.0, existing[0].strength + 0.15)
                    else:
                        levels.append(LiquidityLevel(
                            price=avg_price,
                            type="equal_highs",
                            strength=0.85,
                            timestamp=sh2.timestamp,
                            touches=2
                        ))

        for i, sl1 in enumerate(swing_lows):
            for sl2 in swing_lows[i+1:]:
                if abs(sl1.price - sl2.price) <= tolerance:
                    avg_price = (sl1.price + sl2.price) / 2
                    existing = [l for l in levels if l.type == "equal_lows" and abs(l.price - avg_price) <= tolerance]
                    if existing:
                        existing[0].touches += 1
                        existing[0].strength = min(1.0, existing[0].strength + 0.15)
                    else:
                        levels.append(LiquidityLevel(
                            price=avg_price,
                            type="equal_lows",
                            strength=0.85,
                            timestamp=sl2.timestamp,
                            touches=2
                        ))

        return levels

    def _identify_pois(
        self,
        bias: MarketBias,
        order_blocks: List[OrderBlock],
        fvgs: List[FairValueGap],
        liquidity_levels: List[LiquidityLevel],
        current_price: float,
        swing_range: Tuple[float, float],
        recent_sweeps: List[LiquidityLevel]
    ) -> List[POI]:
        """Identify POIs with improved scoring."""
        pois = []

        swing_high, swing_low = swing_range
        if swing_high == swing_low:
            return pois

        equilibrium = (swing_high + swing_low) / 2

        # Process bullish POIs
        if bias in [MarketBias.BULLISH, MarketBias.NEUTRAL]:
            for ob in order_blocks:
                if ob.type != POIType.BULLISH_OB or ob.mitigated:
                    continue

                score = self.config.poi_ob_score

                ob_mid = (ob.high + ob.low) / 2
                is_discount = ob_mid < equilibrium
                if is_discount:
                    score += self.config.poi_zone_score

                # Check FVG overlap
                overlapping_fvg = None
                for fvg in fvgs:
                    if fvg.type == POIType.BULLISH_FVG and fvg.is_valid:
                        if fvg.low <= ob.high and fvg.high >= ob.low:
                            score += self.config.poi_fvg_overlap_score
                            overlapping_fvg = fvg
                            break

                # Check nearby liquidity
                nearby_liq = None
                for liq in liquidity_levels:
                    if liq.type in ["swing_low", "equal_lows"]:
                        if abs(liq.price - ob.low) <= 10 * self.pip_size:
                            score += self.config.poi_liquidity_score
                            nearby_liq = liq
                            break

                # Check for liquidity sweep
                has_sweep = False
                for sweep in recent_sweeps:
                    if sweep.type == "sweep_low" and abs(sweep.price - ob.low) <= 15 * self.pip_size:
                        score += self.config.poi_sweep_score
                        has_sweep = True
                        break

                if ob.is_valid:
                    score += self.config.poi_fresh_score

                if score >= self.config.poi_min_score:
                    pois.append(POI(
                        price_high=ob.high,
                        price_low=ob.low,
                        type=f"Bullish OB",
                        score=score,
                        timestamp=ob.timestamp,
                        order_block=ob,
                        fvg=overlapping_fvg,
                        liquidity_nearby=nearby_liq,
                        is_discount=is_discount,
                        has_liquidity_sweep=has_sweep
                    ))

        # Process bearish POIs
        if bias in [MarketBias.BEARISH, MarketBias.NEUTRAL]:
            for ob in order_blocks:
                if ob.type != POIType.BEARISH_OB or ob.mitigated:
                    continue

                score = self.config.poi_ob_score

                ob_mid = (ob.high + ob.low) / 2
                is_premium = ob_mid > equilibrium
                if is_premium:
                    score += self.config.poi_zone_score

                overlapping_fvg = None
                for fvg in fvgs:
                    if fvg.type == POIType.BEARISH_FVG and fvg.is_valid:
                        if fvg.low <= ob.high and fvg.high >= ob.low:
                            score += self.config.poi_fvg_overlap_score
                            overlapping_fvg = fvg
                            break

                nearby_liq = None
                for liq in liquidity_levels:
                    if liq.type in ["swing_high", "equal_highs"]:
                        if abs(liq.price - ob.high) <= 10 * self.pip_size:
                            score += self.config.poi_liquidity_score
                            nearby_liq = liq
                            break

                has_sweep = False
                for sweep in recent_sweeps:
                    if sweep.type == "sweep_high" and abs(sweep.price - ob.high) <= 15 * self.pip_size:
                        score += self.config.poi_sweep_score
                        has_sweep = True
                        break

                if ob.is_valid:
                    score += self.config.poi_fresh_score

                if score >= self.config.poi_min_score:
                    pois.append(POI(
                        price_high=ob.high,
                        price_low=ob.low,
                        type=f"Bearish OB",
                        score=score,
                        timestamp=ob.timestamp,
                        order_block=ob,
                        fvg=overlapping_fvg,
                        liquidity_nearby=nearby_liq,
                        is_premium=is_premium,
                        has_liquidity_sweep=has_sweep
                    ))

        pois.sort(key=lambda x: x.score, reverse=True)
        return pois

    def _detect_confirmation(
        self,
        candles: List[Candle],
        direction: str,
        poi: POI
    ) -> Optional[Confirmation]:
        """Detect entry confirmation with stricter requirements."""
        if len(candles) < self.config.choch_lookback + 2:
            return None

        recent = candles[-(self.config.choch_lookback + 1):-1]
        current = candles[-1]
        prev = candles[-2]
        prev2 = candles[-3] if len(candles) > 2 else prev

        # Calculate average candle size for context
        avg_body = sum(abs(c.close - c.open) for c in recent) / len(recent)
        current_body = abs(current.close - current.open)

        if direction == "long":
            swing_high = max(c.high for c in recent)
            swing_low = min(c.low for c in recent)

            # Method 1: CHoCH - must break swing high with strong candle
            if current.close > swing_high:
                # Require strong bullish close
                if current.close > current.open and current_body > avg_body * 0.8:
                    return Confirmation(
                        type=ConfirmationType.CHOCH,
                        timestamp=current.timestamp,
                        entry_price=current.close,
                        sl_price=swing_low,
                        strength=0.9
                    )

            # Method 2: Bullish Engulfing
            # Must be a meaningful engulfing
            if (current.close > current.open and
                prev.close < prev.open and
                current.open <= prev.close and
                current.close >= prev.open and
                current_body > avg_body * 0.9 and  # At least average size
                current.close > prev.open):  # Must close above prev open
                return Confirmation(
                    type=ConfirmationType.ENGULFING,
                    timestamp=current.timestamp,
                    entry_price=current.close,
                    sl_price=min(current.low, prev.low),
                    strength=0.75
                )

            # Method 3: Strong Rejection (hammer)
            body = abs(current.close - current.open)
            lower_wick = min(current.close, current.open) - current.low
            upper_wick = current.high - max(current.close, current.open)

            # Must have very long lower wick and small body
            if (lower_wick > body * 2.5 and
                upper_wick < body * 0.5 and
                current.close > current.open and
                current.close > (current.high + current.low) / 2):  # Close in upper half
                return Confirmation(
                    type=ConfirmationType.REJECTION,
                    timestamp=current.timestamp,
                    entry_price=current.close,
                    sl_price=current.low,
                    strength=0.7
                )

        else:  # Short
            swing_high = max(c.high for c in recent)
            swing_low = min(c.low for c in recent)

            # Method 1: CHoCH
            if current.close < swing_low:
                if current.close < current.open and current_body > avg_body * 0.8:
                    return Confirmation(
                        type=ConfirmationType.CHOCH,
                        timestamp=current.timestamp,
                        entry_price=current.close,
                        sl_price=swing_high,
                        strength=0.9
                    )

            # Method 2: Bearish Engulfing
            if (current.close < current.open and
                prev.close > prev.open and
                current.open >= prev.close and
                current.close <= prev.open and
                current_body > avg_body * 0.9 and
                current.close < prev.open):
                return Confirmation(
                    type=ConfirmationType.ENGULFING,
                    timestamp=current.timestamp,
                    entry_price=current.close,
                    sl_price=max(current.high, prev.high),
                    strength=0.75
                )

            # Method 3: Strong Rejection (inverted hammer / shooting star)
            body = abs(current.close - current.open)
            upper_wick = current.high - max(current.close, current.open)
            lower_wick = min(current.close, current.open) - current.low

            if (upper_wick > body * 2.5 and
                lower_wick < body * 0.5 and
                current.close < current.open and
                current.close < (current.high + current.low) / 2):
                return Confirmation(
                    type=ConfirmationType.REJECTION,
                    timestamp=current.timestamp,
                    entry_price=current.close,
                    sl_price=current.high,
                    strength=0.7
                )

        return None

    def _calculate_tp(
        self,
        entry_price: float,
        sl_price: float,
        is_long: bool,
        liquidity_levels: List[LiquidityLevel]
    ) -> Tuple[float, float, str]:
        """Calculate TP with improved logic."""
        risk = abs(entry_price - sl_price)

        if risk == 0:
            return entry_price, 0, "invalid"

        # Find liquidity targets
        if is_long:
            targets = [l for l in liquidity_levels
                      if l.price > entry_price + risk and l.type in ("swing_high", "equal_highs")]
            targets.sort(key=lambda x: x.price)
        else:
            targets = [l for l in liquidity_levels
                      if l.price < entry_price - risk and l.type in ("swing_low", "equal_lows")]
            targets.sort(key=lambda x: x.price, reverse=True)

        # Find first target with acceptable R:R
        for target in targets[:3]:  # Check first 3 targets
            if is_long:
                potential_rr = (target.price - entry_price) / risk
            else:
                potential_rr = (entry_price - target.price) / risk

            if self.config.min_rr <= potential_rr <= self.config.max_rr:
                buffer = 2 * self.pip_size
                if is_long:
                    tp = target.price - buffer
                else:
                    tp = target.price + buffer
                return tp, potential_rr, f"{target.type}"

        # Fallback to fixed R:R
        if is_long:
            tp = entry_price + (risk * self.config.target_rr)
        else:
            tp = entry_price - (risk * self.config.target_rr)

        return tp, self.config.target_rr, "fixed_rr"

    def on_candle(
        self,
        candle: Candle,
        candles: list[Candle],
    ) -> Optional[StrategySignal]:
        """Process new candle and generate signals."""
        if not self._enabled:
            return None

        ts = candle.timestamp

        # Check for new day
        current_date = ts.date()
        if self._state.current_date != current_date:
            self._reset_daily_state(current_date)

        # === FILTERS ===

        # Session check
        session = self._get_current_session(ts)
        if not session:
            return None

        # Max trades check
        if self._state.trades_today >= self.config.max_trades_per_day:
            return None

        # Consecutive losses check
        if self._state.consecutive_losses >= self.config.max_consecutive_losses:
            logger.warning(f"SMC_V2 [{self.symbol}] Max consecutive losses reached, stopping")
            return None

        # Spread check
        spread_pips = self.current_spread / self.pip_size
        if spread_pips > self.config.max_spread_pips:
            return None

        # === ANALYSIS ===

        # Analyze H4 (optional)
        if len(self.h4_candles) >= 30:
            self._state.h4_bias, _ = self._analyze_market_structure(self.h4_candles, self.atr_h4)

        # Analyze H1
        if len(self.h1_candles) < 50:
            return None

        h1_bias, h1_breaks = self._analyze_market_structure(self.h1_candles, self.atr_h1)
        self._state.h1_bias = h1_bias

        # EMA trend filter
        if self.config.use_ema_filter and self.ema_200 > 0:
            current_price = candle.close
            if current_price > self.ema_200:
                self._state.ema_trend = MarketBias.BULLISH
            elif current_price < self.ema_200:
                self._state.ema_trend = MarketBias.BEARISH
            else:
                self._state.ema_trend = MarketBias.NEUTRAL

        # Determine trading bias - more flexible
        trading_bias = MarketBias.NEUTRAL

        if self.config.use_ema_filter:
            # EMA trend is primary
            if self._state.ema_trend == MarketBias.BULLISH and h1_bias != MarketBias.BEARISH:
                trading_bias = MarketBias.BULLISH
            elif self._state.ema_trend == MarketBias.BEARISH and h1_bias != MarketBias.BULLISH:
                trading_bias = MarketBias.BEARISH
        else:
            # Use H1 bias directly
            if h1_bias != MarketBias.NEUTRAL:
                trading_bias = h1_bias

        # Allow neutral bias for range trading if H4 agrees
        if trading_bias == MarketBias.NEUTRAL and self._state.h4_bias != MarketBias.NEUTRAL:
            if not self.config.require_h4_agreement:
                trading_bias = self._state.h4_bias

        if trading_bias == MarketBias.NEUTRAL:
            return None

        # Find structures
        order_blocks = self._find_order_blocks(self.h1_candles, self.atr_h1)
        fvgs = self._find_fvgs(self.h1_candles, self.atr_h1)
        liquidity_levels = self._find_liquidity_levels(self.h1_candles)

        # Detect liquidity sweeps
        swing_highs, swing_lows = self._find_swing_points(self.h1_candles, lookback=3)
        high_sweeps = self._detect_liquidity_sweep(self.h1_candles, swing_highs, is_high=True)
        low_sweeps = self._detect_liquidity_sweep(self.h1_candles, swing_lows, is_high=False)
        recent_sweeps = high_sweeps + low_sweeps
        self._state.recent_sweeps = recent_sweeps

        # Calculate swing range
        if swing_highs and swing_lows:
            swing_range = (
                max(h.price for h in swing_highs[-3:]),
                min(l.price for l in swing_lows[-3:])
            )
        else:
            swing_range = (candle.high, candle.low)

        # Identify POIs
        current_price = candle.close
        pois = self._identify_pois(
            bias=trading_bias,
            order_blocks=order_blocks,
            fvgs=fvgs,
            liquidity_levels=liquidity_levels,
            current_price=current_price,
            swing_range=swing_range,
            recent_sweeps=recent_sweeps
        )

        self._state.active_pois = pois

        if not pois:
            return None

        # Check if price is in POI zone
        active_poi = None
        for poi in pois:
            # Expand POI zone slightly for entries
            zone_expansion = 2 * self.pip_size
            if poi.price_low - zone_expansion <= current_price <= poi.price_high + zone_expansion:
                active_poi = poi
                break

        if not active_poi:
            return None

        # Momentum filter - check recent price action (relaxed)
        m5_data = self.m5_candles if self.m5_candles else candles
        if len(m5_data) >= 5:
            recent_5 = m5_data[-5:]
            momentum = (recent_5[-1].close - recent_5[0].open) / self.pip_size

            # For longs, we want to see recent bearish or neutral momentum
            # For shorts, we want to see recent bullish or neutral momentum
            if trading_bias == MarketBias.BULLISH and momentum > 20:
                # Price moving up too fast, might be too late
                return None
            if trading_bias == MarketBias.BEARISH and momentum < -20:
                # Price moving down too fast, might be too late
                return None

        self._state.active_poi = active_poi

        # Detect confirmation using M5 or M15 candles
        entry_candles = self.m5_candles if self.m5_candles else candles
        direction = "long" if trading_bias == MarketBias.BULLISH else "short"

        confirmation = self._detect_confirmation(entry_candles, direction, active_poi)

        if not confirmation:
            return None

        # === CALCULATE ENTRY ===

        entry_price = confirmation.entry_price
        is_long = direction == "long"

        # Calculate SL
        if is_long:
            sl = confirmation.sl_price - (2 * self.pip_size)  # Small buffer
        else:
            sl = confirmation.sl_price + (2 * self.pip_size)

        # Validate SL distance
        sl_pips = abs(entry_price - sl) / self.pip_size

        if sl_pips < self.config.min_sl_pips:
            sl_pips = self.config.min_sl_pips
            if is_long:
                sl = entry_price - (sl_pips * self.pip_size)
            else:
                sl = entry_price + (sl_pips * self.pip_size)

        if sl_pips > self.config.max_sl_pips:
            return None

        # Calculate TP
        tp, actual_rr, target_type = self._calculate_tp(
            entry_price, sl, is_long, liquidity_levels
        )

        if actual_rr < self.config.min_rr:
            return None

        # === GENERATE SIGNAL ===

        self._state.trades_today += 1

        tp_pips = abs(tp - entry_price) / self.pip_size

        signal_type = SignalType.LONG if is_long else SignalType.SHORT

        # Calculate confidence based on multiple factors
        confidence = 0.5
        confidence += active_poi.score / 10  # POI score contribution
        confidence += confirmation.strength * 0.2  # Confirmation strength
        if active_poi.has_liquidity_sweep:
            confidence += 0.15
        confidence = min(1.0, confidence)

        logger.info(
            f"SMC_V2 [{self.symbol}] {direction.upper()} | "
            f"POI: {active_poi.score:.1f} | Confirm: {confirmation.type.value} | "
            f"Entry: {entry_price:.5f}, SL: {sl:.5f} ({sl_pips:.1f}p), TP: {tp:.5f} ({tp_pips:.1f}p) | "
            f"R:R: {actual_rr:.2f} | Sweep: {active_poi.has_liquidity_sweep}"
        )

        return StrategySignal(
            signal_type=signal_type,
            symbol=self.symbol,
            price=entry_price,
            stop_loss=sl,
            take_profit=tp,
            confidence=confidence,
            reason=f"SMC_V2 {direction.upper()} - {active_poi.type} ({confirmation.type.value})",
            metadata={
                "session": session,
                "ema_trend": self._state.ema_trend.value,
                "h1_bias": h1_bias.value,
                "poi_score": active_poi.score,
                "confirmation_type": confirmation.type.value,
                "sl_pips": sl_pips,
                "tp_pips": tp_pips,
                "actual_rr": actual_rr,
                "has_sweep": active_poi.has_liquidity_sweep,
                "target_type": target_type,
            }
        )

    def on_tick(self, bid: float, ask: float, timestamp: datetime) -> Optional[StrategySignal]:
        return None

    def should_exit(
        self,
        position: Position,
        current_price: float,
        candles: Optional[list[Candle]] = None
    ) -> Optional[StrategySignal]:
        """Check for exit conditions."""
        if not candles:
            return None

        ts = candles[-1].timestamp
        session = self._get_current_session(ts)

        # Exit before session end
        if session:
            end_hour = self.config.london_end_hour if session == "london" else self.config.ny_end_hour
            minutes_to_end = (end_hour * 60) - (ts.hour * 60 + ts.minute)

            if 0 <= minutes_to_end <= 10:
                return StrategySignal(
                    signal_type=SignalType.EXIT_LONG if position.is_long else SignalType.EXIT_SHORT,
                    symbol=self.symbol,
                    price=current_price,
                    reason="Session end exit"
                )

        return None

    def on_position_closed(self, position: Position, pnl: float) -> None:
        """Track consecutive losses."""
        if pnl < 0:
            self._state.consecutive_losses += 1
        else:
            self._state.consecutive_losses = 0

    def get_status(self) -> dict:
        base_status = super().get_status()
        base_status.update({
            "ema_trend": self._state.ema_trend.value,
            "h4_bias": self._state.h4_bias.value,
            "h1_bias": self._state.h1_bias.value,
            "trades_today": self._state.trades_today,
            "consecutive_losses": self._state.consecutive_losses,
            "active_pois": len(self._state.active_pois),
            "recent_sweeps": len(self._state.recent_sweeps),
            "ema_200": self.ema_200,
        })
        return base_status

    def reset(self) -> None:
        super().reset()
        self._reset_daily_state()
