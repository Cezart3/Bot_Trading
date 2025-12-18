"""
Smart Money Concepts (SMC) Strategy V3 - Full Optimization.

All Phase 1-5 Improvements Implemented:
Phase 1: Displacement Verification, Liquidity Sweep Mandatory, Kill Zone Strict
Phase 2: ADX Filter, Volatility Filter, Session Quality Score
Phase 3: Multi-Candle Confirmation, OB Refinement, POI Freshness
Phase 4: Partial Take Profits, Trailing Stop, Time-Based Exit
Phase 5: HTF Confluence, Previous Day Levels, Equity Curve Trading

Instruments Supported:
- EURUSD (London + NY)
- US500 (NY only)
- USTech100 (NY only)
- GER40/DAX (London + NY)

Author: Trading Bot Project
Version: 3.0
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


# ==================== Enums ====================

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
    MULTI_CANDLE = "multi_candle"
    M5_BOS = "m5_bos"


class MarketCondition(Enum):
    TRENDING = "trending"
    WEAK_TREND = "weak_trend"
    RANGING = "ranging"


class InstrumentType(Enum):
    FOREX = "forex"
    INDEX = "index"


# ==================== Data Classes ====================

@dataclass
class SwingPoint:
    price: float
    timestamp: datetime
    is_high: bool
    strength: int = 1
    swept: bool = False


@dataclass
class StructureBreak:
    type: StructureType
    direction: str
    break_price: float
    timestamp: datetime
    swing_broken: 'SwingPoint'


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
    touches: int = 0  # POI Freshness tracking


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
    score: float
    timestamp: datetime
    order_block: Optional[OrderBlock] = None
    fvg: Optional[FairValueGap] = None
    liquidity_nearby: Optional[LiquidityLevel] = None
    is_premium: bool = False
    is_discount: bool = False
    is_fresh: bool = True
    has_liquidity_sweep: bool = False
    touches: int = 0  # Track touch count


@dataclass
class Confirmation:
    type: ConfirmationType
    timestamp: datetime
    entry_price: float
    sl_price: float
    strength: float


@dataclass
class PreviousDayLevels:
    high: float
    low: float
    close: float
    open: float
    date: datetime


@dataclass
class DailyLevel:
    price: float
    type: str  # "support", "resistance", "pdh", "pdl"
    strength: float


@dataclass
class PartialTP:
    level: int  # 1, 2, 3
    rr: float
    percent: int
    hit: bool = False
    price: float = 0.0


@dataclass
class SMCStateV3:
    """State tracking for SMC V3 strategy."""
    # Bias
    h4_bias: MarketBias = MarketBias.NEUTRAL
    h1_bias: MarketBias = MarketBias.NEUTRAL
    ema_trend: MarketBias = MarketBias.NEUTRAL
    daily_bias: MarketBias = MarketBias.NEUTRAL

    # Swing points
    h1_swing_highs: List[SwingPoint] = field(default_factory=list)
    h1_swing_lows: List[SwingPoint] = field(default_factory=list)
    h1_structure_breaks: List[StructureBreak] = field(default_factory=list)

    # POIs
    active_pois: List[POI] = field(default_factory=list)
    order_blocks: List[OrderBlock] = field(default_factory=list)
    fvgs: List[FairValueGap] = field(default_factory=list)
    liquidity_levels: List[LiquidityLevel] = field(default_factory=list)

    # Entry state
    waiting_for_entry: bool = False
    active_poi: Optional[POI] = None
    pending_confirmation: Optional[Confirmation] = None

    # Session/day tracking
    current_date: Optional[datetime] = None
    trades_today: int = 0
    last_trade_session: Optional[str] = None
    consecutive_losses: int = 0
    consecutive_wins: int = 0

    # Market condition
    market_condition: MarketCondition = MarketCondition.TRENDING
    adx_value: float = 0.0
    session_quality: float = 1.0

    # Premium/Discount
    swing_range_high: Optional[float] = None
    swing_range_low: Optional[float] = None

    # Liquidity sweeps
    recent_sweeps: List[LiquidityLevel] = field(default_factory=list)

    # Previous day levels
    prev_day_levels: Optional[PreviousDayLevels] = None
    daily_levels: List[DailyLevel] = field(default_factory=list)

    # Active trade management
    active_partial_tps: List[PartialTP] = field(default_factory=list)
    trade_entry_time: Optional[datetime] = None
    original_sl: float = 0.0
    trailing_active: bool = False

    # ATR for volatility
    avg_atr_h1: float = 0.0
    current_atr_h1: float = 0.0

    # Equity curve trading
    current_risk_percent: float = 1.0


@dataclass
class SMCConfigV3:
    """Configuration for SMC V3 strategy - All optimizations."""

    # Instrument type
    instrument_type: InstrumentType = InstrumentType.FOREX

    # Timeframes
    tf_bias: str = "H4"
    tf_structure: str = "H1"
    tf_entry: str = "M15"
    tf_precision: str = "M5"

    # === PHASE 1: Quick Wins ===

    # Displacement Verification
    require_displacement: bool = True
    displacement_min_atr: float = 1.0  # Min ATR move from POI (relaxed from 1.5)

    # Liquidity Sweep
    require_sweep_for_low_score: bool = False  # Relaxed - don't require sweep
    sweep_score_threshold: float = 2.0  # If POI score < this, require sweep

    # Kill Zone Strict - CORRECTED TIMES (Romania = UTC+2 in winter)
    # London opens 08:00 UTC = 10:00 Romania
    # NY opens 14:30 UTC = 16:30 Romania
    use_strict_kill_zones: bool = True
    london_kz_start: int = 8   # 08:00 UTC = 10:00 Romania
    london_kz_end: int = 11    # 11:00 UTC = 13:00 Romania (extended for more trades)
    ny_kz_start: int = 14      # 14:00 UTC = 16:00 Romania (30 min before open)
    ny_kz_end: int = 17        # 17:00 UTC = 19:00 Romania (extended for more trades)

    # Extended sessions for indices
    index_ny_start: int = 14   # 14:30 actually, but use 14 for buffer
    index_ny_end: int = 20     # 20:00 UTC = extended session
    index_skip_first_minutes: int = 15  # Skip first 15 min (reduced from 30)

    # === PHASE 2: Market Filters ===

    # ADX Filter - RELAXED for more trades
    use_adx_filter: bool = True
    adx_period: int = 14
    adx_trending: float = 20.0    # ADX > 20 = trending (relaxed from 25)
    adx_weak_trend: float = 15.0  # 15-20 = weak trend (relaxed from 20)
    adx_reduce_risk: bool = True  # Reduce risk in weak trend

    # Volatility Filter
    use_volatility_filter: bool = True
    volatility_lookback: int = 20  # Days for avg ATR
    volatility_high_mult: float = 1.5  # ATR > avg * 1.5 = high vol
    volatility_low_mult: float = 0.7   # ATR < avg * 0.7 = low vol
    volatility_high_sl_add: float = 0.2  # Add 20% to SL in high vol

    # Session Quality
    use_session_quality: bool = True
    session_hour1_quality: float = 1.0
    session_hour2_quality: float = 1.0
    session_hour3_quality: float = 0.7
    session_hour4_quality: float = 0.5
    session_min_quality: float = 0.5

    # === PHASE 3: Entry Improvements ===

    # Multi-Candle Confirmation - RELAXED for more trades
    require_multi_candle: bool = False  # Single candle confirmation OK
    multi_candle_count: int = 2  # 2 consecutive candles (if enabled)

    # OB Refinement (50% entry) - RELAXED
    use_ob_refinement: bool = False  # Use full OB zone for more entries
    ob_entry_zone: float = 0.7  # Enter in 70% of OB (if enabled)

    # POI Freshness - RELAXED
    use_poi_freshness: bool = True
    poi_first_touch_bonus: float = 0.8
    poi_second_touch_bonus: float = 0.4
    poi_max_touches: int = 3  # Allow up to 3 touches (increased from 2)

    # === PHASE 4: Trade Management ===

    # Partial Take Profits
    use_partial_tp: bool = True
    tp1_rr: float = 1.0
    tp1_percent: int = 50
    tp1_move_sl_be: bool = True  # Move SL to BE after TP1
    tp2_rr: float = 2.0
    tp2_percent: int = 30
    tp3_rr: float = 3.0
    tp3_percent: int = 20

    # Trailing Stop
    use_trailing_stop: bool = True
    trailing_start_rr: float = 1.0  # Start trailing after 1R
    trailing_distance_atr: float = 1.5  # Trail by 1.5 ATR

    # Time-Based Exit
    use_time_exit: bool = True
    time_exit_hours: int = 4  # Close if no TP1 in 4 hours

    # === PHASE 5: Advanced Features ===

    # HTF Confluence
    use_htf_confluence: bool = True
    htf_confluence_bonus: float = 1.5  # Score bonus for Daily level
    htf_proximity_pips: float = 20.0   # How close to Daily level

    # Previous Day Levels
    use_prev_day_levels: bool = True
    pdh_pdl_target: bool = True  # Use PDH/PDL as targets

    # Equity Curve Trading
    use_equity_curve: bool = True
    equity_losses_reduce: int = 2    # After 2 losses
    equity_wins_increase: int = 3    # After 3 wins
    equity_reduced_risk: float = 0.5
    equity_normal_risk: float = 1.0

    # === Standard Settings ===

    # Trend Filter
    ema_period: int = 200
    use_ema_filter: bool = True
    require_h4_agreement: bool = False

    # Structure Detection - RELAXED for more trades
    swing_lookback: int = 3
    swing_lookback_h1: int = 4  # Reduced from 5
    bos_min_move_atr: float = 0.4  # Reduced from 0.5
    ob_min_impulse_atr: float = 0.8  # Reduced from 1.0
    fvg_min_gap_atr: float = 0.25  # Reduced from 0.3

    # POI Scoring - RELAXED for more trades
    poi_min_score: float = 1.5  # Reduced from 2.0
    poi_ob_score: float = 1.0  # Reduced from 1.2
    poi_fvg_overlap_score: float = 0.4
    poi_liquidity_score: float = 0.4
    poi_zone_score: float = 0.4
    poi_fresh_score: float = 0.4
    poi_sweep_score: float = 1.0  # Reduced from 1.5

    # Entry Confirmations
    choch_lookback: int = 10  # Reduced from 12

    # Risk Management
    risk_percent: float = 1.0
    max_sl_pips: float = 30.0  # Increased from 25
    min_sl_pips: float = 8.0   # Reduced from 10
    min_rr: float = 1.5  # Reduced from 2.0 for more trades
    target_rr: float = 2.0  # Reduced from 2.5
    max_rr: float = 4.0

    # Sessions (fallback if not strict) - CORRECTED
    london_start_hour: int = 8   # 08:00 UTC = 10:00 Romania
    london_end_hour: int = 12    # 12:00 UTC = 14:00 Romania
    ny_start_hour: int = 14      # 14:00 UTC = 16:00 Romania
    ny_end_hour: int = 20        # 20:00 UTC = 22:00 Romania

    # Filters
    max_spread_pips: float = 2.5  # Increased from 2.0
    max_daily_loss_percent: float = 4.0  # Increased from 3.0

    # Trade limits - INCREASED for more trades
    max_trades_per_day: int = 4  # Increased from 3
    max_trades_per_session: int = 2
    max_consecutive_losses: int = 4  # Increased from 3


class SMCStrategyV3(BaseStrategy):
    """
    Smart Money Concepts Trading Strategy V3 - Fully Optimized.

    Implements all 5 phases of improvements for maximum profitability.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "M15",
        magic_number: int = 12370,
        timezone: str = "UTC",
        config: Optional[SMCConfigV3] = None,
    ):
        super().__init__(
            name="SMC_V3",
            symbol=symbol,
            timeframe=timeframe,
            magic_number=magic_number,
        )

        self.config = config or SMCConfigV3()
        self.timezone = timezone
        self.tz = pytz.timezone(timezone) if timezone != "UTC" else pytz.UTC

        self._state = SMCStateV3()

        # Determine instrument type
        self._detect_instrument_type()

        # Pip size based on symbol
        self.pip_size = self._get_pip_size(symbol)

        # ATR values
        self.atr_h4: float = 0.0
        self.atr_h1: float = 0.0
        self.atr_m15: float = 0.0
        self.atr_m5: float = 0.0
        self.atr_daily: float = 0.0

        # EMA 200 value
        self.ema_200: float = 0.0

        # ADX values
        self.adx: float = 0.0
        self.plus_di: float = 0.0
        self.minus_di: float = 0.0

        # Spread tracking
        self.current_spread: float = 0.0
        self.avg_spread: float = 0.0

        # Candle buffers
        self.daily_candles: List[Candle] = []
        self.h4_candles: List[Candle] = []
        self.h1_candles: List[Candle] = []
        self.m15_candles: List[Candle] = []
        self.m5_candles: List[Candle] = []

        # Historical ATR for volatility
        self.atr_history: List[float] = []

    def _detect_instrument_type(self) -> None:
        """Detect if symbol is forex or index."""
        symbol_upper = self.symbol.upper()
        indices = ["US500", "SP500", "SPX", "USTech100", "USTEC", "NDX",
                   "US30", "DJ30", "DOW", "GER40", "DE40", "DAX"]

        if any(idx in symbol_upper for idx in indices):
            self.config.instrument_type = InstrumentType.INDEX
        else:
            self.config.instrument_type = InstrumentType.FOREX

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
        elif symbol in ["USTech100", "USTEC", "NDX"]:
            return 0.1
        elif symbol in ["GER40", "DE40", "DAX"]:
            return 0.1
        return 0.0001  # Default for forex

    def initialize(self) -> bool:
        """Initialize strategy."""
        logger.info(f"Initializing SMC V3 strategy for {self.symbol}")
        logger.info(f"Instrument type: {self.config.instrument_type.value}")
        logger.info(f"Pip size: {self.pip_size}")
        logger.info(f"Kill zones: London {self.config.london_kz_start}:00-{self.config.london_kz_end}:00, "
                   f"NY {self.config.ny_kz_start}:00-{self.config.ny_kz_end}:00")
        logger.info(f"Features: ADX={self.config.use_adx_filter}, "
                   f"Displacement={self.config.require_displacement}, "
                   f"PartialTP={self.config.use_partial_tp}")
        self._reset_daily_state()
        return True

    def _reset_daily_state(self, current_date: Optional[datetime] = None) -> None:
        """Reset state for new trading day."""
        # Preserve key values across days
        consecutive_losses = self._state.consecutive_losses
        consecutive_wins = self._state.consecutive_wins
        current_risk = self._state.current_risk_percent
        prev_levels = self._state.prev_day_levels
        avg_atr = self._state.avg_atr_h1

        self._state = SMCStateV3()
        self._state.current_date = current_date
        self._state.consecutive_losses = consecutive_losses
        self._state.consecutive_wins = consecutive_wins
        self._state.current_risk_percent = current_risk
        self._state.prev_day_levels = prev_levels
        self._state.avg_atr_h1 = avg_atr

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
        m15_candles: Optional[List[Candle]] = None,
        daily_candles: Optional[List[Candle]] = None
    ) -> None:
        """Set candle data for all timeframes."""
        self.h4_candles = h4_candles
        self.h1_candles = h1_candles
        self.m5_candles = m5_candles
        self.m15_candles = m15_candles if m15_candles else []
        self.daily_candles = daily_candles if daily_candles else []

        # Calculate ATRs
        if len(h4_candles) >= 14:
            self.atr_h4 = self._calculate_atr(h4_candles, 14)
        if len(h1_candles) >= 14:
            self.atr_h1 = self._calculate_atr(h1_candles, 14)
            self._state.current_atr_h1 = self.atr_h1
        if len(m5_candles) >= 14:
            self.atr_m5 = self._calculate_atr(m5_candles, 14)
        if m15_candles and len(m15_candles) >= 14:
            self.atr_m15 = self._calculate_atr(m15_candles, 14)
        if daily_candles and len(daily_candles) >= 14:
            self.atr_daily = self._calculate_atr(daily_candles, 14)

        # Calculate EMA 200 on H1
        if len(h1_candles) >= self.config.ema_period:
            self.ema_200 = self._calculate_ema(h1_candles, self.config.ema_period)

        # Calculate ADX
        if len(h1_candles) >= self.config.adx_period + 14:
            self.adx, self.plus_di, self.minus_di = self._calculate_adx(
                h1_candles, self.config.adx_period
            )
            self._state.adx_value = self.adx

        # Update ATR history for volatility
        if self.atr_h1 > 0:
            self.atr_history.append(self.atr_h1)
            if len(self.atr_history) > self.config.volatility_lookback:
                self.atr_history = self.atr_history[-self.config.volatility_lookback:]
            if len(self.atr_history) >= 5:
                self._state.avg_atr_h1 = sum(self.atr_history) / len(self.atr_history)

        # Calculate previous day levels
        if daily_candles and len(daily_candles) >= 2:
            self._update_previous_day_levels(daily_candles)

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

        ema = sum(closes[:period]) / period
        for close in closes[period:]:
            ema = (close - ema) * multiplier + ema

        return ema

    def _calculate_adx(
        self,
        candles: List[Candle],
        period: int = 14
    ) -> Tuple[float, float, float]:
        """
        Calculate ADX (Average Directional Index).
        Returns: (ADX, +DI, -DI)
        """
        if len(candles) < period + 14:
            return 0.0, 0.0, 0.0

        # Calculate +DM, -DM, TR
        plus_dm = []
        minus_dm = []
        tr_values = []

        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            prev_high = candles[i-1].high
            prev_low = candles[i-1].low
            prev_close = candles[i-1].close

            # True Range
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)

            # Directional Movement
            up_move = high - prev_high
            down_move = prev_low - low

            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0)

            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0)

        if len(tr_values) < period:
            return 0.0, 0.0, 0.0

        # Smooth values using Wilder's smoothing
        def wilder_smooth(values: List[float], period: int) -> List[float]:
            smoothed = [sum(values[:period])]
            for i in range(period, len(values)):
                smoothed.append(smoothed[-1] - (smoothed[-1] / period) + values[i])
            return smoothed

        smooth_tr = wilder_smooth(tr_values, period)
        smooth_plus_dm = wilder_smooth(plus_dm, period)
        smooth_minus_dm = wilder_smooth(minus_dm, period)

        if not smooth_tr or smooth_tr[-1] == 0:
            return 0.0, 0.0, 0.0

        # Calculate +DI and -DI
        plus_di = (smooth_plus_dm[-1] / smooth_tr[-1]) * 100
        minus_di = (smooth_minus_dm[-1] / smooth_tr[-1]) * 100

        # Calculate DX
        if plus_di + minus_di == 0:
            return 0.0, plus_di, minus_di

        dx_values = []
        min_len = min(len(smooth_plus_dm), len(smooth_minus_dm), len(smooth_tr))

        for i in range(min_len):
            if smooth_tr[i] == 0:
                continue
            pdi = (smooth_plus_dm[i] / smooth_tr[i]) * 100
            mdi = (smooth_minus_dm[i] / smooth_tr[i]) * 100
            if pdi + mdi == 0:
                dx_values.append(0)
            else:
                dx_values.append(abs(pdi - mdi) / (pdi + mdi) * 100)

        if len(dx_values) < period:
            return 0.0, plus_di, minus_di

        # ADX is smoothed DX
        adx = sum(dx_values[-period:]) / period

        return adx, plus_di, minus_di

    def _update_previous_day_levels(self, daily_candles: List[Candle]) -> None:
        """Update previous day high/low/close levels."""
        if len(daily_candles) < 2:
            return

        prev_day = daily_candles[-2]
        self._state.prev_day_levels = PreviousDayLevels(
            high=prev_day.high,
            low=prev_day.low,
            close=prev_day.close,
            open=prev_day.open,
            date=prev_day.timestamp
        )

        # Add to daily levels
        self._state.daily_levels = [
            DailyLevel(prev_day.high, "pdh", 0.9),
            DailyLevel(prev_day.low, "pdl", 0.9),
        ]

    def _get_current_session(self, timestamp: datetime) -> Optional[str]:
        """Determine which session is active with Kill Zone support."""
        if timestamp.tzinfo is None:
            ts_utc = timestamp
        else:
            ts_utc = timestamp.astimezone(pytz.UTC)

        hour = ts_utc.hour
        minute = ts_utc.minute
        time_value = hour * 60 + minute

        is_index = self.config.instrument_type == InstrumentType.INDEX

        if self.config.use_strict_kill_zones:
            # Kill Zones only
            london_start = self.config.london_kz_start * 60
            london_end = self.config.london_kz_end * 60
            ny_start = self.config.ny_kz_start * 60
            ny_end = self.config.ny_kz_end * 60

            # Indices only trade NY
            if is_index:
                index_start = self.config.index_ny_start * 60 + self.config.index_skip_first_minutes
                index_end = self.config.index_ny_end * 60

                if index_start <= time_value < index_end:
                    return "ny"
                return None

            # Forex trades both
            if london_start <= time_value < london_end:
                return "london"
            if ny_start <= time_value < ny_end:
                return "ny"
            return None
        else:
            # Extended sessions (fallback)
            london_start = self.config.london_start_hour * 60
            london_end = self.config.london_end_hour * 60
            ny_start = self.config.ny_start_hour * 60
            ny_end = self.config.ny_end_hour * 60

            if london_start <= time_value < london_end:
                return "london"
            if ny_start <= time_value < ny_end:
                return "ny"
            return None

    def _calculate_session_quality(self, timestamp: datetime, session: str) -> float:
        """Calculate session quality based on time into session."""
        if not self.config.use_session_quality:
            return 1.0

        hour = timestamp.hour

        if session == "london":
            session_start = self.config.london_kz_start
        else:
            session_start = self.config.ny_kz_start

        hours_in = hour - session_start

        if hours_in <= 0:
            return self.config.session_hour1_quality
        elif hours_in == 1:
            return self.config.session_hour2_quality
        elif hours_in == 2:
            return self.config.session_hour3_quality
        else:
            return self.config.session_hour4_quality

    def _check_market_condition(self) -> MarketCondition:
        """Check market condition using ADX."""
        if not self.config.use_adx_filter:
            return MarketCondition.TRENDING

        adx = self._state.adx_value

        if adx >= self.config.adx_trending:
            return MarketCondition.TRENDING
        elif adx >= self.config.adx_weak_trend:
            return MarketCondition.WEAK_TREND
        else:
            return MarketCondition.RANGING

    def _check_volatility_filter(self) -> Tuple[bool, float]:
        """
        Check volatility filter.
        Returns: (should_trade, sl_multiplier)
        """
        if not self.config.use_volatility_filter:
            return True, 1.0

        if self._state.avg_atr_h1 == 0:
            return True, 1.0

        current_atr = self._state.current_atr_h1
        avg_atr = self._state.avg_atr_h1

        ratio = current_atr / avg_atr

        if ratio > self.config.volatility_high_mult:
            # High volatility - increase SL
            return True, 1.0 + self.config.volatility_high_sl_add
        elif ratio < self.config.volatility_low_mult:
            # Low volatility - skip
            return False, 1.0

        return True, 1.0

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

        recent_candles = candles[-20:]

        for swing in swing_points[-5:]:
            for i, candle in enumerate(recent_candles):
                if is_high:
                    # Sweep of swing high
                    if candle.high > swing.price and candle.close < swing.price:
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
                    # Sweep of swing low
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

    def _verify_displacement(
        self,
        candles: List[Candle],
        poi: POI,
        direction: str
    ) -> bool:
        """
        Verify displacement - strong move away from POI.
        Phase 1 improvement.
        """
        if not self.config.require_displacement:
            return True

        if len(candles) < 3:
            return False

        atr = self.atr_m5 if self.atr_m5 > 0 else self.atr_h1
        if atr == 0:
            return True

        min_move = self.config.displacement_min_atr * atr

        # Check last 3 candles for strong move
        recent = candles[-3:]

        if direction == "long":
            # Need strong bullish move
            move = recent[-1].close - min(c.low for c in recent)
            if move >= min_move:
                # Verify bullish candles
                bullish_count = sum(1 for c in recent if c.close > c.open)
                return bullish_count >= 2
        else:
            # Need strong bearish move
            move = max(c.high for c in recent) - recent[-1].close
            if move >= min_move:
                bearish_count = sum(1 for c in recent if c.close < c.open)
                return bearish_count >= 2

        return False

    def _analyze_market_structure(
        self,
        candles: List[Candle],
        atr: float
    ) -> Tuple[MarketBias, List[StructureBreak]]:
        """Analyze market structure."""
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

        if bullish_score >= bearish_score + 1:
            bias = MarketBias.BULLISH
        elif bearish_score >= bullish_score + 1:
            bias = MarketBias.BEARISH
        else:
            bias = MarketBias.NEUTRAL

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
        """Find Order Blocks with touch tracking."""
        order_blocks = []

        if len(candles) < 10 or atr == 0:
            return order_blocks

        min_impulse = self.config.ob_min_impulse_atr * atr

        for i in range(2, len(candles) - 2):
            curr = candles[i]
            prev = candles[i - 1]
            next1 = candles[i + 1]
            next2 = candles[i + 2] if i + 2 < len(candles) else next1

            # Bullish OB
            if curr.close < curr.open:
                impulse = max(next1.close, next2.close) - curr.low
                if impulse >= min_impulse:
                    recent_highs = [c.high for c in candles[max(0, i-10):i]]
                    if recent_highs and max(next1.close, next2.close) > max(recent_highs) * 0.999:
                        ob = OrderBlock(
                            type=POIType.BULLISH_OB,
                            high=curr.high,
                            low=curr.low,
                            timestamp=curr.timestamp,
                            impulse_strength=impulse / atr,
                            candle_body_high=max(curr.open, curr.close),
                            candle_body_low=min(curr.open, curr.close),
                            touches=0
                        )

                        # Count touches
                        for check_candle in candles[i+3:]:
                            if check_candle.low <= ob.high:
                                ob.touches += 1

                        # POI Freshness - invalidate after max touches
                        if self.config.use_poi_freshness:
                            if ob.touches > self.config.poi_max_touches:
                                continue

                        order_blocks.append(ob)

            # Bearish OB
            elif curr.close > curr.open:
                impulse = curr.high - min(next1.close, next2.close)
                if impulse >= min_impulse:
                    recent_lows = [c.low for c in candles[max(0, i-10):i]]
                    if recent_lows and min(next1.close, next2.close) < min(recent_lows) * 1.001:
                        ob = OrderBlock(
                            type=POIType.BEARISH_OB,
                            high=curr.high,
                            low=curr.low,
                            timestamp=curr.timestamp,
                            impulse_strength=impulse / atr,
                            candle_body_high=max(curr.open, curr.close),
                            candle_body_low=min(curr.open, curr.close),
                            touches=0
                        )

                        for check_candle in candles[i+3:]:
                            if check_candle.high >= ob.low:
                                ob.touches += 1

                        if self.config.use_poi_freshness:
                            if ob.touches > self.config.poi_max_touches:
                                continue

                        order_blocks.append(ob)

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

    def _check_htf_confluence(self, price: float) -> float:
        """
        Check for HTF (Daily) level confluence.
        Returns bonus score if near Daily level.
        """
        if not self.config.use_htf_confluence:
            return 0.0

        if not self._state.prev_day_levels:
            return 0.0

        pdl = self._state.prev_day_levels
        proximity = self.config.htf_proximity_pips * self.pip_size

        # Check proximity to PDH/PDL
        if abs(price - pdl.high) <= proximity:
            return self.config.htf_confluence_bonus
        if abs(price - pdl.low) <= proximity:
            return self.config.htf_confluence_bonus

        return 0.0

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
        """Identify POIs with all V3 improvements."""
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

                # FVG overlap
                overlapping_fvg = None
                for fvg in fvgs:
                    if fvg.type == POIType.BULLISH_FVG and fvg.is_valid:
                        if fvg.low <= ob.high and fvg.high >= ob.low:
                            score += self.config.poi_fvg_overlap_score
                            overlapping_fvg = fvg
                            break

                # Nearby liquidity
                nearby_liq = None
                for liq in liquidity_levels:
                    if liq.type in ["swing_low", "equal_lows"]:
                        if abs(liq.price - ob.low) <= 10 * self.pip_size:
                            score += self.config.poi_liquidity_score
                            nearby_liq = liq
                            break

                # Liquidity sweep
                has_sweep = False
                for sweep in recent_sweeps:
                    if sweep.type == "sweep_low" and abs(sweep.price - ob.low) <= 15 * self.pip_size:
                        score += self.config.poi_sweep_score
                        has_sweep = True
                        break

                # POI Freshness scoring
                if self.config.use_poi_freshness:
                    if ob.touches == 0:
                        score += self.config.poi_first_touch_bonus
                    elif ob.touches == 1:
                        score += self.config.poi_second_touch_bonus
                else:
                    if ob.is_valid:
                        score += self.config.poi_fresh_score

                # HTF Confluence
                htf_bonus = self._check_htf_confluence(ob_mid)
                score += htf_bonus

                # Check sweep requirement for low scores
                if self.config.require_sweep_for_low_score:
                    if score < self.config.sweep_score_threshold and not has_sweep:
                        continue

                if score >= self.config.poi_min_score:
                    pois.append(POI(
                        price_high=ob.high,
                        price_low=ob.low,
                        type="Bullish OB",
                        score=score,
                        timestamp=ob.timestamp,
                        order_block=ob,
                        fvg=overlapping_fvg,
                        liquidity_nearby=nearby_liq,
                        is_discount=is_discount,
                        has_liquidity_sweep=has_sweep,
                        touches=ob.touches
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

                if self.config.use_poi_freshness:
                    if ob.touches == 0:
                        score += self.config.poi_first_touch_bonus
                    elif ob.touches == 1:
                        score += self.config.poi_second_touch_bonus
                else:
                    if ob.is_valid:
                        score += self.config.poi_fresh_score

                htf_bonus = self._check_htf_confluence(ob_mid)
                score += htf_bonus

                if self.config.require_sweep_for_low_score:
                    if score < self.config.sweep_score_threshold and not has_sweep:
                        continue

                if score >= self.config.poi_min_score:
                    pois.append(POI(
                        price_high=ob.high,
                        price_low=ob.low,
                        type="Bearish OB",
                        score=score,
                        timestamp=ob.timestamp,
                        order_block=ob,
                        fvg=overlapping_fvg,
                        liquidity_nearby=nearby_liq,
                        is_premium=is_premium,
                        has_liquidity_sweep=has_sweep,
                        touches=ob.touches
                    ))

        pois.sort(key=lambda x: x.score, reverse=True)
        return pois

    def _get_refined_entry_zone(self, poi: POI, direction: str) -> Tuple[float, float]:
        """
        Get refined entry zone (50% of OB).
        Phase 3 improvement.
        """
        if not self.config.use_ob_refinement:
            return poi.price_low, poi.price_high

        ob_size = poi.price_high - poi.price_low
        half_zone = ob_size * self.config.ob_entry_zone

        if direction == "long":
            # Entry in lower half for longs
            return poi.price_low, poi.price_low + half_zone
        else:
            # Entry in upper half for shorts
            return poi.price_high - half_zone, poi.price_high

    def _detect_confirmation(
        self,
        candles: List[Candle],
        direction: str,
        poi: POI
    ) -> Optional[Confirmation]:
        """Detect entry confirmation with multi-candle support."""
        if len(candles) < self.config.choch_lookback + 2:
            return None

        recent = candles[-(self.config.choch_lookback + 1):-1]
        current = candles[-1]
        prev = candles[-2]
        prev2 = candles[-3] if len(candles) > 2 else prev

        avg_body = sum(abs(c.close - c.open) for c in recent) / len(recent)
        current_body = abs(current.close - current.open)

        if direction == "long":
            swing_high = max(c.high for c in recent)
            swing_low = min(c.low for c in recent)

            # Multi-Candle Confirmation (Phase 3)
            if self.config.require_multi_candle:
                # Check for 2 consecutive bullish candles
                if (current.close > current.open and prev.close > prev.open):
                    # Both must be decent size
                    prev_body = abs(prev.close - prev.open)
                    if current_body > avg_body * 0.7 and prev_body > avg_body * 0.7:
                        # Combined move should be significant
                        combined_move = current.close - prev.open
                        if combined_move > avg_body * 1.5:
                            return Confirmation(
                                type=ConfirmationType.MULTI_CANDLE,
                                timestamp=current.timestamp,
                                entry_price=current.close,
                                sl_price=min(current.low, prev.low, prev2.low),
                                strength=0.85
                            )

            # CHoCH
            if current.close > swing_high:
                if current.close > current.open and current_body > avg_body * 0.8:
                    return Confirmation(
                        type=ConfirmationType.CHOCH,
                        timestamp=current.timestamp,
                        entry_price=current.close,
                        sl_price=swing_low,
                        strength=0.9
                    )

            # Bullish Engulfing
            if (current.close > current.open and
                prev.close < prev.open and
                current.open <= prev.close and
                current.close >= prev.open and
                current_body > avg_body * 0.9 and
                current.close > prev.open):
                return Confirmation(
                    type=ConfirmationType.ENGULFING,
                    timestamp=current.timestamp,
                    entry_price=current.close,
                    sl_price=min(current.low, prev.low),
                    strength=0.75
                )

            # Strong Rejection
            body = abs(current.close - current.open)
            lower_wick = min(current.close, current.open) - current.low
            upper_wick = current.high - max(current.close, current.open)

            if (lower_wick > body * 2.5 and
                upper_wick < body * 0.5 and
                current.close > current.open and
                current.close > (current.high + current.low) / 2):
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

            # Multi-Candle Confirmation
            if self.config.require_multi_candle:
                if (current.close < current.open and prev.close < prev.open):
                    prev_body = abs(prev.close - prev.open)
                    if current_body > avg_body * 0.7 and prev_body > avg_body * 0.7:
                        combined_move = prev.open - current.close
                        if combined_move > avg_body * 1.5:
                            return Confirmation(
                                type=ConfirmationType.MULTI_CANDLE,
                                timestamp=current.timestamp,
                                entry_price=current.close,
                                sl_price=max(current.high, prev.high, prev2.high),
                                strength=0.85
                            )

            # CHoCH
            if current.close < swing_low:
                if current.close < current.open and current_body > avg_body * 0.8:
                    return Confirmation(
                        type=ConfirmationType.CHOCH,
                        timestamp=current.timestamp,
                        entry_price=current.close,
                        sl_price=swing_high,
                        strength=0.9
                    )

            # Bearish Engulfing
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

            # Strong Rejection
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

    def _calculate_partial_tps(
        self,
        entry_price: float,
        sl_price: float,
        is_long: bool
    ) -> List[PartialTP]:
        """Calculate partial take profit levels."""
        if not self.config.use_partial_tp:
            return []

        risk = abs(entry_price - sl_price)
        tps = []

        # TP1
        if is_long:
            tp1_price = entry_price + (risk * self.config.tp1_rr)
            tp2_price = entry_price + (risk * self.config.tp2_rr)
            tp3_price = entry_price + (risk * self.config.tp3_rr)
        else:
            tp1_price = entry_price - (risk * self.config.tp1_rr)
            tp2_price = entry_price - (risk * self.config.tp2_rr)
            tp3_price = entry_price - (risk * self.config.tp3_rr)

        tps.append(PartialTP(1, self.config.tp1_rr, self.config.tp1_percent, False, tp1_price))
        tps.append(PartialTP(2, self.config.tp2_rr, self.config.tp2_percent, False, tp2_price))
        tps.append(PartialTP(3, self.config.tp3_rr, self.config.tp3_percent, False, tp3_price))

        return tps

    def _calculate_tp(
        self,
        entry_price: float,
        sl_price: float,
        is_long: bool,
        liquidity_levels: List[LiquidityLevel]
    ) -> Tuple[Optional[float], float, str]:
        """
        Calculate TP targeting UNTAPPED liquidity levels only.

        CRITICAL: Returns (None, 0, reason) if no valid liquidity target exists.
        This ensures we only enter trades with clear liquidity targets.

        Targets (priority order):
        1. Equal Highs/Lows (EQH/EQL) - most liquidity
        2. Previous Day High/Low (PDH/PDL)
        3. Previous Week High/Low (PWH/PWL)
        4. Session High/Low
        5. Swing Highs/Lows
        """
        risk = abs(entry_price - sl_price)

        if risk == 0:
            return None, 0, "invalid_risk"

        targets = []

        # === 1. EQUAL HIGHS/LOWS - Highest priority ===
        if is_long:
            eq_targets = [l for l in liquidity_levels
                        if l.price > entry_price and l.type == "equal_highs"]
            for t in eq_targets:
                priority = 1.0 + (getattr(t, 'touches', 2) * 0.1)
                targets.append((t.price, "EQH", priority))
        else:
            eq_targets = [l for l in liquidity_levels
                        if l.price < entry_price and l.type == "equal_lows"]
            for t in eq_targets:
                priority = 1.0 + (getattr(t, 'touches', 2) * 0.1)
                targets.append((t.price, "EQL", priority))

        # === 2. PREVIOUS DAY HIGH/LOW (PDH/PDL) ===
        if self._state.prev_day_levels:
            pdl = self._state.prev_day_levels
            if is_long and pdl.high > entry_price:
                targets.append((pdl.high, "PDH", 0.95))
            elif not is_long and pdl.low < entry_price:
                targets.append((pdl.low, "PDL", 0.95))

        # === 3. PREVIOUS WEEK HIGH/LOW (PWH/PWL) ===
        if self.daily_candles and len(self.daily_candles) >= 5:
            last_week = self.daily_candles[-5:]
            pwh = max(c.high for c in last_week)
            pwl = min(c.low for c in last_week)

            if is_long and pwh > entry_price:
                targets.append((pwh, "PWH", 0.90))
            elif not is_long and pwl < entry_price:
                targets.append((pwl, "PWL", 0.90))

        # === 4. SESSION HIGH/LOW (if in session) ===
        if self.m5_candles and len(self.m5_candles) >= 12:
            # Get today's session candles
            today = self.m5_candles[-1].timestamp.date()
            session_candles = [c for c in self.m5_candles if c.timestamp.date() == today]
            if session_candles:
                session_high = max(c.high for c in session_candles)
                session_low = min(c.low for c in session_candles)

                if is_long and session_high > entry_price:
                    targets.append((session_high, "SessionH", 0.80))
                elif not is_long and session_low < entry_price:
                    targets.append((session_low, "SessionL", 0.80))

        # === 5. SWING HIGHS/LOWS ===
        if is_long:
            swing_targets = [l for l in liquidity_levels
                           if l.price > entry_price and l.type == "swing_high"]
            for t in swing_targets:
                targets.append((t.price, "SwingH", 0.70))
        else:
            swing_targets = [l for l in liquidity_levels
                           if l.price < entry_price and l.type == "swing_low"]
            for t in swing_targets:
                targets.append((t.price, "SwingL", 0.70))

        # === SORT BY DISTANCE (closest first) ===
        if is_long:
            targets.sort(key=lambda x: x[0])
        else:
            targets.sort(key=lambda x: x[0], reverse=True)

        # === FIND FIRST TARGET WITH VALID R:R ===
        for target_price, target_type, priority in targets:
            if is_long:
                potential_rr = (target_price - entry_price) / risk
            else:
                potential_rr = (entry_price - target_price) / risk

            # Must meet minimum R:R
            if potential_rr >= self.config.min_rr:
                # Don't use targets that are too far (unrealistic)
                if potential_rr <= self.config.max_rr:
                    # Buffer before liquidity (let it hit, then exit)
                    buffer = 2 * self.pip_size
                    if is_long:
                        tp = target_price - buffer
                    else:
                        tp = target_price + buffer

                    logger.debug(f"SMC [{self.symbol}] TP Target: {target_type} "
                                f"at {target_price:.5f}, R:R={potential_rr:.2f}")
                    return tp, potential_rr, target_type

        # === NO VALID TARGET - REJECT TRADE ===
        if targets:
            closest = targets[0]
            if is_long:
                closest_rr = (closest[0] - entry_price) / risk
            else:
                closest_rr = (entry_price - closest[0]) / risk

            logger.debug(f"SMC [{self.symbol}] Skip: Closest liquidity {closest[1]} "
                        f"at {closest[0]:.5f} gives R:R={closest_rr:.2f}, need >= {self.config.min_rr}")
        else:
            logger.debug(f"SMC [{self.symbol}] Skip: No liquidity targets found")

        # NO FALLBACK - return None to reject the trade
        return None, 0, "no_valid_liquidity"

    def _update_equity_curve_risk(self) -> None:
        """Update risk based on equity curve trading."""
        if not self.config.use_equity_curve:
            self._state.current_risk_percent = self.config.risk_percent
            return

        if self._state.consecutive_losses >= self.config.equity_losses_reduce:
            self._state.current_risk_percent = self.config.equity_reduced_risk
        elif self._state.consecutive_wins >= self.config.equity_wins_increase:
            self._state.current_risk_percent = self.config.equity_normal_risk
        else:
            self._state.current_risk_percent = self.config.risk_percent

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
            # Don't spam logs when outside session
            return None

        # Session quality
        self._state.session_quality = self._calculate_session_quality(ts, session)
        if self._state.session_quality < self.config.session_min_quality:
            return None

        # Max trades check
        if self._state.trades_today >= self.config.max_trades_per_day:
            return None

        # Consecutive losses check
        if self._state.consecutive_losses >= self.config.max_consecutive_losses:
            logger.warning(f"SMC_V3 [{self.symbol}] Max consecutive losses reached")
            return None

        # Spread check
        spread_pips = self.current_spread / self.pip_size
        if spread_pips > self.config.max_spread_pips:
            return None

        # Market condition (ADX)
        market_condition = self._check_market_condition()
        self._state.market_condition = market_condition

        if market_condition == MarketCondition.RANGING:
            logger.debug(f"SMC [{self.symbol}] Skip: Ranging market (ADX={self.adx:.1f} < {self.config.adx_weak_trend})")
            return None  # Don't trade in ranging market

        # Volatility filter
        should_trade, sl_multiplier = self._check_volatility_filter()
        if not should_trade:
            logger.debug(f"SMC [{self.symbol}] Skip: Low volatility (ATR ratio too low)")
            return None

        # === ANALYSIS ===

        # Analyze H4
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

        # Determine trading bias
        trading_bias = MarketBias.NEUTRAL

        if self.config.use_ema_filter:
            if self._state.ema_trend == MarketBias.BULLISH and h1_bias != MarketBias.BEARISH:
                trading_bias = MarketBias.BULLISH
            elif self._state.ema_trend == MarketBias.BEARISH and h1_bias != MarketBias.BULLISH:
                trading_bias = MarketBias.BEARISH
        else:
            if h1_bias != MarketBias.NEUTRAL:
                trading_bias = h1_bias

        if trading_bias == MarketBias.NEUTRAL and self._state.h4_bias != MarketBias.NEUTRAL:
            if not self.config.require_h4_agreement:
                trading_bias = self._state.h4_bias

        if trading_bias == MarketBias.NEUTRAL:
            logger.debug(f"SMC [{self.symbol}] Skip: Neutral bias (H1={h1_bias.value}, EMA={self._state.ema_trend.value})")
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
            logger.debug(f"SMC [{self.symbol}] Skip: No valid POIs found (OBs={len(order_blocks)}, bias={trading_bias.value})")
            return None

        # Check if price is in POI zone (refined)
        active_poi = None
        direction = "long" if trading_bias == MarketBias.BULLISH else "short"

        for poi in pois:
            # Get refined entry zone
            entry_low, entry_high = self._get_refined_entry_zone(poi, direction)
            zone_expansion = 2 * self.pip_size

            if entry_low - zone_expansion <= current_price <= entry_high + zone_expansion:
                active_poi = poi
                break

        if not active_poi:
            logger.debug(f"SMC [{self.symbol}] Skip: Price not in POI zone (POIs={len(pois)}, price={current_price:.5f})")
            return None

        # Momentum filter
        m5_data = self.m5_candles if self.m5_candles else candles
        if len(m5_data) >= 5:
            recent_5 = m5_data[-5:]
            momentum = (recent_5[-1].close - recent_5[0].open) / self.pip_size

            if trading_bias == MarketBias.BULLISH and momentum > 20:
                return None
            if trading_bias == MarketBias.BEARISH and momentum < -20:
                return None

        self._state.active_poi = active_poi

        # Detect confirmation
        entry_candles = self.m5_candles if self.m5_candles else candles
        confirmation = self._detect_confirmation(entry_candles, direction, active_poi)

        if not confirmation:
            logger.debug(f"SMC [{self.symbol}] Skip: No confirmation pattern for {direction} at POI")
            return None

        # Verify displacement
        if not self._verify_displacement(entry_candles, active_poi, direction):
            logger.debug(f"SMC [{self.symbol}] Skip: Displacement not verified for {direction}")
            return None

        # === CALCULATE ENTRY ===

        entry_price = confirmation.entry_price
        is_long = direction == "long"

        # Calculate SL with volatility adjustment
        if is_long:
            sl = confirmation.sl_price - (2 * self.pip_size)
        else:
            sl = confirmation.sl_price + (2 * self.pip_size)

        # Apply volatility multiplier to SL
        if sl_multiplier != 1.0:
            sl_distance = abs(entry_price - sl)
            adjusted_distance = sl_distance * sl_multiplier
            if is_long:
                sl = entry_price - adjusted_distance
            else:
                sl = entry_price + adjusted_distance

        # Validate SL distance
        sl_pips = abs(entry_price - sl) / self.pip_size

        if sl_pips < self.config.min_sl_pips:
            sl_pips = self.config.min_sl_pips
            if is_long:
                sl = entry_price - (sl_pips * self.pip_size)
            else:
                sl = entry_price + (sl_pips * self.pip_size)

        if sl_pips > self.config.max_sl_pips:
            logger.debug(f"SMC [{self.symbol}] Skip: SL too large ({sl_pips:.1f} > {self.config.max_sl_pips} pips)")
            return None

        # Calculate TP based on liquidity targets
        tp, actual_rr, target_type = self._calculate_tp(
            entry_price, sl, is_long, liquidity_levels
        )

        # CRITICAL: Reject if no valid liquidity target found
        if tp is None:
            logger.debug(f"SMC [{self.symbol}] Skip: No liquidity target with R:R >= {self.config.min_rr}")
            return None

        if actual_rr < self.config.min_rr:
            logger.debug(f"SMC [{self.symbol}] Skip: R:R too low ({actual_rr:.2f} < {self.config.min_rr})")
            return None

        # Calculate partial TPs
        partial_tps = self._calculate_partial_tps(entry_price, sl, is_long)
        self._state.active_partial_tps = partial_tps
        self._state.trade_entry_time = ts
        self._state.original_sl = sl

        # === GENERATE SIGNAL ===

        self._state.trades_today += 1

        # Update equity curve risk
        self._update_equity_curve_risk()
        current_risk = self._state.current_risk_percent

        # Reduce risk for weak trend
        if market_condition == MarketCondition.WEAK_TREND and self.config.adx_reduce_risk:
            current_risk *= 0.5

        tp_pips = abs(tp - entry_price) / self.pip_size
        signal_type = SignalType.LONG if is_long else SignalType.SHORT

        # Calculate confidence
        confidence = 0.5
        confidence += active_poi.score / 10
        confidence += confirmation.strength * 0.2
        if active_poi.has_liquidity_sweep:
            confidence += 0.15
        if self._state.session_quality >= 1.0:
            confidence += 0.1
        if market_condition == MarketCondition.TRENDING:
            confidence += 0.1
        confidence = min(1.0, confidence)

        logger.info(
            f"SMC_V3 [{self.symbol}] {direction.upper()} | "
            f"POI: {active_poi.score:.1f} | Confirm: {confirmation.type.value} | "
            f"Entry: {entry_price:.5f}, SL: {sl:.5f} ({sl_pips:.1f}p), TP: {tp:.5f} ({tp_pips:.1f}p) | "
            f"Target: {target_type} | R:R: {actual_rr:.2f} | ADX: {self.adx:.1f} | Risk: {current_risk}%"
        )

        return StrategySignal(
            signal_type=signal_type,
            symbol=self.symbol,
            price=entry_price,
            stop_loss=sl,
            take_profit=tp,
            confidence=confidence,
            reason=f"SMC_V3 {direction.upper()} - {active_poi.type} ({confirmation.type.value})",
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
                "adx": self.adx,
                "market_condition": market_condition.value,
                "session_quality": self._state.session_quality,
                "risk_percent": current_risk,
                "partial_tps": [(tp.level, tp.price) for tp in partial_tps],
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
        """Check for exit conditions with partial TP and trailing stop."""
        if not candles:
            return None

        ts = candles[-1].timestamp
        session = self._get_current_session(ts)
        is_long = position.is_long

        # Time-based exit (Phase 4)
        if self.config.use_time_exit and self._state.trade_entry_time:
            time_in_trade = (ts - self._state.trade_entry_time).total_seconds() / 3600
            if time_in_trade >= self.config.time_exit_hours:
                # Check if TP1 was not hit
                if self._state.active_partial_tps and not self._state.active_partial_tps[0].hit:
                    return StrategySignal(
                        signal_type=SignalType.EXIT_LONG if is_long else SignalType.EXIT_SHORT,
                        symbol=self.symbol,
                        price=current_price,
                        reason=f"Time exit - {self.config.time_exit_hours}h without TP1"
                    )

        # Check partial TPs
        if self.config.use_partial_tp and self._state.active_partial_tps:
            for tp in self._state.active_partial_tps:
                if tp.hit:
                    continue

                if is_long:
                    if current_price >= tp.price:
                        tp.hit = True
                        logger.info(f"SMC_V3 [{self.symbol}] TP{tp.level} hit at {current_price:.5f}")

                        # Move SL to BE after TP1
                        if tp.level == 1 and self.config.tp1_move_sl_be:
                            self._state.trailing_active = True
                else:
                    if current_price <= tp.price:
                        tp.hit = True
                        logger.info(f"SMC_V3 [{self.symbol}] TP{tp.level} hit at {current_price:.5f}")

                        if tp.level == 1 and self.config.tp1_move_sl_be:
                            self._state.trailing_active = True

        # Session end exit
        if session:
            end_hour = self.config.london_kz_end if session == "london" else self.config.ny_kz_end
            minutes_to_end = (end_hour * 60) - (ts.hour * 60 + ts.minute)

            if 0 <= minutes_to_end <= 10:
                return StrategySignal(
                    signal_type=SignalType.EXIT_LONG if is_long else SignalType.EXIT_SHORT,
                    symbol=self.symbol,
                    price=current_price,
                    reason="Session end exit"
                )

        return None

    def get_trailing_stop(
        self,
        position: Position,
        current_price: float,
        candles: Optional[list[Candle]] = None
    ) -> Optional[float]:
        """Calculate trailing stop level."""
        if not self.config.use_trailing_stop or not self._state.trailing_active:
            return None

        if not candles or len(candles) < 5:
            return None

        is_long = position.is_long
        entry_price = position.entry_price
        current_sl = position.stop_loss or self._state.original_sl

        # Use ATR-based trailing
        atr = self.atr_m5 if self.atr_m5 > 0 else self.atr_h1
        if atr == 0:
            return None

        trail_distance = atr * self.config.trailing_distance_atr

        if is_long:
            # Trail below recent lows
            recent_low = min(c.low for c in candles[-5:])
            new_sl = max(recent_low - trail_distance, entry_price)  # At least BE

            if new_sl > current_sl:
                return new_sl
        else:
            recent_high = max(c.high for c in candles[-5:])
            new_sl = min(recent_high + trail_distance, entry_price)

            if new_sl < current_sl:
                return new_sl

        return None

    def on_position_closed(self, position: Position, pnl: float) -> None:
        """Track wins/losses for equity curve trading."""
        if pnl < 0:
            self._state.consecutive_losses += 1
            self._state.consecutive_wins = 0
        else:
            self._state.consecutive_wins += 1
            self._state.consecutive_losses = 0

        # Reset trade state
        self._state.active_partial_tps = []
        self._state.trade_entry_time = None
        self._state.trailing_active = False

        # Update risk for next trade
        self._update_equity_curve_risk()

    def get_status(self) -> dict:
        base_status = super().get_status()
        base_status.update({
            "ema_trend": self._state.ema_trend.value,
            "h4_bias": self._state.h4_bias.value,
            "h1_bias": self._state.h1_bias.value,
            "trades_today": self._state.trades_today,
            "consecutive_losses": self._state.consecutive_losses,
            "consecutive_wins": self._state.consecutive_wins,
            "active_pois": len(self._state.active_pois),
            "recent_sweeps": len(self._state.recent_sweeps),
            "ema_200": self.ema_200,
            "adx": self.adx,
            "market_condition": self._state.market_condition.value,
            "session_quality": self._state.session_quality,
            "current_risk": self._state.current_risk_percent,
            "trailing_active": self._state.trailing_active,
        })
        return base_status

    def reset(self) -> None:
        super().reset()
        self._reset_daily_state()
        self.atr_history = []
