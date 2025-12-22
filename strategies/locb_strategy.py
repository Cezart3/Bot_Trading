"""
LOCB-Sniper-Reverse Strategy V3.1.
Enhanced reversal strategy with advanced confirmations (CHoCH, iFVG, Engulfing).
Optimized for 2-3% monthly returns with balanced trade frequency.

V3.1 Changes (2025-12-22):
- Opening candle: M15 (was M5) - more stable range
- Confirmation/FVG detection: M5 (was M1) - less noise
- R:R: 2.5 (was 1.8) - better risk/reward with cleaner signals

Changes from V2:
- Added CHoCH, iFVG, Engulfing confirmations
- Added retest logic (3-phase workflow)
- Adaptive displacement based on ATR
- More trades per day (3) for better accumulation
"""

import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from models.candle import Candle
from models.position import Position
from strategies.base_strategy import BaseStrategy, SignalType, StrategySignal
from utils.logger import get_logger

logger = get_logger(__name__)


class ConfirmationType(Enum):
    """Types of confirmation patterns."""
    NONE = "none"
    CHOCH = "CHoCH"
    IFVG = "iFVG"
    ENGULFING = "Engulfing"
    STRONG_CANDLE = "StrongCandle"


class SessionPhase(Enum):
    """Phase of the session state machine."""
    BUILDING_OC = "building_oc"
    WAITING_FAKEOUT = "waiting_fakeout"
    WAITING_RETEST = "waiting_retest"
    WAITING_CONFIRM = "waiting_confirm"
    TRADE_TAKEN = "trade_taken"


@dataclass
class SessionState:
    name: str
    phase: SessionPhase = SessionPhase.BUILDING_OC
    oc_high: Optional[float] = None
    oc_low: Optional[float] = None
    oc_valid: bool = False
    max_extension: float = 0.0
    min_extension: float = 999999.0
    fakeout_direction: Optional[str] = None  # "up" or "down"
    retest_level: Optional[float] = None
    trade_direction: Optional[str] = None  # "long" or "short"
    trades_in_session: int = 0
    opening_candles: List[Candle] = field(default_factory=list)
    candles_since_fakeout: int = 0
    candles_since_retest: int = 0


@dataclass
class LOCBConfig:
    """Configuration for LOCB Strategy V3.1 - M15/M5 timeframes."""
    # Opening Range - NOW M15 (3 x M5 candles)
    min_oc_range: float = 3.0      # Min OC range in pips (adjusted for M15)
    max_oc_range: float = 45.0     # Max OC range in pips (adjusted for M15)
    oc_candles: int = 3            # Number of M5 candles for opening range (3 x 5min = 15min)
    oc_minutes: int = 15           # Opening candle duration in minutes

    # Entry - RELAXED settings
    required_displacement: float = 4.0  # Displacement in pips (slightly higher for M5)
    use_adaptive_displacement: bool = False  # DISABLED - use fixed threshold
    ema_period: int = 50                # EMA period (faster for quicker signals)
    require_strong_candle: bool = True  # RE-ENABLED as fallback

    # Advanced Confirmations - RELAXED (now on M5 candles)
    use_advanced_confirmations: bool = True  # Still use CHoCH, iFVG, Engulfing
    choch_lookback: int = 8              # Lookback in M5 candles
    ifvg_lookback: int = 15              # Lookback in M5 candles

    # Retest Logic - DISABLED (was too strict)
    use_retest_logic: bool = False       # DISABLED - enter on fakeout+confirmation directly
    retest_tolerance_pips: float = 5.0   # Increased tolerance if re-enabled
    max_candles_for_retest: int = 12     # Max M5 candles (60 min)
    max_candles_for_confirm: int = 8     # Max M5 candles for confirmation (40 min)

    # Risk Management
    min_sl_pips: float = 8.0       # Minimum SL (increased for M15 OC)
    max_sl_pips: float = 35.0      # Maximum SL (increased for M15 OC)
    rr_ratio: float = 2.5          # R:R ratio (increased for M15/M5 setup)

    # Trade Management
    max_trades_per_day: int = 4    # Max trades per day (increased)
    max_trades_per_session: int = 2 # Max trades per session
    use_breakeven: bool = True     # Move SL to BE at 1:1
    use_partial_tp: bool = False   # Disabled for simplicity

    # Volatility Filter - RELAXED
    use_atr_filter: bool = True
    atr_period: int = 14
    min_atr_pips: float = 2.0      # Min ATR (slightly higher for M5)
    max_atr_pips: float = 60.0     # Max ATR (very relaxed)


class LOCBStrategy(BaseStrategy):
    def __init__(self, symbol: str, timeframe: str = "M5", config: LOCBConfig = None, **kwargs):
        super().__init__(name="LOCB-Sniper-V3.1", symbol=symbol, timeframe=timeframe)

        # Allow pip_size override for different instruments
        self.pip_size = kwargs.get('pip_size', 0.0001)

        # Use provided config or defaults
        self.config = config or LOCBConfig()

        # Sessions (Kill Zones) - Times in Romania timezone (EET/EEST)
        # London Open: 10:00 Romania time
        # NY Open: 16:30 Romania time (NOT 16:00!)
        # Session windows: 3 hours after open
        self.sessions = {
            "london": {"start_hour": 10, "start_minute": 0, "end_hour": 13},   # 10:00-13:00 Romania
            "ny": {"start_hour": 16, "start_minute": 30, "end_hour": 19}       # 16:30-19:30 Romania
        }

        # State
        self.current_date = None
        self.trades_today = 0
        self.session_states = {}
        self.ema_value = None
        self.current_atr = None
        self.m5_candles: List[Candle] = []  # For confirmations (renamed from recent_candles)

    def initialize(self) -> bool:
        return True

    def on_tick(self, bid: float, ask: float, timestamp: datetime) -> Optional[StrategySignal]:
        return None

    def _calculate_ema(self, candles: List[Candle]) -> Optional[float]:
        """Calculate EMA from candles."""
        if len(candles) < self.config.ema_period:
            return None
        closes = [c.close for c in candles[-self.config.ema_period:]]
        return pd.Series(closes).ewm(span=self.config.ema_period, adjust=False).mean().iloc[-1]

    def _calculate_atr(self, candles: List[Candle]) -> float:
        """Calculate ATR in pips."""
        if len(candles) < self.config.atr_period:
            return 0
        ranges = [c.high - c.low for c in candles[-self.config.atr_period:]]
        atr = sum(ranges) / len(ranges)
        return atr / self.pip_size

    def _get_adaptive_displacement(self) -> float:
        """Get displacement threshold based on current ATR."""
        if not self.config.use_adaptive_displacement or not self.current_atr:
            return self.config.required_displacement

        # Scale displacement: 0.5x ATR, but min 3 pips, max 10 pips
        adaptive = self.current_atr * 0.5
        return max(3.0, min(adaptive, 10.0))

    # ==================== ADVANCED CONFIRMATIONS ====================

    def _detect_choch(self, candles: List[Candle], direction: str) -> bool:
        """
        Detect Change of Character (CHoCH).
        For bullish: price breaks above recent swing high
        For bearish: price breaks below recent swing low
        """
        lookback = self.config.choch_lookback
        if len(candles) < lookback + 2:
            return False

        recent = candles[-(lookback + 1):-1]
        current = candles[-1]

        if direction == "long":
            swing_high = max(c.high for c in recent)
            return current.close > swing_high
        else:  # short
            swing_low = min(c.low for c in recent)
            return current.close < swing_low

    def _find_fvgs(self, candles: List[Candle]) -> List[dict]:
        """
        Find Fair Value Gaps in candle data.
        Bullish FVG: Gap between candle[i-2].high and candle[i].low
        Bearish FVG: Gap between candle[i-2].low and candle[i].high
        """
        fvgs = []
        if len(candles) < 3:
            return fvgs

        for i in range(2, len(candles)):
            c0 = candles[i - 2]
            c2 = candles[i]

            # Bullish FVG
            if c2.low > c0.high:
                fvgs.append({
                    "type": "bullish",
                    "high": c2.low,
                    "low": c0.high,
                    "time": c2.timestamp
                })

            # Bearish FVG
            if c2.high < c0.low:
                fvgs.append({
                    "type": "bearish",
                    "high": c0.low,
                    "low": c2.high,
                    "time": c2.timestamp
                })

        return fvgs

    def _detect_ifvg(self, candles: List[Candle], direction: str) -> bool:
        """
        Detect Inversed Fair Value Gap (iFVG).
        iFVG = FVG that gets violated (price closes through it)
        For bullish: bearish FVG violated (bullish sign)
        For bearish: bullish FVG violated (bearish sign)
        """
        lookback = self.config.ifvg_lookback
        if len(candles) < lookback:
            return False

        recent = candles[-lookback:]
        fvgs = self._find_fvgs(recent[:-1])

        if not fvgs:
            return False

        current = candles[-1]

        for fvg in fvgs:
            if direction == "long" and fvg["type"] == "bearish":
                if current.close > fvg["high"]:
                    return True
            elif direction == "short" and fvg["type"] == "bullish":
                if current.close < fvg["low"]:
                    return True

        return False

    def _detect_engulfing(self, candles: List[Candle], direction: str) -> bool:
        """
        Detect Engulfing pattern.
        Bullish Engulfing: current bullish candle engulfs previous bearish
        Bearish Engulfing: current bearish candle engulfs previous bullish
        """
        if len(candles) < 2:
            return False

        prev = candles[-2]
        curr = candles[-1]

        prev_body_high = max(prev.open, prev.close)
        prev_body_low = min(prev.open, prev.close)
        curr_body_high = max(curr.open, curr.close)
        curr_body_low = min(curr.open, curr.close)

        if direction == "long":
            prev_bearish = prev.close < prev.open
            curr_bullish = curr.close > curr.open
            engulfs = curr_body_low <= prev_body_low and curr_body_high >= prev_body_high
            return prev_bearish and curr_bullish and engulfs
        else:  # short
            prev_bullish = prev.close > prev.open
            curr_bearish = curr.close < curr.open
            engulfs = curr_body_low <= prev_body_low and curr_body_high >= prev_body_high
            return prev_bullish and curr_bearish and engulfs

    def _is_strong_candle(self, candle: Candle, direction: str) -> bool:
        """Check if candle is strong (not a doji)."""
        body = abs(candle.close - candle.open)
        total_range = candle.high - candle.low

        if total_range == 0:
            return False

        body_ratio = body / total_range

        if direction == "long":
            return candle.close > candle.open and body_ratio > 0.4
        else:
            return candle.close < candle.open and body_ratio > 0.4

    def _check_confirmation(self, candles: List[Candle], direction: str) -> Tuple[bool, ConfirmationType]:
        """
        Check for any confirmation pattern.
        Returns (confirmed, confirmation_type)
        Priority: CHoCH > iFVG > Engulfing > StrongCandle
        """
        if self.config.use_advanced_confirmations:
            # CHoCH - strongest signal
            if self._detect_choch(candles, direction):
                return True, ConfirmationType.CHOCH

            # iFVG - second strongest
            if self._detect_ifvg(candles, direction):
                return True, ConfirmationType.IFVG

            # Engulfing - good signal
            if self._detect_engulfing(candles, direction):
                return True, ConfirmationType.ENGULFING

        # Fallback to strong candle if enabled
        if self.config.require_strong_candle:
            if self._is_strong_candle(candles[-1], direction):
                return True, ConfirmationType.STRONG_CANDLE

        # If no strong candle required and no advanced confirmations, accept any
        if not self.config.require_strong_candle and not self.config.use_advanced_confirmations:
            return True, ConfirmationType.NONE

        return False, ConfirmationType.NONE

    def _check_retest(self, candle: Candle, level: float, direction: str) -> bool:
        """
        Check if price has retested a level.
        For long: price should touch/approach level from above
        For short: price should touch/approach level from below
        """
        tolerance = self.config.retest_tolerance_pips * self.pip_size

        if direction == "long":
            # Retest of OC_HIGH for buy
            return candle.low <= level + tolerance and candle.close > level
        else:  # short
            # Retest of OC_LOW for sell
            return candle.high >= level - tolerance and candle.close < level

    def on_candle(self, candle: Candle, candles: list[Candle]) -> Optional[StrategySignal]:
        """
        Main candle processing with state machine logic.
        Uses M15 for opening range, M5 for confirmation patterns.
        Phases: BUILDING_OC -> WAITING_FAKEOUT -> WAITING_RETEST -> WAITING_CONFIRM -> TRADE_TAKEN
        """
        ts = candle.timestamp
        self.m5_candles = candles[-50:] if len(candles) >= 50 else candles  # Keep M5 candles for confirmations

        # Reset on new day
        if self.current_date != ts.date():
            self.current_date = ts.date()
            self.trades_today = 0
            self.session_states = {}

        # Check max trades per day
        if self.trades_today >= self.config.max_trades_per_day:
            return None

        # Update EMA
        self.ema_value = self._calculate_ema(candles)

        # Update ATR and check volatility filter
        if self.config.use_atr_filter and len(candles) >= self.config.atr_period:
            self.current_atr = self._calculate_atr(candles)
            if self.current_atr < self.config.min_atr_pips:
                return None  # Market too quiet
            if self.current_atr > self.config.max_atr_pips:
                return None  # Market too volatile

        # Get current session
        s_name = self._get_session(ts)
        if not s_name:
            return None

        # Initialize session state
        if s_name not in self.session_states:
            self.session_states[s_name] = SessionState(name=s_name)

        ss = self.session_states[s_name]

        # Check max trades per session
        if ss.trades_in_session >= self.config.max_trades_per_session:
            return None

        # ==================== STATE MACHINE ====================

        # PHASE 1: Build Opening Range
        if ss.phase == SessionPhase.BUILDING_OC:
            ss.opening_candles.append(candle)
            if len(ss.opening_candles) >= self.config.oc_candles:
                ss.oc_high = max(c.high for c in ss.opening_candles)
                ss.oc_low = min(c.low for c in ss.opening_candles)
                oc_range = (ss.oc_high - ss.oc_low) / self.pip_size

                if self.config.min_oc_range <= oc_range <= self.config.max_oc_range:
                    ss.oc_valid = True
                    ss.phase = SessionPhase.WAITING_FAKEOUT
                    ss.max_extension = ss.oc_high
                    ss.min_extension = ss.oc_low
                    logger.debug(f"[{self.symbol}] OC built (M15): {ss.oc_low:.5f} - {ss.oc_high:.5f} ({oc_range:.1f}p)")
                else:
                    logger.debug(f"[{self.symbol}] OC range {oc_range:.1f}p outside [{self.config.min_oc_range}-{self.config.max_oc_range}]")
            return None

        if not ss.oc_valid:
            return None

        # PHASE 2: Wait for Fakeout (displacement)
        if ss.phase == SessionPhase.WAITING_FAKEOUT:
            ss.max_extension = max(ss.max_extension, candle.high)
            ss.min_extension = min(ss.min_extension, candle.low)

            displacement_threshold = self._get_adaptive_displacement()
            ext_up = (ss.max_extension - ss.oc_high) / self.pip_size
            ext_down = (ss.oc_low - ss.min_extension) / self.pip_size

            # Fakeout UP -> prepare for SHORT
            if ext_up >= displacement_threshold and candle.close < ss.oc_high:
                if not self.ema_value or candle.close < self.ema_value:  # Bearish bias
                    ss.fakeout_direction = "up"
                    ss.trade_direction = "short"
                    ss.retest_level = ss.oc_low  # Will retest LOW for short entry
                    ss.candles_since_fakeout = 0

                    if self.config.use_retest_logic:
                        ss.phase = SessionPhase.WAITING_RETEST
                        logger.debug(f"[{self.symbol}] Fakeout UP detected, waiting for retest of {ss.retest_level:.5f}")
                    else:
                        # Skip retest, go directly to confirmation
                        ss.phase = SessionPhase.WAITING_CONFIRM

            # Fakeout DOWN -> prepare for LONG
            elif ext_down >= displacement_threshold and candle.close > ss.oc_low:
                if not self.ema_value or candle.close > self.ema_value:  # Bullish bias
                    ss.fakeout_direction = "down"
                    ss.trade_direction = "long"
                    ss.retest_level = ss.oc_high  # Will retest HIGH for long entry
                    ss.candles_since_fakeout = 0

                    if self.config.use_retest_logic:
                        ss.phase = SessionPhase.WAITING_RETEST
                        logger.debug(f"[{self.symbol}] Fakeout DOWN detected, waiting for retest of {ss.retest_level:.5f}")
                    else:
                        ss.phase = SessionPhase.WAITING_CONFIRM

            return None

        # PHASE 3: Wait for Retest
        if ss.phase == SessionPhase.WAITING_RETEST:
            ss.candles_since_fakeout += 1

            # Timeout check
            if ss.candles_since_fakeout > self.config.max_candles_for_retest:
                logger.debug(f"[{self.symbol}] Retest timeout after {ss.candles_since_fakeout} candles")
                ss.phase = SessionPhase.WAITING_FAKEOUT  # Reset to look for new fakeout
                ss.fakeout_direction = None
                return None

            # Check for retest
            if self._check_retest(candle, ss.retest_level, ss.trade_direction):
                ss.phase = SessionPhase.WAITING_CONFIRM
                ss.candles_since_retest = 0
                logger.debug(f"[{self.symbol}] Retest detected at {candle.close:.5f}")

            return None

        # PHASE 4: Wait for Confirmation
        if ss.phase == SessionPhase.WAITING_CONFIRM:
            ss.candles_since_retest += 1

            # Timeout check
            if ss.candles_since_retest > self.config.max_candles_for_confirm:
                logger.debug(f"[{self.symbol}] Confirmation timeout after {ss.candles_since_retest} candles")
                ss.phase = SessionPhase.WAITING_FAKEOUT
                ss.fakeout_direction = None
                return None

            # Check for confirmation on M5 candles
            confirmed, confirm_type = self._check_confirmation(self.m5_candles, ss.trade_direction)

            if confirmed:
                logger.debug(f"[{self.symbol}] Confirmation {confirm_type.value} for {ss.trade_direction}")
                signal = self._trigger_trade(candle, ss, confirm_type)
                if signal:
                    ss.phase = SessionPhase.TRADE_TAKEN
                    return signal

            return None

        return None

    def _trigger_trade(self, candle: Candle, ss: SessionState, confirm_type: ConfirmationType) -> Optional[StrategySignal]:
        """Generate trade signal with proper SL/TP."""
        is_long = ss.trade_direction == "long"
        side = SignalType.LONG if is_long else SignalType.SHORT

        # SL placement options:
        # 1. Below/above the OC extreme (safer)
        # 2. Below/above the confirmation candle (tighter)
        buffer = 2 * self.pip_size

        if is_long:
            # SL below OC low or confirmation candle low
            sl = min(ss.oc_low, candle.low) - buffer
        else:
            # SL above OC high or confirmation candle high
            sl = max(ss.oc_high, candle.high) + buffer

        # Calculate SL distance
        sl_distance = abs(candle.close - sl)
        sl_pips = sl_distance / self.pip_size

        # Validate SL distance
        if sl_pips < self.config.min_sl_pips:
            sl_distance = self.config.min_sl_pips * self.pip_size
            sl = candle.close - sl_distance if is_long else candle.close + sl_distance
            sl_pips = self.config.min_sl_pips

        if sl_pips > self.config.max_sl_pips:
            logger.debug(f"[{self.symbol}] Skip: SL {sl_pips:.1f}p > max {self.config.max_sl_pips}")
            return None

        # Calculate TP
        tp_distance = sl_distance * self.config.rr_ratio
        tp = candle.close + tp_distance if is_long else candle.close - tp_distance

        # Update counters
        ss.trades_in_session += 1
        self.trades_today += 1

        logger.info(
            f"[{self.symbol}] SIGNAL (M15/M5): {side.value} @ {candle.close:.5f} | "
            f"SL: {sl:.5f} ({sl_pips:.1f}p) | TP: {tp:.5f} | "
            f"RR: {self.config.rr_ratio} | Confirm: {confirm_type.value}"
        )

        return StrategySignal(
            signal_type=side,
            symbol=self.symbol,
            price=candle.close,
            stop_loss=sl,
            take_profit=tp,
            reason=f"LOCB-{confirm_type.value}-{side.value}",
            metadata={
                "session": ss.name,
                "oc_high": ss.oc_high,
                "oc_low": ss.oc_low,
                "fakeout_direction": ss.fakeout_direction,
                "confirmation": confirm_type.value,
                "atr": self.current_atr,
                "partial_tp": self.config.use_partial_tp
            }
        )

    def should_exit(self, position: Position, current_price: float, candles: Optional[list[Candle]] = None) -> Optional[StrategySignal]:
        """Check for early exit conditions (breakeven)."""
        if not self.config.use_breakeven:
            return None

        risk = abs(position.entry_price - position.stop_loss)

        if not position.metadata.get("be") and risk > 0:
            # Move to breakeven at 1:1
            if position.is_long and current_price >= position.entry_price + risk:
                position.stop_loss = position.entry_price + (1 * self.pip_size)  # Tiny profit to cover spread
                position.metadata["be"] = True
                logger.debug(f"[{self.symbol}] BE triggered for LONG @ {position.entry_price}")
            elif not position.is_long and current_price <= position.entry_price - risk:
                position.stop_loss = position.entry_price - (1 * self.pip_size)
                position.metadata["be"] = True
                logger.debug(f"[{self.symbol}] BE triggered for SHORT @ {position.entry_price}")

        return None

    def _get_session(self, ts: datetime) -> Optional[str]:
        """Get current trading session name based on Romania timezone."""
        for name, times in self.sessions.items():
            start_hour = times["start_hour"]
            start_minute = times["start_minute"]
            end_hour = times["end_hour"]

            # Create session start time
            session_start = ts.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
            session_end = ts.replace(hour=end_hour, minute=0, second=0, microsecond=0)

            if session_start <= ts < session_end:
                return name
        return None

    def reset(self) -> None:
        """Reset strategy state for new day."""
        self.current_date = None
        self.trades_today = 0
        self.session_states = {}
        self.ema_value = None
        self.current_atr = None
        self.m5_candles = []

    def get_config_dict(self) -> dict:
        """Return current config as dictionary (for logging/debugging)."""
        return {
            "version": "V3.1 (M15/M5)",
            "oc_minutes": self.config.oc_minutes,
            "ema_period": self.config.ema_period,
            "displacement": self.config.required_displacement,
            "rr_ratio": self.config.rr_ratio,
            "min_sl_pips": self.config.min_sl_pips,
            "max_sl_pips": self.config.max_sl_pips,
            "max_trades_per_day": self.config.max_trades_per_day,
            "max_trades_per_session": self.config.max_trades_per_session,
            "min_oc_range": self.config.min_oc_range,
            "max_oc_range": self.config.max_oc_range,
        }
