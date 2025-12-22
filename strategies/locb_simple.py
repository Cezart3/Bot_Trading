"""
LOCB Simple Strategy - Mechanical FVG + Engulfing Entry.

Rules:
1. Mark HIGH/LOW of first M15 candle at session open (10:00 London, 16:30 NY Romania time)
2. Switch to M5, wait for breakout above HIGH (buy) or below LOW (sell)
3. Breakout must create a Fair Value Gap (FVG)
4. Wait for retracement into FVG (max 5 M5 candles)
5. Entry on Engulfing candle confirmation
6. SL below/above engulfing candle +/- spread buffer
7. TP at fixed 2.5:1 R:R

V2 Changes (2025-12-22):
- Opening candle: M15 (was M5) - more stable range
- FVG/Breakout detection: M5 (was M1) - less noise
- R:R: 2.5 (was 3.0) - more realistic targets
"""

import pandas as pd
from datetime import datetime, time
from typing import Optional, List
from dataclasses import dataclass, field
from enum import Enum

from models.candle import Candle
from models.position import Position
from strategies.base_strategy import BaseStrategy, SignalType, StrategySignal
from utils.logger import get_logger

logger = get_logger(__name__)


class SessionPhase(Enum):
    """Simple state machine phases."""
    WAITING_OC = "waiting_oc"           # Waiting for opening candle
    WAITING_BREAKOUT = "waiting_breakout"  # Waiting for breakout + FVG
    WAITING_RETEST = "waiting_retest"   # Waiting for retracement into FVG
    WAITING_ENGULF = "waiting_engulf"   # Waiting for engulfing confirmation
    TRADE_DONE = "trade_done"           # Trade taken, session complete


@dataclass
class FVG:
    """Fair Value Gap."""
    type: str  # "bullish" or "bearish"
    high: float
    low: float
    mid: float  # Middle of FVG for retest check
    creation_time: datetime


@dataclass
class SessionState:
    """State for each trading session."""
    name: str
    phase: SessionPhase = SessionPhase.WAITING_OC
    oc_high: Optional[float] = None
    oc_low: Optional[float] = None
    breakout_direction: Optional[str] = None  # "long" or "short"
    active_fvg: Optional[FVG] = None
    candles_since_fvg: int = 0
    retest_done: bool = False
    trade_taken: bool = False


@dataclass
class LOCBSimpleConfig:
    """Configuration for LOCB Simple Strategy V2 (M15/M5 timeframes)."""
    # Session times (Romania timezone = UTC+2 winter, UTC+3 summer)
    # We'll use UTC internally
    london_open_hour: int = 8      # 10:00 Romania = 08:00 UTC (winter)
    london_open_minute: int = 0
    ny_open_hour: int = 14         # 16:30 Romania = 14:30 UTC (winter)
    ny_open_minute: int = 30

    # Opening Candle Settings
    oc_minutes: int = 15           # Opening candle duration (M15)

    # Timeouts (in M5 candles now, not M1)
    max_candles_for_retest: int = 5  # Max M5 candles to wait for FVG retest (25 min)

    # Risk Management
    spread_buffer_pips: float = 0.7  # 7 points (0.7 pips) buffer to avoid premature SL
    rr_ratio: float = 2.5            # Fixed Risk:Reward ratio (was 3.0)
    min_sl_pips: float = 8.0         # Minimum SL in pips

    # Trade limits
    max_trades_per_session: int = 1  # 1 trade per session per pair


class LOCBSimpleStrategy(BaseStrategy):
    """
    Simple mechanical LOCB strategy with FVG + Engulfing entry.
    V2: Uses M15 for opening candle, M5 for FVG/breakout detection.
    """

    def __init__(self, symbol: str, timeframe: str = "M5", config: LOCBSimpleConfig = None, **kwargs):
        super().__init__(name="LOCB-Simple-V2", symbol=symbol, timeframe=timeframe)

        self.pip_size = kwargs.get('pip_size', 0.0001)
        self.config = config or LOCBSimpleConfig()

        # Sessions with their open times in Romania timezone (matches MT5 CSV data)
        # London open: 10:00 Romania time
        # NY open: 16:30 Romania time (NOT 16:00!)
        self.sessions = {
            "london": {"hour": 10, "minute": 0, "end_hour": 13},    # 10:00-13:00 Romania
            "ny": {"hour": 16, "minute": 30, "end_hour": 19}        # 16:30-19:30 Romania
        }

        # State
        self.current_date = None
        self.session_states = {}
        self.m5_candles: List[Candle] = []  # Recent M5 candles for FVG detection

    def initialize(self) -> bool:
        return True

    def on_tick(self, bid: float, ask: float, timestamp: datetime) -> Optional[StrategySignal]:
        return None

    def reset(self) -> None:
        """Reset for new day."""
        self.current_date = None
        self.session_states = {}
        self.m5_candles = []

    # ==================== FVG DETECTION ====================

    def _detect_fvg_on_breakout(self, candles: List[Candle], direction: str) -> Optional[FVG]:
        """
        Detect if the last 3 candles form a Fair Value Gap.

        Bullish FVG: candle[-3].high < candle[-1].low (gap up)
        Bearish FVG: candle[-3].low > candle[-1].high (gap down)
        """
        if len(candles) < 3:
            return None

        c1 = candles[-3]  # First candle
        c3 = candles[-1]  # Third candle (current)

        if direction == "long":
            # Bullish FVG: gap between c1.high and c3.low
            if c3.low > c1.high:
                fvg_high = c3.low
                fvg_low = c1.high
                return FVG(
                    type="bullish",
                    high=fvg_high,
                    low=fvg_low,
                    mid=(fvg_high + fvg_low) / 2,
                    creation_time=c3.timestamp
                )
        else:  # short
            # Bearish FVG: gap between c1.low and c3.high
            if c3.high < c1.low:
                fvg_high = c1.low
                fvg_low = c3.high
                return FVG(
                    type="bearish",
                    high=fvg_high,
                    low=fvg_low,
                    mid=(fvg_high + fvg_low) / 2,
                    creation_time=c3.timestamp
                )

        return None

    def _check_retest_into_fvg(self, candle: Candle, fvg: FVG) -> bool:
        """
        Check if candle retraces into FVG (preferably to middle).

        For bullish FVG: bearish candle should enter the gap
        For bearish FVG: bullish candle should enter the gap
        """
        if fvg.type == "bullish":
            # Need bearish candle entering the FVG from above
            is_bearish = candle.close < candle.open
            enters_fvg = candle.low <= fvg.high and candle.low >= fvg.low
            # Prefer if it reaches at least near the middle
            reaches_middle = candle.low <= fvg.mid + (fvg.high - fvg.mid) * 0.3
            return is_bearish and enters_fvg
        else:  # bearish FVG
            # Need bullish candle entering the FVG from below
            is_bullish = candle.close > candle.open
            enters_fvg = candle.high >= fvg.low and candle.high <= fvg.high
            return is_bullish and enters_fvg

    def _detect_engulfing(self, candles: List[Candle], direction: str) -> bool:
        """
        Detect Engulfing pattern.

        Bullish Engulfing: current bullish candle body engulfs previous bearish body
        Bearish Engulfing: current bearish candle body engulfs previous bullish body
        """
        if len(candles) < 2:
            return False

        prev = candles[-2]
        curr = candles[-1]

        prev_body_high = max(prev.open, prev.close)
        prev_body_low = min(prev.open, prev.close)
        curr_body_high = max(curr.open, curr.close)
        curr_body_low = min(curr.open, curr.close)

        # Minimum body size check (avoid dojis)
        curr_body = abs(curr.close - curr.open)
        if curr_body < 0.5 * self.pip_size:  # At least 0.5 pip body
            return False

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

    # ==================== MAIN LOGIC ====================

    def _get_active_session(self, ts: datetime) -> Optional[str]:
        """
        Check if we're within a trading session window.
        Returns session name if in session, None otherwise.
        Uses Romania timezone (matches MT5 CSV data).
        """
        for name, times in self.sessions.items():
            session_start = ts.replace(hour=times["hour"], minute=times["minute"], second=0, microsecond=0)
            session_end = ts.replace(hour=times["end_hour"], minute=0, second=0, microsecond=0)
            if session_start <= ts < session_end:
                return name
        return None

    def _is_session_open_candle(self, ts: datetime, session_name: str) -> bool:
        """Check if this timestamp is exactly at session open (for M5 OC marking)."""
        times = self.sessions[session_name]
        return ts.hour == times["hour"] and ts.minute == times["minute"]

    def on_candle(self, candle: Candle, candles: list[Candle]) -> Optional[StrategySignal]:
        """
        Main candle processing.
        Uses M15 for opening candle, M5 for FVG/breakout detection.
        """
        ts = candle.timestamp

        # Keep recent M5 candles for pattern detection
        self.m5_candles = candles[-20:] if len(candles) >= 20 else candles

        # Reset on new day
        if self.current_date != ts.date():
            self.current_date = ts.date()
            self.session_states = {}

        # Get current session
        session_name = self._get_active_session(ts)
        if not session_name:
            return None

        # Initialize session state if needed
        if session_name not in self.session_states:
            self.session_states[session_name] = SessionState(name=session_name)

        ss = self.session_states[session_name]

        # Skip if trade already taken this session
        if ss.trade_taken:
            return None

        # ==================== STATE MACHINE ====================

        # PHASE 1: Wait for Opening Candle (M15 = 3 x M5 candles at session open)
        if ss.phase == SessionPhase.WAITING_OC:
            if self._is_session_open_candle(ts, session_name):
                # This is the start of opening range - mark initial HIGH/LOW
                ss.oc_high = candle.high
                ss.oc_low = candle.low
                ss.phase = SessionPhase.WAITING_BREAKOUT
                logger.debug(f"[{self.symbol}] {session_name} OC building started: {ss.oc_low:.5f} - {ss.oc_high:.5f}")
            return None

        # Update OC high/low for first 15 minutes (building M15 equivalent from M5 candles)
        if ss.phase == SessionPhase.WAITING_BREAKOUT:
            times = self.sessions[session_name]
            session_start = ts.replace(hour=times["hour"], minute=times["minute"], second=0, microsecond=0)
            minutes_since_open = (ts - session_start).total_seconds() / 60

            # Still building the OC (first 15 minutes = 3 M5 candles)
            if minutes_since_open < self.config.oc_minutes:
                ss.oc_high = max(ss.oc_high, candle.high) if ss.oc_high else candle.high
                ss.oc_low = min(ss.oc_low, candle.low) if ss.oc_low else candle.low
                return None
            elif minutes_since_open == self.config.oc_minutes:
                # OC just completed - log the final range
                oc_range = (ss.oc_high - ss.oc_low) / self.pip_size
                logger.debug(f"[{self.symbol}] {session_name} OC complete (M15): {ss.oc_low:.5f} - {ss.oc_high:.5f} ({oc_range:.1f}p)")

        # PHASE 2: Wait for Breakout + FVG
        if ss.phase == SessionPhase.WAITING_BREAKOUT:
            if ss.oc_high is None or ss.oc_low is None:
                return None

            # Check for breakout above HIGH (BUY setup)
            if candle.close > ss.oc_high:
                fvg = self._detect_fvg_on_breakout(self.m5_candles, "long")
                if fvg:
                    ss.breakout_direction = "long"
                    ss.active_fvg = fvg
                    ss.candles_since_fvg = 0
                    ss.phase = SessionPhase.WAITING_RETEST
                    logger.debug(f"[{self.symbol}] Bullish breakout + FVG (M5): {fvg.low:.5f} - {fvg.high:.5f}")

            # Check for breakout below LOW (SELL setup)
            elif candle.close < ss.oc_low:
                fvg = self._detect_fvg_on_breakout(self.m5_candles, "short")
                if fvg:
                    ss.breakout_direction = "short"
                    ss.active_fvg = fvg
                    ss.candles_since_fvg = 0
                    ss.phase = SessionPhase.WAITING_RETEST
                    logger.debug(f"[{self.symbol}] Bearish breakout + FVG (M5): {fvg.low:.5f} - {fvg.high:.5f}")

            return None

        # PHASE 3: Wait for Retest into FVG
        if ss.phase == SessionPhase.WAITING_RETEST:
            ss.candles_since_fvg += 1

            # Timeout check
            if ss.candles_since_fvg > self.config.max_candles_for_retest:
                logger.debug(f"[{self.symbol}] FVG retest timeout after {ss.candles_since_fvg} candles")
                # Reset to look for new breakout
                ss.phase = SessionPhase.WAITING_BREAKOUT
                ss.active_fvg = None
                return None

            # Check if candle retests into FVG
            if self._check_retest_into_fvg(candle, ss.active_fvg):
                ss.retest_done = True
                ss.phase = SessionPhase.WAITING_ENGULF
                logger.debug(f"[{self.symbol}] FVG retest detected at {candle.close:.5f}")

            return None

        # PHASE 4: Wait for Engulfing Confirmation
        if ss.phase == SessionPhase.WAITING_ENGULF:
            ss.candles_since_fvg += 1

            # Extended timeout (still within reasonable window)
            if ss.candles_since_fvg > self.config.max_candles_for_retest + 5:
                logger.debug(f"[{self.symbol}] Engulfing timeout")
                ss.phase = SessionPhase.WAITING_BREAKOUT
                ss.active_fvg = None
                ss.retest_done = False
                return None

            # Check for engulfing pattern on M5
            if self._detect_engulfing(self.m5_candles, ss.breakout_direction):
                logger.info(f"[{self.symbol}] Engulfing confirmed (M5) for {ss.breakout_direction.upper()}")
                signal = self._create_trade_signal(candle, ss)
                if signal:
                    ss.trade_taken = True
                    ss.phase = SessionPhase.TRADE_DONE
                    return signal

            return None

        return None

    def _create_trade_signal(self, candle: Candle, ss: SessionState) -> Optional[StrategySignal]:
        """Create trade signal with proper SL/TP."""
        is_long = ss.breakout_direction == "long"
        side = SignalType.LONG if is_long else SignalType.SHORT

        spread_buffer = self.config.spread_buffer_pips * self.pip_size

        if is_long:
            # SL below engulfing candle low - spread buffer
            sl = candle.low - spread_buffer
            entry = candle.close
        else:
            # SL above engulfing candle high + spread buffer
            sl = candle.high + spread_buffer
            entry = candle.close

        # Calculate SL distance and TP
        sl_distance = abs(entry - sl)
        sl_pips = sl_distance / self.pip_size

        # Enforce minimum SL
        if sl_pips < self.config.min_sl_pips:
            sl_distance = self.config.min_sl_pips * self.pip_size
            sl = entry - sl_distance if is_long else entry + sl_distance
            sl_pips = self.config.min_sl_pips

        # TP at 2.5:1 R:R
        tp_distance = sl_distance * self.config.rr_ratio
        tp = entry + tp_distance if is_long else entry - tp_distance

        logger.info(
            f"[{self.symbol}] SIGNAL (M15/M5): {side.value} @ {entry:.5f} | "
            f"SL: {sl:.5f} ({sl_pips:.1f}p) | TP: {tp:.5f} | R:R: {self.config.rr_ratio}"
        )

        return StrategySignal(
            signal_type=side,
            symbol=self.symbol,
            price=entry,
            stop_loss=sl,
            take_profit=tp,
            reason=f"LOCB-FVG-Engulf-{side.value}",
            metadata={
                "session": ss.name,
                "oc_high": ss.oc_high,
                "oc_low": ss.oc_low,
                "fvg_high": ss.active_fvg.high if ss.active_fvg else None,
                "fvg_low": ss.active_fvg.low if ss.active_fvg else None,
            }
        )

    def should_exit(self, position: Position, current_price: float, candles: Optional[list[Candle]] = None) -> Optional[StrategySignal]:
        """No early exit - let SL/TP handle it."""
        return None

    def get_config_dict(self) -> dict:
        """Return config as dictionary."""
        return {
            "version": "V2 (M15/M5)",
            "london_open": f"{self.config.london_open_hour}:{self.config.london_open_minute:02d}",
            "ny_open": f"{self.config.ny_open_hour}:{self.config.ny_open_minute:02d}",
            "oc_minutes": self.config.oc_minutes,
            "rr_ratio": self.config.rr_ratio,
            "min_sl_pips": self.config.min_sl_pips,
            "spread_buffer": self.config.spread_buffer_pips,
            "max_retest_candles": self.config.max_candles_for_retest,
        }
