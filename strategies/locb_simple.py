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
    # Session times (UTC)
    asian_start_hour: int = 0
    asian_end_hour: int = 8        # Asian Range ends at London Open
    
    # Logic:
    # 1. Asian Range (00:00 - 08:00): Mark High/Low
    # 2. London Session (08:00 - 16:00): Look for Breakout of Asian High/Low

    # Timeouts (in M5 candles)
    max_candles_for_retest: int = 12  # Increased to 1 hour (12 x 5m)

    # Risk Management
    spread_buffer_pips: float = 0.7
    rr_ratio: float = 2.5
    min_sl_pips: float = 8.0

    # Trade limits
    max_trades_per_session: int = 1


class LOCBSimpleStrategy(BaseStrategy):
    """
    Asian Range Breakout Strategy (LOCB Modified).
    1. Define Asian Range (00:00 - 08:00 UTC).
    2. Wait for Breakout + FVG + Retest + Engulfing on M5 during London/NY.
    """

    def __init__(self, symbol: str, timeframe: str = "M5", config: LOCBSimpleConfig = None, **kwargs):
        super().__init__(name="LOCB-Asian-Breakout", symbol=symbol, timeframe=timeframe)

        self.pip_size = kwargs.get('pip_size', 0.0001)
        self.config = config or LOCBSimpleConfig()

        self.asian_range = {"start": self.config.asian_start_hour, "end": self.config.asian_end_hour}
        
        # State
        self.current_date = None
        self.session_states = {}
        self.m5_candles: List[Candle] = [] 

    def initialize(self) -> bool:
        return True

    def on_tick(self, bid: float, ask: float, timestamp: datetime) -> Optional[StrategySignal]:
        return None

    def reset(self) -> None:
        self.current_date = None
        self.session_states = {}
        self.m5_candles = []

    # ... (FVG/Engulfing methods remain the same) ...
    def _detect_fvg_on_breakout(self, candles: List[Candle], direction: str) -> Optional[FVG]:
        if len(candles) < 3: return None
        c1, c3 = candles[-3], candles[-1]
        if direction == "long":
            if c3.low > c1.high:
                return FVG("bullish", c3.low, c1.high, (c3.low + c1.high)/2, c3.timestamp)
        else:
            if c3.high < c1.low:
                return FVG("bearish", c1.low, c3.high, (c1.low + c3.high)/2, c3.timestamp)
        return None

    def _check_retest_into_fvg(self, candle: Candle, fvg: FVG) -> bool:
        if fvg.type == "bullish":
            return candle.close < candle.open and (fvg.low <= candle.low <= fvg.high)
        else:
            return candle.close > candle.open and (fvg.low <= candle.high <= fvg.high)

    def _detect_engulfing(self, candles: List[Candle], direction: str) -> bool:
        if len(candles) < 2: return False
        prev, curr = candles[-2], candles[-1]
        curr_body = abs(curr.close - curr.open)
        if curr_body < 0.5 * self.pip_size: return False
        
        prev_body_high = max(prev.open, prev.close)
        prev_body_low = min(prev.open, prev.close)
        curr_body_high = max(curr.open, curr.close)
        curr_body_low = min(curr.open, curr.close)

        if direction == "long":
            return prev.close < prev.open and curr.close > curr.open and \
                   curr_body_low <= prev_body_low and curr_body_high >= prev_body_high
        else:
            return prev.close > prev.open and curr.close < curr.open and \
                   curr_body_low <= prev_body_low and curr_body_high >= prev_body_high

    # ==================== MAIN LOGIC ====================

    def on_candle(self, candle: Candle, candles: list[Candle]) -> Optional[StrategySignal]:
        ts = candle.timestamp
        self.m5_candles = candles[-20:] if len(candles) >= 20 else candles

        if self.current_date != ts.date():
            self.current_date = ts.date()
            self.session_states = {}

        # We only track one "main" session per day now: The post-Asian session
        session_name = "london_breakout"
        if session_name not in self.session_states:
            self.session_states[session_name] = SessionState(name=session_name)
        
        ss = self.session_states[session_name]
        if ss.trade_taken: return None

        # PHASE 1: Build Asian Range (00:00 - 08:00)
        if ss.phase == SessionPhase.WAITING_OC:
            if self.asian_range["start"] <= ts.hour < self.asian_range["end"]:
                # Inside Asian Session -> Update High/Low
                ss.oc_high = max(ss.oc_high, candle.high) if ss.oc_high else candle.high
                ss.oc_low = min(ss.oc_low, candle.low) if ss.oc_low else candle.low
                logger.debug(f"[{self.symbol}] Building Asian Range: {ss.oc_low} - {ss.oc_high}")
            
            elif ts.hour >= self.asian_range["end"]:
                # Asian Session Finished -> Ready for Breakout
                if ss.oc_high and ss.oc_low:
                    ss.phase = SessionPhase.WAITING_BREAKOUT
                    logger.info(f"[{self.symbol}] Asian Range Complete: {ss.oc_low:.5f} - {ss.oc_high:.5f}")
            return None

        # PHASE 2: Wait for Breakout (After 08:00)
        if ss.phase == SessionPhase.WAITING_BREAKOUT:
            # Cutoff time (e.g., 18:00) to stop looking for new trades? Optional.
            if ts.hour >= 18: return None

            if candle.close > ss.oc_high:
                fvg = self._detect_fvg_on_breakout(self.m5_candles, "long")
                if fvg:
                    ss.breakout_direction = "long"
                    ss.active_fvg = fvg
                    ss.candles_since_fvg = 0
                    ss.phase = SessionPhase.WAITING_RETEST
                    logger.info(f"[{self.symbol}] Bullish Asian Breakout + FVG")

            elif candle.close < ss.oc_low:
                fvg = self._detect_fvg_on_breakout(self.m5_candles, "short")
                if fvg:
                    ss.breakout_direction = "short"
                    ss.active_fvg = fvg
                    ss.candles_since_fvg = 0
                    ss.phase = SessionPhase.WAITING_RETEST
                    logger.info(f"[{self.symbol}] Bearish Asian Breakout + FVG")
            return None

        # PHASE 3: Wait for Retest
        if ss.phase == SessionPhase.WAITING_RETEST:
            ss.candles_since_fvg += 1
            if ss.candles_since_fvg > self.config.max_candles_for_retest:
                ss.phase = SessionPhase.WAITING_BREAKOUT
                ss.active_fvg = None
                return None

            if self._check_retest_into_fvg(candle, ss.active_fvg):
                ss.retest_done = True
                ss.phase = SessionPhase.WAITING_ENGULF
                logger.debug(f"[{self.symbol}] Retest into Asian FVG")
            return None

        # PHASE 4: Confirmation
        if ss.phase == SessionPhase.WAITING_ENGULF:
            ss.candles_since_fvg += 1
            if ss.candles_since_fvg > self.config.max_candles_for_retest + 12: # Extra buffer
                ss.phase = SessionPhase.WAITING_BREAKOUT
                ss.active_fvg = None
                return None

            if self._detect_engulfing(self.m5_candles, ss.breakout_direction):
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
