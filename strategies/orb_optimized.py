"""
ORB Optimized Strategy - Independent Session Ranges.

Improvements over v1:
1. Calculates specific ranges for London AND New York independently.
2. Uses shorter, more explosive ranges (1h) instead of full Asian session (8h).
3. Allows trading BOTH sessions in one day with fresh logic.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum

from models.candle import Candle
from models.position import Position
from strategies.base_strategy import BaseStrategy, SignalType, StrategySignal
from utils.logger import get_logger

logger = get_logger(__name__)

class ORBPhase(Enum):
    IDLE = "idle"
    BUILDING_RANGE = "building_range"
    WAITING_BREAKOUT = "waiting_breakout"
    TRADE_TAKEN = "trade_taken"

@dataclass
class SessionConfig:
    name: str
    range_start_hour: int
    range_end_hour: int
    trade_end_hour: int
    max_range_pips: float = 100.0
    min_range_pips: float = 5.0

@dataclass
class ORBOptimizedConfig:
    # Global Risk
    risk_percent: float = 1.0
    rr_ratio: float = 2.0
    use_breakeven: bool = True
    be_trigger_rr: float = 1.0
    
    # Sessions
    sessions: List[SessionConfig] = None
    
    # Common Filters
    min_sl_pips: float = 5.0  # Tighter SL possible with 1h ranges
    max_sl_pips: float = 40.0
    breakout_buffer_pips: float = 1.0
    spread_buffer_pips: float = 1.0
    
    # Trend/Momentum
    use_ema_filter: bool = True
    ema_period: int = 50 # Faster EMA for intraday
    
    def __post_init__(self):
        if self.sessions is None:
            # Default to London (08-09 range) and NY (13-14 range)
            self.sessions = [
                SessionConfig("London", 8, 9, 12),
                SessionConfig("NewYork", 13, 14, 18)
            ]

class ORBOptimizedStrategy(BaseStrategy):
    def __init__(self, symbol: str, timeframe: str = "M15", config: ORBOptimizedConfig = None, **kwargs):
        super().__init__(name="ORB-Optimized", symbol=symbol, timeframe=timeframe)
        self.config = config or ORBOptimizedConfig()
        self.pip_size = kwargs.get('pip_size', 0.0001)
        
        self.reset_day()
        self.ema_value = None

    def initialize(self) -> bool:
        return True

    def on_tick(self, bid: float, ask: float, timestamp: datetime) -> Optional[StrategySignal]:
        return None

    def reset_day(self):
        self.current_session_idx = -1
        self.phase = ORBPhase.IDLE
        self.range_high = None
        self.range_low = None
        self.range_candles = []
        self.trades_in_session = 0
        self.current_date = None

    def _get_active_session(self, hour: int) -> Optional[tuple[int, SessionConfig]]:
        for i, session in enumerate(self.config.sessions):
            # Check if we are in range building time
            if session.range_start_hour <= hour < session.range_end_hour:
                return i, session
            # Check if we are in trading time
            if session.range_end_hour <= hour < session.trade_end_hour:
                return i, session
        return None, None

    def on_candle(self, candle: Candle, candles: list[Candle]) -> Optional[StrategySignal]:
        ts = candle.timestamp
        
        # New Day Reset
        if self.current_date != ts.date():
            self.reset_day()
            self.current_date = ts.date()

        # Update Indicators
        if self.config.use_ema_filter:
            closes = [c.close for c in candles[-self.config.ema_period:]]
            if len(closes) >= self.config.ema_period:
                self.ema_value = pd.Series(closes).ewm(span=self.config.ema_period, adjust=False).mean().iloc[-1]

        # Determine Session State
        session_idx, session = self._get_active_session(ts.hour)
        
        if session is None:
            self.phase = ORBPhase.IDLE
            return None

        # If we switched sessions (e.g. London -> NY), reset range logic
        if session_idx != self.current_session_idx:
            self.current_session_idx = session_idx
            self.phase = ORBPhase.IDLE
            self.range_high = None
            self.range_low = None
            self.trades_in_session = 0
            logger.debug(f"[{self.symbol}] Entering Session: {session.name}")

        # LOGIC: BUILDING RANGE
        if session.range_start_hour <= ts.hour < session.range_end_hour:
            self.phase = ORBPhase.BUILDING_RANGE
            if self.range_high is None:
                self.range_high = candle.high
                self.range_low = candle.low
            else:
                self.range_high = max(self.range_high, candle.high)
                self.range_low = min(self.range_low, candle.low)
            return None

        # LOGIC: TRADING BREAKOUT
        if session.range_end_hour <= ts.hour < session.trade_end_hour:
            
            # Transition from Build -> Wait
            if self.phase == ORBPhase.BUILDING_RANGE:
                if self.range_high is None: return None
                
                range_pips = (self.range_high - self.range_low) / self.pip_size
                if not (session.min_range_pips <= range_pips <= session.max_range_pips):
                    logger.debug(f"[{self.symbol}] Skipped {session.name}: Range {range_pips:.1f} pips invalid.")
                    self.phase = ORBPhase.IDLE # Invalid range
                    return None
                
                self.phase = ORBPhase.WAITING_BREAKOUT
                logger.debug(f"[{self.symbol}] {session.name} Range Set: {self.range_low:.5f}-{self.range_high:.5f}")

            if self.phase == ORBPhase.WAITING_BREAKOUT and self.trades_in_session == 0:
                return self._check_breakout(candle, session)

        return None

    def _check_breakout(self, candle: Candle, session: SessionConfig) -> Optional[StrategySignal]:
        buffer = self.config.breakout_buffer_pips * self.pip_size
        
        # LONG
        if candle.close > self.range_high + buffer:
            # EMA Filter: If Price < EMA, don't buy (counter-trend check)
            if self.config.use_ema_filter and self.ema_value and candle.close < self.ema_value:
                return None
            return self._create_signal(candle, "long")

        # SHORT
        elif candle.close < self.range_low - buffer:
            # EMA Filter: If Price > EMA, don't sell
            if self.config.use_ema_filter and self.ema_value and candle.close > self.ema_value:
                return None
            return self._create_signal(candle, "short")
            
        return None

    def _create_signal(self, candle: Candle, direction: str) -> StrategySignal:
        is_long = direction == "long"
        side = SignalType.LONG if is_long else SignalType.SHORT
        entry = candle.close
        
        # SL logic: Opposite side of range, but constrained
        spread = self.config.spread_buffer_pips * self.pip_size
        raw_sl = (self.range_low - spread) if is_long else (self.range_high + spread)
        
        sl_dist_pips = abs(entry - raw_sl) / self.pip_size
        
        # Normalize SL
        if sl_dist_pips < self.config.min_sl_pips:
            raw_sl = entry - (self.config.min_sl_pips * self.pip_size) if is_long else entry + (self.config.min_sl_pips * self.pip_size)
        elif sl_dist_pips > self.config.max_sl_pips:
            # Cap SL to max (better R:R on wide ranges)
            raw_sl = entry - (self.config.max_sl_pips * self.pip_size) if is_long else entry + (self.config.max_sl_pips * self.pip_size)

        tp_dist = abs(entry - raw_sl) * self.config.rr_ratio
        tp = entry + tp_dist if is_long else entry - tp_dist
        
        self.phase = ORBPhase.TRADE_TAKEN
        self.trades_in_session += 1
        
        return StrategySignal(
            signal_type=side,
            symbol=self.symbol,
            price=entry,
            stop_loss=raw_sl,
            take_profit=tp,
            reason=f"ORB-{direction.upper()}",
            metadata={"be_activated": False}
        )

    def should_exit(self, position: Position, current_price: float, candles: Optional[list[Candle]] = None) -> Optional[StrategySignal]:
        if not self.config.use_breakeven or position.metadata.get("be_activated"): return None
        
        risk = abs(position.entry_price - position.stop_loss)
        pnl_pips = (current_price - position.entry_price) if position.is_long else (position.entry_price - current_price)
        pnl_r = pnl_pips / risk
        
        if pnl_r >= self.config.be_trigger_rr:
             position.stop_loss = position.entry_price + (2 * self.pip_size if position.is_long else -2 * self.pip_size)
             position.metadata["be_activated"] = True
             # logger.info(f"[{position.symbol}] BE Activated at {pnl_r:.2f}R")
        
        return None
