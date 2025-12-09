"""
ORB (Opening Range Breakout) Strategy for Forex.

Optimized for EUR/USD on London Session based on backtesting:
- Session: London (10:00 - 18:00 Romania/Server time)
- Opening Range: First 60 minutes (10:00 - 11:00)
- Risk/Reward: 2:1
- One trade per session maximum

The strategy:
1. Waits for the Opening Range period to complete (first hour of session)
2. Records the High and Low of the Opening Range
3. Enters LONG when price closes above OR High
4. Enters SHORT when price closes below OR Low
5. Stop Loss at opposite end of Opening Range
6. Take Profit at 2x the risk (R:R = 2:1)
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Optional, List
import pytz

from models.candle import Candle
from models.order import OrderSide
from strategies.base_strategy import BaseStrategy, StrategySignal, SignalType
from utils.logger import get_logger

logger = get_logger(__name__)


class ForexSession(Enum):
    """Forex trading sessions."""
    ASIA = "asia"
    LONDON = "london"
    NEW_YORK = "new_york"


@dataclass
class ForexORBConfig:
    """Configuration for Forex ORB Strategy."""

    # Session times (Server/Romania time)
    session_name: str = "LONDON"
    session_start_hour: int = 10
    session_start_minute: int = 0
    session_end_hour: int = 18
    session_end_minute: int = 0

    # Opening Range duration in minutes
    orb_duration_minutes: int = 60

    # Risk management
    risk_reward_ratio: float = 2.0

    # Filters - OPTIMIZED for better win rate
    min_orb_range_pips: float = 10.0   # Minimum OR range (avoid tight ranges with false breakouts)
    max_orb_range_pips: float = 40.0   # Maximum OR range (avoid extreme volatility)

    # Breakout confirmation - helps avoid false breakouts
    breakout_buffer_pips: float = 3.0  # Require price to be X pips beyond OR level (increased from 2)
    wait_for_close: bool = True        # Wait for candle close (already implemented)

    # Time filter - avoid immediate breakouts after OR (often false)
    min_minutes_after_orb: int = 5     # Wait 5 minutes after OR before trading (avoid immediate fakeouts)

    # Trend filter - only trade in direction of higher timeframe trend
    use_trend_filter: bool = True      # Enable/disable trend filter
    trend_ema_period: int = 50         # EMA period for trend detection on H1

    # One trade per session
    max_trades_per_session: int = 1


class ORBState(Enum):
    """State machine for ORB strategy."""
    WAITING_FOR_SESSION = "waiting_for_session"
    BUILDING_ORB = "building_orb"
    ORB_COMPLETE = "orb_complete"
    IN_TRADE = "in_trade"
    SESSION_DONE = "session_done"


@dataclass
class OpeningRange:
    """Opening Range data."""
    high: float
    low: float
    start_time: datetime
    end_time: datetime
    candle_count: int


class ORBForexStrategy(BaseStrategy):
    """
    Opening Range Breakout Strategy for Forex.

    Designed for EUR/USD on London session but configurable for other pairs/sessions.
    """

    def __init__(
        self,
        config: Optional[ForexORBConfig] = None,
        symbol: str = "EURUSD",
        timeframe: str = "M5",
    ):
        """Initialize ORB Forex strategy."""
        super().__init__(
            name="ORB_FOREX",
            symbol=symbol,
            timeframe=timeframe,
            magic_number=3001,
        )
        self.config = config or ForexORBConfig()
        self.pip_size = 0.0001 if "JPY" not in symbol else 0.01

        # State tracking
        self._state = ORBState.WAITING_FOR_SESSION
        self._opening_range: Optional[OpeningRange] = None
        self._orb_candles: List[Candle] = []
        self._current_date: Optional[str] = None
        self._trades_today = 0
        self._h1_candles: List[Candle] = []  # For trend detection
        self._current_trend: Optional[str] = None  # "UP", "DOWN", or None

    def initialize(self) -> bool:
        """Initialize strategy."""
        logger.info(
            f"Initializing {self.name} strategy for {self.symbol} | "
            f"Session: {self.config.session_name} ({self.config.session_start_hour}:00-{self.config.session_end_hour}:00) | "
            f"ORB: {self.config.orb_duration_minutes} min | R:R: {self.config.risk_reward_ratio}:1"
        )
        self._reset_session()
        return True

    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self._reset_session()

    def _reset_session(self) -> None:
        """Reset state for new session."""
        self._state = ORBState.WAITING_FOR_SESSION
        self._opening_range = None
        self._orb_candles = []
        self._trades_today = 0
        logger.debug("Session state reset")

    def update_h1_candles(self, h1_candles: List[Candle]) -> None:
        """Update H1 candles for trend detection."""
        self._h1_candles = h1_candles
        self._update_trend()

    def _calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate EMA for given prices."""
        if len(prices) < period:
            return None

        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period  # Start with SMA

        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _update_trend(self) -> None:
        """Update trend based on H1 EMA."""
        if not self.config.use_trend_filter:
            self._current_trend = None
            return

        if len(self._h1_candles) < self.config.trend_ema_period + 5:
            self._current_trend = None
            return

        # Get close prices
        closes = [c.close for c in self._h1_candles]

        # Calculate EMA
        ema = self._calculate_ema(closes, self.config.trend_ema_period)

        if ema is None:
            self._current_trend = None
            return

        # Current price vs EMA
        current_price = closes[-1]

        if current_price > ema:
            self._current_trend = "UP"
        else:
            self._current_trend = "DOWN"

        logger.debug(f"Trend updated: {self._current_trend} (Price: {current_price:.5f}, EMA50: {ema:.5f})")

    def _is_session_time(self, timestamp: datetime) -> bool:
        """Check if current time is within session hours."""
        current_time = timestamp.time()
        session_start = time(self.config.session_start_hour, self.config.session_start_minute)
        session_end = time(self.config.session_end_hour, self.config.session_end_minute)
        return session_start <= current_time < session_end

    def _is_orb_period(self, timestamp: datetime) -> bool:
        """Check if current time is within Opening Range period."""
        current_time = timestamp.time()
        orb_start = time(self.config.session_start_hour, self.config.session_start_minute)

        # Calculate ORB end time
        orb_end_minutes = self.config.session_start_minute + self.config.orb_duration_minutes
        orb_end_hour = self.config.session_start_hour + orb_end_minutes // 60
        orb_end_minute = orb_end_minutes % 60
        orb_end = time(orb_end_hour, orb_end_minute)

        return orb_start <= current_time < orb_end

    def _build_opening_range(self) -> None:
        """Build Opening Range from collected candles."""
        if not self._orb_candles:
            return

        orb_high = max(c.high for c in self._orb_candles)
        orb_low = min(c.low for c in self._orb_candles)

        self._opening_range = OpeningRange(
            high=orb_high,
            low=orb_low,
            start_time=self._orb_candles[0].timestamp,
            end_time=self._orb_candles[-1].timestamp,
            candle_count=len(self._orb_candles),
        )

        range_pips = (orb_high - orb_low) / self.pip_size
        logger.info(
            f"Opening Range SET | High: {orb_high:.5f} | Low: {orb_low:.5f} | "
            f"Range: {range_pips:.1f} pips | Candles: {len(self._orb_candles)}"
        )

    def _check_orb_valid(self) -> bool:
        """Check if Opening Range is valid for trading."""
        if self._opening_range is None:
            return False

        range_pips = (self._opening_range.high - self._opening_range.low) / self.pip_size

        if range_pips < self.config.min_orb_range_pips:
            logger.debug(f"ORB too small: {range_pips:.1f} pips < {self.config.min_orb_range_pips}")
            return False

        if range_pips > self.config.max_orb_range_pips:
            logger.debug(f"ORB too large: {range_pips:.1f} pips > {self.config.max_orb_range_pips}")
            return False

        return True

    def on_candle(
        self,
        candle: Candle,
        candles: List[Candle],
    ) -> Optional[StrategySignal]:
        """
        Process candle for ORB strategy.

        Args:
            candle: Latest candle.
            candles: Historical candles.

        Returns:
            Trading signal or None.
        """
        # Check for new day
        candle_date = candle.timestamp.strftime("%Y-%m-%d")
        if self._current_date != candle_date:
            self._reset_session()
            self._current_date = candle_date

        # State machine
        if self._state == ORBState.WAITING_FOR_SESSION:
            # Check if session started
            if self._is_session_time(candle.timestamp):
                if self._is_orb_period(candle.timestamp):
                    self._state = ORBState.BUILDING_ORB
                    self._orb_candles = [candle]
                    logger.info(f"Session started, building Opening Range...")

        elif self._state == ORBState.BUILDING_ORB:
            # Continue building ORB
            if self._is_orb_period(candle.timestamp):
                self._orb_candles.append(candle)
            else:
                # ORB period ended, finalize
                self._build_opening_range()
                self._state = ORBState.ORB_COMPLETE

        elif self._state == ORBState.ORB_COMPLETE:
            # Check if session ended
            if not self._is_session_time(candle.timestamp):
                self._state = ORBState.SESSION_DONE
                return None

            # Check if max trades reached
            if self._trades_today >= self.config.max_trades_per_session:
                return None

            # Check if ORB is valid
            if not self._check_orb_valid():
                self._state = ORBState.SESSION_DONE
                return None

            # Check for breakout
            signal = self._check_breakout(candle)
            if signal:
                self._state = ORBState.IN_TRADE
                self._trades_today += 1
                return signal

        elif self._state == ORBState.IN_TRADE:
            # Trade is active, managed by broker
            if not self._is_session_time(candle.timestamp):
                self._state = ORBState.SESSION_DONE

        elif self._state == ORBState.SESSION_DONE:
            # Wait for new session
            if not self._is_session_time(candle.timestamp):
                self._state = ORBState.WAITING_FOR_SESSION

        return None

    def _check_breakout(self, candle: Candle) -> Optional[StrategySignal]:
        """Check for breakout and generate signal with buffer confirmation."""
        if self._opening_range is None:
            return None

        # Check min_minutes_after_orb filter
        if self.config.min_minutes_after_orb > 0:
            minutes_since_orb = (candle.timestamp - self._opening_range.end_time).total_seconds() / 60
            if minutes_since_orb < self.config.min_minutes_after_orb:
                return None

        orb_high = self._opening_range.high
        orb_low = self._opening_range.low

        # Calculate breakout buffer in price
        buffer = self.config.breakout_buffer_pips * self.pip_size

        # LONG breakout - close above OR high + buffer
        if candle.close > orb_high + buffer:
            # Check trend filter - only LONG if trend is UP or filter disabled
            if self.config.use_trend_filter and self._current_trend == "DOWN":
                logger.debug(f"LONG blocked by trend filter (trend: {self._current_trend})")
                return None
            entry_price = candle.close
            stop_loss = orb_low - buffer  # SL below OR low with buffer
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * self.config.risk_reward_ratio)

            signal = StrategySignal(
                signal_type=SignalType.LONG,
                symbol=self.symbol,
                price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=candle.timestamp,
                reason=f"ORB LONG breakout | OR High: {orb_high:.5f} + {self.config.breakout_buffer_pips} pips buffer",
                metadata={
                    "strategy": "ORB_FOREX",
                    "session": self.config.session_name,
                    "orb_high": orb_high,
                    "orb_low": orb_low,
                    "orb_range_pips": (orb_high - orb_low) / self.pip_size,
                    "risk_pips": risk / self.pip_size,
                    "buffer_pips": self.config.breakout_buffer_pips,
                },
            )

            logger.info(
                f"SIGNAL: LONG @ {entry_price:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}"
            )
            return signal

        # SHORT breakout - close below OR low - buffer
        elif candle.close < orb_low - buffer:
            # Check trend filter - only SHORT if trend is DOWN or filter disabled
            if self.config.use_trend_filter and self._current_trend == "UP":
                logger.debug(f"SHORT blocked by trend filter (trend: {self._current_trend})")
                return None

            entry_price = candle.close
            stop_loss = orb_high + buffer  # SL above OR high with buffer
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * self.config.risk_reward_ratio)

            signal = StrategySignal(
                signal_type=SignalType.SHORT,
                symbol=self.symbol,
                price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=candle.timestamp,
                reason=f"ORB SHORT breakout | OR Low: {orb_low:.5f} - {self.config.breakout_buffer_pips} pips buffer",
                metadata={
                    "strategy": "ORB_FOREX",
                    "session": self.config.session_name,
                    "orb_high": orb_high,
                    "orb_low": orb_low,
                    "orb_range_pips": (orb_high - orb_low) / self.pip_size,
                    "risk_pips": risk / self.pip_size,
                    "buffer_pips": self.config.breakout_buffer_pips,
                },
            )

            logger.info(
                f"SIGNAL: SHORT @ {entry_price:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}"
            )
            return signal

        return None

    def on_tick(self, bid: float, ask: float, timestamp: datetime) -> Optional[StrategySignal]:
        """Not used for this candle-based strategy."""
        return None

    def get_status(self) -> dict:
        """Get current strategy status."""
        or_info = None
        if self._opening_range:
            or_info = {
                "high": self._opening_range.high,
                "low": self._opening_range.low,
                "range_pips": (self._opening_range.high - self._opening_range.low) / self.pip_size,
                "candles": self._opening_range.candle_count,
            }

        return {
            "strategy": self.name,
            "symbol": self.symbol,
            "session": self.config.session_name,
            "state": self._state.value,
            "opening_range": or_info,
            "trades_today": self._trades_today,
            "orb_candles": len(self._orb_candles),
            "config": {
                "session_hours": f"{self.config.session_start_hour}:00-{self.config.session_end_hour}:00",
                "orb_duration": f"{self.config.orb_duration_minutes} min",
                "risk_reward": f"{self.config.risk_reward_ratio}:1",
            },
        }


# Pre-configured strategies for common setups
def create_eurusd_london_strategy() -> ORBForexStrategy:
    """
    Create optimized EUR/USD London session strategy.

    IMPORTANT: Hours are in SERVER TIME (MT5 server)!
    TeleTrade server is UTC+4 (2 hours ahead of Romania EET)

    Session mapping:
    - Romania 10:00 = Server 12:00
    - Romania 19:00 = Server 21:00

    Optimizations applied:
    - min_orb_range: 10 pips (avoid tight ranges with false breakouts)
    - max_orb_range: 40 pips (avoid extreme volatility)
    - breakout_buffer: 2 pips (confirm real breakout, not just a touch)
    """
    config = ForexORBConfig(
        session_name="LONDON",
        # SERVER TIME (TeleTrade = Romania + 2 hours)
        session_start_hour=12,  # 12:00 server = 10:00 Romania
        session_start_minute=0,
        session_end_hour=21,    # 21:00 server = 19:00 Romania
        session_end_minute=0,
        orb_duration_minutes=60,  # First hour (12:00-13:00 server = 10:00-11:00 Romania)
        risk_reward_ratio=2.0,
        # OPTIMIZED filters for profitability
        min_orb_range_pips=10.0,   # Avoid tight ranges
        max_orb_range_pips=40.0,   # Avoid extreme volatility
        breakout_buffer_pips=3.0,  # Buffer for confirmation (increased from 2)
        wait_for_close=True,
        min_minutes_after_orb=5,   # Wait 5 min after ORB to avoid fakeouts
        use_trend_filter=True,     # Only trade with H1 trend
        trend_ema_period=50,       # EMA 50 on H1 for trend
        max_trades_per_session=1,
    )
    return ORBForexStrategy(config=config, symbol="EURUSD", timeframe="M5")


def create_eurusd_newyork_strategy() -> ORBForexStrategy:
    """Create EUR/USD New York session strategy."""
    config = ForexORBConfig(
        session_name="NEW_YORK",
        session_start_hour=15,  # 15:00 Romania time
        session_start_minute=0,
        session_end_hour=22,    # 22:00 Romania time
        session_end_minute=0,
        orb_duration_minutes=60,
        risk_reward_ratio=2.0,
        min_orb_range_pips=10.0,
        max_orb_range_pips=40.0,
        breakout_buffer_pips=2.0,
        wait_for_close=True,
        min_minutes_after_orb=0,
        max_trades_per_session=1,
    )
    return ORBForexStrategy(config=config, symbol="EURUSD", timeframe="M5")
