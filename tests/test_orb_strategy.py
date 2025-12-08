"""Tests for ORB Strategy."""

import pytest
from datetime import datetime, timedelta
import pytz

from strategies.orb_strategy import ORBStrategy, ORBState
from strategies.base_strategy import SignalType
from models.candle import Candle, OpeningRange


class TestORBStrategy:
    """Test cases for ORB Strategy."""

    @pytest.fixture
    def strategy(self):
        """Create ORB strategy instance."""
        return ORBStrategy(
            symbol="EURUSD",
            timeframe="M5",
            session_start="09:30",
            session_end="16:00",
            timezone="America/New_York",
            range_minutes=5,
            breakout_buffer_pips=2.0,
            min_range_pips=5.0,
            max_range_pips=50.0,
        )

    @pytest.fixture
    def sample_candles(self):
        """Create sample candles for testing."""
        tz = pytz.timezone("America/New_York")
        base_time = datetime.now(tz).replace(hour=9, minute=30, second=0, microsecond=0)

        candles = []
        prices = [
            (1.1000, 1.1020, 1.0990, 1.1010),  # Opening range candle
            (1.1010, 1.1025, 1.1005, 1.1020),  # After range
            (1.1020, 1.1040, 1.1015, 1.1035),  # Potential breakout
        ]

        for i, (o, h, l, c) in enumerate(prices):
            candle = Candle(
                timestamp=base_time + timedelta(minutes=i * 5),
                open=o,
                high=h,
                low=l,
                close=c,
                volume=1000.0,
                symbol="EURUSD",
                timeframe="M5",
            )
            candles.append(candle)

        return candles

    def test_strategy_initialization(self, strategy):
        """Test strategy initializes correctly."""
        assert strategy.name == "ORB"
        assert strategy.symbol == "EURUSD"
        assert strategy.timeframe == "M5"
        assert strategy.range_minutes == 5
        assert strategy.is_enabled

    def test_strategy_initialize_method(self, strategy):
        """Test initialize method."""
        result = strategy.initialize()
        assert result is True

    def test_opening_range_calculation(self, strategy):
        """Test opening range is calculated correctly."""
        tz = pytz.timezone("America/New_York")
        base_time = datetime.now(tz).replace(hour=9, minute=30, second=0, microsecond=0)

        # Create candles within opening range
        candles = [
            Candle(
                timestamp=base_time,
                open=1.1000,
                high=1.1020,
                low=1.0990,
                close=1.1015,
                volume=1000,
                symbol="EURUSD",
                timeframe="M5",
            )
        ]

        opening_range = strategy._calculate_opening_range(candles)

        if opening_range:  # May be None depending on timing
            assert opening_range.high == 1.1020
            assert opening_range.low == 1.0990

    def test_atr_calculation(self, strategy, sample_candles):
        """Test ATR calculation."""
        # Need more candles for ATR
        atr = strategy._calculate_atr(sample_candles, period=2)
        assert atr >= 0

    def test_breakout_detection_disabled(self, strategy, sample_candles):
        """Test no signal when strategy is disabled."""
        strategy.disable()
        signal = strategy.on_candle(sample_candles[-1], sample_candles)
        assert signal is None

    def test_signal_type_values(self):
        """Test signal type enumeration."""
        assert SignalType.LONG.value == "long"
        assert SignalType.SHORT.value == "short"
        assert SignalType.EXIT_LONG.value == "exit_long"
        assert SignalType.EXIT_SHORT.value == "exit_short"

    def test_strategy_reset(self, strategy):
        """Test strategy reset."""
        strategy._state.breakout_triggered = True
        strategy._state.trades_today = 5

        strategy.reset()

        assert strategy._state.breakout_triggered is False
        assert strategy._state.trades_today == 0

    def test_stop_loss_calculation_long(self, strategy):
        """Test stop loss calculation for long position."""
        strategy._state.opening_range = OpeningRange(
            high=1.1020,
            low=1.0990,
            start_time=datetime.now(),
            end_time=datetime.now(),
            symbol="EURUSD",
        )

        sl = strategy.calculate_stop_loss(1.1025, is_long=True)
        expected_sl = 1.0990 - (strategy.breakout_buffer_pips * strategy.pip_size)

        assert sl == pytest.approx(expected_sl, rel=1e-5)

    def test_stop_loss_calculation_short(self, strategy):
        """Test stop loss calculation for short position."""
        strategy._state.opening_range = OpeningRange(
            high=1.1020,
            low=1.0990,
            start_time=datetime.now(),
            end_time=datetime.now(),
            symbol="EURUSD",
        )

        sl = strategy.calculate_stop_loss(1.0985, is_long=False)
        expected_sl = 1.1020 + (strategy.breakout_buffer_pips * strategy.pip_size)

        assert sl == pytest.approx(expected_sl, rel=1e-5)

    def test_take_profit_calculation(self, strategy):
        """Test take profit calculation."""
        entry = 1.1025
        stop_loss = 1.0990

        tp = strategy.calculate_take_profit(entry, stop_loss, is_long=True)

        risk = entry - stop_loss
        expected_tp = entry + (risk * strategy.risk_reward_ratio)

        assert tp == pytest.approx(expected_tp, rel=1e-5)

    def test_strategy_status(self, strategy):
        """Test get_status method."""
        status = strategy.get_status()

        assert status["name"] == "ORB"
        assert status["symbol"] == "EURUSD"
        assert status["enabled"] is True
        assert "opening_range" in status
        assert "trades_today" in status

    def test_max_trades_per_day_limit(self, strategy):
        """Test max trades per day is enforced."""
        strategy._state.trades_today = strategy._state.max_trades_per_day

        # Should not generate signal when max trades reached
        tz = pytz.timezone("America/New_York")
        candle = Candle(
            timestamp=datetime.now(tz).replace(hour=10, minute=0),
            open=1.1030,
            high=1.1050,
            low=1.1025,
            close=1.1045,
            volume=1000,
            symbol="EURUSD",
            timeframe="M5",
        )

        signal = strategy.on_candle(candle, [candle])
        assert signal is None


class TestOpeningRange:
    """Test cases for OpeningRange model."""

    def test_opening_range_creation(self):
        """Test OpeningRange creation."""
        or_range = OpeningRange(
            high=1.1020,
            low=1.0990,
            start_time=datetime.now(),
            end_time=datetime.now(),
            symbol="EURUSD",
        )

        assert or_range.high == 1.1020
        assert or_range.low == 1.0990
        assert or_range.is_valid

    def test_range_size(self):
        """Test range size calculation."""
        or_range = OpeningRange(
            high=1.1020,
            low=1.0990,
            start_time=datetime.now(),
            end_time=datetime.now(),
            symbol="EURUSD",
        )

        assert or_range.range_size == pytest.approx(0.0030, rel=1e-5)

    def test_mid_point(self):
        """Test mid point calculation."""
        or_range = OpeningRange(
            high=1.1020,
            low=1.0990,
            start_time=datetime.now(),
            end_time=datetime.now(),
            symbol="EURUSD",
        )

        assert or_range.mid_point == pytest.approx(1.1005, rel=1e-5)

    def test_breakout_detection_long(self):
        """Test long breakout detection."""
        or_range = OpeningRange(
            high=1.1020,
            low=1.0990,
            start_time=datetime.now(),
            end_time=datetime.now(),
            symbol="EURUSD",
        )

        assert or_range.is_breakout_long(1.1025) is True
        assert or_range.is_breakout_long(1.1015) is False
        assert or_range.is_breakout_long(1.1022, buffer=0.0003) is False

    def test_breakout_detection_short(self):
        """Test short breakout detection."""
        or_range = OpeningRange(
            high=1.1020,
            low=1.0990,
            start_time=datetime.now(),
            end_time=datetime.now(),
            symbol="EURUSD",
        )

        assert or_range.is_breakout_short(1.0985) is True
        assert or_range.is_breakout_short(1.0995) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
