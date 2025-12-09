"""Tests for ORB Forex Strategy - London Session."""

import pytest
from datetime import datetime, time
from strategies.orb_forex_strategy import (
    ORBForexStrategy,
    ForexORBConfig,
    ORBState,
    OpeningRange,
    create_eurusd_london_strategy,
)
from strategies.base_strategy import SignalType
from models.candle import Candle


class TestForexORBConfig:
    """Test ForexORBConfig defaults and customization."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ForexORBConfig()
        assert config.session_name == "LONDON"
        assert config.session_start_hour == 10
        assert config.session_end_hour == 18
        assert config.orb_duration_minutes == 60
        assert config.risk_reward_ratio == 2.0
        assert config.min_orb_range_pips == 10.0
        assert config.max_orb_range_pips == 40.0
        assert config.breakout_buffer_pips == 2.0
        assert config.max_trades_per_session == 1

    def test_custom_config(self):
        """Test custom configuration."""
        config = ForexORBConfig(
            session_name="CUSTOM",
            session_start_hour=9,
            risk_reward_ratio=3.0,
            min_orb_range_pips=15.0,
        )
        assert config.session_name == "CUSTOM"
        assert config.session_start_hour == 9
        assert config.risk_reward_ratio == 3.0
        assert config.min_orb_range_pips == 15.0


class TestORBForexStrategy:
    """Test ORB Forex Strategy logic."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return create_eurusd_london_strategy()

    @pytest.fixture
    def initialized_strategy(self):
        """Create and initialize strategy."""
        strategy = create_eurusd_london_strategy()
        strategy.initialize()
        return strategy

    def test_strategy_initialization(self, strategy):
        """Test strategy initializes correctly."""
        assert strategy.name == "ORB_FOREX"
        assert strategy.symbol == "EURUSD"
        assert strategy.timeframe == "M5"
        assert strategy.pip_size == 0.0001
        assert strategy._state == ORBState.WAITING_FOR_SESSION

    def test_initialize_method(self, strategy):
        """Test initialize method."""
        result = strategy.initialize()
        assert result is True

    def test_session_time_detection(self, initialized_strategy):
        """Test session time detection."""
        # Within session (11:00)
        ts_within = datetime(2025, 12, 9, 11, 0)
        assert initialized_strategy._is_session_time(ts_within) is True

        # Before session (9:00)
        ts_before = datetime(2025, 12, 9, 9, 0)
        assert initialized_strategy._is_session_time(ts_before) is False

        # After session (19:00)
        ts_after = datetime(2025, 12, 9, 19, 0)
        assert initialized_strategy._is_session_time(ts_after) is False

    def test_orb_period_detection(self, initialized_strategy):
        """Test ORB period detection."""
        # Within ORB (10:30)
        ts_orb = datetime(2025, 12, 9, 10, 30)
        assert initialized_strategy._is_orb_period(ts_orb) is True

        # After ORB (11:30)
        ts_after_orb = datetime(2025, 12, 9, 11, 30)
        assert initialized_strategy._is_orb_period(ts_after_orb) is False

    def test_orb_valid_range_filter(self, initialized_strategy):
        """Test ORB range validation."""
        # Valid range (20 pips)
        initialized_strategy._opening_range = OpeningRange(
            high=1.1020,
            low=1.1000,  # 20 pips range
            start_time=datetime.now(),
            end_time=datetime.now(),
            candle_count=12,
        )
        assert initialized_strategy._check_orb_valid() is True

        # Too small range (5 pips < 10 min)
        initialized_strategy._opening_range = OpeningRange(
            high=1.1005,
            low=1.1000,  # 5 pips range
            start_time=datetime.now(),
            end_time=datetime.now(),
            candle_count=12,
        )
        assert initialized_strategy._check_orb_valid() is False

        # Too large range (50 pips > 40 max)
        initialized_strategy._opening_range = OpeningRange(
            high=1.1050,
            low=1.1000,  # 50 pips range
            start_time=datetime.now(),
            end_time=datetime.now(),
            candle_count=12,
        )
        assert initialized_strategy._check_orb_valid() is False

    def test_breakout_signal_long(self, initialized_strategy):
        """Test LONG breakout signal generation with buffer."""
        # Setup OR
        initialized_strategy._state = ORBState.ORB_COMPLETE
        initialized_strategy._opening_range = OpeningRange(
            high=1.1020,
            low=1.1000,
            start_time=datetime.now(),
            end_time=datetime.now(),
            candle_count=12,
        )

        # Candle that closes above OR high + buffer (2 pips)
        # OR high = 1.1020, buffer = 2 pips = 0.0002
        # Need close > 1.1022
        breakout_candle = Candle(
            timestamp=datetime(2025, 12, 9, 11, 30),
            open=1.1018,
            high=1.1030,
            low=1.1015,
            close=1.1025,  # > 1.1022
            volume=100,
            symbol="EURUSD",
            timeframe="M5",
        )

        signal = initialized_strategy._check_breakout(breakout_candle)
        assert signal is not None
        assert signal.signal_type == SignalType.LONG
        assert signal.price == 1.1025
        assert signal.stop_loss < 1.1000  # Below OR low with buffer

    def test_breakout_signal_short(self, initialized_strategy):
        """Test SHORT breakout signal generation with buffer."""
        # Setup OR
        initialized_strategy._state = ORBState.ORB_COMPLETE
        initialized_strategy._opening_range = OpeningRange(
            high=1.1020,
            low=1.1000,
            start_time=datetime.now(),
            end_time=datetime.now(),
            candle_count=12,
        )

        # Candle that closes below OR low - buffer (2 pips)
        # OR low = 1.1000, buffer = 2 pips = 0.0002
        # Need close < 1.0998
        breakout_candle = Candle(
            timestamp=datetime(2025, 12, 9, 11, 30),
            open=1.1002,
            high=1.1005,
            low=1.0990,
            close=1.0995,  # < 1.0998
            volume=100,
            symbol="EURUSD",
            timeframe="M5",
        )

        signal = initialized_strategy._check_breakout(breakout_candle)
        assert signal is not None
        assert signal.signal_type == SignalType.SHORT
        assert signal.price == 1.0995
        assert signal.stop_loss > 1.1020  # Above OR high with buffer

    def test_no_signal_within_range(self, initialized_strategy):
        """Test no signal when price within OR range."""
        initialized_strategy._state = ORBState.ORB_COMPLETE
        initialized_strategy._opening_range = OpeningRange(
            high=1.1020,
            low=1.1000,
            start_time=datetime.now(),
            end_time=datetime.now(),
            candle_count=12,
        )

        # Candle within range (no breakout)
        no_breakout_candle = Candle(
            timestamp=datetime(2025, 12, 9, 11, 30),
            open=1.1008,
            high=1.1015,
            low=1.1005,
            close=1.1012,  # Within range
            volume=100,
            symbol="EURUSD",
            timeframe="M5",
        )

        signal = initialized_strategy._check_breakout(no_breakout_candle)
        assert signal is None

    def test_get_status(self, initialized_strategy):
        """Test get_status method."""
        status = initialized_strategy.get_status()
        assert status["strategy"] == "ORB_FOREX"
        assert status["symbol"] == "EURUSD"
        assert status["session"] == "LONDON"
        assert "state" in status
        assert "config" in status

    def test_strategy_reset(self, initialized_strategy):
        """Test strategy reset."""
        initialized_strategy._state = ORBState.IN_TRADE
        initialized_strategy._trades_today = 1
        initialized_strategy._opening_range = OpeningRange(
            high=1.1020, low=1.1000,
            start_time=datetime.now(), end_time=datetime.now(),
            candle_count=12,
        )

        initialized_strategy._reset_session()

        assert initialized_strategy._state == ORBState.WAITING_FOR_SESSION
        assert initialized_strategy._trades_today == 0
        assert initialized_strategy._opening_range is None


class TestRiskCalculation:
    """Test risk and lot size calculations."""

    def test_lot_size_calculation_100k_05_percent(self):
        """Test lot size for $100k account with 0.5% risk."""
        account_balance = 100_000.0
        risk_percent = 0.5
        pip_value_per_lot = 10.0

        # Risk amount should be $500
        risk_amount = account_balance * (risk_percent / 100)
        assert risk_amount == 500.0

        # For 30 pip SL, lot size should be 1.67
        sl_pips = 30
        lot_size = risk_amount / (sl_pips * pip_value_per_lot)
        assert round(lot_size, 2) == 1.67

        # Verify actual risk
        actual_risk = sl_pips * pip_value_per_lot * round(lot_size, 2)
        assert abs(actual_risk - 500) < 5  # Within $5 tolerance

    def test_lot_size_different_sl_distances(self):
        """Test lot size calculation for different SL distances."""
        account_balance = 100_000.0
        risk_percent = 0.5
        pip_value_per_lot = 10.0
        risk_amount = 500.0

        test_cases = [
            (20, 2.50),  # 20 pip SL -> 2.5 lots
            (25, 2.00),  # 25 pip SL -> 2.0 lots
            (30, 1.67),  # 30 pip SL -> 1.67 lots
            (40, 1.25),  # 40 pip SL -> 1.25 lots
            (50, 1.00),  # 50 pip SL -> 1.0 lots
        ]

        for sl_pips, expected_lots in test_cases:
            lot_size = risk_amount / (sl_pips * pip_value_per_lot)
            lot_size = round(lot_size, 2)
            assert lot_size == expected_lots, f"SL {sl_pips} pips: expected {expected_lots}, got {lot_size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
