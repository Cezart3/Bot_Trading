"""Tests for Risk Manager."""

import pytest
from datetime import date

from utils.risk_manager import RiskManager, DailyStats
from models.trade import Trade
from datetime import datetime, timedelta


class TestRiskManager:
    """Test cases for Risk Manager."""

    @pytest.fixture
    def risk_manager(self):
        """Create risk manager instance."""
        return RiskManager(
            max_risk_per_trade=1.0,
            max_daily_loss=3.0,
            max_positions=3,
            stop_loss_atr_multiplier=1.5,
            take_profit_atr_multiplier=2.0,
        )

    def test_initialization(self, risk_manager):
        """Test risk manager initialization."""
        assert risk_manager.max_risk_per_trade == 1.0
        assert risk_manager.max_daily_loss == 3.0
        assert risk_manager.max_positions == 3

    def test_initialize_daily_stats(self, risk_manager):
        """Test daily stats initialization."""
        risk_manager.initialize_daily_stats(10000.0)

        stats = risk_manager.get_daily_stats()
        assert stats is not None
        assert stats.starting_balance == 10000.0
        assert stats.current_balance == 10000.0
        assert stats.date == date.today()

    def test_update_balance(self, risk_manager):
        """Test balance update."""
        risk_manager.initialize_daily_stats(10000.0)
        risk_manager.update_balance(10500.0)

        stats = risk_manager.get_daily_stats()
        assert stats.current_balance == 10500.0
        assert stats.peak_balance == 10500.0

    def test_can_trade_success(self, risk_manager):
        """Test can_trade when conditions are met."""
        risk_manager.initialize_daily_stats(10000.0)

        can_trade, reason = risk_manager.can_trade(current_positions=0)
        assert can_trade is True
        assert reason == "OK"

    def test_can_trade_max_positions(self, risk_manager):
        """Test can_trade when max positions reached."""
        risk_manager.initialize_daily_stats(10000.0)

        can_trade, reason = risk_manager.can_trade(current_positions=3)
        assert can_trade is False
        assert "Max positions" in reason

    def test_can_trade_daily_loss_limit(self, risk_manager):
        """Test can_trade when daily loss limit reached."""
        risk_manager.initialize_daily_stats(10000.0)
        risk_manager.update_balance(9600.0)  # 4% loss

        can_trade, reason = risk_manager.can_trade(current_positions=0)
        assert can_trade is False
        assert "Daily loss limit" in reason

    def test_calculate_position_size(self, risk_manager):
        """Test position size calculation."""
        lot_size = risk_manager.calculate_position_size(
            account_balance=10000.0,
            stop_loss_pips=50.0,
            pip_value=10.0,  # $10 per pip per lot
            min_lot=0.01,
            max_lot=10.0,
            lot_step=0.01,
        )

        # Risk = 1% of 10000 = 100
        # Position = 100 / (50 * 10) = 0.2 lots
        assert lot_size == pytest.approx(0.2, rel=0.1)

    def test_calculate_position_size_min_lot(self, risk_manager):
        """Test position size respects minimum lot."""
        lot_size = risk_manager.calculate_position_size(
            account_balance=100.0,  # Small account
            stop_loss_pips=500.0,  # Large SL
            pip_value=10.0,
            min_lot=0.01,
            max_lot=10.0,
            lot_step=0.01,
        )

        assert lot_size >= 0.01

    def test_calculate_stop_loss_long(self, risk_manager):
        """Test stop loss calculation for long position."""
        sl = risk_manager.calculate_stop_loss(
            entry_price=1.1000,
            atr=0.0050,
            is_long=True,
            pip_size=0.0001,
        )

        expected = 1.1000 - (0.0050 * 1.5)  # ATR * multiplier
        assert sl == pytest.approx(expected, rel=1e-5)

    def test_calculate_stop_loss_short(self, risk_manager):
        """Test stop loss calculation for short position."""
        sl = risk_manager.calculate_stop_loss(
            entry_price=1.1000,
            atr=0.0050,
            is_long=False,
            pip_size=0.0001,
        )

        expected = 1.1000 + (0.0050 * 1.5)
        assert sl == pytest.approx(expected, rel=1e-5)

    def test_calculate_take_profit_long(self, risk_manager):
        """Test take profit calculation for long position."""
        tp = risk_manager.calculate_take_profit(
            entry_price=1.1000,
            atr=0.0050,
            is_long=True,
            pip_size=0.0001,
        )

        expected = 1.1000 + (0.0050 * 2.0)  # ATR * multiplier
        assert tp == pytest.approx(expected, rel=1e-5)

    def test_record_trade_win(self, risk_manager):
        """Test recording a winning trade."""
        risk_manager.initialize_daily_stats(10000.0)

        trade = Trade(
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            exit_price=1.1100,
            entry_time=datetime.now() - timedelta(hours=1),
            exit_time=datetime.now(),
            pnl=100.0,
        )

        risk_manager.record_trade(trade)

        stats = risk_manager.get_daily_stats()
        assert stats.trades_count == 1
        assert stats.wins == 1
        assert stats.total_pnl == 100.0

    def test_record_trade_loss(self, risk_manager):
        """Test recording a losing trade."""
        risk_manager.initialize_daily_stats(10000.0)

        trade = Trade(
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            exit_price=1.0950,
            entry_time=datetime.now() - timedelta(hours=1),
            exit_time=datetime.now(),
            pnl=-50.0,
        )

        risk_manager.record_trade(trade)

        stats = risk_manager.get_daily_stats()
        assert stats.trades_count == 1
        assert stats.losses == 1
        assert stats.total_pnl == -50.0

    def test_drawdown_tracking(self, risk_manager):
        """Test drawdown tracking."""
        risk_manager.initialize_daily_stats(10000.0)
        risk_manager.update_balance(10500.0)  # New peak
        risk_manager.update_balance(10200.0)  # Drawdown

        stats = risk_manager.get_daily_stats()
        # Drawdown = (10500 - 10200) / 10500 = 2.86%
        expected_dd = ((10500 - 10200) / 10500) * 100
        assert stats.max_drawdown == pytest.approx(expected_dd, rel=0.1)

    def test_get_risk_status(self, risk_manager):
        """Test risk status report."""
        risk_manager.initialize_daily_stats(10000.0)
        risk_manager.update_balance(9900.0)

        status = risk_manager.get_risk_status()

        assert "status" in status
        assert "daily_pnl" in status
        assert "daily_return_percent" in status
        assert status["status"] == "active"


class TestDailyStats:
    """Test cases for DailyStats."""

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        stats = DailyStats(
            date=date.today(),
            starting_balance=10000.0,
            current_balance=10500.0,
            trades_count=10,
            wins=6,
            losses=4,
        )

        assert stats.win_rate == 60.0

    def test_win_rate_no_trades(self):
        """Test win rate with no trades."""
        stats = DailyStats(
            date=date.today(),
            starting_balance=10000.0,
            current_balance=10000.0,
        )

        assert stats.win_rate == 0.0

    def test_daily_return(self):
        """Test daily return calculation."""
        stats = DailyStats(
            date=date.today(),
            starting_balance=10000.0,
            current_balance=10500.0,
        )

        assert stats.daily_return == 5.0

    def test_daily_loss_percent(self):
        """Test daily loss percentage calculation."""
        stats = DailyStats(
            date=date.today(),
            starting_balance=10000.0,
            current_balance=9700.0,
        )

        assert stats.daily_loss_percent == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
