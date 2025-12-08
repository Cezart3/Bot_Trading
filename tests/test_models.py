"""Tests for data models."""

import pytest
from datetime import datetime, timedelta

from models.candle import Candle
from models.order import Order, OrderSide, OrderType, OrderStatus
from models.position import Position, PositionStatus
from models.trade import Trade, TradeResult


class TestCandle:
    """Test cases for Candle model."""

    @pytest.fixture
    def bullish_candle(self):
        """Create a bullish candle."""
        return Candle(
            timestamp=datetime.now(),
            open=1.1000,
            high=1.1050,
            low=1.0980,
            close=1.1040,
            volume=1000.0,
            symbol="EURUSD",
            timeframe="M5",
        )

    @pytest.fixture
    def bearish_candle(self):
        """Create a bearish candle."""
        return Candle(
            timestamp=datetime.now(),
            open=1.1040,
            high=1.1050,
            low=1.0980,
            close=1.1000,
            volume=1000.0,
            symbol="EURUSD",
            timeframe="M5",
        )

    def test_bullish_detection(self, bullish_candle):
        """Test bullish candle detection."""
        assert bullish_candle.is_bullish is True
        assert bullish_candle.is_bearish is False

    def test_bearish_detection(self, bearish_candle):
        """Test bearish candle detection."""
        assert bearish_candle.is_bearish is True
        assert bearish_candle.is_bullish is False

    def test_body_size(self, bullish_candle):
        """Test body size calculation."""
        expected = abs(1.1040 - 1.1000)
        assert bullish_candle.body_size == pytest.approx(expected, rel=1e-5)

    def test_total_range(self, bullish_candle):
        """Test total range calculation."""
        expected = 1.1050 - 1.0980
        assert bullish_candle.total_range == pytest.approx(expected, rel=1e-5)

    def test_upper_wick(self, bullish_candle):
        """Test upper wick calculation."""
        expected = 1.1050 - 1.1040  # high - close (bullish)
        assert bullish_candle.upper_wick == pytest.approx(expected, rel=1e-5)

    def test_lower_wick(self, bullish_candle):
        """Test lower wick calculation."""
        expected = 1.1000 - 1.0980  # open - low (bullish)
        assert bullish_candle.lower_wick == pytest.approx(expected, rel=1e-5)

    def test_mid_price(self, bullish_candle):
        """Test mid price calculation."""
        expected = (1.1050 + 1.0980) / 2
        assert bullish_candle.mid_price == pytest.approx(expected, rel=1e-5)

    def test_contains_price(self, bullish_candle):
        """Test price containment check."""
        assert bullish_candle.contains_price(1.1020) is True
        assert bullish_candle.contains_price(1.1100) is False
        assert bullish_candle.contains_price(1.0950) is False

    def test_to_dict(self, bullish_candle):
        """Test dictionary conversion."""
        data = bullish_candle.to_dict()
        assert data["open"] == 1.1000
        assert data["high"] == 1.1050
        assert data["low"] == 1.0980
        assert data["close"] == 1.1040
        assert data["symbol"] == "EURUSD"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "timestamp": "2024-01-15T10:00:00",
            "open": 1.1000,
            "high": 1.1050,
            "low": 1.0980,
            "close": 1.1040,
            "volume": 1000,
            "symbol": "EURUSD",
            "timeframe": "M5",
        }
        candle = Candle.from_dict(data)
        assert candle.open == 1.1000
        assert candle.symbol == "EURUSD"


class TestOrder:
    """Test cases for Order model."""

    def test_market_buy_order(self):
        """Test market buy order creation."""
        order = Order.market_buy(
            symbol="EURUSD",
            quantity=0.1,
            stop_loss=1.0950,
            take_profit=1.1100,
            comment="Test order",
        )

        assert order.symbol == "EURUSD"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 0.1
        assert order.is_buy is True
        assert order.is_sell is False

    def test_market_sell_order(self):
        """Test market sell order creation."""
        order = Order.market_sell(
            symbol="EURUSD",
            quantity=0.1,
            stop_loss=1.1100,
            take_profit=1.0950,
        )

        assert order.side == OrderSide.SELL
        assert order.is_sell is True
        assert order.is_buy is False

    def test_stop_buy_order(self):
        """Test stop buy order creation."""
        order = Order.stop_buy(
            symbol="EURUSD",
            quantity=0.1,
            stop_price=1.1050,
            stop_loss=1.0990,
            take_profit=1.1150,
        )

        assert order.order_type == OrderType.STOP
        assert order.stop_price == 1.1050

    def test_order_status(self):
        """Test order status properties."""
        order = Order.market_buy("EURUSD", 0.1)

        assert order.is_pending is True
        assert order.is_filled is False
        assert order.is_active is True

        order.status = OrderStatus.FILLED
        assert order.is_filled is True
        assert order.is_pending is False

    def test_remaining_quantity(self):
        """Test remaining quantity calculation."""
        order = Order.market_buy("EURUSD", 0.1)
        order.filled_quantity = 0.03

        assert order.remaining_quantity == pytest.approx(0.07, rel=1e-5)

    def test_order_to_dict(self):
        """Test order dictionary conversion."""
        order = Order.market_buy("EURUSD", 0.1, stop_loss=1.0950)
        data = order.to_dict()

        assert data["symbol"] == "EURUSD"
        assert data["side"] == "buy"
        assert data["order_type"] == "market"
        assert data["stop_loss"] == 1.0950


class TestPosition:
    """Test cases for Position model."""

    @pytest.fixture
    def long_position(self):
        """Create a long position."""
        return Position(
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050,
            stop_loss=1.0950,
            take_profit=1.1100,
        )

    @pytest.fixture
    def short_position(self):
        """Create a short position."""
        return Position(
            symbol="EURUSD",
            side="short",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.0950,
            stop_loss=1.1050,
            take_profit=1.0900,
        )

    def test_long_position_properties(self, long_position):
        """Test long position properties."""
        assert long_position.is_long is True
        assert long_position.is_short is False
        assert long_position.is_open is True

    def test_short_position_properties(self, short_position):
        """Test short position properties."""
        assert short_position.is_short is True
        assert short_position.is_long is False

    def test_unrealized_pnl_long(self, long_position):
        """Test unrealized PnL for long position."""
        # Price moved from 1.1000 to 1.1050, qty 0.1
        expected_pnl = (1.1050 - 1.1000) * 0.1
        assert long_position.unrealized_pnl == pytest.approx(expected_pnl, rel=1e-5)

    def test_unrealized_pnl_short(self, short_position):
        """Test unrealized PnL for short position."""
        # Price moved from 1.1000 to 1.0950 (favorable for short), qty 0.1
        expected_pnl = (1.1000 - 1.0950) * 0.1
        assert short_position.unrealized_pnl == pytest.approx(expected_pnl, rel=1e-5)

    def test_risk_reward_ratio(self, long_position):
        """Test risk/reward ratio calculation."""
        # Entry: 1.1000, SL: 1.0950, TP: 1.1100
        # Risk: 50 pips, Reward: 100 pips
        rr = long_position.risk_reward_ratio
        assert rr == pytest.approx(2.0, rel=1e-2)

    def test_close_position(self, long_position):
        """Test position close."""
        long_position.close(exit_price=1.1050, pnl=50.0)

        assert long_position.status == PositionStatus.CLOSED
        assert long_position.exit_price == 1.1050
        assert long_position.pnl == 50.0
        assert long_position.closed_at is not None


class TestTrade:
    """Test cases for Trade model."""

    def test_winning_trade(self):
        """Test winning trade creation."""
        trade = Trade(
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            exit_price=1.1100,
            entry_time=datetime.now() - timedelta(hours=1),
            exit_time=datetime.now(),
            pnl=100.0,
            pnl_percent=0.5,
            stop_loss=1.0950,
            take_profit=1.1100,
            exit_reason="tp",
        )

        assert trade.result == TradeResult.WIN
        assert trade.is_win is True
        assert trade.is_loss is False

    def test_losing_trade(self):
        """Test losing trade creation."""
        trade = Trade(
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            exit_price=1.0950,
            entry_time=datetime.now() - timedelta(minutes=30),
            exit_time=datetime.now(),
            pnl=-50.0,
            stop_loss=1.0950,
            exit_reason="sl",
        )

        assert trade.result == TradeResult.LOSS
        assert trade.is_loss is True

    def test_trade_duration(self):
        """Test trade duration calculation."""
        entry = datetime.now() - timedelta(hours=2)
        exit = datetime.now()

        trade = Trade(
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            exit_price=1.1050,
            entry_time=entry,
            exit_time=exit,
            pnl=50.0,
        )

        assert trade.duration_minutes == pytest.approx(120, rel=0.1)

    def test_net_pnl(self):
        """Test net PnL calculation."""
        trade = Trade(
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            exit_price=1.1050,
            entry_time=datetime.now() - timedelta(hours=1),
            exit_time=datetime.now(),
            pnl=50.0,
            commission=5.0,
            swap=2.0,
        )

        assert trade.net_pnl == pytest.approx(43.0, rel=1e-5)

    def test_r_multiple(self):
        """Test R-multiple calculation."""
        trade = Trade(
            symbol="EURUSD",
            side="long",
            quantity=1.0,
            entry_price=1.1000,
            exit_price=1.1100,
            entry_time=datetime.now() - timedelta(hours=1),
            exit_time=datetime.now(),
            pnl=0.01,  # 100 pips profit
            stop_loss=1.0950,  # 50 pips risk
        )

        # R-multiple = PnL / Risk
        # Risk = (1.1000 - 1.0950) * 1.0 = 0.005
        # R = 0.01 / 0.005 = 2.0
        assert trade.r_multiple == pytest.approx(2.0, rel=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
