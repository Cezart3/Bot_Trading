"""Abstract base class for broker integrations."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from models.candle import Candle
from models.order import Order, OrderSide, OrderType
from models.position import Position


class BaseBroker(ABC):
    """
    Abstract base class defining the interface for all broker integrations.
    All broker implementations (MT5, NinjaTrader, etc.) must inherit from this class.
    """

    def __init__(self, name: str):
        """Initialize broker with name."""
        self.name = name
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if broker is connected."""
        return self._connected

    # ==================== Connection Methods ====================

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the broker.

        Returns:
            bool: True if connection successful, False otherwise.
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the broker.

        Returns:
            bool: True if disconnection successful, False otherwise.
        """
        pass

    @abstractmethod
    def ping(self) -> bool:
        """
        Check if connection is alive.

        Returns:
            bool: True if connection is alive, False otherwise.
        """
        pass

    # ==================== Account Methods ====================

    @abstractmethod
    def get_account_balance(self) -> float:
        """
        Get current account balance.

        Returns:
            float: Account balance.
        """
        pass

    @abstractmethod
    def get_account_equity(self) -> float:
        """
        Get current account equity.

        Returns:
            float: Account equity.
        """
        pass

    @abstractmethod
    def get_account_margin(self) -> float:
        """
        Get used margin.

        Returns:
            float: Used margin.
        """
        pass

    @abstractmethod
    def get_account_free_margin(self) -> float:
        """
        Get free margin available.

        Returns:
            float: Free margin.
        """
        pass

    @abstractmethod
    def get_account_info(self) -> dict:
        """
        Get complete account information.

        Returns:
            dict: Account information including balance, equity, margin, etc.
        """
        pass

    # ==================== Market Data Methods ====================

    @abstractmethod
    def get_current_price(self, symbol: str) -> tuple[float, float]:
        """
        Get current bid/ask price for a symbol.

        Args:
            symbol: Trading symbol.

        Returns:
            tuple: (bid, ask) prices.
        """
        pass

    @abstractmethod
    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[Candle]:
        """
        Get historical candle data.

        Args:
            symbol: Trading symbol.
            timeframe: Candle timeframe (e.g., 'M1', 'M5', 'H1', 'D1').
            count: Number of candles to retrieve.
            start_time: Start time for data retrieval.
            end_time: End time for data retrieval.

        Returns:
            list[Candle]: List of candle objects.
        """
        pass

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> dict:
        """
        Get symbol information (pip value, lot size, etc.).

        Args:
            symbol: Trading symbol.

        Returns:
            dict: Symbol information.
        """
        pass

    @abstractmethod
    def get_symbols(self) -> list[str]:
        """
        Get list of available symbols.

        Returns:
            list[str]: List of symbol names.
        """
        pass

    # ==================== Order Methods ====================

    @abstractmethod
    def place_order(self, order: Order) -> Optional[str]:
        """
        Place a new order.

        Args:
            order: Order object with all order details.

        Returns:
            Optional[str]: Broker order ID if successful, None otherwise.
        """
        pass

    @abstractmethod
    def modify_order(
        self,
        order_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        price: Optional[float] = None,
    ) -> bool:
        """
        Modify an existing order.

        Args:
            order_id: Broker order ID.
            stop_loss: New stop loss price.
            take_profit: New take profit price.
            price: New order price (for pending orders).

        Returns:
            bool: True if modification successful, False otherwise.
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Broker order ID.

        Returns:
            bool: True if cancellation successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.

        Args:
            order_id: Broker order ID.

        Returns:
            Optional[Order]: Order object if found, None otherwise.
        """
        pass

    @abstractmethod
    def get_pending_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """
        Get all pending orders.

        Args:
            symbol: Filter by symbol (optional).

        Returns:
            list[Order]: List of pending orders.
        """
        pass

    # ==================== Position Methods ====================

    @abstractmethod
    def get_positions(self, symbol: Optional[str] = None) -> list[Position]:
        """
        Get all open positions.

        Args:
            symbol: Filter by symbol (optional).

        Returns:
            list[Position]: List of open positions.
        """
        pass

    @abstractmethod
    def get_position(self, position_id: str) -> Optional[Position]:
        """
        Get position by ID.

        Args:
            position_id: Broker position ID.

        Returns:
            Optional[Position]: Position object if found, None otherwise.
        """
        pass

    @abstractmethod
    def close_position(
        self, position_id: str, volume: Optional[float] = None
    ) -> bool:
        """
        Close an open position.

        Args:
            position_id: Broker position ID.
            volume: Volume to close (None for full close).

        Returns:
            bool: True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def modify_position(
        self,
        position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        """
        Modify position's SL/TP.

        Args:
            position_id: Broker position ID.
            stop_loss: New stop loss price.
            take_profit: New take profit price.

        Returns:
            bool: True if modification successful, False otherwise.
        """
        pass

    # ==================== Utility Methods ====================

    def calculate_lot_size(
        self,
        symbol: str,
        risk_amount: float,
        stop_loss_pips: float,
    ) -> float:
        """
        Calculate position size based on risk.

        Args:
            symbol: Trading symbol.
            risk_amount: Amount to risk in account currency.
            stop_loss_pips: Stop loss distance in pips.

        Returns:
            float: Calculated lot size.
        """
        symbol_info = self.get_symbol_info(symbol)
        pip_value = symbol_info.get("pip_value", 10)
        min_lot = symbol_info.get("volume_min", 0.01)
        max_lot = symbol_info.get("volume_max", 100)
        lot_step = symbol_info.get("volume_step", 0.01)

        if stop_loss_pips <= 0:
            return min_lot

        lot_size = risk_amount / (stop_loss_pips * pip_value)

        # Round to lot step
        lot_size = round(lot_size / lot_step) * lot_step

        # Clamp to min/max
        lot_size = max(min_lot, min(max_lot, lot_size))

        return round(lot_size, 2)

    @abstractmethod
    def get_server_time(self) -> datetime:
        """
        Get broker server time.

        Returns:
            datetime: Server time.
        """
        pass
