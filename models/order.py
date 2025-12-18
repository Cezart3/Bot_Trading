"""Order data model."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class OrderType(Enum):
    """Order type enumeration."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status enumeration."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """
    Represents a trading order.

    Attributes:
        symbol: Trading symbol
        side: Buy or Sell
        order_type: Market, Limit, Stop, etc.
        quantity: Order quantity/lot size
        price: Order price (for limit orders)
        stop_price: Stop price (for stop orders)
        stop_loss: Stop loss price
        take_profit: Take profit price
        status: Current order status
        order_id: Unique order identifier
        broker_order_id: Broker's order ID
        created_at: Order creation timestamp
        filled_at: Order fill timestamp
        filled_price: Actual fill price
        filled_quantity: Filled quantity
        comment: Order comment/tag
    """

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    broker_order_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    comment: str = ""
    strategy_name: str = ""
    magic_number: int = 0

    @property
    def is_buy(self) -> bool:
        """Check if order is a buy order."""
        return self.side == OrderSide.BUY

    @property
    def is_sell(self) -> bool:
        """Check if order is a sell order."""
        return self.side == OrderSide.SELL

    @property
    def is_pending(self) -> bool:
        """Check if order is pending."""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]

    @property
    def is_filled(self) -> bool:
        """Check if order is filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_active(self) -> bool:
        """Check if order is active (not terminal state)."""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL,
        ]

    @property
    def remaining_quantity(self) -> float:
        """Calculate remaining quantity to fill."""
        return self.quantity - self.filled_quantity

    def to_dict(self) -> dict:
        """Convert order to dictionary."""
        return {
            "order_id": self.order_id,
            "broker_order_id": self.broker_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "filled_price": self.filled_price,
            "filled_quantity": self.filled_quantity,
            "comment": self.comment,
            "strategy_name": self.strategy_name,
            "magic_number": self.magic_number,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Order":
        """Create Order from dictionary."""
        return cls(
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            order_type=OrderType(data["order_type"]),
            quantity=float(data["quantity"]),
            price=data.get("price"),
            stop_price=data.get("stop_price"),
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            status=OrderStatus(data.get("status", "pending")),
            order_id=data.get("order_id", str(uuid.uuid4())),
            broker_order_id=data.get("broker_order_id"),
            comment=data.get("comment", ""),
            strategy_name=data.get("strategy_name", ""),
            magic_number=data.get("magic_number", 0),
        )

    @classmethod
    def market_buy(
        cls,
        symbol: str,
        quantity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "",
    ) -> "Order":
        """Create a market buy order."""
        return cls(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment,
        )

    @classmethod
    def market_sell(
        cls,
        symbol: str,
        quantity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "",
    ) -> "Order":
        """Create a market sell order."""
        return cls(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment,
        )

    @classmethod
    def limit_buy(
        cls,
        symbol: str,
        quantity: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "",
    ) -> "Order":
        """Create a limit buy order (buy at or below price)."""
        return cls(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment,
        )

    @classmethod
    def limit_sell(
        cls,
        symbol: str,
        quantity: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "",
    ) -> "Order":
        """Create a limit sell order (sell at or above price)."""
        return cls(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment,
        )

    @classmethod
    def stop_buy(
        cls,
        symbol: str,
        quantity: float,
        stop_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "",
    ) -> "Order":
        """Create a stop buy order (for breakout)."""
        return cls(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            quantity=quantity,
            stop_price=stop_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment,
        )

    @classmethod
    def stop_sell(
        cls,
        symbol: str,
        quantity: float,
        stop_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "",
    ) -> "Order":
        """Create a stop sell order (for breakout)."""
        return cls(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=quantity,
            stop_price=stop_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment,
        )
