"""Position data model."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class PositionStatus(Enum):
    """Position status enumeration."""

    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"


@dataclass
class Position:
    """
    Represents an open trading position.

    Attributes:
        symbol: Trading symbol
        side: Long or Short (buy/sell)
        quantity: Position size
        entry_price: Average entry price
        current_price: Current market price
        stop_loss: Stop loss price
        take_profit: Take profit price
        status: Position status
        position_id: Unique position identifier
        broker_position_id: Broker's position ID
        opened_at: Position open timestamp
        closed_at: Position close timestamp
        exit_price: Exit price if closed
        pnl: Realized profit/loss
        commission: Trading commission
    """

    symbol: str
    side: str  # "long" or "short"
    quantity: float
    entry_price: float
    current_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    broker_position_id: Optional[str] = None
    opened_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    comment: str = ""
    strategy_name: str = ""
    magic_number: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.side.lower() == "long"

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.side.lower() == "short"

    @property
    def is_open(self) -> bool:
        """Check if position is open."""
        return self.status == PositionStatus.OPEN

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.current_price == 0:
            return 0.0
        if self.is_long:
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl_percent(self) -> float:
        """Calculate unrealized P&L as percentage."""
        if self.entry_price == 0:
            return 0.0
        if self.is_long:
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.current_price) / self.entry_price) * 100

    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk/reward ratio if SL and TP are set."""
        if not self.stop_loss or not self.take_profit:
            return None
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else None

    @property
    def distance_to_sl(self) -> Optional[float]:
        """Calculate distance to stop loss."""
        if not self.stop_loss or self.current_price == 0:
            return None
        if self.is_long:
            return self.current_price - self.stop_loss
        else:
            return self.stop_loss - self.current_price

    @property
    def distance_to_tp(self) -> Optional[float]:
        """Calculate distance to take profit."""
        if not self.take_profit or self.current_price == 0:
            return None
        if self.is_long:
            return self.take_profit - self.current_price
        else:
            return self.current_price - self.take_profit

    def update_price(self, price: float) -> None:
        """Update current price."""
        self.current_price = price

    def close(self, exit_price: float, pnl: float) -> None:
        """Close the position."""
        self.exit_price = exit_price
        self.pnl = pnl
        self.status = PositionStatus.CLOSED
        self.closed_at = datetime.now()

    def to_dict(self) -> dict:
        """Convert position to dictionary."""
        return {
            "position_id": self.position_id,
            "broker_position_id": self.broker_position_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "status": self.status.value,
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "commission": self.commission,
            "swap": self.swap,
            "comment": self.comment,
            "strategy_name": self.strategy_name,
            "magic_number": self.magic_number,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Position":
        """Create Position from dictionary."""
        opened_at = data.get("opened_at")
        if isinstance(opened_at, str):
            opened_at = datetime.fromisoformat(opened_at)
        elif opened_at is None:
            opened_at = datetime.now()

        return cls(
            symbol=data["symbol"],
            side=data["side"],
            quantity=float(data["quantity"]),
            entry_price=float(data["entry_price"]),
            current_price=float(data.get("current_price", 0)),
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            status=PositionStatus(data.get("status", "open")),
            position_id=data.get("position_id", str(uuid.uuid4())),
            broker_position_id=data.get("broker_position_id"),
            opened_at=opened_at,
            comment=data.get("comment", ""),
            strategy_name=data.get("strategy_name", ""),
            magic_number=data.get("magic_number", 0),
            metadata=data.get("metadata", {}),
        )
