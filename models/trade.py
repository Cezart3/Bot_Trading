"""Trade (closed position) data model for history and analysis."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class TradeResult(Enum):
    """Trade result enumeration."""

    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


@dataclass
class Trade:
    """
    Represents a completed trade (closed position).
    Used for trade history and performance analysis.

    Attributes:
        symbol: Trading symbol
        side: Long or Short
        quantity: Position size
        entry_price: Entry price
        exit_price: Exit price
        entry_time: Entry timestamp
        exit_time: Exit timestamp
        pnl: Realized P&L
        pnl_percent: P&L as percentage
        commission: Total commission
        swap: Swap charges
        stop_loss: Stop loss used
        take_profit: Take profit used
        result: Win/Loss/Breakeven
        exit_reason: Why trade was closed
        strategy_name: Strategy that generated the trade
    """

    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    result: TradeResult = TradeResult.BREAKEVEN
    exit_reason: str = ""  # "tp", "sl", "manual", "trailing", "session_end"
    strategy_name: str = ""
    magic_number: int = 0
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    notes: str = ""

    def __post_init__(self):
        """Determine trade result after initialization."""
        if self.pnl > 0:
            self.result = TradeResult.WIN
        elif self.pnl < 0:
            self.result = TradeResult.LOSS
        else:
            self.result = TradeResult.BREAKEVEN

    @property
    def is_win(self) -> bool:
        """Check if trade was a win."""
        return self.result == TradeResult.WIN

    @property
    def is_loss(self) -> bool:
        """Check if trade was a loss."""
        return self.result == TradeResult.LOSS

    @property
    def duration(self) -> float:
        """Calculate trade duration in seconds."""
        return (self.exit_time - self.entry_time).total_seconds()

    @property
    def duration_minutes(self) -> float:
        """Calculate trade duration in minutes."""
        return self.duration / 60

    @property
    def net_pnl(self) -> float:
        """Calculate net P&L (after commission and swap)."""
        return self.pnl - self.commission - self.swap

    @property
    def r_multiple(self) -> Optional[float]:
        """Calculate R-multiple (risk multiple) if SL was set."""
        if not self.stop_loss:
            return None
        risk = abs(self.entry_price - self.stop_loss) * self.quantity
        if risk == 0:
            return None
        return self.pnl / risk

    def to_dict(self) -> dict:
        """Convert trade to dictionary."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "net_pnl": self.net_pnl,
            "commission": self.commission,
            "swap": self.swap,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "result": self.result.value,
            "exit_reason": self.exit_reason,
            "strategy_name": self.strategy_name,
            "magic_number": self.magic_number,
            "duration_minutes": self.duration_minutes,
            "r_multiple": self.r_multiple,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trade":
        """Create Trade from dictionary."""
        entry_time = data["entry_time"]
        exit_time = data["exit_time"]
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)
        if isinstance(exit_time, str):
            exit_time = datetime.fromisoformat(exit_time)

        return cls(
            symbol=data["symbol"],
            side=data["side"],
            quantity=float(data["quantity"]),
            entry_price=float(data["entry_price"]),
            exit_price=float(data["exit_price"]),
            entry_time=entry_time,
            exit_time=exit_time,
            pnl=float(data["pnl"]),
            pnl_percent=float(data.get("pnl_percent", 0)),
            commission=float(data.get("commission", 0)),
            swap=float(data.get("swap", 0)),
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            exit_reason=data.get("exit_reason", ""),
            strategy_name=data.get("strategy_name", ""),
            magic_number=data.get("magic_number", 0),
            trade_id=data.get("trade_id", str(uuid.uuid4())),
            notes=data.get("notes", ""),
        )
