"""
NinjaTrader broker implementation.
Uses socket communication to connect to NinjaTrader 8.

NOTE: This requires a custom NinjaTrader addon/indicator that listens for
commands on a TCP socket. You can use the NinjaTrader ATI (Automated Trading Interface)
or create a custom addon.
"""

import json
import socket
from datetime import datetime
from typing import Optional
import threading
import queue

from brokers.base_broker import BaseBroker
from models.candle import Candle
from models.order import Order, OrderSide, OrderStatus, OrderType
from models.position import Position, PositionStatus
from utils.logger import get_logger

logger = get_logger(__name__)


class NinjaTraderBroker(BaseBroker):
    """
    NinjaTrader 8 broker implementation.
    Communicates with NinjaTrader via TCP socket.

    Requires a NinjaTrader addon that:
    1. Listens on a TCP port for commands
    2. Executes trading operations
    3. Returns market data and account info
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        account: str = "Sim101",
        timeout: float = 10.0,
    ):
        """
        Initialize NinjaTrader broker.

        Args:
            host: NinjaTrader host address.
            port: NinjaTrader socket port.
            account: NinjaTrader account name.
            timeout: Socket timeout in seconds.
        """
        super().__init__("NinjaTrader")
        self.host = host
        self.port = port
        self.account = account
        self.timeout = timeout
        self._socket: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._response_queue = queue.Queue()

    def _send_command(self, command: dict) -> Optional[dict]:
        """
        Send command to NinjaTrader and receive response.

        Args:
            command: Command dictionary to send.

        Returns:
            Response dictionary or None if failed.
        """
        if not self._connected or not self._socket:
            logger.error("Not connected to NinjaTrader")
            return None

        with self._lock:
            try:
                # Send command as JSON
                message = json.dumps(command) + "\n"
                self._socket.sendall(message.encode("utf-8"))

                # Receive response
                response = b""
                while True:
                    chunk = self._socket.recv(4096)
                    if not chunk:
                        break
                    response += chunk
                    if b"\n" in response:
                        break

                if response:
                    return json.loads(response.decode("utf-8").strip())
                return None

            except socket.timeout:
                logger.error("Socket timeout while communicating with NinjaTrader")
                return None
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse NinjaTrader response: {e}")
                return None
            except Exception as e:
                logger.error(f"Error communicating with NinjaTrader: {e}")
                return None

    # ==================== Connection Methods ====================

    def connect(self) -> bool:
        """Establish connection to NinjaTrader."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.timeout)
            self._socket.connect((self.host, self.port))

            # Send handshake
            response = self._send_command({
                "action": "connect",
                "account": self.account,
            })

            if response and response.get("status") == "ok":
                self._connected = True
                logger.info(f"Connected to NinjaTrader at {self.host}:{self.port}")
                return True
            else:
                logger.error(f"NinjaTrader handshake failed: {response}")
                self._socket.close()
                return False

        except ConnectionRefusedError:
            logger.error(f"Connection refused - is NinjaTrader running with the socket addon?")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to NinjaTrader: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from NinjaTrader."""
        if self._socket:
            try:
                self._send_command({"action": "disconnect"})
                self._socket.close()
            except Exception:
                pass
        self._socket = None
        self._connected = False
        logger.info("Disconnected from NinjaTrader")
        return True

    def ping(self) -> bool:
        """Check if NinjaTrader connection is alive."""
        response = self._send_command({"action": "ping"})
        return response is not None and response.get("status") == "ok"

    # ==================== Account Methods ====================

    def get_account_balance(self) -> float:
        """Get account balance."""
        response = self._send_command({
            "action": "get_account",
            "account": self.account,
        })
        if response:
            return float(response.get("balance", 0))
        return 0.0

    def get_account_equity(self) -> float:
        """Get account equity."""
        response = self._send_command({
            "action": "get_account",
            "account": self.account,
        })
        if response:
            return float(response.get("equity", 0))
        return 0.0

    def get_account_margin(self) -> float:
        """Get used margin."""
        response = self._send_command({
            "action": "get_account",
            "account": self.account,
        })
        if response:
            return float(response.get("margin_used", 0))
        return 0.0

    def get_account_free_margin(self) -> float:
        """Get free margin."""
        response = self._send_command({
            "action": "get_account",
            "account": self.account,
        })
        if response:
            return float(response.get("buying_power", 0))
        return 0.0

    def get_account_info(self) -> dict:
        """Get complete account information."""
        response = self._send_command({
            "action": "get_account",
            "account": self.account,
        })
        return response if response else {}

    # ==================== Market Data Methods ====================

    def get_current_price(self, symbol: str) -> tuple[float, float]:
        """Get current bid/ask price."""
        response = self._send_command({
            "action": "get_price",
            "symbol": symbol,
        })
        if response:
            return (
                float(response.get("bid", 0)),
                float(response.get("ask", 0)),
            )
        return (0.0, 0.0)

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[Candle]:
        """Get historical candle data."""
        command = {
            "action": "get_candles",
            "symbol": symbol,
            "timeframe": timeframe,
            "count": count,
        }
        if start_time:
            command["start_time"] = start_time.isoformat()
        if end_time:
            command["end_time"] = end_time.isoformat()

        response = self._send_command(command)

        if not response or "candles" not in response:
            return []

        candles = []
        for c in response["candles"]:
            candle = Candle(
                timestamp=datetime.fromisoformat(c["timestamp"]),
                open=float(c["open"]),
                high=float(c["high"]),
                low=float(c["low"]),
                close=float(c["close"]),
                volume=float(c.get("volume", 0)),
                symbol=symbol,
                timeframe=timeframe,
            )
            candles.append(candle)

        return candles

    def get_symbol_info(self, symbol: str) -> dict:
        """Get symbol information."""
        response = self._send_command({
            "action": "get_symbol_info",
            "symbol": symbol,
        })
        return response if response else {}

    def get_symbols(self) -> list[str]:
        """Get list of available symbols."""
        response = self._send_command({"action": "get_symbols"})
        if response and "symbols" in response:
            return response["symbols"]
        return []

    # ==================== Order Methods ====================

    def place_order(self, order: Order) -> Optional[str]:
        """Place a new order."""
        command = {
            "action": "place_order",
            "account": self.account,
            "symbol": order.symbol,
            "side": "buy" if order.is_buy else "sell",
            "order_type": order.order_type.value,
            "quantity": order.quantity,
            "comment": order.comment,
        }

        if order.price:
            command["price"] = order.price
        if order.stop_price:
            command["stop_price"] = order.stop_price
        if order.stop_loss:
            command["stop_loss"] = order.stop_loss
        if order.take_profit:
            command["take_profit"] = order.take_profit

        response = self._send_command(command)

        if response and response.get("status") == "ok":
            order_id = response.get("order_id")
            order.broker_order_id = order_id
            order.status = OrderStatus.SUBMITTED
            logger.info(f"Order placed on NinjaTrader: {order_id}")
            return order_id

        logger.error(f"Failed to place order: {response}")
        return None

    def modify_order(
        self,
        order_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        price: Optional[float] = None,
    ) -> bool:
        """Modify an existing order."""
        command = {
            "action": "modify_order",
            "order_id": order_id,
        }
        if stop_loss:
            command["stop_loss"] = stop_loss
        if take_profit:
            command["take_profit"] = take_profit
        if price:
            command["price"] = price

        response = self._send_command(command)
        return response is not None and response.get("status") == "ok"

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        response = self._send_command({
            "action": "cancel_order",
            "order_id": order_id,
        })
        return response is not None and response.get("status") == "ok"

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        response = self._send_command({
            "action": "get_order",
            "order_id": order_id,
        })
        if response and "order" in response:
            o = response["order"]
            return Order(
                symbol=o["symbol"],
                side=OrderSide.BUY if o["side"] == "buy" else OrderSide.SELL,
                order_type=OrderType(o["order_type"]),
                quantity=float(o["quantity"]),
                price=o.get("price"),
                stop_loss=o.get("stop_loss"),
                take_profit=o.get("take_profit"),
                status=OrderStatus(o.get("status", "pending")),
                broker_order_id=order_id,
            )
        return None

    def get_pending_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """Get all pending orders."""
        command = {
            "action": "get_pending_orders",
            "account": self.account,
        }
        if symbol:
            command["symbol"] = symbol

        response = self._send_command(command)

        if not response or "orders" not in response:
            return []

        orders = []
        for o in response["orders"]:
            order = Order(
                symbol=o["symbol"],
                side=OrderSide.BUY if o["side"] == "buy" else OrderSide.SELL,
                order_type=OrderType(o["order_type"]),
                quantity=float(o["quantity"]),
                price=o.get("price"),
                stop_loss=o.get("stop_loss"),
                take_profit=o.get("take_profit"),
                status=OrderStatus(o.get("status", "pending")),
                broker_order_id=o["order_id"],
            )
            orders.append(order)

        return orders

    # ==================== Position Methods ====================

    def get_positions(self, symbol: Optional[str] = None) -> list[Position]:
        """Get all open positions."""
        command = {
            "action": "get_positions",
            "account": self.account,
        }
        if symbol:
            command["symbol"] = symbol

        response = self._send_command(command)

        if not response or "positions" not in response:
            return []

        positions = []
        for p in response["positions"]:
            position = Position(
                symbol=p["symbol"],
                side="long" if p["side"] == "long" else "short",
                quantity=float(p["quantity"]),
                entry_price=float(p["entry_price"]),
                current_price=float(p.get("current_price", 0)),
                stop_loss=p.get("stop_loss"),
                take_profit=p.get("take_profit"),
                status=PositionStatus.OPEN,
                broker_position_id=p["position_id"],
                pnl=float(p.get("unrealized_pnl", 0)),
            )
            positions.append(position)

        return positions

    def get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID."""
        response = self._send_command({
            "action": "get_position",
            "position_id": position_id,
        })
        if response and "position" in response:
            p = response["position"]
            return Position(
                symbol=p["symbol"],
                side="long" if p["side"] == "long" else "short",
                quantity=float(p["quantity"]),
                entry_price=float(p["entry_price"]),
                current_price=float(p.get("current_price", 0)),
                stop_loss=p.get("stop_loss"),
                take_profit=p.get("take_profit"),
                status=PositionStatus.OPEN,
                broker_position_id=position_id,
                pnl=float(p.get("unrealized_pnl", 0)),
            )
        return None

    def close_position(
        self, position_id: str, volume: Optional[float] = None
    ) -> bool:
        """Close an open position."""
        command = {
            "action": "close_position",
            "position_id": position_id,
        }
        if volume:
            command["volume"] = volume

        response = self._send_command(command)
        return response is not None and response.get("status") == "ok"

    def modify_position(
        self,
        position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        """Modify position's SL/TP."""
        command = {
            "action": "modify_position",
            "position_id": position_id,
        }
        if stop_loss:
            command["stop_loss"] = stop_loss
        if take_profit:
            command["take_profit"] = take_profit

        response = self._send_command(command)
        return response is not None and response.get("status") == "ok"

    def get_server_time(self) -> datetime:
        """Get NinjaTrader server time."""
        response = self._send_command({"action": "get_time"})
        if response and "time" in response:
            return datetime.fromisoformat(response["time"])
        return datetime.now()
