"""MetaTrader 5 broker implementation."""

from datetime import datetime
from typing import Optional
import pandas as pd

from brokers.base_broker import BaseBroker
from models.candle import Candle
from models.order import Order, OrderSide, OrderStatus, OrderType
from models.position import Position, PositionStatus
from utils.logger import get_logger

logger = get_logger(__name__)

# MT5 timeframe mapping
TIMEFRAME_MAP = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 10080,
    "MN1": 43200,
}


class MT5Broker(BaseBroker):
    """
    MetaTrader 5 broker implementation.
    Provides connection and trading functionality for MT5 platform.
    """

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        path: Optional[str] = None,
        timeout: int = 60000,
    ):
        """
        Initialize MT5 broker.

        Args:
            login: MT5 account number.
            password: MT5 account password.
            server: MT5 broker server name.
            path: Path to MT5 terminal (optional).
            timeout: Connection timeout in milliseconds.
        """
        super().__init__("MetaTrader5")
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.timeout = timeout
        self._mt5 = None

    def _get_mt5(self):
        """Lazy import of MetaTrader5 module."""
        if self._mt5 is None:
            try:
                import MetaTrader5 as mt5
                self._mt5 = mt5
            except ImportError:
                logger.error("MetaTrader5 package not installed. Run: pip install MetaTrader5")
                raise
        return self._mt5

    def _get_mt5_timeframe(self, timeframe: str) -> int:
        """Convert string timeframe to MT5 constant."""
        mt5 = self._get_mt5()
        timeframe_constants = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1,
        }
        return timeframe_constants.get(timeframe.upper(), mt5.TIMEFRAME_M5)

    def _get_filling_type(self, symbol: str) -> int:
        """Get the correct filling type for a symbol."""
        mt5 = self._get_mt5()
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return mt5.ORDER_FILLING_IOC

        # Check which filling modes are supported
        filling_mode = symbol_info.filling_mode
        if filling_mode & 1:  # FOK supported
            return mt5.ORDER_FILLING_FOK
        elif filling_mode & 2:  # IOC supported
            return mt5.ORDER_FILLING_IOC
        else:  # Return/BOC
            return mt5.ORDER_FILLING_RETURN

    # ==================== Connection Methods ====================

    def connect(self) -> bool:
        """Establish connection to MT5."""
        mt5 = self._get_mt5()

        # Initialize MT5
        init_kwargs = {"timeout": self.timeout}
        if self.path:
            init_kwargs["path"] = self.path

        if not mt5.initialize(**init_kwargs):
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False

        # Login to account
        if not mt5.login(self.login, password=self.password, server=self.server):
            logger.error(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False

        self._connected = True
        logger.info(f"Connected to MT5 account {self.login} on {self.server}")
        return True

    def disconnect(self) -> bool:
        """Disconnect from MT5."""
        if self._mt5:
            self._mt5.shutdown()
        self._connected = False
        logger.info("Disconnected from MT5")
        return True

    def ping(self) -> bool:
        """Check if MT5 connection is alive."""
        if not self._connected:
            return False
        mt5 = self._get_mt5()
        info = mt5.account_info()
        return info is not None

    # ==================== Account Methods ====================

    def get_account_balance(self) -> float:
        """Get account balance."""
        mt5 = self._get_mt5()
        info = mt5.account_info()
        return info.balance if info else 0.0

    def get_account_equity(self) -> float:
        """Get account equity."""
        mt5 = self._get_mt5()
        info = mt5.account_info()
        return info.equity if info else 0.0

    def get_account_margin(self) -> float:
        """Get used margin."""
        mt5 = self._get_mt5()
        info = mt5.account_info()
        return info.margin if info else 0.0

    def get_account_free_margin(self) -> float:
        """Get free margin."""
        mt5 = self._get_mt5()
        info = mt5.account_info()
        return info.margin_free if info else 0.0

    def get_account_info(self) -> dict:
        """Get complete account information."""
        mt5 = self._get_mt5()
        info = mt5.account_info()
        if not info:
            return {}
        return {
            "login": info.login,
            "name": info.name,
            "server": info.server,
            "currency": info.currency,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "margin_free": info.margin_free,
            "margin_level": info.margin_level,
            "profit": info.profit,
            "leverage": info.leverage,
        }

    # ==================== Market Data Methods ====================

    def get_current_price(self, symbol: str) -> tuple[float, float]:
        """Get current bid/ask price."""
        mt5 = self._get_mt5()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get price for {symbol}")
            return (0.0, 0.0)
        return (tick.bid, tick.ask)

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[Candle]:
        """Get historical candle data."""
        mt5 = self._get_mt5()
        tf = self._get_mt5_timeframe(timeframe)

        if start_time and end_time:
            rates = mt5.copy_rates_range(symbol, tf, start_time, end_time)
        elif start_time:
            rates = mt5.copy_rates_from(symbol, tf, start_time, count)
        else:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)

        if rates is None or len(rates) == 0:
            logger.warning(f"No candle data received for {symbol}")
            return []

        candles = []
        for rate in rates:
            candle = Candle(
                timestamp=datetime.fromtimestamp(rate["time"]),
                open=float(rate["open"]),
                high=float(rate["high"]),
                low=float(rate["low"]),
                close=float(rate["close"]),
                volume=float(rate["tick_volume"]),
                symbol=symbol,
                timeframe=timeframe,
            )
            candles.append(candle)

        return candles

    def get_symbol_info(self, symbol: str) -> dict:
        """Get symbol information."""
        mt5 = self._get_mt5()
        info = mt5.symbol_info(symbol)
        if not info:
            return {}

        # Calculate pip value
        point = info.point
        digits = info.digits
        pip_size = point * 10 if digits in [3, 5] else point

        return {
            "name": info.name,
            "description": info.description,
            "point": info.point,
            "digits": info.digits,
            "pip_size": pip_size,
            "spread": info.spread,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "contract_size": info.trade_contract_size,
            "pip_value": info.trade_tick_value * (pip_size / point),
            "margin_initial": info.margin_initial,
        }

    def get_symbols(self) -> list[str]:
        """Get list of available symbols."""
        mt5 = self._get_mt5()
        symbols = mt5.symbols_get()
        return [s.name for s in symbols] if symbols else []

    # ==================== Order Methods ====================

    def place_order(self, order: Order) -> Optional[str]:
        """Place a new order."""
        mt5 = self._get_mt5()

        # Get current price
        bid, ask = self.get_current_price(order.symbol)
        price = ask if order.is_buy else bid

        # Map order type
        if order.order_type == OrderType.MARKET:
            action = mt5.TRADE_ACTION_DEAL
            mt5_type = mt5.ORDER_TYPE_BUY if order.is_buy else mt5.ORDER_TYPE_SELL
        elif order.order_type == OrderType.LIMIT:
            action = mt5.TRADE_ACTION_PENDING
            mt5_type = mt5.ORDER_TYPE_BUY_LIMIT if order.is_buy else mt5.ORDER_TYPE_SELL_LIMIT
            price = order.price
        elif order.order_type == OrderType.STOP:
            action = mt5.TRADE_ACTION_PENDING
            mt5_type = mt5.ORDER_TYPE_BUY_STOP if order.is_buy else mt5.ORDER_TYPE_SELL_STOP
            price = order.stop_price
        else:
            logger.error(f"Unsupported order type: {order.order_type}")
            return None

        # Build request
        request = {
            "action": action,
            "symbol": order.symbol,
            "volume": order.quantity,
            "type": mt5_type,
            "price": price,
            "deviation": 20,
            "magic": order.magic_number,
            "comment": order.comment or "TradingBot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_type(order.symbol),
        }

        # Add SL/TP if specified
        if order.stop_loss:
            request["sl"] = order.stop_loss
        if order.take_profit:
            request["tp"] = order.take_profit

        # Send order
        result = mt5.order_send(request)

        if result is None:
            logger.error(f"Order send failed: {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return None

        order.broker_order_id = str(result.order)
        order.status = OrderStatus.FILLED if action == mt5.TRADE_ACTION_DEAL else OrderStatus.SUBMITTED
        order.filled_price = result.price
        order.filled_quantity = result.volume

        logger.info(f"Order placed: {result.order} at {result.price}")
        return str(result.order)

    def modify_order(
        self,
        order_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        price: Optional[float] = None,
    ) -> bool:
        """Modify an existing pending order."""
        mt5 = self._get_mt5()

        orders = mt5.orders_get(ticket=int(order_id))
        if not orders:
            logger.error(f"Order {order_id} not found")
            return False

        order = orders[0]

        request = {
            "action": mt5.TRADE_ACTION_MODIFY,
            "order": int(order_id),
            "symbol": order.symbol,
            "price": price if price else order.price_open,
            "sl": stop_loss if stop_loss else order.sl,
            "tp": take_profit if take_profit else order.tp,
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order modification failed: {result}")
            return False

        logger.info(f"Order {order_id} modified successfully")
        return True

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        mt5 = self._get_mt5()

        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": int(order_id),
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order cancellation failed: {result}")
            return False

        logger.info(f"Order {order_id} cancelled")
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        mt5 = self._get_mt5()

        orders = mt5.orders_get(ticket=int(order_id))
        if not orders:
            return None

        o = orders[0]
        return Order(
            symbol=o.symbol,
            side=OrderSide.BUY if o.type in [0, 2, 4] else OrderSide.SELL,
            order_type=OrderType.LIMIT if o.type in [2, 3] else OrderType.STOP,
            quantity=o.volume_current,
            price=o.price_open,
            stop_loss=o.sl if o.sl > 0 else None,
            take_profit=o.tp if o.tp > 0 else None,
            status=OrderStatus.SUBMITTED,
            broker_order_id=str(o.ticket),
            comment=o.comment,
            magic_number=o.magic,
        )

    def get_pending_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """Get all pending orders."""
        mt5 = self._get_mt5()

        if symbol:
            orders = mt5.orders_get(symbol=symbol)
        else:
            orders = mt5.orders_get()

        if not orders:
            return []

        result = []
        for o in orders:
            order = Order(
                symbol=o.symbol,
                side=OrderSide.BUY if o.type in [0, 2, 4] else OrderSide.SELL,
                order_type=OrderType.LIMIT if o.type in [2, 3] else OrderType.STOP,
                quantity=o.volume_current,
                price=o.price_open,
                stop_loss=o.sl if o.sl > 0 else None,
                take_profit=o.tp if o.tp > 0 else None,
                status=OrderStatus.SUBMITTED,
                broker_order_id=str(o.ticket),
                comment=o.comment,
                magic_number=o.magic,
            )
            result.append(order)

        return result

    # ==================== Position Methods ====================

    def get_positions(self, symbol: Optional[str] = None) -> list[Position]:
        """Get all open positions."""
        mt5 = self._get_mt5()

        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if not positions:
            return []

        result = []
        for p in positions:
            position = Position(
                symbol=p.symbol,
                side="long" if p.type == 0 else "short",
                quantity=p.volume,
                entry_price=p.price_open,
                current_price=p.price_current,
                stop_loss=p.sl if p.sl > 0 else None,
                take_profit=p.tp if p.tp > 0 else None,
                status=PositionStatus.OPEN,
                broker_position_id=str(p.ticket),
                pnl=p.profit,
                swap=p.swap,
                comment=p.comment,
                magic_number=p.magic,
            )
            result.append(position)

        return result

    def get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID."""
        mt5 = self._get_mt5()

        positions = mt5.positions_get(ticket=int(position_id))
        if not positions:
            return None

        p = positions[0]
        return Position(
            symbol=p.symbol,
            side="long" if p.type == 0 else "short",
            quantity=p.volume,
            entry_price=p.price_open,
            current_price=p.price_current,
            stop_loss=p.sl if p.sl > 0 else None,
            take_profit=p.tp if p.tp > 0 else None,
            status=PositionStatus.OPEN,
            broker_position_id=str(p.ticket),
            pnl=p.profit,
            swap=p.swap,
            comment=p.comment,
            magic_number=p.magic,
        )

    def close_position(
        self, position_id: str, volume: Optional[float] = None
    ) -> bool:
        """Close an open position."""
        mt5 = self._get_mt5()

        positions = mt5.positions_get(ticket=int(position_id))
        if not positions:
            logger.error(f"Position {position_id} not found")
            return False

        p = positions[0]
        close_volume = volume if volume else p.volume

        # Get current price
        bid, ask = self.get_current_price(p.symbol)
        price = bid if p.type == 0 else ask  # Bid for long, Ask for short

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "volume": close_volume,
            "type": mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY,
            "position": int(position_id),
            "price": price,
            "deviation": 20,
            "magic": p.magic,
            "comment": "Close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_type(p.symbol),
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Position close failed: {result}")
            return False

        logger.info(f"Position {position_id} closed at {result.price}")
        return True

    def modify_position(
        self,
        position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        """Modify position's SL/TP."""
        mt5 = self._get_mt5()

        positions = mt5.positions_get(ticket=int(position_id))
        if not positions:
            logger.error(f"Position {position_id} not found")
            return False

        p = positions[0]

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": p.symbol,
            "position": int(position_id),
            "sl": stop_loss if stop_loss else p.sl,
            "tp": take_profit if take_profit else p.tp,
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Position modification failed: {result}")
            return False

        logger.info(f"Position {position_id} modified: SL={stop_loss}, TP={take_profit}")
        return True

    def get_server_time(self) -> datetime:
        """Get MT5 server time."""
        mt5 = self._get_mt5()
        tick = mt5.symbol_info_tick(self.get_symbols()[0] if self.get_symbols() else "EURUSD")
        if tick:
            return datetime.fromtimestamp(tick.time)
        return datetime.now()
