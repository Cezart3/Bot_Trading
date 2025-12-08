"""
Multi-Account Manager for Prop Trading Bot.

Allows running the same strategy on multiple prop accounts simultaneously,
potentially from different providers.

Features:
- Independent risk management per account
- Separate trade tracking
- Consolidated reporting
- Provider-agnostic design
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from pathlib import Path
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.logger import get_logger
from utils.risk_manager import RiskManager, PropAccountLimits

logger = get_logger(__name__)


class AccountProvider(Enum):
    """Supported prop trading providers."""
    MT5 = "mt5"
    NINJATRADER = "ninjatrader"
    TRADOVATE = "tradovate"
    TOPSTEP = "topstep"
    APEX = "apex"
    FTMO = "ftmo"
    MFF = "mff"  # MyForexFunds
    TFT = "tft"  # The Funded Trader
    GENERIC = "generic"


@dataclass
class AccountConfig:
    """Configuration for a single trading account."""

    account_id: str
    name: str  # Friendly name like "FTMO Account 1"
    provider: AccountProvider
    broker: str  # MT5 server, NinjaTrader connection, etc.

    # Connection settings
    login: str = ""
    password: str = ""
    server: str = ""

    # Account settings
    initial_balance: float = 10000.0
    currency: str = "USD"

    # Provider-specific limits
    max_daily_drawdown: float = 4.0  # %
    max_account_drawdown: float = 10.0  # %
    max_positions: int = 1

    # Risk settings (can override global)
    risk_per_trade: float = 1.0  # %
    reduced_risk: float = 0.5  # %

    # Symbols to trade on this account
    symbols: list[str] = field(default_factory=lambda: ["NVDA", "AMD", "TSLA"])

    # Status
    enabled: bool = True
    connected: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "account_id": self.account_id,
            "name": self.name,
            "provider": self.provider.value,
            "broker": self.broker,
            "login": self.login,
            "server": self.server,
            "initial_balance": self.initial_balance,
            "currency": self.currency,
            "max_daily_drawdown": self.max_daily_drawdown,
            "max_account_drawdown": self.max_account_drawdown,
            "max_positions": self.max_positions,
            "risk_per_trade": self.risk_per_trade,
            "reduced_risk": self.reduced_risk,
            "symbols": self.symbols,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AccountConfig":
        """Create from dictionary."""
        data["provider"] = AccountProvider(data.get("provider", "generic"))
        return cls(**data)


@dataclass
class AccountState:
    """Runtime state for an account."""

    config: AccountConfig
    risk_manager: RiskManager
    current_balance: float = 0.0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    trades_today: int = 0
    total_trades: int = 0
    last_update: Optional[datetime] = None
    error_message: str = ""
    broker_instance: Any = None  # Reference to broker connection


class MultiAccountManager:
    """
    Manages multiple prop trading accounts.

    Usage:
        manager = MultiAccountManager()
        manager.add_account(AccountConfig(...))
        manager.add_account(AccountConfig(...))
        manager.start_all()
    """

    def __init__(self, config_path: str = "config/accounts.json"):
        """
        Initialize multi-account manager.

        Args:
            config_path: Path to accounts configuration file.
        """
        self.config_path = Path(config_path)
        self.accounts: Dict[str, AccountState] = {}
        self._lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._running = False

    def add_account(self, config: AccountConfig) -> None:
        """
        Add a new account to manage.

        Args:
            config: Account configuration.
        """
        # Create risk manager with account-specific limits
        limits = PropAccountLimits(
            max_daily_drawdown=config.max_daily_drawdown,
            max_account_drawdown=config.max_account_drawdown,
            normal_risk_per_trade=config.risk_per_trade,
            reduced_risk_per_trade=config.reduced_risk,
            max_positions=config.max_positions,
        )

        risk_manager = RiskManager(
            initial_account_balance=config.initial_balance,
            limits=limits,
        )

        state = AccountState(
            config=config,
            risk_manager=risk_manager,
            current_balance=config.initial_balance,
        )

        with self._lock:
            self.accounts[config.account_id] = state

        logger.info(f"Added account: {config.name} ({config.provider.value})")

    def remove_account(self, account_id: str) -> bool:
        """Remove an account."""
        with self._lock:
            if account_id in self.accounts:
                del self.accounts[account_id]
                logger.info(f"Removed account: {account_id}")
                return True
        return False

    def get_account(self, account_id: str) -> Optional[AccountState]:
        """Get account state by ID."""
        return self.accounts.get(account_id)

    def list_accounts(self) -> list[dict]:
        """List all accounts with their status."""
        result = []
        for account_id, state in self.accounts.items():
            result.append({
                "account_id": account_id,
                "name": state.config.name,
                "provider": state.config.provider.value,
                "enabled": state.config.enabled,
                "connected": state.config.connected,
                "balance": state.current_balance,
                "daily_pnl": state.daily_pnl,
                "total_pnl": state.total_pnl,
                "trades_today": state.trades_today,
            })
        return result

    def save_config(self) -> None:
        """Save accounts configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "accounts": [
                state.config.to_dict()
                for state in self.accounts.values()
            ]
        }

        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.accounts)} accounts to {self.config_path}")

    def load_config(self) -> None:
        """Load accounts configuration from file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return

        with open(self.config_path) as f:
            data = json.load(f)

        for account_data in data.get("accounts", []):
            config = AccountConfig.from_dict(account_data)
            self.add_account(config)

        logger.info(f"Loaded {len(self.accounts)} accounts from {self.config_path}")

    def connect_account(self, account_id: str) -> bool:
        """
        Connect to a specific account's broker.

        Args:
            account_id: Account identifier.

        Returns:
            True if connected successfully.
        """
        state = self.accounts.get(account_id)
        if not state:
            logger.error(f"Account not found: {account_id}")
            return False

        config = state.config

        try:
            if config.provider == AccountProvider.MT5:
                from brokers.mt5_broker import MT5Broker
                broker = MT5Broker()
                connected = broker.connect(
                    login=int(config.login),
                    password=config.password,
                    server=config.server,
                )
                if connected:
                    state.broker_instance = broker
                    state.config.connected = True
                    state.current_balance = broker.get_balance()
                    logger.info(f"Connected to {config.name} | Balance: ${state.current_balance:,.2f}")
                    return True

            elif config.provider == AccountProvider.NINJATRADER:
                from brokers.ninjatrader_broker import NinjaTraderBroker
                broker = NinjaTraderBroker()
                connected = broker.connect()
                if connected:
                    state.broker_instance = broker
                    state.config.connected = True
                    logger.info(f"Connected to {config.name}")
                    return True

            else:
                logger.warning(f"Provider {config.provider.value} not yet implemented")
                return False

        except Exception as e:
            state.error_message = str(e)
            logger.error(f"Failed to connect {config.name}: {e}")
            return False

        return False

    def connect_all(self) -> dict[str, bool]:
        """
        Connect to all enabled accounts.

        Returns:
            Dictionary of account_id -> connection status.
        """
        results = {}

        for account_id, state in self.accounts.items():
            if state.config.enabled:
                results[account_id] = self.connect_account(account_id)
            else:
                results[account_id] = False

        connected = sum(1 for v in results.values() if v)
        logger.info(f"Connected to {connected}/{len(results)} accounts")

        return results

    def disconnect_all(self) -> None:
        """Disconnect from all accounts."""
        for account_id, state in self.accounts.items():
            if state.broker_instance:
                try:
                    state.broker_instance.disconnect()
                    state.config.connected = False
                    logger.info(f"Disconnected from {state.config.name}")
                except Exception as e:
                    logger.error(f"Error disconnecting {state.config.name}: {e}")

    def get_consolidated_report(self) -> dict:
        """
        Get consolidated report across all accounts.

        Returns:
            Dictionary with aggregated statistics.
        """
        total_balance = 0.0
        total_daily_pnl = 0.0
        total_pnl = 0.0
        total_trades = 0
        trades_today = 0

        account_reports = []

        for account_id, state in self.accounts.items():
            total_balance += state.current_balance
            total_daily_pnl += state.daily_pnl
            total_pnl += state.total_pnl
            total_trades += state.total_trades
            trades_today += state.trades_today

            risk_status = state.risk_manager.get_risk_status()

            account_reports.append({
                "account_id": account_id,
                "name": state.config.name,
                "provider": state.config.provider.value,
                "balance": state.current_balance,
                "daily_pnl": state.daily_pnl,
                "total_pnl": state.total_pnl,
                "trades_today": state.trades_today,
                "can_trade": risk_status.get("can_trade", False),
                "daily_dd": risk_status.get("daily_loss_percent", 0),
                "account_dd": risk_status.get("account_drawdown_percent", 0),
            })

        return {
            "timestamp": datetime.now().isoformat(),
            "total_accounts": len(self.accounts),
            "connected_accounts": sum(1 for s in self.accounts.values() if s.config.connected),
            "total_balance": total_balance,
            "total_daily_pnl": total_daily_pnl,
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "trades_today": trades_today,
            "accounts": account_reports,
        }

    def print_status(self) -> None:
        """Print status of all accounts."""
        report = self.get_consolidated_report()

        print("\n" + "=" * 80)
        print("              MULTI-ACCOUNT STATUS")
        print("=" * 80)

        print(f"\n  Total Accounts: {report['total_accounts']}")
        print(f"  Connected: {report['connected_accounts']}")
        print(f"  Total Balance: ${report['total_balance']:,.2f}")
        print(f"  Today's P&L: ${report['total_daily_pnl']:+,.2f}")
        print(f"  Total P&L: ${report['total_pnl']:+,.2f}")

        print("\n  " + "-" * 76)
        print(f"  {'Account':<20} | {'Provider':<12} | {'Balance':>12} | {'Daily P&L':>12} | {'Status':<10}")
        print("  " + "-" * 76)

        for acc in report["accounts"]:
            status = "ACTIVE" if acc["can_trade"] else "STOPPED"
            print(
                f"  {acc['name']:<20} | {acc['provider']:<12} | "
                f"${acc['balance']:>11,.0f} | ${acc['daily_pnl']:>+11,.0f} | {status:<10}"
            )

        print("  " + "-" * 76)
        print("=" * 80 + "\n")


def create_sample_config():
    """Create a sample multi-account configuration."""

    manager = MultiAccountManager()

    # Example: FTMO Account
    manager.add_account(AccountConfig(
        account_id="ftmo_001",
        name="FTMO $100k",
        provider=AccountProvider.MT5,
        broker="FTMO-Server",
        login="123456",
        password="password123",
        server="FTMO-Demo",
        initial_balance=100000.0,
        max_daily_drawdown=5.0,
        max_account_drawdown=10.0,
        symbols=["NVDA", "AMD", "TSLA"],
    ))

    # Example: TopStep Account
    manager.add_account(AccountConfig(
        account_id="topstep_001",
        name="TopStep $50k",
        provider=AccountProvider.NINJATRADER,
        broker="TopStep",
        initial_balance=50000.0,
        max_daily_drawdown=4.0,
        max_account_drawdown=8.0,
        symbols=["ES", "NQ"],
    ))

    # Example: Personal MT5 Account
    manager.add_account(AccountConfig(
        account_id="mt5_demo",
        name="MT5 Demo",
        provider=AccountProvider.MT5,
        broker="MetaQuotes",
        login="9042470",
        password="NiJp@w1c",
        server="MetaQuotes-Demo",
        initial_balance=10000.0,
        symbols=["NVDA", "AMD", "TSLA"],
    ))

    manager.save_config()
    print("Sample configuration created at config/accounts.json")

    return manager


if __name__ == "__main__":
    # Create sample config
    manager = create_sample_config()
    manager.print_status()
