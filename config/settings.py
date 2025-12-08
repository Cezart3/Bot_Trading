"""
Application settings using Pydantic Settings.
All configuration is loaded from environment variables or .env file.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MT5Settings(BaseSettings):
    """MetaTrader 5 connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="MT5_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    login: int = Field(default=0, description="MT5 account number")
    password: str = Field(default="", description="MT5 account password")
    server: str = Field(default="", description="MT5 broker server")
    path: str = Field(
        default=r"C:\Program Files\MetaTrader 5\terminal64.exe",
        description="Path to MT5 terminal",
    )
    timeout: int = Field(default=60000, description="Connection timeout in ms")


class NinjaTraderSettings(BaseSettings):
    """NinjaTrader connection settings."""

    model_config = SettingsConfigDict(env_prefix="NT_")

    host: str = Field(default="127.0.0.1", description="NinjaTrader host")
    port: int = Field(default=5555, description="NinjaTrader port")
    account: str = Field(default="Sim101", description="NinjaTrader account")


class ORBSettings(BaseSettings):
    """Open Range Breakout strategy settings."""

    model_config = SettingsConfigDict(env_prefix="ORB_")

    range_minutes: int = Field(
        default=5, description="Minutes to calculate opening range"
    )
    session_start: str = Field(default="09:30", description="Trading session start")
    session_end: str = Field(default="16:00", description="Trading session end")
    timezone: str = Field(default="America/New_York", description="Trading timezone")
    breakout_buffer_pips: float = Field(
        default=2.0, description="Buffer above/below range for entry"
    )
    use_atr_filter: bool = Field(default=True, description="Use ATR for volatility filter")
    min_range_pips: float = Field(default=5.0, description="Minimum range size in pips")
    max_range_pips: float = Field(default=50.0, description="Maximum range size in pips")


class RiskSettings(BaseSettings):
    """Risk management settings."""

    model_config = SettingsConfigDict(env_prefix="")

    max_risk_per_trade: float = Field(
        default=1.0, description="Max risk per trade in %"
    )
    max_daily_loss: float = Field(default=3.0, description="Max daily loss in %")
    max_positions: int = Field(default=3, description="Max concurrent positions")
    stop_loss_atr_multiplier: float = Field(
        default=1.5, description="SL as ATR multiplier"
    )
    take_profit_atr_multiplier: float = Field(
        default=2.0, description="TP as ATR multiplier"
    )
    use_trailing_stop: bool = Field(default=False, description="Enable trailing stop")
    trailing_stop_pips: float = Field(default=10.0, description="Trailing stop in pips")


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    default_symbol: str = Field(default="EURUSD", description="Default trading symbol")
    default_timeframe: str = Field(default="M5", description="Default timeframe")
    default_lot_size: float = Field(default=0.01, description="Default lot size")

    # Mode
    trading_mode: Literal["live", "paper", "demo", "backtest"] = Field(
        default="paper", description="Trading mode"
    )
    active_broker: Literal["mt5", "ninjatrader"] = Field(
        default="mt5", description="Active broker"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_to_file: bool = Field(default=True, description="Log to file")
    log_path: str = Field(default="logs/", description="Log directory")

    # Database
    database_url: str = Field(
        default="sqlite:///data/trades.db", description="Database URL"
    )

    # Sub-settings
    mt5: MT5Settings = Field(default_factory=MT5Settings)
    ninjatrader: NinjaTraderSettings = Field(default_factory=NinjaTraderSettings)
    orb: ORBSettings = Field(default_factory=ORBSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
