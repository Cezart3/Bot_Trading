"""Logging configuration using loguru."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

# Remove default handler
logger.remove()

# Flag to track if logging is set up
_logging_configured = False


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_path: str = "logs/",
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_to_file: Whether to log to files.
        log_path: Directory for log files.
        rotation: When to rotate log files.
        retention: How long to keep log files.
    """
    global _logging_configured

    if _logging_configured:
        return

    # Console handler with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        level=level,
        colorize=True,
    )

    if log_to_file:
        # Create log directory
        log_dir = Path(log_path)
        log_dir.mkdir(parents=True, exist_ok=True)

        # General log file
        logger.add(
            log_dir / "trading_bot_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

        # Error log file
        logger.add(
            log_dir / "errors_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level="ERROR",
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

        # Trade log file (for order execution)
        logger.add(
            log_dir / "trades_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
            level="INFO",
            rotation=rotation,
            retention=retention,
            filter=lambda record: "trade" in record["extra"],
            compression="zip",
        )

    _logging_configured = True
    logger.info("Logging configured successfully")


def get_logger(name: Optional[str] = None):
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__).

    Returns:
        Logger instance.
    """
    if name:
        return logger.bind(name=name)
    return logger


def log_trade(message: str) -> None:
    """
    Log a trade-related message to the trades log.

    Args:
        message: Trade message to log.
    """
    logger.bind(trade=True).info(message)


class TradeLogger:
    """Context manager for trade logging."""

    def __init__(self, trade_type: str, symbol: str):
        """
        Initialize trade logger.

        Args:
            trade_type: Type of trade (BUY, SELL, CLOSE).
            symbol: Trading symbol.
        """
        self.trade_type = trade_type
        self.symbol = symbol
        self.logger = logger.bind(trade=True)

    def __enter__(self):
        """Enter context."""
        self.logger.info(f"=== {self.trade_type} {self.symbol} START ===")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if exc_type:
            self.logger.error(f"=== {self.trade_type} {self.symbol} FAILED: {exc_val} ===")
        else:
            self.logger.info(f"=== {self.trade_type} {self.symbol} COMPLETE ===")
        return False

    def log(self, message: str) -> None:
        """Log a message within trade context."""
        self.logger.info(f"[{self.trade_type}] {message}")
