"""Data loader for backtesting - loads historical data from various sources."""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import requests
import io

from models.candle import Candle
from utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Loads historical price data from various sources.

    Supports:
    - CSV files
    - MT5 historical data
    - Yahoo Finance (free)
    - Alpha Vantage (free with API key)
    """

    def __init__(self, data_dir: str = "data"):
        """Initialize data loader."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_mt5_csv(
        self,
        filepath: str,
        symbol: str = "",
        timeframe: str = "M1",
    ) -> list[Candle]:
        """
        Load candles from MT5 exported CSV file.
        Format: 2022.01.02,17:03,1.136900,1.136900,1.136900,1.136900,0
        """
        logger.info(f"Loading MT5 data from {filepath}")

        # Citim fără header
        df = pd.read_csv(
            filepath, 
            header=None, 
            names=["date", "time", "open", "high", "low", "close", "volume"]
        )

        # Combinăm data și ora într-un singur timestamp
        df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])

        candles = []
        for _, row in df.iterrows():
            candle = Candle(
                timestamp=row["timestamp"].to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                symbol=symbol,
                timeframe=timeframe,
            )
            candles.append(candle)

        logger.info(f"Loaded {len(candles)} candles from MT5 CSV")
        return candles

    def load_from_csv(
        self,
        filepath: str,
        symbol: str = "",
        timeframe: str = "M5",
        date_column: str = "timestamp",
        date_format: Optional[str] = None,
    ) -> list[Candle]:
        """
        Load candles from CSV file.

        Expected CSV columns: timestamp, open, high, low, close, volume
        """
        logger.info(f"Loading data from {filepath}")

        df = pd.read_csv(filepath)

        # Parse dates
        if date_format:
            df[date_column] = pd.to_datetime(df[date_column], format=date_format)
        else:
            df[date_column] = pd.to_datetime(df[date_column])

        # Sort by date
        df = df.sort_values(date_column)

        candles = []
        for _, row in df.iterrows():
            candle = Candle(
                timestamp=row[date_column].to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0)),
                symbol=symbol,
                timeframe=timeframe,
            )
            candles.append(candle)

        logger.info(f"Loaded {len(candles)} candles from CSV")
        return candles

    def download_yahoo_finance(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "5m",
    ) -> list[Candle]:
        """
        Download data from Yahoo Finance.
        """
        # Format symbol for yfinance (e.g., EURUSD -> EURUSD=X)
        if len(symbol) == 6 and symbol.isalpha():
            y_symbol = f"{symbol}=X"
        else:
            y_symbol = symbol

        logger.info(f"Downloading {y_symbol} from Yahoo Finance ({interval})")

        try:
            import yfinance as yf

            # Limit download period for intraday data as per yfinance constraints
            if 'm' in interval or 'h' in interval:
                actual_start = max(start_date, datetime.now() - timedelta(days=59))
                if actual_start > start_date:
                    logger.warning(f"yfinance 5m data is limited to 60 days. Fetching from {actual_start.date()}")
            else:
                actual_start = start_date

            ticker = yf.Ticker(y_symbol)
            df = ticker.history(
                start=actual_start,
                end=end_date,
                interval=interval,
            )

            if df.empty:
                logger.warning(f"No data received for {y_symbol}")
                return []

            # Ensure timezone is UTC
            df.index = df.index.tz_convert('UTC')

            candles = []
            for timestamp, row in df.iterrows():
                candle = Candle(
                    timestamp=timestamp.to_pydatetime(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                    symbol=symbol,
                    timeframe=interval.upper(),
                )
                candles.append(candle)

            logger.info(f"Downloaded {len(candles)} candles from Yahoo Finance")
            return candles

        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
            return []
        except Exception as e:
            logger.error(f"Error downloading from Yahoo Finance: {e}")
            return []

    def resample_data(self, candles: list[Candle], timeframe: str, fill_gaps: bool = False) -> list[Candle]:
        """Resample candle data to a higher timeframe using pandas."""
        if not candles:
            return []
        
        df = pd.DataFrame([c.to_dict() for c in candles])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        resample_rule = {
            'M1': '1min', # Added M1 for resampling
            'M5': '5min',
            'H1': '1h',
            'H4': '4h',
            'D1': '1d',
        }.get(timeframe, '1h') # Default to 1h if not found

        # If fill_gaps is True, forward fill missing values
        if fill_gaps:
            resampled_df = df.resample(resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).ffill() # Forward fill missing data
        else:
            resampled_df = df.resample(resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

        resampled_candles = []
        for timestamp, row in resampled_df.iterrows():
            resampled_candles.append(Candle(
                timestamp=timestamp.to_pydatetime(),
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                symbol=candles[0].symbol,
                timeframe=timeframe
            ))
        
        logger.info(f"Resampled {len(candles)} M5 candles to {len(resampled_candles)} {timeframe} candles")
        return resampled_candles

    def fetch_and_resample_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> dict[str, list[Candle]]:
        """
        Fetches M5 data as primary, optionally fetches 1m data if period is short,
        and resamples higher timeframes. Handles yfinance 1m data limitations.
        """
        
        m1_candles: List[Candle] = []
        
        # Always fetch 5m data first as the reliable base
        m5_base_candles = self.download_yahoo_finance(symbol, start_date, end_date, "5m")
        if not m5_base_candles:
            logger.error(f"[{symbol}] Failed to fetch 5m data. Cannot proceed.")
            return {}
        
        m5_candles = m5_base_candles # M5 is always available if we get here

        # Now, try to get 1m data if the period is within yfinance limits
        if (end_date - start_date).days <= 60:
            logger.info(f"[{symbol}] Attempting to fetch 1m data for the last 60 days.")
            try:
                m1_candles = self.download_yahoo_finance(symbol, start_date, end_date, "1m")
            except Exception as e:
                logger.warning(f"[{symbol}] Failed to fetch 1m data: {e}. Will use resampled 1m from 5m.")
        else:
            logger.warning(f"[{symbol}] 1m data not requested (period > 60 days). Will use resampled 1m from 5m.")
        
        # If 1m data was not successfully fetched for the full period, create sparse 1m from 5m
        if not m1_candles or len(m1_candles) < len(m5_base_candles) * 4: # Heuristic: if 1m is very sparse, use resampled
            logger.warning(f"[{symbol}] Using resampled 1m data from 5m candles.")
            m1_candles = self.resample_data(m5_base_candles, "M1", fill_gaps=True) 

        # Now resample higher timeframes from the primary M5 candles
        h1_candles = self.resample_data(m5_candles, "H1")
        h4_candles = self.resample_data(m5_candles, "H4")
        daily_candles = self.resample_data(m5_candles, "D1")

        return {
            "m1": m1_candles,
            "m5": m5_candles,
            "h1": h1_candles,
            "h4": h4_candles,
            "d1": daily_candles,
        }
    
    def generate_sample_data(
        self,
        symbol: str = "US500",
        timeframe: str = "M5",
        days: int = 30,
        start_price: float = 5000.0,
        volatility: float = 0.001,
        session_start: str = "09:30",
        session_end: str = "16:00",
    ) -> list[Candle]:
        """
        Generate sample/synthetic data for testing.

        This creates realistic-looking price data with:
        - Opening gaps
        - Intraday trends
        - Normal distribution of returns
        """
        import numpy as np
        import pytz

        logger.info(f"Generating {days} days of sample data for {symbol}")

        np.random.seed(42)  # For reproducibility

        tz = pytz.timezone("America/New_York")
        candles = []

        current_price = start_price

        # Parse session times
        start_h, start_m = map(int, session_start.split(":"))
        end_h, end_m = map(int, session_end.split(":"))

        # Calculate candles per session (5-minute candles)
        session_minutes = (end_h * 60 + end_m) - (start_h * 60 + start_m)
        candles_per_day = session_minutes // 5

        start_date = datetime.now(tz) - timedelta(days=days)

        for day in range(days):
            current_date = start_date + timedelta(days=day)

            # Skip weekends
            if current_date.weekday() >= 5:
                continue

            # Opening gap (random)
            gap = np.random.normal(0, volatility * 2) * current_price
            current_price += gap

            # Generate intraday candles
            for candle_num in range(candles_per_day):
                minutes_offset = candle_num * 5
                candle_time = current_date.replace(
                    hour=start_h,
                    minute=start_m,
                    second=0,
                    microsecond=0,
                ) + timedelta(minutes=minutes_offset)

                # Random price movement
                returns = np.random.normal(0, volatility)

                # Add some trend bias based on time of day
                if candle_num < 3:  # First 15 minutes - higher volatility
                    returns *= 1.5
                elif candle_num > candles_per_day - 6:  # Last 30 minutes
                    returns *= 1.2

                # Generate OHLC
                open_price = current_price
                close_price = open_price * (1 + returns)

                # High/Low with some randomness
                range_size = abs(returns) + np.random.uniform(0.0001, 0.001)

                if close_price > open_price:  # Bullish
                    high_price = close_price + (range_size * current_price * np.random.uniform(0.2, 0.5))
                    low_price = open_price - (range_size * current_price * np.random.uniform(0.2, 0.5))
                else:  # Bearish
                    high_price = open_price + (range_size * current_price * np.random.uniform(0.2, 0.5))
                    low_price = close_price - (range_size * current_price * np.random.uniform(0.2, 0.5))

                # Volume (higher at open and close)
                base_volume = 10000
                if candle_num < 3 or candle_num > candles_per_day - 6:
                    volume = base_volume * np.random.uniform(1.5, 3.0)
                else:
                    volume = base_volume * np.random.uniform(0.5, 1.5)

                candle = Candle(
                    timestamp=candle_time,
                    open=round(open_price, 2),
                    high=round(high_price, 2),
                    low=round(low_price, 2),
                    close=round(close_price, 2),
                    volume=round(volume),
                    symbol=symbol,
                    timeframe=timeframe,
                )
                candles.append(candle)

                current_price = close_price

        logger.info(f"Generated {len(candles)} sample candles")
        return candles

    def save_to_csv(self, candles: list[Candle], filename: str) -> str:
        """Save candles to CSV file."""
        filepath = self.data_dir / filename

        data = [c.to_dict() for c in candles]
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)

        logger.info(f"Saved {len(candles)} candles to {filepath}")
        return str(filepath)

    def load_from_mt5(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        mt5_broker,
    ) -> list[Candle]:
        """Load historical data from MT5."""
        if not mt5_broker.is_connected:
            logger.error("MT5 not connected")
            return []

        candles = mt5_broker.get_candles(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_date,
            end_time=end_date,
        )

        logger.info(f"Loaded {len(candles)} candles from MT5")
        return candles

    def download_stock_data(
        self,
        symbol: str,
        days: int = 60,
        interval: str = "5m",
    ) -> list[Candle]:
        """
        Download stock data from Yahoo Finance.

        For stocks like NVDA, AMD, TSLA.
        Yahoo Finance 5m data is limited to 60 days.

        Args:
            symbol: Stock ticker (NVDA, AMD, TSLA, etc.)
            days: Number of days (max 60 for 5m data)
            interval: Data interval (5m recommended)

        Returns:
            List of candles.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=min(days, 60))

        return self.download_yahoo_finance(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )

    def generate_stock_sample_data(
        self,
        symbol: str = "NVDA",
        timeframe: str = "M5",
        days: int = 30,
        start_price: float = 140.0,
        volatility: float = 0.002,
    ) -> list[Candle]:
        """
        Generate realistic stock sample data for testing.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe
            days: Number of trading days
            start_price: Starting price
            volatility: Price volatility (0.002 = 0.2% typical for stocks)

        Returns:
            List of candles.
        """
        return self.generate_sample_data(
            symbol=symbol,
            timeframe=timeframe,
            days=days,
            start_price=start_price,
            volatility=volatility,
            session_start="09:30",
            session_end="16:00",
        )
