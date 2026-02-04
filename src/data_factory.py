# src/data_factory.py
import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
from config import Settings
from src.utils import logger

def fetch_data(symbol: str, num_bars: int) -> pd.DataFrame:
    """
    Fetches historical M5 data for a given symbol from MT5.
    """
    # Ensure connection
    if not mt5.initialize():
        logger.error(f"MT5 initialization failed in fetch_data. Error: {mt5.last_error()}")
        return pd.DataFrame()

    rates = mt5.copy_rates_from_pos(symbol, Settings.TIMEFRAME, 0, num_bars)
    
    if rates is None:
        logger.error(f"Failed to fetch data for {symbol} (Error: {mt5.last_error()})")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers the features required by the AI model.
    """
    if df.empty:
        return df

    # Copy to avoid SettingWithCopy warnings
    df = df.copy()

    # 1. Log Returns (Momentum)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

    # 2. Distance from 50 EMA (Trend)
    ema50 = ta.ema(df['close'], length=50)
    df['dist_ema'] = (df['close'] - ema50) / df['close']

    # 3. RSI (Oscillator) - Scaled 0-1
    df['rsi'] = ta.rsi(df['close'], length=14) / 100.0

    # 4. Volatility (ATR / Close)
    atr = ta.atr(df['high'], df['low'], df['close'], length=Settings.ATR_PERIOD)
    df['volatility'] = atr / df['close']

    # 5. Time Context (Hour scaled 0-1)
    df['hour'] = df.index.hour / 23.0

    # Drop NaNs created by indicators (e.g., EMA need 50 bars)
    df.dropna(inplace=True)

    # Ensure we only have the required columns for the model + OHLC used for trading logic
    # The model only sees Settings.FEATURES
    return df
