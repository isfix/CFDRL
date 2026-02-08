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
    Uses chunked fetching to maximize retrieval within terminal limits.
    """
    # Ensure connection
    if not mt5.initialize(path=Settings.MT5_PATH, login=Settings.MT5_LOGIN, password=Settings.MT5_PASSWORD, server=Settings.MT5_SERVER):
        logger.error(f"MT5 initialization failed in fetch_data. Path: {Settings.MT5_PATH}")
        logger.error(f"Error: {mt5.last_error()}")
        return pd.DataFrame()

    # Check terminal limit
    terminal_info = mt5.terminal_info()
    max_bars = terminal_info.maxbars if terminal_info else 0
    if num_bars > max_bars:
        logger.warning(f"Requested {num_bars} bars, but terminal 'Max bars in chart' is {max_bars}.")
        logger.warning("You may need to increase this setting in MT5 (Tools > Options > Charts).")
    
    # Chunked Fetch Strategy (Backwards)
    all_rates = []
    total_fetched = 0
    
    # Start from "now" + buffer (to ensure we get the latest candle)
    # Actually, copy_rates_range 'date_to' is exclusive usually.
    from datetime import datetime, timedelta
    current_date = datetime.now() + timedelta(minutes=Settings.TIMEFRAME * 2) 
    
    # 30 day chunks
    chunk_size_days = 30
    
    logger.info(f"Fetching approximately {num_bars} bars for {symbol}...")
    
    while total_fetched < num_bars:
        start_date = current_date - timedelta(days=chunk_size_days)
        
        rates = mt5.copy_rates_range(symbol, Settings.TIMEFRAME, start_date, current_date)
        
        if rates is None or len(rates) == 0:
            logger.warning(f"No data received between {start_date} and {current_date}. Stopping fetch.")
            break
            
        all_rates.insert(0, rates)
        total_fetched += len(rates)
        
        current_date = start_date
        
        # Safety break if we are getting very few bars (end of history)
        if len(rates) < 10: 
            break
            
    if not all_rates:
        logger.error(f"Failed to fetch data for {symbol}.")
        return pd.DataFrame()
        
    rates = np.concatenate(all_rates)
    
    # Truncate if we got too many (from the start/oldest)
    if len(rates) > num_bars:
         rates = rates[-num_bars:]
         
    logger.info(f"Successfully fetched {len(rates)} bars for {symbol}.")
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # Remove duplicates just in case
    df = df[~df.index.duplicated(keep='last')]
    
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers the features required by the AI model.
    """
    if df.empty:
        return df

    # Copy to avoid SettingWithCopy warnings
    df = df.copy()

    # 1. Log Returns (Momentum) - Scaled x1000
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1)) * 1000.0

    # 2. Distance from 50 EMA (Trend) - Scaled x1000
    ema50 = ta.ema(df['close'], length=50)
    df['dist_ema'] = ((df['close'] - ema50) / df['close']) * 1000.0

    # 3. RSI (Oscillator) - Scaled 0-1
    df['rsi'] = ta.rsi(df['close'], length=14) / 100.0

    # 4. ROC (Velocity) - Scaled x1000 (New "Godlike" Feature)
    # Rate of change over 3 bars to detect immediate momentum
    df['roc'] = ta.roc(df['close'], length=3) * 10.0 # ROC is usually 0.01-0.5, *10 makes it 0.1-5.0
    df.fillna(0, inplace=True) # Handle initial NaNs form ROC
    
    # 5. Volatility (ATR / Close) - Scaled x1000
    atr = ta.atr(df['high'], df['low'], df['close'], length=Settings.ATR_PERIOD)
    df['volatility'] = (atr / df['close']) * 1000.0

    # 5. Time Context (Hour scaled 0-1)
    df['hour'] = df.index.hour / 23.0

    # Bollinger Bands (Step 3: Squeeze Filter)
    # Appends BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
    bb = ta.bbands(df['close'], length=20, std=2)
    bb.columns = ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0']
    df = pd.concat([df, bb], axis=1)

    # Drop NaNs created by indicators (e.g., EMA need 50 bars)
    df.dropna(inplace=True)

    # Ensure we only have the required columns for the model + OHLC used for trading logic
    # The model only sees Settings.FEATURES
    return df
