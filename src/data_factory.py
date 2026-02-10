# src/data_factory.py
import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
from config import Settings
from src.utils import logger

# Safety Limit to prevent OOM / DoS
MAX_BARS_SAFETY_LIMIT = 1000000 

def fetch_data(symbol: str, num_bars: int) -> pd.DataFrame:
    """
    Fetches historical M5 data for a given symbol from MT5.
    Uses chunked fetching to maximize retrieval within terminal limits.
    """
    if num_bars > MAX_BARS_SAFETY_LIMIT:
        logger.warning(f"Requested {num_bars} bars exceeds safety limit. Capping at {MAX_BARS_SAFETY_LIMIT}.")
        num_bars = MAX_BARS_SAFETY_LIMIT
        
    # Ensure connection
    if not mt5.initialize(path=Settings.MT5_PATH, login=Settings.MT5_LOGIN, password=Settings.MT5_PASSWORD, server=Settings.MT5_SERVER):
        logger.error(f"MT5 initialization failed in fetch_data. Path: {Settings.MT5_PATH}")
        logger.error(f"Error: {mt5.last_error()}")
        return pd.DataFrame()

    # Check terminal limit
    terminal_info = mt5.terminal_info()
    max_bars = terminal_info.maxbars if terminal_info else 0
    
    if max_bars > 0 and num_bars > max_bars:
        logger.warning(f"Requested {num_bars} bars, but terminal 'Max bars in chart' is {max_bars}.")
        logger.warning("You may need to increase this setting in MT5 (Tools > Options > Charts).")
    
    # Chunked Fetch Strategy (Backwards)
    all_rates = []
    total_fetched = 0
    
    from datetime import datetime, timedelta
    current_date = datetime.now() + timedelta(minutes=Settings.TIMEFRAME * 2) 
    
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
        
        if len(rates) < 10: 
            break
            
    if not all_rates:
        logger.error(f"Failed to fetch data for {symbol}.")
        return pd.DataFrame()
        
    rates = np.concatenate(all_rates)
    
    if len(rates) > num_bars:
         rates = rates[-num_bars:]
         
    logger.info(f"Successfully fetched {len(rates)} bars for {symbol}.")
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='last')]
    
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 26: Research-driven feature engineering.
    
    Key changes from previous version:
    - Standardized returns instead of raw price ratios
    - ATR-normalized candle structure
    - Regime indicators as features (not hard filters)
    - Correlation-filtered feature set
    """
    if df.empty:
        return df

    df = df.copy()
    
    # --- BASE CALCULATIONS ---
    df['mid_price'] = (df['high'] + df['low']) / 2.0
    
    # ATR (base for normalization)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr'] = df['atr'].fillna(method='bfill').fillna(df['close'] * 0.001)  # Fallback
    
    # Relative volatility (for backtester compatibility)
    df['volatility'] = df['atr'] / df['close']
    
    # --- 1. STANDARDIZED RETURNS (Core Features) ---
    for lag in [1, 2, 5, 10, 20]:
        raw_ret = df['close'].pct_change(lag)
        roll_mean = raw_ret.rolling(Settings.STANDARDIZE_WINDOW).mean()
        roll_std = raw_ret.rolling(Settings.STANDARDIZE_WINDOW).std()
        # Avoid division by zero
        roll_std = roll_std.replace(0, np.nan).fillna(raw_ret.std())
        df[f'ret_close_{lag}'] = (raw_ret - roll_mean) / roll_std
    
    # --- 2. CANDLE STRUCTURE (ATR-Normalized) ---
    df['body_ratio'] = (df['close'] - df['open']) / df['atr']
    df['upper_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['atr']
    df['lower_wick_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['atr']
    df['range_ratio'] = (df['high'] - df['low']) / df['atr']
    
    # --- 3. VOLATILITY FEATURES ---
    rolling_vol = df['close'].pct_change().rolling(Settings.VOLATILITY_WINDOW).std()
    rolling_vol_mean = rolling_vol.rolling(100).mean()
    df['volatility_ratio'] = rolling_vol / rolling_vol_mean.replace(0, np.nan).fillna(rolling_vol.mean())
    df['atr_normalized'] = df['atr'] / df['close']
    
    # --- 4. MOMENTUM (Normalized) ---
    rsi = ta.rsi(df['close'], length=14)
    df['rsi_14'] = rsi / 100.0  # Scale to [0, 1]
    
    macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        macd_col = [c for c in macd_df.columns if 'MACDh' in c or 'MACD_' in c]
        signal_col = [c for c in macd_df.columns if 'MACDs' in c]
        if macd_col and signal_col:
            df['macd_signal_dist'] = (macd_df[macd_col[0]] - macd_df[signal_col[0]]) / df['atr']
        else:
            df['macd_signal_dist'] = 0.0
    else:
        df['macd_signal_dist'] = 0.0
    
    roc = ta.roc(df['close'], length=10)
    df['roc_10'] = roc / 100.0 if roc is not None else 0.0
    
    # --- 5. TREND REGIME (Learned by agent, not hard-filtered) ---
    ema_50 = ta.ema(df['close'], length=50)
    ema_200 = ta.ema(df['close'], length=200)
    df['ema_200'] = ema_200  # Keep for backtester/live reference
    
    df['dist_ema_50'] = (df['close'] - ema_50) / df['atr']
    df['dist_ema_200'] = (df['close'] - ema_200) / df['atr']
    df['ema_slope_50'] = (ema_50 - ema_50.shift(5)) / df['atr']
    
    # --- 6. BOLLINGER BAND REGIME ---
    bb = ta.bbands(df['close'], length=20, std=2)
    if bb is not None and not bb.empty:
        bb_cols = bb.columns
        bbl = bb[bb_cols[0]]  # Lower
        bbm = bb[bb_cols[1]]  # Mid
        bbu = bb[bb_cols[2]]  # Upper
        bb_range = (bbu - bbl).replace(0, np.nan).fillna(1e-8)
        df['bb_position'] = (df['close'] - bbl) / bb_range
        df['bb_width'] = (bbu - bbl) / df['close']
    else:
        df['bb_position'] = 0.5
        df['bb_width'] = 0.0
    
    # --- 7. ADX / DI (Learned, not filtered) ---
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx_df is not None and not adx_df.empty:
        adx_cols = adx_df.columns
        df['adx_normalized'] = adx_df[adx_cols[0]] / 100.0  # ADX
        # DI+ - DI-
        if len(adx_cols) >= 3:
            df['di_diff'] = (adx_df[adx_cols[1]] - adx_df[adx_cols[2]]) / 100.0
        else:
            df['di_diff'] = 0.0
    else:
        df['adx_normalized'] = 0.0
        df['di_diff'] = 0.0
    
    # --- CLEANUP ---
    # Fill NaNs (created by lags/indicators)
    df.fillna(0, inplace=True)
    
    # Replace infinities
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Clip extreme values to prevent gradient explosion
    for feat in Settings.FEATURES:
        if feat in df.columns:
            df[feat] = df[feat].clip(-10, 10)
    
    # Drop initial rows that are invalid due to large lags (EMA 200 needs 200+ bars)
    df = df.iloc[250:]
    
    return df
