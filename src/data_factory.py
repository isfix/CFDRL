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
    Dynamically generates features listed in Settings.FEATURES.
    """
    if df.empty:
        return df

    # Copy to avoid SettingWithCopy warnings
    df = df.copy()

    # --- MID-PRICE (Central to our noise reduction strategy) ---
    df['mid_price'] = (df['high'] + df['low']) / 2.0
    
    # --- INDICATORS FOR FILTERS (Calculated regardless of Feature usage) ---
    # Bollinger Bands (Squeeze Filter)
    bb = ta.bbands(df['mid_price'], length=20, std=2)
    bb.columns = ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0']
    df = pd.concat([df, bb], axis=1)

    # ADX (Trend Filter)
    # Always calculate ADX as it's used in filters, even if not in FEATURES
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx_df['ADX_14'] / 100.0
    
    # --- DYNAMIC FEATURE GENERATION ---
    for feature in Settings.FEATURES:
        if feature in df.columns:
            continue
            
        try:
            # Parse Feature Name
            parts = feature.split('_')
            type_ = parts[0] # rel, spread, ratio, diff
            
            # 1. Relations (Ratio): rel_A_B -> A / B
            if type_ == 'rel':
                col_a = '_'.join(parts[1:-1]) # Handle mid_price (2 words)
                col_b = parts[-1]
                
                # Fix for multi-word columns like mid_price
                # If parts has 3 items: rel, mid, price? No.
                # parts: ['rel', 'mid', 'price', 'close'] -> col_a: 'mid_price', col_b: 'close'
                # parts: ['rel', 'close', 'mid', 'price'] -> col_a: 'close', col_b: 'mid_price'?
                # This split logic is fragile. 
                # Better: check if known columns exist.
                known_cols = ['open', 'high', 'low', 'close', 'mid_price']
                
                # Re-parse robustly
                # Find which known col is at the start of the rest?
                rest = feature.replace('rel_', '')
                
                # Try to find split point
                c1, c2 = None, None
                for k in known_cols:
                    if rest.startswith(k + '_'):
                        c1 = k
                        c2 = rest.replace(k + '_', '')
                        break
                
                if c1 and c2 in known_cols:
                    df[feature] = df[c1] / df[c2]
                    
            # 2. Spreads (Diff): spread_A_B -> A - B
            elif type_ == 'spread':
                rest = feature.replace('spread_', '')
                c1, c2 = None, None
                known_cols = ['open', 'high', 'low', 'close', 'mid_price']
                for k in known_cols:
                    if rest.startswith(k + '_'):
                        c1 = k
                        c2 = rest.replace(k + '_', '')
                        break
                
                if c1 and c2 in known_cols:
                    df[feature] = df[c1] - df[c2]

            # 3. Ratio Lag: ratio_A_B_lagN -> A / B.shift(N) (or A/A.shift(N) if B is missing)
            elif type_ == 'ratio':
                rest = feature.replace('ratio_', '')
                if '_lag' in rest:
                    base, lag_str = rest.split('_lag')
                    lag = int(lag_str)
                    
                    # Try to split base into c1, c2
                    known_cols = ['open', 'high', 'low', 'close', 'mid_price']
                    c1, c2 = None, None
                    
                    # Check if base is a single column first
                    if base in known_cols:
                        df[feature] = df[base] / df[base].shift(lag)
                    else:
                        for k in known_cols:
                            if base.startswith(k + '_'):
                                c1 = k
                                c2 = base.replace(k + '_', '')
                                break
                        
                        if c1 and c2 in known_cols:
                            df[feature] = df[c1] / df[c2].shift(lag)

            # 4. Diff Lag: diff_A_B_lagN -> A - B.shift(N) (or A - A.shift(N))
            elif type_ == 'diff':
                rest = feature.replace('diff_', '')
                if '_lag' in rest:
                    base, lag_str = rest.split('_lag')
                    lag = int(lag_str)
                    
                    known_cols = ['open', 'high', 'low', 'close', 'mid_price']
                    c1, c2 = None, None
                    
                    # Check if base is a single column first
                    if base in known_cols:
                        df[feature] = df[base] - df[base].shift(lag)
                    else:
                        for k in known_cols:
                            if base.startswith(k + '_'):
                                c1 = k
                                c2 = base.replace(k + '_', '')
                                break
                        
                        if c1 and c2 in known_cols:
                             df[feature] = df[c1] - df[c2].shift(lag)
        except Exception as e:
            logger.warning(f"Could not generate feature {feature}: {e}")

    # Fill NaNs (created by lags/indicators)
    df.fillna(0, inplace=True)
    
    # Drop initial rows that might be 0/invalid due to large lags
    # Using a safe margin
    df = df.iloc[50:]
    
    return df
