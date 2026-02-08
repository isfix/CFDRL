# src/data_factory.py
import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
import re
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
    
    # Only warn if max_bars is actually reported (>0) and we exceed it
    if max_bars > 0 and num_bars > max_bars:
        logger.warning(f"Requested {num_bars} bars, but terminal 'Max bars in chart' is {max_bars}.")
        logger.warning("You may need to increase this setting in MT5 (Tools > Options > Charts).")
    
    # Chunked Fetch Strategy (Backwards)
    all_rates = []
    total_fetched = 0
    
    # Start from "now" + buffer (to ensure we get the latest candle)
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
    
    # Known columns for parsing validation
    known_cols = ['open', 'high', 'low', 'close', 'mid_price']

    # --- DYNAMIC FEATURE GENERATION (Robust Regex) ---
    for feature in Settings.FEATURES:
        if feature in df.columns:
            continue
            
        try:
            # ROBUST PARSING via Regex
            # 1. Relations: rel_A_B -> A / B
            # 2. Spreads: spread_A_B -> A - B
            # 3. Ratio Lag: ratio_A_B_lagN or ratio_A_lagN
            # 4. Diff Lag: diff_A_B_lagN or diff_A_lagN
            
            if feature.startswith('rel_'):
                # rel_(colA)_(colB)
                # Greedy match for known columns
                content = feature[4:] # Strip 'rel_'
                c1, c2 = None, None
                for k in known_cols:
                    if content.startswith(k + '_'):
                        c1 = k
                        c2 = content[len(k)+1:]
                        break
                if c1 and c2 in known_cols:
                    df[feature] = df[c1] / df[c2]

            elif feature.startswith('spread_'):
                content = feature[7:] # Strip 'spread_'
                c1, c2 = None, None
                for k in known_cols:
                    if content.startswith(k + '_'):
                        c1 = k
                        c2 = content[len(k)+1:]
                        break
                if c1 and c2 in known_cols:
                    df[feature] = df[c1] - df[c2]

            elif feature.startswith('ratio_'):
                # ratio_close_lag1 or ratio_close_mid_price_lag1
                content = feature[6:]
                match = re.match(r'(.+)_lag(\d+)$', content)
                if match:
                    base_part = match.group(1)
                    lag = int(match.group(2))
                    
                    if base_part in known_cols:
                        df[feature] = df[base_part] / df[base_part].shift(lag)
                    else:
                        # try split base_part into c1_c2
                        c1, c2 = None, None
                        for k in known_cols:
                            if base_part.startswith(k + '_'):
                                c1 = k
                                c2 = base_part[len(k)+1:]
                                break
                        if c1 and c2 in known_cols:
                             df[feature] = df[c1] / df[c2].shift(lag)

            elif feature.startswith('diff_'):
                # diff_close_lag1 or diff_close_mid_price_lag1
                content = feature[5:]
                match = re.match(r'(.+)_lag(\d+)$', content)
                if match:
                    base_part = match.group(1)
                    lag = int(match.group(2))
                    
                    if base_part in known_cols:
                        df[feature] = df[base_part] - df[base_part].shift(lag)
                    else:
                        # try split base_part into c1_c2
                        c1, c2 = None, None
                        for k in known_cols:
                            if base_part.startswith(k + '_'):
                                c1 = k
                                c2 = base_part[len(k)+1:]
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
