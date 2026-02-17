# src/data_factory.py
"""
Data pipeline: MT5 data fetching + feature engineering (35 features).

Features:
- Time structure (6): hour/dow cyclical, session flags
- Returns & Momentum (5): standardized returns, RSI, ROC
- Mean Reversion (4): z-scores, BB position, VWAP deviation
- Trend (4): EMA distances, EMA slope, ADX
- Volatility (4): ATR normalized, BB width, vol ratio, ATR percentile
- Candle (3): body/wick ratios
- Volume Dynamics (3): volume ratio/trend, price-volume correlation
- Session Context (4): session high/low distances, range position/ratio
- Multi-TF (2): H1 trend, H1 momentum
"""

import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
from config import Settings
from src.utils import logger

MAX_BARS_SAFETY_LIMIT = 1000000

# Feature list — 35 features total
FEATURES = [
    # --- Time Structure (6) ---
    'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos',
    'session_london', 'session_nyc',
    # --- Returns & Momentum (5) ---
    'ret_1', 'ret_5', 'ret_20',
    'rsi_14', 'roc_10',
    # --- Mean Reversion (4) ---
    'zscore_20', 'zscore_50',
    'bb_position', 'vwap_deviation',
    # --- Trend Regime (4) ---
    'dist_ema_20', 'dist_ema_50',
    'ema_slope_20', 'adx_14',
    # --- Volatility (4) ---
    'atr_normalized', 'bb_width',
    'volatility_ratio', 'atr_percentile',
    # --- Candle Structure (3) ---
    'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio',
    # --- Volume Dynamics (3) ---
    'volume_ratio', 'volume_trend', 'price_volume_corr',
    # --- Session Context (4) ---
    'dist_session_high', 'dist_session_low',
    'session_range_position', 'session_range_ratio',
    # --- Multi-TF Context (2) ---
    'h1_trend', 'h1_momentum',
]


# ================================================================
# DATA FETCHING
# ================================================================

def fetch_data(symbol: str, num_bars: int) -> pd.DataFrame:
    """
    Fetch historical data from MT5 using chunked backwards retrieval.
    Returns DataFrame with datetime index and OHLCV columns.
    """
    if num_bars > MAX_BARS_SAFETY_LIMIT:
        logger.warning(f"Requested {num_bars} bars exceeds safety limit. Capping at {MAX_BARS_SAFETY_LIMIT}.")
        num_bars = MAX_BARS_SAFETY_LIMIT

    if not mt5.initialize(path=Settings.MT5_PATH, login=Settings.MT5_LOGIN,
                         password=Settings.MT5_PASSWORD, server=Settings.MT5_SERVER):
        logger.error(f"MT5 initialization failed: {mt5.last_error()}")
        return pd.DataFrame()

    terminal_info = mt5.terminal_info()
    max_bars = terminal_info.maxbars if terminal_info else 0
    if max_bars > 0 and num_bars > max_bars:
        logger.warning(f"Requested {num_bars} bars, terminal max is {max_bars}.")

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

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[~df.index.duplicated(keep='last')]
    return df


# ================================================================
# RESAMPLING
# ================================================================

def resample_ohlcv(df, rule):
    """Resample to higher timeframe (e.g. '15min', '1h')."""
    if rule is None:
        return df
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'tick_volume' in df.columns:
        agg['tick_volume'] = 'sum'
    if 'real_volume' in df.columns:
        agg['real_volume'] = 'sum'
    return df.resample(rule).agg(agg).dropna()


# ================================================================
# FEATURE ENGINEERING (35 features)
# ================================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 35 features for intraday trading.
    Expects OHLCV data with datetime index (M5 or M15).
    """
    if df.empty:
        return df

    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)

    df['mid_price'] = (df['high'] + df['low']) / 2.0

    # 1. TIME STRUCTURE
    hour = df.index.hour + df.index.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    dow = df.index.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 5.0)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 5.0)
    utc_hour = df.index.hour
    df['session_london'] = ((utc_hour >= 7) & (utc_hour < 16)).astype(float)
    df['session_nyc'] = ((utc_hour >= 13) & (utc_hour < 22)).astype(float)

    # 2. BASE
    atr = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr'] = atr.bfill().fillna(df['close'] * 0.001)

    # Relative volatility (for backtester/live compatibility)
    df['volatility'] = df['atr'] / df['close']

    # 3. RETURNS & MOMENTUM
    std_window = 60
    for lag in [1, 5, 20]:
        raw_ret = df['close'].pct_change(lag)
        rm = raw_ret.rolling(std_window).mean()
        rs = raw_ret.rolling(std_window).std().replace(0, np.nan).fillna(raw_ret.std())
        df[f'ret_{lag}'] = (raw_ret - rm) / rs

    rsi = ta.rsi(df['close'], length=14)
    df['rsi_14'] = (rsi / 100.0) if rsi is not None else 0.5

    roc = ta.roc(df['close'], length=10)
    df['roc_10'] = (roc / 100.0) if roc is not None else 0.0

    # 4. MEAN REVERSION
    for window in [20, 50]:
        roll_mean = df['close'].rolling(window).mean()
        roll_std = df['close'].rolling(window).std().replace(0, np.nan).fillna(1e-8)
        df[f'zscore_{window}'] = (df['close'] - roll_mean) / roll_std

    bb = ta.bbands(df['close'], length=20, std=2)
    if bb is not None and not bb.empty:
        bbl, bbm, bbu = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
        bb_range = (bbu - bbl).replace(0, np.nan).fillna(1e-8)
        df['bb_position'] = (df['close'] - bbl) / bb_range
        df['bb_width'] = (bbu - bbl) / df['close']
    else:
        df['bb_position'], df['bb_width'] = 0.5, 0.0

    _compute_vwap(df)

    # 5. TREND REGIME
    ema_20 = ta.ema(df['close'], length=20)
    ema_50 = ta.ema(df['close'], length=50)
    df['dist_ema_20'] = (df['close'] - ema_20) / df['atr']
    df['dist_ema_50'] = (df['close'] - ema_50) / df['atr']
    df['ema_slope_20'] = (ema_20 - ema_20.shift(5)) / df['atr']

    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx_df is not None and not adx_df.empty:
        df['adx_14'] = adx_df.iloc[:, 0] / 100.0
    else:
        df['adx_14'] = 0.0

    # 6. VOLATILITY
    df['atr_normalized'] = df['atr'] / df['close']
    rolling_vol = df['close'].pct_change().rolling(20).std()
    rv_mean = rolling_vol.rolling(100).mean().replace(0, np.nan).fillna(rolling_vol.mean())
    df['volatility_ratio'] = rolling_vol / rv_mean
    atr_roll_min = df['atr'].rolling(100, min_periods=1).min()
    atr_roll_max = df['atr'].rolling(100, min_periods=1).max()
    atr_range = (atr_roll_max - atr_roll_min).replace(0, np.nan).fillna(1e-8)
    df['atr_percentile'] = (df['atr'] - atr_roll_min) / atr_range

    # 7. CANDLE STRUCTURE
    df['body_ratio'] = (df['close'] - df['open']) / df['atr']
    df['upper_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['atr']
    df['lower_wick_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['atr']

    # 8. VOLUME DYNAMICS
    vol_col = 'tick_volume' if 'tick_volume' in df.columns else 'real_volume'
    if vol_col in df.columns:
        tv = df[vol_col].astype(float).replace(0, np.nan).fillna(1)
        vol_avg = tv.rolling(20).mean().replace(0, np.nan).fillna(tv.mean())
        df['volume_ratio'] = tv / vol_avg
        df['volume_trend'] = (tv.rolling(10).mean() - tv.rolling(20).mean()) / (tv.rolling(20).mean() + 1e-8)
        df['price_volume_corr'] = df['close'].pct_change().rolling(10).corr(tv.pct_change())
    else:
        df['volume_ratio'], df['volume_trend'], df['price_volume_corr'] = 1.0, 0.0, 0.0

    # 9. SESSION CONTEXT
    _compute_session_levels(df)

    # 10. MULTI-TF CONTEXT
    _compute_mtf_context(df)

    # CLEANUP
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    for feat in FEATURES:
        if feat in df.columns:
            df[feat] = df[feat].clip(-10, 10)

    df = df.iloc[250:]  # Drop warmup rows
    return df


def _compute_vwap(df):
    """VWAP deviation (resets each day)."""
    vol_col = 'tick_volume' if 'tick_volume' in df.columns else 'real_volume'
    tv = df[vol_col].astype(float).replace(0, 1) if vol_col in df.columns else pd.Series(1.0, index=df.index)
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    dg = df.index.date
    cum_tp_vol = (tp * tv).groupby(dg).cumsum()
    cum_vol = tv.groupby(dg).cumsum().replace(0, np.nan).fillna(1)
    vwap = cum_tp_vol / cum_vol
    df['vwap_deviation'] = (df['close'] - vwap) / df['atr']


def _compute_session_levels(df):
    """Session-relative price levels (fully vectorized)."""
    dg = df.index.date
    session_high = df['high'].groupby(dg).cummax()
    session_low = df['low'].groupby(dg).cummin()
    session_range = (session_high - session_low).replace(0, np.nan).fillna(1e-8)

    df['dist_session_high'] = (session_high - df['close']) / df['atr']
    df['dist_session_low'] = (df['close'] - session_low) / df['atr']
    df['session_range_position'] = (df['close'] - session_low) / session_range

    daily_high = df['high'].resample('D').max()
    daily_low = df['low'].resample('D').min()
    daily_range = (daily_high - daily_low).dropna()
    avg_range = daily_range.rolling(20, min_periods=1).mean()
    ratio = daily_range / avg_range.replace(0, np.nan).fillna(1e-8)
    date_idx = pd.Series(df.index.date, index=df.index)
    df['session_range_ratio'] = date_idx.map(ratio.to_dict()).fillna(1.0).values


def _compute_mtf_context(df):
    """H1 context from intraday data."""
    h1_ema = ta.ema(df['close'], length=48)
    df['h1_trend'] = (h1_ema - h1_ema.shift(12)) / df['atr']
    h1_rsi = ta.rsi(df['close'], length=56)
    df['h1_momentum'] = (h1_rsi / 100.0) if h1_rsi is not None else 0.5
