# config.py
import MetaTrader5 as mt5

class Settings:
    # --- MT5 Connection ---
    MT5_LOGIN = 000000000
    MT5_PASSWORD = "PASSWOED"
    MT5_SERVER = "Server-Demo01"
    MT5_PATH = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"

    # --- TRADING PAIRS & PROFILES ---
    PAIR_CONFIGS = {
        # --- Commodities ---
        'XAUUSD': {
            'spread': 0.20,
            'commission': 0.0,
            'scaling_factor': 5.0,
            'contract_size': 100,
            'reward_type': 'trend'
        },
        # --- Major Forex (USD base/quote) ---
        'EURUSD': {
            'spread': 0.00010,
            'commission': 0.0,
            'scaling_factor': 10000.0,
            'contract_size': 100000,
            'reward_type': 'mean_reversion'
        },
        'GBPUSD': {
            'spread': 0.00020,
            'commission': 0.0,
            'scaling_factor': 10000.0,
            'contract_size': 100000,
            'reward_type': 'trend'
        },
        'AUDUSD': {
            'spread': 0.00015,
            'commission': 0.0,
            'scaling_factor': 10000.0,
            'contract_size': 100000,
            'reward_type': 'mean_reversion'
        },
        'NZDUSD': {
            'spread': 0.00020,
            'commission': 0.0,
            'scaling_factor': 10000.0,
            'contract_size': 100000,
            'reward_type': 'mean_reversion'
        },
        'USDCAD': {
            'spread': 0.00020,
            'commission': 0.0,
            'scaling_factor': 10000.0,
            'contract_size': 100000,
            'reward_type': 'mean_reversion'
        },
        'USDCHF': {
            'spread': 0.00015,
            'commission': 0.0,
            'scaling_factor': 10000.0,
            'contract_size': 100000,
            'reward_type': 'mean_reversion'
        },
        # --- JPY pairs (1 pip = 0.01, scaling_factor = 100) ---
        'USDJPY': {
            'spread': 0.010,
            'commission': 0.0,
            'scaling_factor': 100.0,
            'contract_size': 100000,
            'reward_type': 'trend'
        },
        'GBPJPY': {
            'spread': 0.020,
            'commission': 0.0,
            'scaling_factor': 100.0,
            'contract_size': 100000,
            'reward_type': 'trend'
        },
        'EURJPY': {
            'spread': 0.015,
            'commission': 0.0,
            'scaling_factor': 100.0,
            'contract_size': 100000,
            'reward_type': 'trend'
        },
        'AUDJPY': {
            'spread': 0.015,
            'commission': 0.0,
            'scaling_factor': 100.0,
            'contract_size': 100000,
            'reward_type': 'trend'
        },
        'CADJPY': {
            'spread': 0.015,
            'commission': 0.0,
            'scaling_factor': 100.0,
            'contract_size': 100000,
            'reward_type': 'trend'
        },
        # --- Non-JPY Cross Pairs (1 pip = 0.0001, scaling_factor = 10000) ---
        'EURGBP': {
            'spread': 0.00015,
            'commission': 0.0,
            'scaling_factor': 10000.0,
            'contract_size': 100000,
            'reward_type': 'mean_reversion'
        },
        'EURAUD': {
            'spread': 0.00025,
            'commission': 0.0,
            'scaling_factor': 10000.0,
            'contract_size': 100000,
            'reward_type': 'trend'
        },
        'GBPAUD': {
            'spread': 0.00030,
            'commission': 0.0,
            'scaling_factor': 10000.0,
            'contract_size': 100000,
            'reward_type': 'trend'
        },
        'EURCAD': {
            'spread': 0.00025,
            'commission': 0.0,
            'scaling_factor': 10000.0,
            'contract_size': 100000,
            'reward_type': 'mean_reversion'
        },
        'GBPCAD': {
            'spread': 0.00030,
            'commission': 0.0,
            'scaling_factor': 10000.0,
            'contract_size': 100000,
            'reward_type': 'trend'
        },
        # --- Indices (CFD, 1 lot = 1 contract) ---
        'NAS100': {
            'spread': 1.5,
            'commission': 0.0,
            'scaling_factor': 1.0,
            'contract_size': 1,
            'reward_type': 'trend'
        },
        'US30': {
            'spread': 3.0,
            'commission': 0.0,
            'scaling_factor': 1.0,
            'contract_size': 1,
            'reward_type': 'trend'
        },
    }
    
    PAIRS = list(PAIR_CONFIGS.keys())

    # --- CORE TRADING PARAMETERS ---
    # M15 timeframe (alpha proven at 55.1% directional accuracy)
    TIMEFRAME = mt5.TIMEFRAME_M15
    MAGIC_NUMBER_BASE = 202400

    # --- FEATURE SET (35 features, alpha-proven at 55.1%) ---
    FEATURES = [
        # --- Time Structure (6) ---
        'hour_sin', 'hour_cos',
        'dow_sin', 'dow_cos',
        'session_london', 'session_nyc',
        # --- Returns & Momentum (5) ---
        'ret_1', 'ret_5', 'ret_20',
        'rsi_14', 'roc_10',
        # --- Mean Reversion Signals (4) ---
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

    # --- AI MODEL ARCHITECTURE ---
    INPUT_DIM = len(FEATURES)  # 35 features
    SEQUENCE_LENGTH = 48       # 48 M15 bars = 12 hours lookback
    ENCODER_DIM = 64           # Wider for 35 features
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.15

    # Semantic Action Space (HOLD/BUY/SELL/CLOSE)
    OUTPUT_DIM = 4
    ACTION_NAMES = ['HOLD', 'BUY', 'SELL', 'CLOSE']
    # ACTION_MAP not used anymore — actions are semantic, not position fractions
    MAX_LOT_SIZE = 0.05

    # --- TRAINING HYPERPARAMETERS ---
    EPOCHS = 30
    BATCH_SIZE = 64            # Smaller for sequential environment
    LEARNING_RATE = 0.0003     # Slightly higher for faster convergence
    GAMMA = 0.95               # Long horizon for holding trades
    EPSILON_START = 1.0
    EPSILON_DECAY = 0.9997     # Per-step decay
    EPSILON_MIN = 0.05
    EPSILON_DECAY_PER_STEP = True  # Decay every step, not epoch

    # DDQN Target Network
    TARGET_UPDATE_FREQ = 500
    TAU = 0.005

    # LR Scheduling
    LR_SCHEDULER_PATIENCE = 5
    LR_SCHEDULER_FACTOR = 0.5

    # PER (Prioritized Experience Replay)
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    MEMORY_CAPACITY = 100000

    # Online Learning (Live Adaptation)
    ONLINE_LR = 1e-6
    ONLINE_BUFFER_SIZE = 1000
    ONLINE_UPDATE_STEPS = 3
    LIVE_EPSILON = 0.05

    # Reward Parameters
    TRANSACTION_COST_BPS = 2   # Applied only on position CHANGE
    REWARD_CLIP = 10.0

    # Rolling Window Training (in M15 bars)
    TRAIN_WINDOW = 50000       # ~521 trading days of M15
    VAL_WINDOW = 8000          # ~83 trading days
    TEST_WINDOW = 8000         # ~83 trading days
    EARLY_STOP_PATIENCE = 10

    # Feature Engineering
    STANDARDIZE_WINDOW = 60
    FEATURE_WARMUP = 250       # Bars to skip for indicator warmup

    # --- DATA PARAMETERS ---
    TRAIN_DATA_BARS = 450000   # M5 bars fetched, resampled to M15
    INIT_DATA_BARS = 500

    # --- RISK MANAGEMENT ---
    ATR_PERIOD = 14
    ATR_SL_MULTIPLIER = 2.5
    ATR_TP_MULTIPLIER = 3.0
    SPREAD_FILTER_POINTS = 30
    MAX_DAILY_LOSS_PCT = 2.0   # Max daily drawdown percent
    MAX_POSITION_SIZE = 0.1    # Max lot size
    MAX_CONCURRENT_TRADES = 1  # Only 1 position at a time

    # P2: Overnight Financing (CFD-specific)
    SWAP_RATE_ANNUAL = 0.05
