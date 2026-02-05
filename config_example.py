# config.py
import MetaTrader5 as mt5

class Settings:
    # --- MT5 Connection ---
    # (Fill these with your actual credentials)
    MT5_LOGIN = id
    MT5_PASSWORD = "password"
    MT5_SERVER = "Server-Example"
    MT5_PATH = "C:\\Program Files\\MetaTrader 5\\terminal64.exe" # Adjust path if needed

    # --- TRADING PAIRS ---
    # The list of symbols the bot will train and trade.
    # To add a pair, just add its name here.
    PAIRS = ['XAUUSD', 'EURUSD', 'GBPUSD']

    # --- CORE TRADING PARAMETERS ---
    TIMEFRAME = mt5.TIMEFRAME_M5
    MAGIC_NUMBER_BASE = 202400  # Base magic number. Bot will add index (e.g., XAUUSD=202400, EURUSD=202401)

    # --- AI MODEL ARCHITECTURE ---
    # These define the structure of the LSTM brain.
    SEQUENCE_LENGTH = 60  # Lookback window: 60 candles = 5 hours on M5
    FEATURES = ['log_ret', 'dist_ema', 'rsi', 'volatility', 'hour']
    INPUT_DIM = len(FEATURES)
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.2
    OUTPUT_DIM = 3  # 3 Actions: 0=Hold, 1=Buy, 2=Sell

    # --- TRAINING HYPERPARAMETERS ---
    # Controls how the AI learns.
    EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    GAMMA = 0.99  # Discount factor for future rewards in RL
    EPSILON_START = 1.0  # Exploration rate (starts at 100%)
    EPSILON_DECAY = 0.995 # Decay rate per epoch
    EPSILON_MIN = 0.01   # Minimum exploration rate
    REPLAY_MEMORY_SIZE = 10000 # How many experiences to store for training

    # --- DATA PARAMETERS ---
    TRAIN_DATA_BARS = 450000 # Increased for Deep Learning
    TRAIN_SPLIT_INDEX = 420000 # First 420k for training
    TEST_SPLIT_INDEX = 420000 # Remaining for testing (420k to end)
    INIT_DATA_BARS = 200    # Number of bars to fetch on bot startup

    # --- RISK MANAGEMENT ---
    # Controls how trades are managed.
    ATR_PERIOD = 14
    ATR_SL_MULTIPLIER = 2.5 # Stop Loss = 2.5 * ATR
    SPREAD_FILTER_POINTS = 30 # For XAUUSD, 30 points = $0.30. Don't trade if spread is wider.
