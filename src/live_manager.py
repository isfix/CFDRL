# src/live_manager.py
import MetaTrader5 as mt5
import torch
import pandas as pd
import time
from datetime import datetime
import os
import sys

# Add parent directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Settings
from src import data_factory
from src.brain import QNetwork
from src.utils import logger
import os
from concurrent.futures import ThreadPoolExecutor

# --- Global State ---
active_models = {}  # {symbol: model_instance}
market_state = {}   # {symbol: DataFrame}

def load_models():
    """Load trained PyTorch models for each pair."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading models on {device}...")
    
    for symbol in Settings.PAIRS:
        model_path = f"models/{symbol}_brain.pth"
        if not os.path.exists(model_path):
            logger.error(f"Model not found for {symbol} at {model_path}. Skipping.")
            continue
            
        model = QNetwork().to(device)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval() # Set to evaluation mode
            active_models[symbol] = model
            logger.info(f"Loaded model for {symbol}")
        except Exception as e:
            logger.error(f"Failed to load model for {symbol}: {e}")

def init_market_state():
    """Initialize market data buffer for feature calculation."""
    logger.info("Initializing market state...")
    for symbol in Settings.PAIRS:
        # We need enough data to calculate indicators (e.g. EMA 50) + Sequence Length
        # 200 bars is safe enough for INIT
        df = data_factory.fetch_data(symbol, Settings.INIT_DATA_BARS)
        if not df.empty:
            market_state[symbol] = df
            logger.info(f"Initialized state for {symbol} with {len(df)} bars")
        else:
            logger.error(f"Failed to fetch init data for {symbol}")

def update_market_state():
    """
    Efficiency Hack: Only fetch last few bars and append to state.
    """
    def process_symbol(symbol):
        try:
            if symbol not in market_state:
                return

            # Fetch last 2 bars (Current open candle + last closed candle)
            # We really only need the latest closed candle to update our history
            new_data = data_factory.fetch_data(symbol, 2)
            if new_data.empty:
                return

            current_df = market_state[symbol]

            # Merge and Remove duplicates based on index (Time)
            # Efficiency: Check index of new data against current data before concatenating
            # Remove rows from current_df that are in new_data (to be updated)
            mask = ~current_df.index.isin(new_data.index)
            combined = pd.concat([current_df[mask], new_data])

            # Keep window size reasonable (don't let it grow effectively infinite)
            # We need at least INIT_DATA_BARS
            if len(combined) > Settings.INIT_DATA_BARS + 50:
                combined = combined.iloc[-Settings.INIT_DATA_BARS:]

            market_state[symbol] = combined
        except Exception as e:
            logger.error(f"Error updating state for {symbol}: {e}")

    with ThreadPoolExecutor() as executor:
        executor.map(process_symbol, Settings.PAIRS)

def get_signal(symbol):
    """
    Prepares data and asks the AI for a decision.
    Returns: Tuple(Action, DataFrame)
    """
    if symbol not in active_models or symbol not in market_state:
        return 0, None # Hold defaults
        
    df = market_state[symbol]
    
    # Recalculate features on the updated window
    df_features = data_factory.prepare_features(df)
    
    if len(df_features) < Settings.SEQUENCE_LENGTH:
        logger.warning(f"Not enough data for {symbol} inference.")
        return 0, df_features
        
    # Get the last sequence
    # input shape: (1, seq_len, input_dim)
    seq_window = df_features[Settings.FEATURES].iloc[-Settings.SEQUENCE_LENGTH:].values
    seq_tensor = torch.FloatTensor(seq_window).unsqueeze(0).to(next(active_models[symbol].parameters()).device)
    
    with torch.no_grad():
        q_values = active_models[symbol](seq_tensor)
        action = torch.argmax(q_values).item()
        
    return action, df_features

def execute_trade(symbol, signal, df_features=None):
    """
    Executes trade logic based on signal and current position.
    Signal: 0=Hold, 1=Buy, 2=Sell
    """
    # 1. Check existing positions
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        logger.error(f"Failed to get positions for {symbol}, error code: {mt5.last_error()}")
        return

    # Assuming one position per pair for simplicity
    current_pos = positions[0] if len(positions) > 0 else None
    pos_type = current_pos.type if current_pos else None # 0=Buy, 1=Sell
    
    # Use config-based magic number
    try:
        idx = Settings.PAIRS.index(symbol)
        magic = Settings.MAGIC_NUMBER_BASE + idx
    except ValueError:
        magic = Settings.MAGIC_NUMBER_BASE

    # --- Time Filter ---
    # Close all positions after 20:00 (8 PM)
    if datetime.now().hour >= 20:
        if current_pos:
            close_position(current_pos, symbol)
            logger.info(f"Time filter: Closed position for {symbol}")
        return

    # --- Logic ---
    # Need Latest Price for SL/TP
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return

    # --- FIX START: Spread Filter ---
    spread_points = (tick.ask - tick.bid) / tick.point
    if spread_points > Settings.SPREAD_FILTER_POINTS:
        logger.warning(f"Spread too high ({spread_points} > {Settings.SPREAD_FILTER_POINTS}). Skipping trade.")
        return
    # --- FIX END ---
        
    # Calculate ATR for SL
    # We can peek at the latest volatility from our market_state
    # If not readily available in state without processing, fetch or calc
    # To save time, let's roughly calc or use the last row of processed features if available
    # But prepare_features was called in get_signal.
    # Let's just re-calculate ATR on the fly efficiently or grab it from the df if we saved it.
    # For now, let's trust the signal mostly, but if we need precise SL:
    # We can access market_state[symbol] -> calc ATR.
    
    # Simple ATR calc on last 14 bars
    df = market_state[symbol]
    
    # If we are strictly following blueprint, we use "ATR * Settings.ATR_SL_MULTIPLIER"
    # We already have 'volatility' (ATR/Close) in features.
    # volatility = ATR / Close  =>  ATR = volatility * Close
    # Let's stick to the simplest valid logic.
    
    # Fetch/Calc ATR
    # Use passed features if available to avoid redundant calculation
    if df_features is not None and not df_features.empty:
        df_feat = df_features
    else:
        df_feat = data_factory.prepare_features(df)

    if df_feat.empty: return
    last_row = df_feat.iloc[-1]
    last_volatility = last_row['volatility'] # ATR/Close
    current_atr = last_volatility * tick.ask # approx
    
    sl_points = current_atr * Settings.ATR_SL_MULTIPLIER
    
    # MT5 Order Request Template
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.01, # Fixed lot size for demo
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick.ask,
        "sl": 0.0,
        "tp": 0.0,
        "deviation": 10,
        "magic": magic,
        "comment": "AI Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    if signal == 1: # BUY
        # If we have a SELL, close it first
        if pos_type == 1: 
            close_position(current_pos, symbol)
            # Then open Buy? Or wait next candle? 
            # Strategy says "Opposes current position: Close". It implies flip or just close.
            # Let's simple Close for now.
            logger.info(f"{symbol}: Signal BUY. Closed existing SELL.")
            return 
            
        # If no position, Open BUY
        if not current_pos:
            request["type"] = mt5.ORDER_TYPE_BUY
            request["price"] = tick.ask
            request["sl"] = tick.ask - sl_points
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"{symbol}: Buy failed: {result.comment}")
            else:
                logger.info(f"{symbol}: BUY executed. SL@{request['sl']}")

    elif signal == 2: # SELL
        # If we have a BUY, close it
        if pos_type == 0:
            close_position(current_pos, symbol)
            logger.info(f"{symbol}: Signal SELL. Closed existing BUY.")
            return
            
        # If no position, Open SELL
        if not current_pos:
            request["type"] = mt5.ORDER_TYPE_SELL
            request["price"] = tick.bid
            request["sl"] = tick.bid + sl_points
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"{symbol}: Sell failed: {result.comment}")
            else:
                logger.info(f"{symbol}: SELL executed. SL@{request['sl']}")

    # Signal 0 (HOLD) - Do nothing.

def process_pair(symbol):
    """
    Orchestrates signal generation and trade execution for a single pair.
    Designed to be run in parallel.
    """
    try:
        sig, df_features = get_signal(symbol)

        # Convert int signal to string for log
        sig_str = "HOLD"
        if sig == 1: sig_str = "BUY"
        elif sig == 2: sig_str = "SELL"

        if sig != 0:
            logger.info(f"{symbol} Signal: {sig_str}")

        execute_trade(symbol, sig, df_features=df_features)
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")

def close_position(position, symbol):
    tick = mt5.symbol_info_tick(symbol)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
        "position": position.ticket,
        "price": tick.bid if position.type == 0 else tick.ask,
        "deviation": 10,
        "magic": position.magic,
        "comment": "AI Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    mt5.order_send(request)

# --- Main Daemon ---
if __name__ == "__main__":
    if not mt5.initialize():
        logger.error("MT5 init failed")
        exit()
        
    logger.info("Starting Multi-Pair Bot...")
    load_models()
    init_market_state()
    
    logger.info("Bot Running. Monitoring market...")
    
    try:
        while True:
            time.sleep(1)
            
            # Check for new M5 Candle
            # Simple check: current minute % 5 == 0 AND second == 0
            # To be more robust, we should track 'last_processed_bar_time'
            # But adhering to the blueprint's simplicity:
            now = datetime.now()
            
            # We want to run ONCE per candle close. 
            # Let's say we check every second. If minute % 5 == 0 and second < 5.
            # But better: just update state every X seconds and check if data changed?
            # Blueprint says: "compare current time minutes % 5 == 0"
            pass_check = (now.minute % 5 == 0) and (now.second < 2) # 2 second window to catch it
            
            if pass_check:
                logger.info(f"New Candle Detected: {now.strftime('%H:%M:%S')}")
                update_market_state()
                
                # Parallelize signal generation and trade execution
                with ThreadPoolExecutor() as executor:
                    list(executor.map(process_pair, Settings.PAIRS))
                
                # Sleep enough to avoid duplicate triggers for this same candle minute
                # We just need to wait until we are past the 2-second window
                while datetime.now().second < 2:
                    time.sleep(0.5)
                
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    finally:
        mt5.shutdown()
