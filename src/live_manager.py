# src/live_manager.py
"""
Phase 27: Live Manager with Online Learning + Take-Profit.

P0: Online learning (Da Costa & Gebbie OGD) — adapts to live market
P1: Take-Profit via ATR_TP_MULTIPLIER
"""
import MetaTrader5 as mt5
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
import sys
import copy
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Settings
from src import data_factory
from src.brain import QNetwork
from src.per_memory import PrioritizedReplayBuffer
from src.utils import logger
from concurrent.futures import ThreadPoolExecutor

# --- Global State ---
active_models = {}         # {symbol: model_instance}
online_optimizers = {}     # {symbol: optimizer} — for online learning
online_buffers = {}        # {symbol: PrioritizedReplayBuffer}
market_state = {}          # {symbol: DataFrame}
prev_actions = {}          # {symbol: action_idx} — track position changes

def load_models():
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
            model.eval()
            active_models[symbol] = model
            
            # P0: Initialize online learning components
            online_optimizers[symbol] = optim.Adam(
                model.parameters(), lr=Settings.ONLINE_LR
            )
            online_buffers[symbol] = PrioritizedReplayBuffer(
                capacity=Settings.ONLINE_BUFFER_SIZE, alpha=Settings.PER_ALPHA
            )
            prev_actions[symbol] = 3  # Start with HOLD
            
            logger.info(f"Loaded model for {symbol} (online learning enabled)")
        except Exception as e:
            logger.error(f"Failed to load model for {symbol}: {e}")

def init_market_state():
    logger.info("Initializing market state...")
    for symbol in Settings.PAIRS:
        df = data_factory.fetch_data(symbol, Settings.INIT_DATA_BARS)
        if not df.empty:
            market_state[symbol] = df
            logger.info(f"Initialized state for {symbol} with {len(df)} bars")
        else:
            logger.error(f"Failed to fetch init data for {symbol}")

def update_market_state():
    def process_symbol(symbol):
        try:
            if symbol not in market_state:
                return
            new_data = data_factory.fetch_data(symbol, 2)
            if new_data.empty:
                return
            current_df = market_state[symbol]
            mask = ~current_df.index.isin(new_data.index)
            combined = pd.concat([current_df[mask], new_data])
            if len(combined) > Settings.INIT_DATA_BARS + 50:
                combined = combined.iloc[-Settings.INIT_DATA_BARS:]
            market_state[symbol] = combined
        except Exception as e:
            logger.error(f"Error updating state for {symbol}: {e}")

    with ThreadPoolExecutor() as executor:
        executor.map(process_symbol, Settings.PAIRS)

def online_learn(symbol, state, action_idx, reward, next_state):
    """
    P0: Online Gradient Descent — adapt model to live market.
    Small buffer + few gradient steps with tiny LR.
    """
    if symbol not in online_buffers or symbol not in active_models:
        return
    
    buffer = online_buffers[symbol]
    model = active_models[symbol]
    optimizer = online_optimizers[symbol]
    device = next(model.parameters()).device
    
    # Store experience
    buffer.push(state, action_idx, reward, next_state, False)
    
    if len(buffer) < 32:
        return
    
    # Mini-batch gradient steps
    model.train()
    loss_fn = nn.MSELoss(reduction='none')
    action_map = torch.tensor(Settings.ACTION_MAP, device=device)
    
    for _ in range(Settings.ONLINE_UPDATE_STEPS):
        states, actions, rewards, next_states, _, idxs, is_weights = buffer.sample(32, beta=0.6)
        
        st = torch.FloatTensor(states).to(device)
        nst = torch.FloatTensor(next_states).to(device)
        at = torch.LongTensor(actions).to(device)
        rt = torch.FloatTensor(rewards).to(device)
        wt = torch.FloatTensor(is_weights).to(device)
        
        current_q = model(st).gather(1, at.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = model(nst).max(1)[0]
            target_q = rt + Settings.GAMMA * next_q
        
        loss = (loss_fn(current_q, target_q) * wt).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        td_errors = torch.abs(target_q - current_q).detach().cpu().numpy()
        buffer.update_priorities(idxs, td_errors)
    
    model.eval()

def get_signal(symbol):
    """Returns: (action_idx, position_fraction, df_features)"""
    if symbol not in active_models or symbol not in market_state:
        return 3, 0.0, None
        
    df = market_state[symbol]
    df_features = data_factory.prepare_features(df)
    
    if len(df_features) < Settings.SEQUENCE_LENGTH:
        logger.warning(f"Not enough data for {symbol} inference.")
        return 3, 0.0, df_features
        
    seq_window = df_features[Settings.FEATURES].iloc[-Settings.SEQUENCE_LENGTH:].values
    device = next(active_models[symbol].parameters()).device
    seq_tensor = torch.FloatTensor(seq_window).unsqueeze(0).to(device)
    
    with torch.no_grad():
        q_values = active_models[symbol](seq_tensor)
        # P28: Live epsilon for exploration (Zengeler & Handmann)
        if random.random() < Settings.LIVE_EPSILON:
            action_idx = random.randint(0, Settings.OUTPUT_DIM - 1)
        else:
            action_idx = torch.argmax(q_values).item()
    
    position_fraction = Settings.ACTION_MAP[action_idx]
    return action_idx, position_fraction, df_features

def execute_trade(symbol, action_idx, position_fraction, df_features=None):
    """P1: Includes Take-Profit alongside Stop-Loss."""
    if abs(position_fraction) < 0.01:
        return
    
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        logger.error(f"Failed to get positions for {symbol}, error: {mt5.last_error()}")
        return

    current_pos = positions[0] if len(positions) > 0 else None
    pos_type = current_pos.type if current_pos else None

    try:
        idx = Settings.PAIRS.index(symbol)
        magic = Settings.MAGIC_NUMBER_BASE + idx
    except ValueError:
        magic = Settings.MAGIC_NUMBER_BASE

    tick = mt5.symbol_info_tick(symbol)
    sym_info = mt5.symbol_info(symbol)
    if not tick or not sym_info:
        return
    point = sym_info.point

    # Spread filter (safety gate)
    spread_points = (tick.ask - tick.bid) / point
    if spread_points > Settings.SPREAD_FILTER_POINTS:
        logger.warning(f"Spread too high ({spread_points:.0f}). Skipping.")
        return

    lot_size = round(abs(position_fraction) * Settings.MAX_LOT_SIZE, 2)
    lot_size = max(lot_size, 0.01)

    # ATR for SL and TP
    if df_features is not None and not df_features.empty and 'atr_normalized' in df_features.columns:
        current_atr = df_features.iloc[-1]['atr_normalized'] * tick.ask
    else:
        current_atr = tick.ask * 0.002

    sl_dist = current_atr * Settings.ATR_SL_MULTIPLIER
    tp_dist = current_atr * Settings.ATR_TP_MULTIPLIER  # P1: Take-Profit

    is_buy = position_fraction > 0
    is_sell = position_fraction < 0

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick.ask,
        "sl": 0.0,
        "tp": 0.0,
        "deviation": 10,
        "magic": magic,
        "comment": f"AI P27 {position_fraction:+.2f}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    if is_buy:
        if pos_type == 1:
            close_position(current_pos, symbol)
            current_pos = None

        if not current_pos:
            request["type"] = mt5.ORDER_TYPE_BUY
            request["price"] = tick.ask
            request["sl"] = tick.ask - sl_dist
            request["tp"] = tick.ask + tp_dist  # P1: TP
            request["volume"] = lot_size
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"{symbol}: Buy failed: {result.comment}")
            else:
                logger.info(f"{symbol}: BUY {lot_size} lots. SL@{request['sl']:.5f} TP@{request['tp']:.5f}")

    elif is_sell:
        if pos_type == 0:
            close_position(current_pos, symbol)
            current_pos = None

        if not current_pos:
            request["type"] = mt5.ORDER_TYPE_SELL
            request["price"] = tick.bid
            request["sl"] = tick.bid + sl_dist
            request["tp"] = tick.bid - tp_dist  # P1: TP
            request["volume"] = lot_size
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"{symbol}: Sell failed: {result.comment}")
            else:
                logger.info(f"{symbol}: SELL {lot_size} lots. SL@{request['sl']:.5f} TP@{request['tp']:.5f}")

def process_pair(symbol):
    """Orchestrates signal + execution + online learning feedback."""
    try:
        action_idx, position_fraction, df_features = get_signal(symbol)

        action_labels = {0: "STRONG SELL", 1: "SELL", 2: "WEAK SELL", 
                        3: "HOLD", 4: "WEAK BUY", 5: "BUY", 6: "STRONG BUY"}
        sig_str = action_labels.get(action_idx, "UNKNOWN")

        if action_idx != 3:
            logger.info(f"{symbol} Signal: {sig_str} (pos={position_fraction:+.2f})")

        execute_trade(symbol, action_idx, position_fraction, df_features=df_features)
        
        # P28 FIX: Online learning feedback
        # Reward MUST match training formula: (vol_scale * position * price_change - cost).clamp(-5, 5)
        if df_features is not None and len(df_features) >= Settings.SEQUENCE_LENGTH + 1:
            prev_action = prev_actions.get(symbol, 3)
            prev_position = Settings.ACTION_MAP[prev_action]
            
            if abs(prev_position) > 0.01:
                profile = Settings.PAIR_CONFIGS.get(symbol, {})
                sf = profile.get('scaling_factor', 1.0)
                
                prices = df_features['mid_price'].values
                price_change = (prices[-1] - prices[-2]) * sf
                
                # P30: Match training reward (simple PnL - cost, no vol scaling)
                cost = Settings.TRANSACTION_COST_BPS * 0.0001 * sf * abs(prev_position)
                reward = max(-10.0, min(10.0, prev_position * price_change - cost))
                
                # Get state sequences for online buffer
                features = df_features[Settings.FEATURES].values
                state = features[-Settings.SEQUENCE_LENGTH - 1 : -1]
                next_state = features[-Settings.SEQUENCE_LENGTH:]
                
                if len(state) == Settings.SEQUENCE_LENGTH and len(next_state) == Settings.SEQUENCE_LENGTH:
                    online_learn(symbol, state, prev_action, reward, next_state)
            
            prev_actions[symbol] = action_idx
            
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")

def close_position(position, symbol):
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return
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
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"{symbol}: Close failed: {result.comment}")
    else:
        logger.info(f"{symbol}: Position closed. PnL: {result.profit}")

# --- Main Daemon ---
if __name__ == "__main__":
    if not mt5.initialize(
        path=Settings.MT5_PATH,
        login=Settings.MT5_LOGIN,
        password=Settings.MT5_PASSWORD,
        server=Settings.MT5_SERVER
    ):
        logger.error(f"MT5 init failed.")
        exit()
        
    logger.info("Starting Multi-Pair Bot (Phase 27)...")
    load_models()
    init_market_state()
    
    logger.info("Bot Running. Monitoring market...")
    
    try:
        while True:
            time.sleep(1)
            now = datetime.now()
            pass_check = (now.minute % 5 == 0) and (now.second < 2)
            
            if pass_check:
                logger.info(f"New Candle: {now.strftime('%H:%M:%S')}")
                update_market_state()
                
                with ThreadPoolExecutor() as executor:
                    list(executor.map(process_pair, Settings.PAIRS))
                
                # Periodically save online-updated models (every hour)
                if now.minute == 0:
                    for sym, model in active_models.items():
                        backup_path = f"models/{sym}_brain_live.pth"
                        torch.save(model.state_dict(), backup_path)
                        logger.info(f"Saved online model checkpoint: {backup_path}")
                
                while datetime.now().second < 2:
                    time.sleep(0.5)
                
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
        # Save final online-updated models
        for sym, model in active_models.items():
            torch.save(model.state_dict(), f"models/{sym}_brain_live.pth")
            logger.info(f"Saved final online model: {sym}")
    finally:
        mt5.shutdown()
