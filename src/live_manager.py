# src/live_manager.py
"""
Live Manager for position-aware QNetwork with 4 semantic actions.

Actions: HOLD(0), BUY(1), SELL(2), CLOSE(3)
Features: Online learning, ATR-based SL/TP, spread filter, position tracking.
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
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Settings
from src import data_factory
from src.brain import QNetwork
from src.per_memory import PrioritizedReplayBuffer
from src.utils import logger
from concurrent.futures import ThreadPoolExecutor

# Semantic actions (must match training)
ACT_HOLD, ACT_BUY, ACT_SELL, ACT_CLOSE = 0, 1, 2, 3
ACTION_NAMES = ['HOLD', 'BUY', 'SELL', 'CLOSE']

# --- Global State ---
active_models = {}         # {symbol: model_instance}
online_optimizers = {}     # {symbol: optimizer}
online_buffers = {}        # {symbol: PrioritizedReplayBuffer}
market_state = {}          # {symbol: DataFrame}
prev_actions = {}          # {symbol: action_idx}

# Position tracking (mirrors TradingEnv state for trade_state input)
live_positions = {}        # {symbol: {'pos': 0.0, 'entry_price': 0.0, 'bars_held': 0}}


def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading models on {device}...")

    for symbol in Settings.PAIRS:
        # Try multiple model path formats
        candidates = [
            f"models/best_{symbol}.pt",
            f"models/{symbol}_brain.pth",
            f"models/{symbol}_brain_live.pth",
        ]

        model_path = None
        for path in candidates:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            logger.error(f"Model not found for {symbol}. Tried: {candidates}. Skipping.")
            continue

        model = QNetwork().to(device)
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            active_models[symbol] = model

            # Initialize online learning components
            online_optimizers[symbol] = optim.Adam(
                model.parameters(), lr=Settings.ONLINE_LR
            )
            online_buffers[symbol] = PrioritizedReplayBuffer(
                capacity=Settings.ONLINE_BUFFER_SIZE, alpha=Settings.PER_ALPHA
            )
            prev_actions[symbol] = ACT_HOLD
            live_positions[symbol] = {'pos': 0.0, 'entry_price': 0.0, 'bars_held': 0}

            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Loaded model for {symbol} from {model_path} ({total_params:,} params)")
        except Exception as e:
            logger.error(f"Failed to load model for {symbol}: {e}")


def init_market_state():
    logger.info("Initializing market state...")
    for symbol in Settings.PAIRS:
        df = data_factory.fetch_data(symbol, Settings.INIT_DATA_BARS)
        if not df.empty:
            market_state[symbol] = df
            logger.info(f"  {symbol}: {len(df)} bars loaded (to {df.index[-1]})")
        else:
            logger.error(f"  {symbol}: Failed to get initial data.")


def update_market_state():
    """Fetch latest bars and append to market_state."""
    def process_symbol(symbol):
        try:
            df_new = data_factory.fetch_data(symbol, 10)
            if not df_new.empty and symbol in market_state:
                df_old = market_state[symbol]
                combined = pd.concat([df_old, df_new])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.sort_index()
                # Keep only what we need
                max_keep = Settings.INIT_DATA_BARS + 100
                if len(combined) > max_keep:
                    combined = combined.iloc[-max_keep:]
                market_state[symbol] = combined
        except Exception as e:
            logger.error(f"Error updating state for {symbol}: {e}")

    with ThreadPoolExecutor() as executor:
        executor.map(process_symbol, Settings.PAIRS)


def get_trade_state(symbol):
    """Build trade_state vector matching the training environment."""
    pos_info = live_positions.get(symbol, {'pos': 0.0, 'entry_price': 0.0, 'bars_held': 0})

    # Check actual MT5 position to stay in sync
    positions = mt5.positions_get(symbol=symbol)
    if positions and len(positions) > 0:
        mt5_pos = positions[0]
        # Sync direction from MT5
        if mt5_pos.type == 0:  # BUY
            pos_info['pos'] = 1.0
        elif mt5_pos.type == 1:  # SELL
            pos_info['pos'] = -1.0
        if pos_info['entry_price'] == 0.0:
            pos_info['entry_price'] = mt5_pos.price_open
    elif pos_info['pos'] != 0.0:
        # MT5 says no position but we think we have one: reset
        pos_info['pos'] = 0.0
        pos_info['entry_price'] = 0.0
        pos_info['bars_held'] = 0

    live_positions[symbol] = pos_info

    # Compute unrealized PnL
    profile = Settings.PAIR_CONFIGS.get(symbol, {})
    sf = profile.get('scaling_factor', 10000.0)
    upnl = 0.0
    if pos_info['pos'] != 0 and pos_info['entry_price'] > 0:
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            current_price = (tick.bid + tick.ask) / 2.0
            upnl = pos_info['pos'] * (current_price - pos_info['entry_price']) * sf

    trade_state = np.array([
        pos_info['pos'],
        min(pos_info['bars_held'] / 48.0, 1.0),
        np.clip(upnl / 100.0, -1, 1)
    ], dtype=np.float32)

    return trade_state


def get_signal(symbol):
    """Returns: (action_idx, df_features)"""
    if symbol not in active_models or symbol not in market_state:
        return ACT_HOLD, None

    df = market_state[symbol]

    # Resample M5 -> M15 and compute features
    df_m15 = data_factory.resample_ohlcv(df, rule='15min')
    df_features = data_factory.prepare_features(df_m15)

    if len(df_features) < Settings.SEQUENCE_LENGTH:
        logger.warning(f"Not enough data for {symbol} inference.")
        return ACT_HOLD, df_features

    # Market state tensor
    seq_window = df_features[Settings.FEATURES].iloc[-Settings.SEQUENCE_LENGTH:].values
    device = next(active_models[symbol].parameters()).device
    mkt_tensor = torch.FloatTensor(seq_window).unsqueeze(0).to(device)

    # Trade state tensor
    trade_state = get_trade_state(symbol)
    ts_tensor = torch.FloatTensor(trade_state).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = active_models[symbol](mkt_tensor, ts_tensor)
        # Live epsilon for exploration
        if random.random() < Settings.LIVE_EPSILON:
            action_idx = random.randint(0, Settings.OUTPUT_DIM - 1)
        else:
            action_idx = q_values.argmax(dim=1).item()

    return action_idx, df_features


def execute_trade(symbol, action_idx, df_features=None):
    """Execute semantic action: BUY, SELL, CLOSE, or HOLD (no-op)."""
    if action_idx == ACT_HOLD:
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

    # Spread filter
    spread_points = (tick.ask - tick.bid) / point
    if spread_points > Settings.SPREAD_FILTER_POINTS:
        logger.warning(f"Spread too high ({spread_points:.0f}). Skipping.")
        return

    lot_size = Settings.MAX_POSITION_SIZE

    # ATR for SL and TP
    if df_features is not None and not df_features.empty and 'atr' in df_features.columns:
        current_atr = df_features.iloc[-1]['atr']
    else:
        current_atr = tick.ask * 0.002

    sl_dist = current_atr * Settings.ATR_SL_MULTIPLIER
    tp_dist = current_atr * Settings.ATR_TP_MULTIPLIER

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
        "comment": f"AI {ACTION_NAMES[action_idx]}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    if action_idx == ACT_CLOSE:
        # Close any existing position
        if current_pos:
            close_position(current_pos, symbol)
            live_positions[symbol] = {'pos': 0.0, 'entry_price': 0.0, 'bars_held': 0}
        return

    if action_idx == ACT_BUY:
        # Close short first if open
        if pos_type == 1:  # SELL position
            close_position(current_pos, symbol)
            current_pos = None

        if not current_pos:
            request["type"] = mt5.ORDER_TYPE_BUY
            request["price"] = tick.ask
            request["sl"] = tick.ask - sl_dist
            request["tp"] = tick.ask + tp_dist
            request["volume"] = lot_size

            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"{symbol}: Buy failed: {result.comment}")
            else:
                logger.info(f"{symbol}: BUY {lot_size} lots. SL@{request['sl']:.5f} TP@{request['tp']:.5f}")
                live_positions[symbol] = {
                    'pos': 1.0,
                    'entry_price': tick.ask,
                    'bars_held': 0
                }

    elif action_idx == ACT_SELL:
        # Close long first if open
        if pos_type == 0:  # BUY position
            close_position(current_pos, symbol)
            current_pos = None

        if not current_pos:
            request["type"] = mt5.ORDER_TYPE_SELL
            request["price"] = tick.bid
            request["sl"] = tick.bid + sl_dist
            request["tp"] = tick.bid - tp_dist
            request["volume"] = lot_size

            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"{symbol}: Sell failed: {result.comment}")
            else:
                logger.info(f"{symbol}: SELL {lot_size} lots. SL@{request['sl']:.5f} TP@{request['tp']:.5f}")
                live_positions[symbol] = {
                    'pos': -1.0,
                    'entry_price': tick.bid,
                    'bars_held': 0
                }


def online_learn(symbol, state_mkt, state_ts, action_idx, reward, next_state_mkt, next_state_ts):
    """Online gradient descent — adapt model to live market with trade_state."""
    if symbol not in online_buffers or symbol not in active_models:
        return

    buffer = online_buffers[symbol]
    model = active_models[symbol]
    optimizer = online_optimizers[symbol]
    device = next(model.parameters()).device

    # Store experience as (mkt_state, trade_state) tuple
    buffer.push((state_mkt, state_ts), action_idx, reward,
                (next_state_mkt, next_state_ts), False)

    if len(buffer) < 32:
        return

    model.train()
    loss_fn = nn.MSELoss(reduction='none')

    for _ in range(Settings.ONLINE_UPDATE_STEPS):
        states, actions, rewards, next_states, _, idxs, is_weights = buffer.sample(32, beta=0.6)

        mkt = torch.FloatTensor(np.array([s[0] for s in states])).to(device)
        ts = torch.FloatTensor(np.array([s[1] for s in states])).to(device)
        nmkt = torch.FloatTensor(np.array([s[0] for s in next_states])).to(device)
        nts = torch.FloatTensor(np.array([s[1] for s in next_states])).to(device)
        at = torch.LongTensor(actions).to(device)
        rt = torch.FloatTensor(rewards).to(device)
        wt = torch.FloatTensor(is_weights).to(device)

        with torch.enable_grad():
            current_q = model(mkt, ts).gather(1, at.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = model(nmkt, nts).max(1)[0]
                target_q = rt + Settings.GAMMA * next_q

            loss = (loss_fn(current_q, target_q) * wt).mean()
            optimizer.zero_grad()
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        td_errors = torch.abs(target_q - current_q).detach().cpu().numpy()
        buffer.update_priorities(idxs, td_errors)

    model.eval()


def process_pair(symbol):
    """Orchestrates signal + execution + online learning feedback."""
    try:
        action_idx, df_features = get_signal(symbol)
        sig_str = ACTION_NAMES[action_idx]

        if action_idx != ACT_HOLD:
            logger.info(f"{symbol} Signal: {sig_str}")

        execute_trade(symbol, action_idx, df_features=df_features)

        # Online learning feedback
        if df_features is not None and len(df_features) >= Settings.SEQUENCE_LENGTH + 1:
            prev_action = prev_actions.get(symbol, ACT_HOLD)
            pos_info = live_positions.get(symbol, {'pos': 0.0, 'entry_price': 0.0, 'bars_held': 0})

            if pos_info['pos'] != 0:
                profile = Settings.PAIR_CONFIGS.get(symbol, {})
                sf = profile.get('scaling_factor', 10000.0)

                prices = df_features['close'].values
                price_change = (prices[-1] - prices[-2]) * sf

                cost = Settings.TRANSACTION_COST_BPS * 0.0001 * sf
                reward = np.clip(pos_info['pos'] * price_change - cost, -10.0, 10.0)

                # Build state sequences for online buffer
                features = df_features[Settings.FEATURES].values
                state_mkt = features[-Settings.SEQUENCE_LENGTH - 1: -1]
                next_state_mkt = features[-Settings.SEQUENCE_LENGTH:]

                trade_state = get_trade_state(symbol)
                # Approximate prev trade state
                prev_trade_state = trade_state.copy()

                if (len(state_mkt) == Settings.SEQUENCE_LENGTH and
                        len(next_state_mkt) == Settings.SEQUENCE_LENGTH):
                    online_learn(symbol, state_mkt, prev_trade_state,
                                 prev_action, reward, next_state_mkt, trade_state)

            prev_actions[symbol] = action_idx

            # Update bars_held for position tracking
            if pos_info['pos'] != 0:
                pos_info['bars_held'] += 1
                live_positions[symbol] = pos_info

    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        import traceback
        traceback.print_exc()


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
        logger.info(f"{symbol}: Position closed. {result.comment}")
        live_positions[symbol] = {'pos': 0.0, 'entry_price': 0.0, 'bars_held': 0}


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

    logger.info("Starting Live Bot...")
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

                # Save online-updated models hourly & garbage collect
                if now.minute == 0:
                    for sym, model in active_models.items():
                        backup_path = f"models/{sym}_brain_live.pth"
                        torch.save(model.state_dict(), backup_path)
                        logger.info(f"Saved online model checkpoint: {backup_path}")
                    gc.collect()
                    try:
                        import psutil
                        rss = psutil.Process().memory_info().rss / 1024 / 1024
                        logger.info(f"Memory: {rss:.0f} MB")
                    except ImportError:
                        pass

                while datetime.now().second < 2:
                    time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
        for sym, model in active_models.items():
            torch.save(model.state_dict(), f"models/{sym}_brain_live.pth")
            logger.info(f"Saved final online model: {sym}")
    finally:
        mt5.shutdown()
