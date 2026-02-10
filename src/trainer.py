# src/trainer.py
"""
Phase 27: Research-Driven Trainer with Critical Fixes.

P0 Fixes:
- True DDQN with target network (soft-update, τ=0.005)
- Corrected volatility scaling formula (1/vol inverse)
P1 Fixes:
- Removed warm-start leakage (independent windows)
- Fixed Sharpe calculation (trade-only returns)
P2 Fixes:
- Added LR scheduling (ReduceLROnPlateau)
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import MetaTrader5 as mt5
from tqdm import tqdm
import argparse
from config import Settings
from src import data_factory
from src.brain import QNetwork
from src.per_memory import PrioritizedReplayBuffer
from src.utils import logger
from torch.utils.data import Dataset, DataLoader
import random

class TradingDataset(Dataset):
    """Dataset providing (state, next_state, curr_price, next_price, volatility)."""
    def __init__(self, feature_data, close_prices, volatilities, seq_len):
        self.feature_data = feature_data
        self.close_prices = close_prices
        self.volatilities = volatilities
        self.seq_len = seq_len
        self.valid_indices = range(seq_len, len(feature_data) - 1)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        return {
            'state': torch.FloatTensor(self.feature_data[i - self.seq_len : i].copy()),
            'next_state': torch.FloatTensor(self.feature_data[i - self.seq_len + 1 : i + 1].copy()),
            'curr_price': torch.tensor(self.close_prices[i-1], dtype=torch.float32),
            'next_price': torch.tensor(self.close_prices[i], dtype=torch.float32),
            'volatility': torch.tensor(self.volatilities[i-1], dtype=torch.float32)
        }

class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on device: {self.device}")
        self.loss_fn = nn.MSELoss(reduction='none')

    def soft_update(self, policy_net, target_net, tau=Settings.TAU):
        """Soft-update target network: θ_target = τ·θ_policy + (1-τ)·θ_target"""
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def train_model(self, symbol: str):
        logger.info(f"Starting Phase 27 training for {symbol}...")
        
        # 1. Fetch & Prepare Data
        df = data_factory.fetch_data(symbol, Settings.TRAIN_DATA_BARS)
        if df.empty:
            logger.error(f"No data for {symbol}. Skipping.")
            return

        df = data_factory.prepare_features(df)
        if df.empty:
            logger.error(f"Not enough data after feature engineering for {symbol}.")
            return

        logger.info(f"Total usable data: {len(df)} bars after feature engineering.")

        # 2. Rolling Windows
        total_bars = len(df)
        window_total = Settings.TRAIN_WINDOW + Settings.VAL_WINDOW + Settings.TEST_WINDOW
        
        if total_bars < window_total:
            logger.warning(f"Data ({total_bars}) < window requirement ({window_total}). Using proportional split.")
            train_end = int(total_bars * 0.7)
            val_end = int(total_bars * 0.85)
            windows = [(0, train_end, val_end, total_bars)]
        else:
            windows = []
            start = 0
            while start + window_total <= total_bars:
                train_end = start + Settings.TRAIN_WINDOW
                val_end = train_end + Settings.VAL_WINDOW
                test_end = val_end + Settings.TEST_WINDOW
                windows.append((start, train_end, val_end, test_end))
                start += Settings.TEST_WINDOW
            if len(windows) > 5:
                windows = windows[-5:]
        
        logger.info(f"Rolling Windows: {len(windows)} study periods.")
        
        # Track best model across all windows
        best_val_sharpe = -np.inf
        best_model_state = None
        
        profile = Settings.PAIR_CONFIGS[symbol]
        SCALING_FACTOR = profile['scaling_factor']
        
        for w_idx, (w_start, w_train_end, w_val_end, w_test_end) in enumerate(windows):
            logger.info(f"\n--- Window {w_idx+1}/{len(windows)}: "
                       f"Train[{w_start}:{w_train_end}] "
                       f"Val[{w_train_end}:{w_val_end}] "
                       f"Test[{w_val_end}:{w_test_end}] ---")
            
            df_train = df.iloc[w_start:w_train_end].copy()
            df_val = df.iloc[w_train_end:w_val_end].copy()
            
            train_features = df_train[Settings.FEATURES].values
            train_prices = df_train['mid_price'].values
            train_vols = df_train['volatility_ratio'].values if 'volatility_ratio' in df_train.columns else np.ones(len(df_train))
            train_dataset = TradingDataset(train_features, train_prices, train_vols, Settings.SEQUENCE_LENGTH)
            
            val_features = df_val[Settings.FEATURES].values
            val_prices = df_val['mid_price'].values
            val_vols = df_val['volatility_ratio'].values if 'volatility_ratio' in df_val.columns else np.ones(len(df_val))
            val_dataset = TradingDataset(val_features, val_prices, val_vols, Settings.SEQUENCE_LENGTH)
            
            if len(train_dataset) < Settings.BATCH_SIZE:
                logger.warning(f"Window {w_idx+1}: Not enough training data ({len(train_dataset)}). Skipping.")
                continue
            
            # Populate PER
            memory = PrioritizedReplayBuffer(capacity=len(train_dataset), alpha=Settings.PER_ALPHA)
            temp_loader = DataLoader(train_dataset, batch_size=4096, shuffle=False)
            
            logger.info(f"Populating Replay Buffer with {len(train_dataset)} transitions...")
            for batch in tqdm(temp_loader, desc="Filling Memory"):
                states = batch['state'].numpy()
                next_states = batch['next_state'].numpy()
                curr_prices_batch = batch['curr_price'].numpy()
                next_prices_batch = batch['next_price'].numpy()
                vols_batch = batch['volatility'].numpy()
                for i in range(len(states)):
                    memory.push(states[i], next_states[i], curr_prices_batch[i], next_prices_batch[i], vols_batch[i])

            # P1 FIX: NO warm-start — train each window independently (Fischer & Krauss)
            policy_net = QNetwork().to(self.device)
            
            # P0 FIX: Create Target Network (True DDQN)
            target_net = QNetwork().to(self.device)
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()
            
            optimizer = optim.Adam(policy_net.parameters(), lr=Settings.LEARNING_RATE)
            
            # P28 FIX: LR Scheduling on val Sharpe (not loss)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', 
                patience=Settings.LR_SCHEDULER_PATIENCE, 
                factor=Settings.LR_SCHEDULER_FACTOR,
                verbose=True
            )
            
            epsilon = Settings.EPSILON_START
            steps_per_epoch = len(train_dataset) // Settings.BATCH_SIZE
            
            # Early stopping
            best_val_loss = np.inf
            patience_counter = 0
            global_step = 0
            
            action_map_tensor = torch.tensor(Settings.ACTION_MAP, device=self.device)
            
            for epoch in range(Settings.EPOCHS):
                total_loss = 0
                beta = Settings.PER_BETA_START + (1.0 - Settings.PER_BETA_START) * (epoch / Settings.EPOCHS)
                
                pbar = tqdm(range(steps_per_epoch), desc=f"W{w_idx+1} Epoch {epoch+1}/{Settings.EPOCHS}")
                
                for _ in pbar:
                    states_s, next_states_s, curr_prices_s, next_prices_s, vols_s, idxs, is_weights = memory.sample(Settings.BATCH_SIZE, beta)
                    
                    state_tensor = torch.FloatTensor(states_s).to(self.device)
                    next_state_tensor = torch.FloatTensor(next_states_s).to(self.device)
                    curr_price = torch.FloatTensor(curr_prices_s).to(self.device)
                    next_price = torch.FloatTensor(next_prices_s).to(self.device)
                    vol_tensor = torch.FloatTensor(vols_s).to(self.device)
                    weights_tensor = torch.FloatTensor(is_weights).to(self.device)
                    
                    batch_size = state_tensor.size(0)

                    # Epsilon-Greedy Action
                    if random.random() < epsilon:
                        action_tensor = torch.randint(0, Settings.OUTPUT_DIM, (batch_size,), device=self.device)
                    else:
                        with torch.no_grad():
                            q_values = policy_net(state_tensor)
                            action_tensor = torch.argmax(q_values, dim=1)
                    
                    # Map action to position fraction
                    position = action_map_tensor[action_tensor]
                    
                    # Price change
                    price_change = (next_price - curr_price) * SCALING_FACTOR
                    
                    # P0 FIX: Corrected Volatility Scaling
                    # vol_tensor is volatility_ratio (current_vol / avg_vol), normalized around 1.0
                    # High vol (2.0) → scale 0.5 (half position), Low vol (0.5) → scale 2.0
                    vol_safe = vol_tensor.clamp(min=0.2)
                    vol_scale = (1.0 / vol_safe).clamp(0.2, 3.0)
                    
                    # Transaction cost
                    cost = Settings.TRANSACTION_COST_BPS * 0.0001 * SCALING_FACTOR * torch.abs(position)
                    
                    # Final reward
                    reward_tensor = (vol_scale * position * price_change - cost).clamp(-5.0, 5.0)
                    
                    # --- P0 FIX: True DDQN Update ---
                    current_q = policy_net(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)
                    
                    with torch.no_grad():
                        # DDQN: Select action from POLICY net, evaluate with TARGET net
                        next_actions = policy_net(next_state_tensor).argmax(1)
                        next_q = target_net(next_state_tensor).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                        target_q = reward_tensor + (Settings.GAMMA * next_q)
                    
                    loss_element = self.loss_fn(current_q, target_q)
                    loss = (loss_element * weights_tensor).mean()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    # Update PER priorities
                    td_errors = torch.abs(target_q - current_q).detach().cpu().numpy()
                    memory.update_priorities(idxs, td_errors)
                    
                    total_loss += loss.item()
                    global_step += 1
                    
                    # P0 FIX: Soft-update target network
                    if global_step % Settings.TARGET_UPDATE_FREQ == 0:
                        self.soft_update(policy_net, target_net)
                
                # Decay Epsilon
                if epsilon > Settings.EPSILON_MIN:
                    epsilon *= Settings.EPSILON_DECAY
                
                avg_loss = total_loss / steps_per_epoch
                
                # Validation
                val_sharpe = self.validate(policy_net, val_dataset, symbol)
                
                # P28 FIX: LR Scheduler on val Sharpe (maximize)
                scheduler.step(val_sharpe)
                current_lr = optimizer.param_groups[0]['lr']
                
                logger.info(f"W{w_idx+1} Epoch {epoch+1} done. "
                           f"Loss: {avg_loss:.6f}, Eps: {epsilon:.4f}, LR: {current_lr:.2e}")
                
                # Early Stopping check
                
                # Early Stopping
                if avg_loss < best_val_loss:
                    best_val_loss = avg_loss
                    patience_counter = 0
                    if val_sharpe > best_val_sharpe:
                        best_val_sharpe = val_sharpe
                        best_model_state = {k: v.clone() for k, v in policy_net.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= Settings.EARLY_STOP_PATIENCE:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                        break
        
        # Save best model
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{symbol}_brain.pth"
        
        if best_model_state is not None:
            torch.save(best_model_state, model_path)
            logger.info(f"Best model saved to {model_path} (Val Sharpe: {best_val_sharpe:.4f})")
        else:
            torch.save(policy_net.state_dict(), model_path)
            logger.info(f"Model saved to {model_path} (no validation improvement detected)")

    def validate(self, policy_net, val_dataset, symbol):
        """P1 FIX: Sharpe calculated on trade-only returns (not zero-position bars)."""
        policy_net.eval()
        val_loader = DataLoader(val_dataset, batch_size=Settings.BATCH_SIZE, shuffle=False)
        
        profile = Settings.PAIR_CONFIGS[symbol]
        SCALING_FACTOR = profile['scaling_factor']
        action_map = torch.tensor(Settings.ACTION_MAP, device=self.device)
        
        all_returns = []
        total_trades = 0
        
        with torch.no_grad():
            for batch in val_loader:
                state_tensor = batch['state'].to(self.device).float()
                curr_prices = batch['curr_price'].to(self.device).float()
                next_prices = batch['next_price'].to(self.device).float()
                
                q_values = policy_net(state_tensor)
                actions = torch.argmax(q_values, dim=1)
                positions = action_map[actions]
                
                price_change = (next_prices - curr_prices) * SCALING_FACTOR
                bar_returns = positions * price_change
                
                total_trades += (actions != 3).sum().item()
                all_returns.extend(bar_returns.cpu().numpy().tolist())
        
        returns_arr = np.array(all_returns)
        net_pnl = returns_arr.sum()
        
        # P1 FIX: Only include bars where position != 0 for Sharpe
        trade_returns = returns_arr[returns_arr != 0]
        if len(trade_returns) > 1 and np.std(trade_returns) > 0:
            sharpe = (np.mean(trade_returns) / np.std(trade_returns)) * np.sqrt(252 * 12 * 24)
        else:
            sharpe = 0.0
        
        logger.info(f"[VALIDATION] Trades: {total_trades}, Net PnL: {net_pnl:.5f}, "
                    f"Sharpe (trade-only): {sharpe:.4f}")
        
        policy_net.train()
        return sharpe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 27 Deep RL Trainer")
    parser.add_argument("--pair", type=str, required=True, help="Symbol to train (e.g., EURUSD)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    args = parser.parse_args()
    
    symbol = args.pair.upper()
    
    if args.epochs:
        Settings.EPOCHS = args.epochs
    
    if symbol not in Settings.PAIR_CONFIGS:
        logger.error(f"Symbol {symbol} not found in Settings.PAIR_CONFIGS!")
        logger.info(f"Available: {list(Settings.PAIR_CONFIGS.keys())}")
        exit()

    if not mt5.initialize(
        path=Settings.MT5_PATH,
        login=Settings.MT5_LOGIN,
        password=Settings.MT5_PASSWORD,
        server=Settings.MT5_SERVER
    ):
        logger.error(f"MT5 init failed. Path: {Settings.MT5_PATH}")
        logger.error(f"Error: {mt5.last_error()}")
        exit()
    
    trainer = Trainer()
    
    try:
        trainer.train_model(symbol)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    finally:
        mt5.shutdown()
