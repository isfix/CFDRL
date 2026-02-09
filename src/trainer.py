# src/trainer.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from src.per_memory import PrioritizedReplayBuffer # Phase 16: PER
from src.utils import logger
import os
from torch.utils.data import Dataset, DataLoader
import random

class TradingDataset(Dataset):
    def __init__(self, feature_data, close_prices, seq_len):
        self.feature_data = feature_data
        self.close_prices = close_prices
        self.seq_len = seq_len
        self.valid_indices = range(seq_len, len(feature_data) - 1)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        state_window = self.feature_data[i - self.seq_len : i]
        next_state_window = self.feature_data[i - self.seq_len + 1 : i + 1]
        curr_price = self.close_prices[i-1]
        next_price = self.close_prices[i]

        return {
            'state': torch.FloatTensor(state_window),
            'next_state': torch.FloatTensor(next_state_window),
            'curr_price': torch.tensor(curr_price, dtype=torch.float32),
            'next_price': torch.tensor(next_price, dtype=torch.float32)
        }

class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on device: {self.device}")
        
        # PER uses Importance Sampling Weights, so we need reduction='none'
        self.loss_fn = nn.MSELoss(reduction='none')

    def train_model(self, symbol: str):
        logger.info(f"Starting training for {symbol}...")
        
        # 1. Fetch Data
        df = data_factory.fetch_data(symbol, Settings.TRAIN_DATA_BARS)
        if df.empty:
            logger.error(f"No data for {symbol}. Skipping.")
            return

        df = data_factory.prepare_features(df)
        if df.empty:
            logger.error(f"Not enough data after feature engineering for {symbol}.")
            return

        # 2. Prepare Sliding Window Dataset
        feature_data = df[Settings.FEATURES].values
        # PAPER IMPLEMENTATION: Train on Mid-Price to learn "True Trend"
        close_prices = df['mid_price'].values
        
        dataset = TradingDataset(feature_data, close_prices, Settings.SEQUENCE_LENGTH)
        
        # 3. Populate Prioritized Replay Buffer (Offline RL -> Online-style Memory)
        memory_capacity = len(dataset)
        memory = PrioritizedReplayBuffer(capacity=memory_capacity, alpha=Settings.PER_ALPHA)
        
        logger.info(f"Populating Replay Buffer with {memory_capacity} transitions...")
        temp_loader = DataLoader(dataset, batch_size=4096, shuffle=False)
        
        for batch in tqdm(temp_loader, desc="Filling Memory"):
            states = batch['state'].numpy()
            next_states = batch['next_state'].numpy()
            curr_prices = batch['curr_price'].numpy()
            next_prices = batch['next_price'].numpy()
            
            for i in range(len(states)):
                # Store tuple (state, next_state, curr_price, next_price, dummy_done)
                data = (states[i], next_states[i], curr_prices[i], next_prices[i], False)
                memory.push(0, 0, 0, 0, 0) # Placeholder args to init tree node
                # Direct override to store our custom data in the buffer
                # Accessing the internal data array of SumTree via the push index logic is tricky 
                # because push already incremented 'write'. 
                # But 'push' sets max priority.
                # Actually, our `push` implementation in per_memory.py takes (s,a,r,s,d) and stores as a tuple.
                # We should just pass our tuple as 'state' and ignore others, or modify push.
                # Let's trust that we can retrieve what we put in.
                
                # We modified per_memory.py to store `data` tuple.
                # memory.tree.add(max_prio, data)
                # So we just need to pass meaningful data.
                # Let's assume memory.push() stores the arguments as a tuple.
                # Wait, looking at my `per_memory.py`:
                # data = (state, action, reward, next_state, done)
                # self.tree.add(max_prio, data)
                
                # So I should pass:
                # state -> states[i]
                # action -> next_states[i] (HACK: Storing next_state in action slot)
                # reward -> curr_prices[i] (HACK: Storing curr_price in reward slot)
                # next_state -> next_prices[i] (HACK: Storing next_price in next_state slot)
                # done -> False
                
                memory.push(states[i], next_states[i], curr_prices[i], next_prices[i], False)

        # Initialize Model & Optimizer
        policy_net = QNetwork().to(self.device)
        optimizer = optim.Adam(policy_net.parameters(), lr=Settings.LEARNING_RATE)
        
        epsilon = Settings.EPSILON_START
        
        # 4. Training Loop using PER
        steps_per_epoch = len(dataset) // Settings.BATCH_SIZE
        
        for epoch in range(Settings.EPOCHS):
            total_loss = 0
            
            # Annealing Beta (0.4 -> 1.0)
            beta = 0.4 + (0.6 * (epoch / Settings.EPOCHS))
            
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{Settings.EPOCHS}")
            
            for _ in pbar:
                # A. Sample from PER
                # Returns: (states, actions, rewards, next_states, dones, idxs, is_weights)
                # But remember our HACK above!
                # states -> state
                # actions -> next_state
                # rewards -> curr_price
                # next_states -> next_price
                
                states, next_states_hack, curr_prices_hack, next_prices_hack, _, idxs, is_weights = memory.sample(Settings.BATCH_SIZE, beta)
                
                state_tensor = torch.FloatTensor(states).to(self.device)
                next_state_tensor = torch.FloatTensor(next_states_hack).to(self.device)
                curr_price = torch.FloatTensor(curr_prices_hack).to(self.device)
                next_price = torch.FloatTensor(next_prices_hack).to(self.device)
                weights_tensor = torch.FloatTensor(is_weights).to(self.device)
                
                batch_size = state_tensor.size(0)

                # --- Step D: Epsilon-Greedy Action ---
                if random.random() < epsilon:
                    action_tensor = torch.randint(0, Settings.OUTPUT_DIM, (batch_size,), device=self.device)
                else:
                    with torch.no_grad():
                        q_values = policy_net(state_tensor)
                        action_tensor = torch.argmax(q_values, dim=1)
                
                # --- Step E: Reward Calculation ---
                profile = Settings.PAIR_CONFIGS[symbol]
                SCALING_FACTOR = profile['scaling_factor']
                SPREAD = profile['spread']
                COMMISSION = profile['commission']

                # 1. Raw PnL
                price_diff = next_price - curr_price
                
                # 2. Normalized PnL
                norm_pnl = price_diff * SCALING_FACTOR
                
                # 3. Normalized Costs
                norm_spread = SPREAD * SCALING_FACTOR
                norm_comm = COMMISSION * SCALING_FACTOR
                total_cost = norm_spread + norm_comm
                
                # 3. Base Penalty
                base_penalty = -0.5 # Aggressive Mode: Waiting is expensive
                reward_tensor = torch.full((batch_size,), base_penalty, device=self.device)
                
                # Masks
                is_buy = (action_tensor == 1)
                is_sell = (action_tensor == 2)
                
                # 4. Final Reward Calculation (Hybrid Tanh)
                buy_net = (norm_pnl - total_cost)
                buy_final = torch.tanh(buy_net)
                reward_tensor[is_buy] = buy_final[is_buy]
                
                sell_net = (-norm_pnl - total_cost)
                sell_final = torch.tanh(sell_net)
                reward_tensor[is_sell] = sell_final[is_sell]
                
                # --- Step F: Learning (DQN Update) ---
                current_q = policy_net(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    next_q = policy_net(next_state_tensor).max(1)[0]
                    target_q = reward_tensor + (Settings.GAMMA * next_q)
                
                # Loss with IS Weights
                loss_element = self.loss_fn(current_q, target_q)
                loss = (loss_element * weights_tensor).mean()
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # --- Step G: Update Priorities ---
                # TD Error = |target - current|
                td_errors = torch.abs(target_q - current_q).detach().cpu().numpy()
                memory.update_priorities(idxs, td_errors)
                
                total_loss += loss.item()
            
            # Decay Epsilon
            if epsilon > Settings.EPSILON_MIN:
                epsilon *= Settings.EPSILON_DECAY
            
            avg_loss = total_loss / steps_per_epoch
            logger.info(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.6f}, Epsilon: {epsilon:.4f}")

        # 4. Save Model
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{symbol}_brain.pth"
        torch.save(policy_net.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep RL Trainer")
    parser.add_argument("--pair", type=str, required=True, help="Symbol to train (e.g., EURUSD)")
    args = parser.parse_args()
    
    symbol = args.pair.upper()
    
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
