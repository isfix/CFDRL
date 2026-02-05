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
from config import Settings
from src import data_factory
from src.brain import QNetwork
from src.utils import logger
import os
from torch.utils.data import Dataset, DataLoader

class TradingDataset(Dataset):
    def __init__(self, feature_data, close_prices, seq_len):
        self.feature_data = feature_data
        self.close_prices = close_prices
        self.seq_len = seq_len
        # Valid indices match the original loop: range(Settings.SEQUENCE_LENGTH, len(df) - 1)
        # i goes from seq_len to len(df) - 2
        # feature_data length is len(df)
        self.valid_indices = range(seq_len, len(feature_data) - 1)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Map idx (0 to len-1) to the actual index i in feature_data
        i = self.valid_indices[idx]

        # State: i - seq_len : i
        state_window = self.feature_data[i - self.seq_len : i]

        # Next State: i - seq_len + 1 : i + 1
        next_state_window = self.feature_data[i - self.seq_len + 1 : i + 1]

        # Prices for reward calculation
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
        
        self.loss_fn = nn.MSELoss()

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
        # Convert DataFrame to numpy for faster access
        # We need the FEATURES columns for input, and CLOSE price for reward calculation
        feature_data = df[Settings.FEATURES].values
        close_prices = df['close'].values
        
        dataset = TradingDataset(feature_data, close_prices, Settings.SEQUENCE_LENGTH)
        dataloader = DataLoader(dataset, batch_size=Settings.BATCH_SIZE, shuffle=True, drop_last=True)
        
        # Initialize Model & Optimizer
        policy_net = QNetwork().to(self.device)
        optimizer = optim.Adam(policy_net.parameters(), lr=Settings.LEARNING_RATE)
        
        epsilon = Settings.EPSILON_START
        
        # 3. Training Loop
        for epoch in range(Settings.EPOCHS):
            total_loss = 0
            
            # Use tqdm for progress bar
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Settings.EPOCHS} - {symbol}")
            
            for batch in pbar:
                state_tensor = batch['state'].to(self.device)
                next_state_tensor = batch['next_state'].to(self.device)
                curr_price = batch['curr_price'].to(self.device)
                next_price = batch['next_price'].to(self.device)
                
                batch_size = state_tensor.size(0)

                # --- Step D: Epsilon-Greedy Action (Vectorized) ---
                random_vals = torch.rand(batch_size, device=self.device)
                random_mask = random_vals < epsilon
                
                random_actions = torch.randint(0, Settings.OUTPUT_DIM, (batch_size,), device=self.device)
                
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    model_actions = torch.argmax(q_values, dim=1)
                
                action_tensor = torch.where(random_mask, random_actions, model_actions)
                
                # --- Step E: Reward Calculation (Vectorized) ---
                # Recall Reward based on price change
                price_diff = next_price - curr_price
                
                # Spread cost approximation
                spread_penalty = 0.0002 * curr_price
                
                reward_tensor = torch.zeros(batch_size, device=self.device)
                
                # Buy
                reward_tensor[action_tensor == 1] = price_diff[action_tensor == 1] - spread_penalty[action_tensor == 1]
                # Sell
                reward_tensor[action_tensor == 2] = -price_diff[action_tensor == 2] - spread_penalty[action_tensor == 2]
                # Hold -> 0
                
                # --- Step F: Learning (DQN Update) ---
                # Q(s, a) = r + gamma * max(Q(s', a'))
                
                # Get Q-value for the taken action
                current_q = policy_net(state_tensor).gather(1, action_tensor.unsqueeze(1))
                
                # Get Max Q for next state
                with torch.no_grad():
                    next_q = policy_net(next_state_tensor).max(1)[0]
                    target_q = reward_tensor + (Settings.GAMMA * next_q)
                
                # Loss
                loss = self.loss_fn(current_q.squeeze(1), target_q)
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Decay Epsilon
            if epsilon > Settings.EPSILON_MIN:
                epsilon *= Settings.EPSILON_DECAY
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.6f}, Epsilon: {epsilon:.4f}")

        # 4. Save Model
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{symbol}_brain.pth"
        torch.save(policy_net.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    if not mt5.initialize():
        logger.error("MT5 init failed")
        exit()
    
    trainer = Trainer()
    
    try:
        for symbol in Settings.PAIRS:
            trainer.train_model(symbol)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    finally:
        mt5.shutdown()
