# src/trainer.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import MetaTrader5 as mt5
from tqdm import tqdm
from config import Settings
from src import data_factory
from src.brain import QNetwork
from src.utils import logger
import os

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
        
        # Create sequences
        # X: (N, seq_len, input_dim)
        # We can't use a standard DataLoader easily because we need random access for replay memory
        # But for this simple DQN, we can just iterate or sample.
        # Let's pre-generate valid indices where a full sequence exists
        valid_indices = range(Settings.SEQUENCE_LENGTH, len(df) - 1) # -1 because we need 'next' state for reward
        
        # Initialize Model & Optimizer
        policy_net = QNetwork().to(self.device)
        optimizer = optim.Adam(policy_net.parameters(), lr=Settings.LEARNING_RATE)
        
        epsilon = Settings.EPSILON_START
        
        # 3. Training Loop
        for epoch in range(Settings.EPOCHS):
            total_loss = 0
            
            # Use tqdm for progress bar
            pbar = tqdm(valid_indices, desc=f"Epoch {epoch+1}/{Settings.EPOCHS} - {symbol}")
            
            for i in pbar:
                # Current State (Sequence ending at i)
                # Slice: i - seq_len : i
                state_window = feature_data[i - Settings.SEQUENCE_LENGTH : i]
                state_tensor = torch.FloatTensor(state_window).unsqueeze(0).to(self.device) # (1, seq, dim)
                
                # Next State (Sequence ending at i+1)
                next_state_window = feature_data[i - Settings.SEQUENCE_LENGTH + 1 : i + 1]
                next_state_tensor = torch.FloatTensor(next_state_window).unsqueeze(0).to(self.device)
                
                # --- Step D: Epsilon-Greedy Action ---
                if random.random() < epsilon:
                    action = random.randint(0, Settings.OUTPUT_DIM - 1)
                else:
                    with torch.no_grad():
                        q_values = policy_net(state_tensor)
                        action = torch.argmax(q_values).item()
                
                # --- Step E: Recall Reward ---
                # Reward based on price change from i to i+1
                current_close = close_prices[i-1] # index aligns with end of state_window? 
                # Be careful with indices. 
                # feature_data[i] corresponds to the row at index i.
                # state_window is [i-seq : i], so the last row is i-1.
                # So current price is close_prices[i-1].
                # Next price is close_prices[i].
                
                curr_price = close_prices[i-1]
                next_price = close_prices[i]
                
                price_diff = next_price - curr_price
                
                # Spread cost approximation (in price units)
                # Settings.SPREAD_FILTER_POINTS is in points (e.g. 30 points). 
                # We need to convert points to price. 
                # For simplicity in this example, let's assume a fixed small cost or 0 for training signal.
                # Or better: check symbol digits. But for now, let's use a simplified logical reward.
                # --- FIX START ---
                # Estimate spread cost based on config
                # We convert points to price. For XAUUSD, 1 point = 0.01 (usually), but let's be safe.
                # A rough estimate for Gold spread is 0.20 to 0.30.
                # Let's use a percentage estimate if we don't know point value, 
                # OR use the config value assuming it equates to price difference.
                
                # Better approach for training: Penalize every trade entry.
                # Let's assume a cost of ~2 pips (0.0002 for forex, 0.20 for gold)
                # We can approximate this using the Close price * 0.0001 (1 pip approx) * Spread Factor
                
                # Simplified robust fix:
                spread_penalty = 0.0002 * curr_price # Approx 2 pips cost relative to price
                
                reward = 0
                if action == 1: # Buy
                    reward = price_diff - spread_penalty
                elif action == 2: # Sell
                    reward = -price_diff - spread_penalty
                elif action == 0: # Hold
                    reward = 0 # No penalty for holding
                # --- FIX END ---
                
                # Normalize reward slightly to avoid exploding gradients if prices are huge (like BTC)
                # or just clip later.
                
                reward_tensor = torch.FloatTensor([reward]).to(self.device)
                action_tensor = torch.LongTensor([action]).to(self.device)
                
                # --- Step F: Learning (DQN Update) ---
                # Q(s, a) = r + gamma * max(Q(s', a'))
                
                # Get Q-value for the taken action
                current_q = policy_net(state_tensor).gather(1, action_tensor.unsqueeze(1))
                
                # Get Max Q for next state
                with torch.no_grad():
                    next_q = policy_net(next_state_tensor).max(1)[0]
                    target_q = reward_tensor + (Settings.GAMMA * next_q)
                
                # Loss
                loss = self.loss_fn(current_q.squeeze(), target_q.squeeze())
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Decay Epsilon
            if epsilon > Settings.EPSILON_MIN:
                epsilon *= Settings.EPSILON_DECAY
            
            avg_loss = total_loss / len(valid_indices)
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
