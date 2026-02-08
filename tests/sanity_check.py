import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Settings
from src import data_factory
from src.brain import QNetwork
from src.trainer import TradingDataset # Now importable

def run_sanity_check(symbol="EURUSD", tuning_file=None):
    if tuning_file is None:
        tuning_file = f"{symbol}_Tuning.csv"
        
    print("--- 5-Minute Sanity Check (Overfit Test) ---")
    
    # 1. Load Data
    csv_path = os.path.join(os.path.dirname(__file__), "data", tuning_file)
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found. Please run create_tuning_data.py first.")
        # Fallback for testing immediately without generating tuning data: try to fetch from MT5 locally if file missing?
        # Better to fail and ask user to generate.
        return

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=['time'], index_col='time')
    
    # 2. Slice Tiny Dataset (First 2000 bars)
    print("Slicing first 2000 bars for Overfit Test...")
    df_small = df.iloc[:2000].copy()
    
    # 3. Prepare Features
    df_small = data_factory.prepare_features(df_small)
    print(f"Data Shape after features: {df_small.shape}")
    print(f"Feature Sample (First Row):\n{df_small[Settings.FEATURES].iloc[0]}")
    
    # 4. Create Dataset/Loader
    feature_data = df_small[Settings.FEATURES].values
    close_prices = df_small['close'].values
    dataset = TradingDataset(feature_data, close_prices, Settings.SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True) # Small batch for small data
    
    # 5. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = QNetwork().to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0001) # Lower LR for stability with larger inputs
    loss_fn = nn.MSELoss()
    
    print(f"Training on {device} for 50 Epochs...")
    
    # 6. Training Loop (Simplified)
    loss_history = []
    
    for epoch in range(50):
        total_loss = 0
        for batch in dataloader:
            state = batch['state'].to(device)
            next_state = batch['next_state'].to(device)
            curr_price = batch['curr_price'].to(device)
            next_price = batch['next_price'].to(device)
            
            # Forward
            q_values = policy_net(state)
            
            # Pseudo-Optimal Actions (We want to see if it learns best path)
            # Actually we use Epsilon Greedy usually, but for Overfit we want to see Loss drop.
            # We'll use the same Logic as Trainer but with Fixed high epsilon? 
            # No, for overfitting we want it to minimize error on Q-values.
            # Let's use standard DQN update.
            
            # We assume "best" action implies we know the future. 
            # But here we just run the standard algo and hope it converges.
            
            # Action Selection (Argmax for now to test convergence logic?? No, Training needs exploration usually)
            # For strict overfit test, we can just run the standard loop.
            
            # ... (Copying simplified logic from Trainer) ...
            
            # For brevity, let's just train to predict a dummy target? 
            # No, we want to test the REWARD FUNCTION pipeline!
            
            # Epsilon = 0.1 (Mostly exploit to show it learned)
            epsilon = 0.1
            
            batch_size = state.size(0)
            
            rand = torch.rand(batch_size, device=device)
            rand_mask = rand < epsilon
            rand_act = torch.randint(0, 3, (batch_size,), device=device)
            
            with torch.no_grad():
                model_act = torch.argmax(policy_net(state), dim=1)
                
            actions = torch.where(rand_mask, rand_act, model_act)
            
            # --- Reward Calc (Reused from logic) ---
            # --- Reward Calc (Reused from logic) ---
            # Profile: Use XAUUSD config manually or from Settings
            if symbol in Settings.PAIR_CONFIGS:
                profile = Settings.PAIR_CONFIGS[symbol]
                SCALING = profile['scaling_factor']
                SPREAD = profile['spread']
            else:
                SCALING = 5.0 
                SPREAD = 0.20
            
            diff = next_price - curr_price
            norm_pnl = diff * SCALING
            cost = SPREAD * SCALING
            
            # GODLIKE: Higher penalty for inaction to force scalping
            base_penalty = -0.5 
            rewards = torch.full((batch_size,), base_penalty, device=device)
            
            is_buy = (actions == 1)
            is_sell = (actions == 2)
            
            # Buy Logic (Consistency Focus)
            buy_r = (norm_pnl - cost)
            buy_final = torch.where(
                buy_r > 0, 
                buy_r + 1.0,  # Bonus for winning
                buy_r * 2.0   # Punishment for losing
            )
            rewards[is_buy] = torch.clamp(buy_final[is_buy], -2.0, 5.0)
            
            # Sell Logic
            sell_r = (-norm_pnl - cost)
            sell_final = torch.where(
                sell_r > 0, 
                sell_r + 1.0, 
                sell_r * 2.0
            )
            rewards[is_sell] = torch.clamp(sell_final[is_sell], -2.0, 5.0)
            
            # Update
            curr_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = policy_net(next_state).max(1)[0]
                target_q = rewards + (0.85 * next_q) # Gamma 0.85
                
            loss = loss_fn(curr_q, target_q)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss {avg_loss:.6f}")
            
    # Check Result
    final_loss = loss_history[-1]
    print(f"Final Loss: {final_loss:.6f}")
    if final_loss < 0.1: # Arbitrary threshold for "Converged"
        print("SUCCESS: Model is overfitting/learning! Math pipeline is likely valid.")
    else:
        print("WARNING: Model failed to overfit even 2000 bars. Check Inputs/Reward Function.")

if __name__ == "__main__":
    run_sanity_check()
