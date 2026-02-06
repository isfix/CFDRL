import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Settings
from src import data_factory

def run_reward_debug(symbol="EURUSD", tuning_file=None):
    if tuning_file is None:
        tuning_file = f"{symbol}_Tuning.csv"
        
    print("--- Static Reward Debugger ---")
    
    csv_path = os.path.join(os.path.dirname(__file__), "data", tuning_file)
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path, parse_dates=['time'], index_col='time')
    print(f"Loaded {len(df)} bars.")
    
    # Load Config for the Symbol
    if symbol in Settings.PAIR_CONFIGS:
        profile = Settings.PAIR_CONFIGS[symbol]
        SCALING = profile['scaling_factor']
        SPREAD = profile['spread']
        COMMISSION = profile['commission']
    else:
        print(f"Warning: {symbol} not in PAIR_CONFIGS. Using Defaults.")
        # Fallback to defaults or user arguments if we had them
        SCALING = 1.0
        SPREAD = 0.0001
        COMMISSION = 0.0
    
    norm_spread = SPREAD * SCALING
    norm_comm = COMMISSION * SCALING
    total_cost = norm_spread + norm_comm
    
    print(f"Config: Spread={SPREAD}, Comm={COMMISSION}, Scaling={SCALING}")
    print(f"Per Trade Cost Penalty = {total_cost:.4f} reward units")
    
    cumulative_reward = 0.0
    win_count = 0
    loss_count = 0
    
    # Simulation: Assume Perfect Direction Prediction for every candle
    # If Next Close > Curr Close: We Buy
    # If Next Close < Curr Close: We Sell
    
    closes = df['close'].values
    
    for t in range(len(closes) - 1):
        curr_price = closes[t]
        next_price = closes[t+1]
        
        diff = next_price - curr_price
        
        if diff > 0: 
            # Perfect Bull trade
            # Reward = (Diff * Scale) - Cost
            # If > 0, * 10
            r = (diff * SCALING) - total_cost
            if r > 0:
                reward = r * 10
                win_count += 1
            else:
                reward = r # Small win wiped by cost
                loss_count += 1
                
        elif diff < 0:
            # Perfect Bear trade
            # Reward = (-Diff * Scale) - Cost
            r = (-diff * SCALING) - total_cost
            if r > 0:
                reward = r * 10
                win_count += 1
            else:
                reward = r
                loss_count += 1
                
        else:
            reward = -0.1 # Hold penalty
            
        cumulative_reward += reward
        
    print(f"\n--- Results (Perfect Strategy) ---")
    print(f"Total Potential Reward: {cumulative_reward:.2f}")
    print(f"Profitable Candles: {win_count}")
    print(f"Unprofitable Candles: {loss_count}")
    
    if cumulative_reward > 1000:
        print("Verdict: PASSED. Massive potential for profit exists.")
    elif cumulative_reward > 0:
        print("Verdict: MARGINAL. Profit possible but difficult.")
    else:
        print("Verdict: IMPOSSIBLE. Costs > Volatility. Increase Scaling or Reduce Commission.")

if __name__ == "__main__":
    run_reward_debug()
