import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Settings
from src import data_factory
from src.brain import QNetwork
from src.trainer import TradingDataset

def objective(trial):
    # 1. Hyperparameters to Tune
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.99)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    scaling_factor = trial.suggest_float("scaling_factor", 1.0, 10.0) # Tuning the reward scale itself!
    
    # 2. Load Data (Golden Slice directly)
    tuning_file = "EURUSD_Tuning.csv" # Default for now, could be dynamic
    csv_path = os.path.join(os.path.dirname(__file__), "data", tuning_file)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{tuning_file} not found in tests/data")
        
    df = pd.read_csv(csv_path, parse_dates=['time'], index_col='time')
    # Use only first 10k bars for speed
    df = df.iloc[:10000].copy()
    
    df = data_factory.prepare_features(df)
    feature_data = df[Settings.FEATURES].values
    close_prices = df['close'].values
    
    dataset = TradingDataset(feature_data, close_prices, Settings.SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    # 4. Short Training Loop (10 Epochs)
    epochs = 10
    
    for epoch in range(epochs):
        total_loss = 0
        total_reward_metric = 0 # To track performance
        
        for batch in dataloader:
            state = batch['state'].to(device)
            next_state = batch['next_state'].to(device)
            curr_price = batch['curr_price'].to(device)
            next_price = batch['next_price'].to(device)
            
            # Simple Forward
            q_values = model(state)
            
            # Action Selection (Epsilon 0.5 - High Exploration for tuning)
            # Or just Argmax? Training needs exploration.
            epsilon = 0.3
            batch_size_curr = state.size(0)
            
            rand_act = torch.randint(0, 3, (batch_size_curr,), device=device)
            model_act = torch.argmax(q_values, dim=1)
            mask = torch.rand(batch_size_curr, device=device) < epsilon
            actions = torch.where(mask, rand_act, model_act)
            
            # --- Reward Calc (Using Trial Scaling for 'scaling_factor', but others from Config) ---
            # We want to allow Optuna to tune scaling_factor, but base SPREAD should come from config
            if "EURUSD" in Settings.PAIR_CONFIGS: # Assuming EURUSD for now or make symbol dynamic
                 profile = Settings.PAIR_CONFIGS["EURUSD"]
                 SPREAD = profile['spread']
            else:
                 SPREAD = 0.0001
                 
            diff = next_price - curr_price
            norm_pnl = diff * scaling_factor
            cost = SPREAD * scaling_factor
            
            is_buy = (actions == 1)
            is_sell = (actions == 2)
            
            rewards = torch.full((batch_size_curr,), -0.1, device=device)
            
            buy_r = (norm_pnl - cost)
            rewards[is_buy] = torch.where(buy_r > 0, buy_r * 10, buy_r)[is_buy]
            
            sell_r = (-norm_pnl - cost)
            rewards[is_sell] = torch.where(sell_r > 0, sell_r * 10, sell_r)[is_sell]
            
            # Update
            curr_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = model(next_state).max(1)[0]
                target_q = rewards + (gamma * next_q)
                
            loss = loss_fn(curr_q, target_q)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_reward_metric += rewards.sum().item()
            
        avg_loss = total_loss / len(dataloader)
        avg_reward = total_reward_metric / len(dataloader)
        
        # Report intermediate result to Optuna
        # We maximize Reward (or minimize Loss). Let's maximize Reward.
        trial.report(avg_reward, epoch)
        
        # Handle Pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return avg_reward

if __name__ == "__main__":
    print("--- Optuna Hyperparameter Optimization ---")
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "data", "EURUSD_Tuning.csv")):
        print("Dataset missing. Run create_tuning_data.py first.")
        exit()
        
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=20)
    
    print("\n--- Best Params ---")
    print(study.best_params)
