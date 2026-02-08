import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Settings
from src import data_factory
from src.brain import QNetwork
from src.utils import logger

class Backtester:
    def __init__(self, symbol, initial_balance=10000):
        self.symbol = symbol
        self.balance = initial_balance
        self.equity = initial_balance
        self.equity_curve = [initial_balance]
        self.trades = []
        self.win_count = 0
        self.loss_count = 0
        
        # Costs (Dynamic from Config)
        self.lot_size = 0.01
        
        if symbol in Settings.PAIR_CONFIGS:
             profile = Settings.PAIR_CONFIGS[symbol]
             self.contract_size = profile['contract_size']
             
             # Calculate Costs in $
             # Spread Cost = Spread * Contract * Lot
             self.spread_cost_per_trade = profile['spread'] * self.contract_size * self.lot_size
             self.comm_per_round_trip = profile['commission'] * self.contract_size * self.lot_size
             
        elif "USD" in symbol and "XAU" not in symbol:
             # Fallback Forex
             self.contract_size = 100000
             self.spread_cost_per_trade = 0.10
             self.comm_per_round_trip = 0.15
        else:
             # Fallback Gold
             self.contract_size = 100
             self.spread_cost_per_trade = 0.20
             self.comm_per_round_trip = 0.07
             
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self):
        self.model = QNetwork().to(self.device)
        model_path = f"models/{self.symbol}_brain.pth"
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval() # Set to eval mode (No Dropout)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model {model_path} not found!")
            exit()

    # run() and close_position() removed as they are superseded by run_backtest() logic


    def print_stats(self):
        print("\n--- RESULTS ---")
        print(f"Final Balance: ${self.balance:.2f}")
        print(f"Total Trades: {len(self.trades)}")
        print(f"Win Rate: {self.win_count / len(self.trades) * 100 if self.trades else 0:.1f}%")
        
    def plot_equity(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve)
        plt.title(f"Equity Curve - {self.symbol}")
        plt.ylabel("Balance ($)")
        plt.xlabel("Bars")
        plt.grid()
        plt.show()

# --- Redefining Run to include logic correctly ---
# I will use a more monolithic run method to ensure variable scope is correct
    
def run_backtest(symbol):
    tester = Backtester(symbol)
    tester.load_model()
    
    # ... Fetch Data ...
    df = data_factory.fetch_data(symbol, Settings.TRAIN_DATA_BARS)
    if df.empty: return

    test_start_idx = Settings.TEST_SPLIT_INDEX
    if len(df) < test_start_idx + 100: return
    
    df_test = df.iloc[test_start_idx:].copy()
    df_test = data_factory.prepare_features(df_test)
    if len(df_test) < Settings.SEQUENCE_LENGTH: return

    feature_data = df_test[Settings.FEATURES].values
    opens = df_test['open'].values
    highs = df_test['high'].values
    lows = df_test['low'].values
    closes = df_test['close'].values
    times = df_test.index
    atrs = df_test['volatility'].values * closes
    
    position = 0 # 0, 1, -1
    entry_price = 0.0
    stop_loss = 0.0
    
    # Statistics
    trades = []
    
    print(f"Running simulation on {len(df_test)} bars...")
    
    for t in tqdm(range(Settings.SEQUENCE_LENGTH, len(df_test) - 1)):
        # Data at t is CLOSED. We decide.
        # Execution is at t+1 OPEN.
        
        state_tensor = torch.FloatTensor(feature_data[t - Settings.SEQUENCE_LENGTH : t]).unsqueeze(0).to(tester.device)
        
        with torch.no_grad():
            q = tester.model(state_tensor)
            action = torch.argmax(q).item()
            
        next_open = opens[t+1]
        next_high = highs[t+1]
        next_low = lows[t+1]
        next_time = times[t+1]
        atr = atrs[t]
        
        # 1. CHECK EXIT for Existing Position
        pnl = 0
        trade_closed = False
        
        if position != 0:
            exit_price = 0.0
            
            # Time Exit
            if next_time.hour >= 20 and next_time.minute == 0:
                exit_price = next_open
                trade_closed = True
            
            # SL Hit (Assumes hit at SL price exactly - slippage ignored)
            elif position == 1 and next_low <= stop_loss:
                exit_price = stop_loss
                trade_closed = True
            elif position == -1 and next_high >= stop_loss:
                exit_price = stop_loss
                trade_closed = True
                
            # Logic Flip (handled below, but if closed here, we update)
            
            if trade_closed:
                # Calc PnL
                raw_pnl = (exit_price - entry_price) * tester.contract_size * tester.lot_size if position == 1 else (entry_price - exit_price) * tester.contract_size * tester.lot_size
                # Deduct Costs
                net_pnl = raw_pnl - tester.spread_cost_per_trade - tester.comm_per_round_trip
                
                tester.balance += net_pnl
                trades.append(net_pnl)
                position = 0
                
        # 2. ENTRY / REVERSAL Logic
        # Only if not just closed (or maybe we can reverse? kept simple)
        if not trade_closed:
            if action == 1: # Buy Signal
                if position == -1: # Reverse Short -> Long
                    # Close Short
                    exit_price = next_open
                    raw_pnl = (entry_price - exit_price) * tester.contract_size * tester.lot_size
                    net_pnl = raw_pnl - tester.spread_cost_per_trade - tester.comm_per_round_trip
                    tester.balance += net_pnl
                    trades.append(net_pnl)
                    
                    # Open Long
                    position = 1
                    entry_price = next_open
                    stop_loss = entry_price - (atr * 2.5)
                    
                elif position == 0: # Open Long
                    position = 1
                    entry_price = next_open
                    stop_loss = entry_price - (atr * 2.5)
            
            elif action == 2: # Sell Signal
                if position == 1: # Reverse Long -> Short
                    # Close Long
                    exit_price = next_open
                    raw_pnl = (exit_price - entry_price) * tester.contract_size * tester.lot_size
                    net_pnl = raw_pnl - tester.spread_cost_per_trade - tester.comm_per_round_trip
                    tester.balance += net_pnl
                    trades.append(net_pnl)
                    
                    # Open Short
                    position = -1
                    entry_price = next_open
                    stop_loss = entry_price + (atr * 2.5)
                    
                elif position == 0: # Open Short
                    position = -1
                    entry_price = next_open
                    stop_loss = entry_price + (atr * 2.5)
        
        # Tracking
        tester.equity_curve.append(tester.balance)
        
    tester.trades = trades
    tester.win_count = sum(1 for x in trades if x > 0)
    print("Backtest Complete.")
    tester.print_stats()
    tester.plot_equity()

if __name__ == "__main__":
    if not mt5.initialize(path=Settings.MT5_PATH, login=Settings.MT5_LOGIN, password=Settings.MT5_PASSWORD, server=Settings.MT5_SERVER):
         print("MT5 Init Failed")
         exit()
         
    # Interactive Mode
    user_symbol = input(f"Enter symbol to backtest (Available: {Settings.PAIRS}, Default: XAUUSD): ").strip().upper()
    
    if not user_symbol:
        user_symbol = "XAUUSD"
        
    run_backtest(user_symbol)
