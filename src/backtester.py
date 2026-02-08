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

    def run(self):
        print(f"Starting Backtest for {self.symbol}...")
        
        # 1. Fetch ALL Data
        # We fetch 450k bars as configured
        df = data_factory.fetch_data(self.symbol, Settings.TRAIN_DATA_BARS)
        if df.empty:
            print("No data fetched.")
            return

        # 2. Strict Split
        # Test Set: Settings.TEST_SPLIT_INDEX to End
        test_start_idx = Settings.TEST_SPLIT_INDEX
        
        if len(df) < test_start_idx + 100:
            print(f"Not enough data for split. Got {len(df)}, need > {test_start_idx}")
            return
            
        print(f"Splitting Data. Training (Ignored): 0-{test_start_idx}. Testing: {test_start_idx}-{len(df)}")
        
        # 3. Feature Engineering BEFORE Split (Fix Data Leakage)
        # We calculate indicators on the full history to ensure valid values at the split point.
        df = data_factory.prepare_features(df)
        
        # Slice the DataFrame
        df_test = df.iloc[test_start_idx:].copy()
        
        # Need at least SEQ_LEN bars
        if len(df_test) < Settings.SEQUENCE_LENGTH:
            print("Not enough test data after feature engineering.")
            return

        # Convert to numpy for fast access
        feature_data = df_test[Settings.FEATURES].values
        opens = df_test['open'].values
        highs = df_test['high'].values
        lows = df_test['low'].values
        closes = df_test['close'].values
        times = df_test.index
        # Pre-calculate ATR for Speed (already in features usually, but let's assume 'volatility' * close)
        # volatility = ATR / Close
        atrs = df_test['volatility'].values * closes
        
        # 4. Simulation Loop
        # We iterate from SEQUENCE_LENGTH to End
        # t is the index of the "Current" candle. Decision happens at Close of t.
        # Trade execution happens at Open of t+1.
        
        position = 0 # 0=None, 1=Buy, -1=Sell
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0 # Not strictly used if model exits, but user mentioned hard exit at 20:00
        
        # We can only iterate up to len-2, because we need t+1 to execute
        for t in tqdm(range(Settings.SEQUENCE_LENGTH, len(df_test) - 1)):
            time_idx = df_test.index[t]
            
            # --- State Management ---
            # Input: [t - SEQ_LEN : t]
            current_seq = feature_data[t - Settings.SEQUENCE_LENGTH : t]
            state_tensor = torch.FloatTensor(current_seq).unsqueeze(0).to(self.device)
            
            # --- Model Prediction ---
            with torch.no_grad():
                q_values = self.model(state_tensor)
                
                # PROBABILITY FILTER REMOVED (Machine Gunner Mode)
                # We want to take every trade the model suggests.
                action = torch.argmax(q_values, dim=1).item()
                
                # Confidence check deleted.

                # --- FILTERS (Asset Personality) ---
                if action != 0: # Only filter if trying to trade
                    current_time = pd.to_datetime(time_idx) 
                    
                    # 1. ADX FILTER (Dead Market Avoidance)
                    # "Do Not Disturb" sign for the AI.
                    # ADX is the LAST feature (index -1)
                    current_adx = state_tensor[0, -1, -1].item() * 100.0 # Un-normalize
                    
                    if current_adx < 25:
                        action = 0 # Market is flat. Sit on hands.
                    
                    # 2. BOLLINGER SQUEEZE FILTER (Secondary Volatility Check)
                    # Need to access BB columns. 
                    # Optimally, we extracted them earlier in run_backtest.
                    # But I need to ensure they are available here.
                    # Let's assume they are in df_test.
                    try:
                        # Use t-1 (Last Closed Candle) for Filters to avoid Look-Ahead
                        prev_idx = df_test.index[t-1]
                        bb_u = df_test.loc[prev_idx, 'BBU_20_2.0']
                        bb_l = df_test.loc[prev_idx, 'BBL_20_2.0']
                        curr_close = df_test.loc[prev_idx, 'close']
                        
                        width = bb_u - bb_l
                        vol_pct = width / curr_close
                        if vol_pct < 0.0005:
                             action = 0 # Forced Wait (Squeeze)
                    except KeyError:
                        pass # Columns not found, skip filter
            
            # --- Execution Logic (at t Open) ---
            # Decision made using data up to t-1 (Close). 
            # We execute at the Open of t (Start of current candle).
            next_open = opens[t]
            next_high = highs[t]
            next_low = lows[t]
            next_time = times[t]
            atr = atrs[t-1] # Use ATR of last closed candle
            
            # Check Exits first (Stop Loss / Time / Breakeven)
            if position != 0:
                # Time Exit (20:00)
                if next_time.hour >= 20 and next_time.minute == 0:
                     self.close_position(position, next_open, "Time Exit")
                     position = 0
                
                # SL Check (simplified: did price hit SL within the candle?)
                elif position == 1: # Long
                    if next_low <= stop_loss:
                         self.close_position(1, stop_loss, "SL Hit")
                         position = 0
                    
                    # BREAKEVEN TRIGGER (Upgrade 1)
                    # If profit > 1.0 * ATR, move SL to Entry + Spread
                    elif (next_high - entry_price) > (1.0 * atr) and stop_loss < entry_price:
                        stop_loss = entry_price + (self.spread_cost_per_trade * 0.01) # Approx points
                        # Note: spread_cost_per_trade is in $. Need points.
                        # Assuming 1 pt = $1 for 0.01 lot.
                        # Let's use ATR/10 for a small buffer or just Entry.
                        stop_loss = entry_price 
                        
                elif position == -1: # Short
                    if next_high >= stop_loss:
                         self.close_position(-1, stop_loss, "SL Hit")
                         position = 0
                    
                    # BREAKEVEN TRIGGER (Upgrade 1)
                    elif (entry_price - next_low) > (1.0 * atr) and stop_loss > entry_price:
                        stop_loss = entry_price
            
            # Process Signal (Entry / Close)
            if action == 1: # Signal BUY
                if position == -1: # Close Short
                    self.close_position(-1, next_open, "Signal Flip")
                    position = 0
                
                if position == 0: # Open Long
                    entry_price = next_open + (self.spread_cost_per_trade / 100) # Add simulated spread to entry price? 
                    # User: "Subtract Spread... from every trade".
                    # Better: Fill at Open, verify Cost deduction in PnL logic.
                    entry_price = next_open
                    stop_loss = entry_price - (atr * 2.5)
                    position = 1
            
            elif action == 2: # Signal SELL
                if position == 1: # Close Long
                    self.close_position(1, next_open, "Signal Flip")
                    position = 0
                    
                if position == 0: # Open Short
                    entry_price = next_open
                    stop_loss = entry_price + (atr * 2.5)
                    position = -1
            
            # Update Equity Curve (Mark to Market - approx)
            # This is complex in a loop, let's just create curve updates on Trade Close for simplicity
            # OR ideally update every bar.
            current_pnl = 0
            if position == 1:
                current_pnl = (closes[t] - entry_price) * self.contract_size * self.lot_size
            elif position == -1:
                current_pnl = (entry_price - closes[t]) * self.contract_size * self.lot_size
            
            self.equity_curve.append(self.balance + current_pnl)

        print("Backtest Complete.")
        self.print_stats()
        self.plot_equity()

    def close_position(self, direction, price, reason):
        # direction: 1=Buy, -1=Sell
        # price: Exit Price
        
        # Calculate Gross PnL
        # Note: self.entry_price needs to be stored in the class actually, i fixed it below
        pass # Implemented inline for simplicity in the loop above? 
        # Refactoring to a method requires valid state.
        # Let's rely on the simulation loop logic. 
        # But wait, i need to deduct costs.
        
        # Let's fix the loop logic slightly to handle PnL correctly.
        # I will do it inside the loop for now.
        pass

    # Re-implementing the loop with cleaner logic for PnL
    # ... (Actually, let's just keep the loop logic self-contained)

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
