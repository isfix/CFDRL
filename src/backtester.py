# src/backtester.py
"""
Phase 27: Backtester with all P0/P1/P2 fixes.

P1: Take-Profit (ATR-based symmetric TP/SL)
P2: Swap costs (overnight financing), latency model (1-bar execution delay)
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Settings
from src import data_factory
from src.brain import QNetwork
from src.utils import logger

class Backtester:
    def __init__(self, symbol, initial_balance=10000):
        self.symbol = symbol
        self.balance = initial_balance
        self.equity_curve = [initial_balance]
        
        if symbol in Settings.PAIR_CONFIGS:
            profile = Settings.PAIR_CONFIGS[symbol]
            self.contract_size = profile['contract_size']
            self.spread = profile['spread']
            self.commission = profile['commission']
        elif "USD" in symbol and "XAU" not in symbol:
            self.contract_size = 100000
            self.spread = 0.0001
            self.commission = 0.0
        else:
            self.contract_size = 100
            self.spread = 0.20
            self.commission = 0.0
              
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self):
        self.model = QNetwork().to(self.device)
        model_path = f"models/{self.symbol}_brain.pth"
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model {model_path} not found!")
            exit()

    def print_stats(self, trades):
        if not trades:
            print("\n--- RESULTS ---")
            print("No trades executed.")
            return
            
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        
        print("\n--- RESULTS ---")
        print(f"Final Balance: ${self.balance:.2f}")
        print(f"Total Trades: {len(trades)}")
        print(f"Wins: {len(wins)} | Losses: {len(losses)}")
        print(f"Win Rate: {len(wins) / len(trades) * 100:.1f}%")
        print(f"Total PnL: ${sum(trades):.2f}")
        print(f"Avg Trade: ${np.mean(trades):.4f}")
        print(f"Max Win: ${max(trades):.4f}")
        print(f"Max Loss: ${min(trades):.4f}")
        
        if len(trades) > 1 and np.std(trades) > 0:
            sharpe = (np.mean(trades) / np.std(trades)) * np.sqrt(len(trades))
            print(f"Sharpe Ratio (trade-level): {sharpe:.4f}")
        
        # Max Drawdown
        peak = self.equity_curve[0]
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
        print(f"Max Drawdown: {max_dd * 100:.2f}%")
        
    def plot_equity(self):
        plt.figure(figsize=(14, 6))
        plt.plot(self.equity_curve, linewidth=0.8)
        plt.title(f"Equity Curve - {self.symbol} (Phase 27)")
        plt.ylabel("Balance ($)")
        plt.xlabel("Bars")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def run_backtest(symbol):
    tester = Backtester(symbol)
    tester.load_model()
    
    df = data_factory.fetch_data(symbol, Settings.TRAIN_DATA_BARS)
    if df.empty:
        return

    test_start_idx = len(df) - Settings.TEST_WINDOW
    if test_start_idx < 0:
        test_start_idx = int(len(df) * 0.8)
    
    df_test_raw = df.iloc[test_start_idx:].copy()
    df_test = data_factory.prepare_features(df_test_raw)
    if len(df_test) < Settings.SEQUENCE_LENGTH + 2:
        print("Not enough data after feature engineering.")
        return

    feature_data = df_test[Settings.FEATURES].values
    opens = df_test['open'].values
    highs = df_test['high'].values
    lows = df_test['low'].values
    closes = df_test['close'].values
    times = df_test.index
    atrs = df_test['volatility'].values * closes  # Absolute ATR

    # State
    position = 0.0
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0        # P1: TP tracking
    current_lot = 0.0
    entry_bar = 0             # P2: Track position duration for swap costs
    
    trades = []
    swap_costs_total = 0.0
    tp_hits = 0
    sl_hits = 0
    action_counts = {i: 0 for i in range(Settings.OUTPUT_DIM)}
    
    # P2: Pending orders (latency model — 1 bar execution delay)
    pending_order = None  # (bar_index, target_position)
    
    print(f"Running simulation on {len(df_test)} bars...")
    
    for t in tqdm(range(Settings.SEQUENCE_LENGTH, len(df_test) - 2)):
        # --- P2: Execute pending order from previous bar ---
        if pending_order is not None:
            order_bar, target_pos = pending_order
            pending_order = None
            
            exec_bar = t  # Execute at current bar's open (1-bar delay)
            slippage = np.random.uniform(0, 0.5) * tester.spread
            
            # Close existing if direction change
            if position != 0 and np.sign(target_pos) != np.sign(position):
                if position > 0:
                    exit_price = opens[exec_bar] - slippage
                    raw_pnl = (exit_price - entry_price) * tester.contract_size * current_lot
                else:
                    exit_price = opens[exec_bar] + slippage
                    raw_pnl = (entry_price - exit_price) * tester.contract_size * current_lot
                
                spread_cost = tester.spread * tester.contract_size * current_lot
                comm_cost = tester.commission * tester.contract_size * current_lot
                
                # P2: Swap cost for multi-day hold
                bars_held = t - entry_bar
                days_held = bars_held / (12 * 24)  # M5 bars per day
                swap_cost = 0.0
                if days_held >= 1.0:
                    position_value = abs(entry_price * tester.contract_size * current_lot)
                    swap_cost = position_value * Settings.SWAP_RATE_ANNUAL / 365.0 * days_held
                    swap_costs_total += swap_cost
                
                net_pnl = raw_pnl - spread_cost - comm_cost - swap_cost
                tester.balance += net_pnl
                trades.append(net_pnl)
                position = 0.0
                current_lot = 0.0
            
            # Open new position
            if abs(target_pos) > 0.01 and position == 0:
                current_lot = round(abs(target_pos) * Settings.MAX_LOT_SIZE, 2)
                current_lot = max(current_lot, 0.01)
                atr = atrs[exec_bar]
                
                if target_pos > 0:
                    position = target_pos
                    entry_price = opens[exec_bar] + slippage
                    stop_loss = entry_price - (atr * Settings.ATR_SL_MULTIPLIER)
                    take_profit = entry_price + (atr * Settings.ATR_TP_MULTIPLIER)  # P1
                else:
                    position = target_pos
                    entry_price = opens[exec_bar] - slippage
                    stop_loss = entry_price + (atr * Settings.ATR_SL_MULTIPLIER)
                    take_profit = entry_price - (atr * Settings.ATR_TP_MULTIPLIER)  # P1
                
                entry_bar = exec_bar
        
        # --- AI Decision (creates pending order with 1-bar delay) ---
        state_tensor = torch.FloatTensor(
            feature_data[t - Settings.SEQUENCE_LENGTH : t]
        ).unsqueeze(0).to(tester.device)
        
        with torch.no_grad():
            q = tester.model(state_tensor)
            action_idx = torch.argmax(q).item()
        
        action_counts[action_idx] += 1
        target_position = Settings.ACTION_MAP[action_idx]
        
        # P2: Queue order for next bar (latency model)
        if abs(target_position - position) > 0.01:
            pending_order = (t, target_position)
        
        # --- SL / TP CHECK on current bar ---
        next_high = highs[t+1]
        next_low = lows[t+1]
        
        # P1: Take-Profit check
        if position > 0:
            if next_high >= take_profit:
                raw_pnl = (take_profit - entry_price) * tester.contract_size * current_lot
                spread_cost = tester.spread * tester.contract_size * current_lot
                net_pnl = raw_pnl - spread_cost
                tester.balance += net_pnl
                trades.append(net_pnl)
                tp_hits += 1
                position = 0.0
                current_lot = 0.0
                pending_order = None  # Cancel pending if TP hit
            elif next_low <= stop_loss:
                raw_pnl = (stop_loss - entry_price) * tester.contract_size * current_lot
                spread_cost = tester.spread * tester.contract_size * current_lot
                net_pnl = raw_pnl - spread_cost
                tester.balance += net_pnl
                trades.append(net_pnl)
                sl_hits += 1
                position = 0.0
                current_lot = 0.0
                pending_order = None
        elif position < 0:
            if next_low <= take_profit:
                raw_pnl = (entry_price - take_profit) * tester.contract_size * current_lot
                spread_cost = tester.spread * tester.contract_size * current_lot
                net_pnl = raw_pnl - spread_cost
                tester.balance += net_pnl
                trades.append(net_pnl)
                tp_hits += 1
                position = 0.0
                current_lot = 0.0
                pending_order = None
            elif next_high >= stop_loss:
                raw_pnl = (entry_price - stop_loss) * tester.contract_size * current_lot
                spread_cost = tester.spread * tester.contract_size * current_lot
                net_pnl = raw_pnl - spread_cost
                tester.balance += net_pnl
                trades.append(net_pnl)
                sl_hits += 1
                position = 0.0
                current_lot = 0.0
                pending_order = None
        
        tester.equity_curve.append(tester.balance)
    
    # --- Results ---
    print("\n--- ACTION DISTRIBUTION ---")
    action_labels = ["STRONG SELL", "SELL", "WEAK SELL", "HOLD", "WEAK BUY", "BUY", "STRONG BUY"]
    total_bars = sum(action_counts.values())
    for idx, label in enumerate(action_labels):
        count = action_counts[idx]
        pct = count / total_bars * 100 if total_bars > 0 else 0
        print(f"  {label:>12}: {count:>6} ({pct:>5.1f}%)")
    
    print(f"\n--- EXIT BREAKDOWN ---")
    signal_exits = len(trades) - tp_hits - sl_hits
    print(f"  TP Hits: {tp_hits}")
    print(f"  SL Hits: {sl_hits}")
    print(f"  Signal Exits: {signal_exits}")
    print(f"  Total Swap Costs: ${swap_costs_total:.4f}")
    
    tester.print_stats(trades)
    tester.plot_equity()


if __name__ == "__main__":
    if not mt5.initialize(
        path=Settings.MT5_PATH,
        login=Settings.MT5_LOGIN,
        password=Settings.MT5_PASSWORD,
        server=Settings.MT5_SERVER
    ):
        logger.error(f"MT5 init failed.")
        exit()
    
    print(f"Available pairs: {Settings.PAIRS}")
    user_symbol = input(f"Enter symbol to backtest (Default: XAUUSD): ").strip().upper() or "XAUUSD"
    
    if user_symbol not in Settings.PAIR_CONFIGS:
        print(f"Symbol {user_symbol} not configured!")
        exit()
    
    try:
        run_backtest(user_symbol)
    except KeyboardInterrupt:
        print("\nBacktest interrupted.")
    finally:
        mt5.shutdown()
