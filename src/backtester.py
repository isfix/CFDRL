# src/backtester.py
"""
Backtester for position-aware QNetwork with semantic 4-action space.

Simulates: latency (1-bar delay), spread, commission, ATR-based SL/TP, swap costs.
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

# Semantic actions (must match training)
ACT_HOLD, ACT_BUY, ACT_SELL, ACT_CLOSE = 0, 1, 2, 3
ACTION_NAMES = ['HOLD', 'BUY', 'SELL', 'CLOSE']


class Backtester:
    def __init__(self, symbol, initial_balance=10000):
        self.symbol = symbol
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.equity_curve = [initial_balance]

        if symbol in Settings.PAIR_CONFIGS:
            profile = Settings.PAIR_CONFIGS[symbol]
            self.contract_size = profile['contract_size']
            self.spread = profile['spread']
            self.commission = profile['commission']
            self.scaling_factor = profile['scaling_factor']
        elif "USD" in symbol and "XAU" not in symbol:
            self.contract_size = 100000
            self.spread = 0.0001
            self.commission = 0.0
            self.scaling_factor = 10000.0
        else:
            self.contract_size = 100
            self.spread = 0.20
            self.commission = 0.0
            self.scaling_factor = 10.0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self):
        """Load model — tries best_{symbol}.pt first, then {symbol}_brain.pth."""
        candidates = [
            f"models/best_{self.symbol}.pt",
            f"models/{self.symbol}_brain.pth",
        ]
        for path in candidates:
            if os.path.exists(path):
                self.model = QNetwork().to(self.device)
                state_dict = torch.load(path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"Loaded model from {path} ({total_params:,} params)")
                return
        print(f"No model found for {self.symbol}! Tried: {candidates}")
        exit()

    def print_stats(self, trades):
        if not trades:
            print("\n--- RESULTS ---")
            print("No trades executed.")
            return

        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        total_pnl = sum(trades)
        roi = (self.balance - self.initial_balance) / self.initial_balance * 100

        print("\n" + "=" * 60)
        print(f"  BACKTEST RESULTS — {self.symbol}")
        print("=" * 60)
        print(f"  Final Balance:  ${self.balance:,.2f}  (ROI: {roi:+.1f}%)")
        print(f"  Total PnL:      ${total_pnl:,.2f}")
        print(f"  Total Trades:   {len(trades)}")
        print(f"  Wins:           {len(wins)}  |  Losses: {len(losses)}")
        if trades:
            print(f"  Win Rate:       {len(wins) / len(trades) * 100:.1f}%")
            print(f"  Avg Trade:      ${np.mean(trades):.4f}")
            print(f"  Max Win:        ${max(trades):.4f}")
            print(f"  Max Loss:       ${min(trades):.4f}")

        if len(trades) > 1 and np.std(trades) > 0:
            sharpe = (np.mean(trades) / np.std(trades)) * np.sqrt(len(trades))
            print(f"  Sharpe (trade): {sharpe:.4f}")

        # Max Drawdown
        peak = self.equity_curve[0]
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
        print(f"  Max Drawdown:   {max_dd * 100:.2f}%")

        # Daily Sharpe from equity curve
        if len(self.equity_curve) > 96:
            eq = np.array(self.equity_curve)
            # 96 M15 bars per day
            daily_returns = []
            for i in range(96, len(eq), 96):
                daily_returns.append(eq[i] - eq[i - 96])
            if len(daily_returns) > 5:
                dr = np.array(daily_returns)
                daily_sharpe = (dr.mean() / (dr.std() + 1e-8)) * np.sqrt(252)
                print(f"  Sharpe (daily): {daily_sharpe:.4f}")
        print("=" * 60)

    def plot_equity(self):
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Equity curve
        axes[0].plot(self.equity_curve, linewidth=0.8)
        axes[0].axhline(y=self.initial_balance, color='r', linestyle='--', alpha=0.5)
        axes[0].set_title(f"Equity Curve — {self.symbol}")
        axes[0].set_ylabel("Balance ($)")
        axes[0].set_xlabel("Bars (M15)")
        axes[0].grid(alpha=0.3)

        # Drawdown
        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak * 100
        axes[1].fill_between(range(len(dd)), dd, alpha=0.4, color='red')
        axes[1].set_title(f"Drawdown — {self.symbol}")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].set_xlabel("Bars (M15)")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        suffix = '_oos' if hasattr(self, '_oos') and self._oos else ''
        fname = f"backtest_{self.symbol}{suffix}.png"
        plt.savefig(fname, dpi=150)
        plt.show()
        print(f"Chart saved: {fname}")


def run_backtest(symbol, oos=False, train_bars=400000):
    tester = Backtester(symbol)
    tester._oos = oos
    tester.load_model()

    pair_cfg = Settings.PAIR_CONFIGS.get(symbol, Settings.PAIR_CONFIGS['EURUSD'])
    SF = pair_cfg['scaling_factor']

    if oos:
        # Out-of-sample: fetch 2x the training bars, test on the OLDER half
        total_fetch = train_bars * 2
        print(f"\n=== OUT-OF-SAMPLE MODE ===")
        print(f"Fetching {total_fetch} M5 bars (training used latest {train_bars})")
        df = data_factory.fetch_data(symbol, total_fetch)
        if df.empty:
            print("Failed to fetch data.")
            return
        # The model was trained on the LATEST train_bars, so use everything BEFORE that
        cutoff = len(df) - train_bars
        if cutoff <= 0:
            print(f"Not enough data: got {len(df)} bars, need >{train_bars}")
            return
        df_test_raw = df.iloc[:cutoff].copy()
        print(f"OOS test data: {len(df_test_raw)} M5 bars")
        print(f"  OOS Period: {df_test_raw.index[0]} to {df_test_raw.index[-1]}")
        print(f"  (Training data starts at: {df.index[cutoff]})")
    else:
        # Default: fetch and use last 20% as test (in-sample validation)
        df = data_factory.fetch_data(symbol, Settings.TRAIN_DATA_BARS)
        if df.empty:
            print("Failed to fetch data.")
            return
        test_start_idx = int(len(df) * 0.8)
        df_test_raw = df.iloc[test_start_idx:].copy()

    print(f"Raw test data: {len(df_test_raw)} M5 bars from {df_test_raw.index[0]} to {df_test_raw.index[-1]}")

    # Resample M5 -> M15 and compute features
    df_m15 = data_factory.resample_ohlcv(df_test_raw, rule='15min')
    df_test = data_factory.prepare_features(df_m15)

    if len(df_test) < Settings.SEQUENCE_LENGTH + 10:
        print(f"Not enough data after features: {len(df_test)} bars")
        return

    print(f"Test data: {len(df_test)} M15 bars from {df_test.index[0]} to {df_test.index[-1]}")

    # Extract arrays
    available_feats = [f for f in Settings.FEATURES if f in df_test.columns]
    if len(available_feats) < len(Settings.FEATURES):
        missing = set(Settings.FEATURES) - set(available_feats)
        print(f"Warning: missing {len(missing)} features: {missing}")

    feature_data = df_test[available_feats].values.astype(np.float32)
    closes = df_test['close'].values
    opens = df_test['open'].values
    highs = df_test['high'].values
    lows = df_test['low'].values
    times = df_test.index

    # Compute ATR for SL/TP
    atr_col = df_test.get('atr')
    if atr_col is not None:
        atrs = atr_col.values
    else:
        # Fallback: compute from high-low range
        atrs = (highs - lows).astype(np.float64)
        atrs = pd.Series(atrs).rolling(14, min_periods=1).mean().values

    # ----------------------------------------------------------------
    # State tracking
    # ----------------------------------------------------------------
    position = 0.0        # +1, -1, or 0
    entry_price = 0.0
    bars_held = 0
    stop_loss = 0.0
    take_profit = 0.0
    current_lot = 0.0
    entry_bar = 0

    trades = []
    swap_costs_total = 0.0
    tp_hits = 0
    sl_hits = 0
    action_counts = {i: 0 for i in range(Settings.OUTPUT_DIM)}

    # 1-bar execution delay
    pending_action = None

    print(f"\nSimulating {len(df_test) - Settings.SEQUENCE_LENGTH - 2} bars...")

    for t in tqdm(range(Settings.SEQUENCE_LENGTH, len(df_test) - 2)):
        cp = closes[t - 1]

        # --- Execute pending action from previous bar ---
        if pending_action is not None:
            action = pending_action
            pending_action = None

            exec_price = opens[t]
            slippage = np.random.uniform(0, 0.5) * tester.spread

            # Close existing position if needed
            if action == ACT_CLOSE and position != 0:
                if position > 0:
                    exit_price = exec_price - slippage
                    raw_pnl = (exit_price - entry_price) * tester.contract_size * current_lot
                else:
                    exit_price = exec_price + slippage
                    raw_pnl = (entry_price - exit_price) * tester.contract_size * current_lot

                spread_cost = tester.spread * tester.contract_size * current_lot
                comm_cost = tester.commission * tester.contract_size * current_lot

                # Swap cost for multi-day hold
                days_held = bars_held / 96.0  # 96 M15 bars per day
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
                bars_held = 0

            # BUY: close short (if any), open long
            elif action == ACT_BUY and position <= 0:
                # Close short first
                if position < 0:
                    exit_price = exec_price + slippage
                    raw_pnl = (entry_price - exit_price) * tester.contract_size * current_lot
                    spread_cost = tester.spread * tester.contract_size * current_lot
                    comm_cost = tester.commission * tester.contract_size * current_lot
                    days_held_d = bars_held / 96.0
                    swap_cost = 0.0
                    if days_held_d >= 1.0:
                        pv = abs(entry_price * tester.contract_size * current_lot)
                        swap_cost = pv * Settings.SWAP_RATE_ANNUAL / 365.0 * days_held_d
                        swap_costs_total += swap_cost
                    net_pnl = raw_pnl - spread_cost - comm_cost - swap_cost
                    tester.balance += net_pnl
                    trades.append(net_pnl)
                    position = 0.0
                    current_lot = 0.0

                # Open long
                current_lot = Settings.MAX_POSITION_SIZE
                entry_price = exec_price + slippage
                position = 1.0
                bars_held = 0
                entry_bar = t
                atr = atrs[t] if t < len(atrs) else atrs[-1]
                stop_loss = entry_price - (atr * Settings.ATR_SL_MULTIPLIER)
                take_profit = entry_price + (atr * Settings.ATR_TP_MULTIPLIER)

            # SELL: close long (if any), open short
            elif action == ACT_SELL and position >= 0:
                # Close long first
                if position > 0:
                    exit_price = exec_price - slippage
                    raw_pnl = (exit_price - entry_price) * tester.contract_size * current_lot
                    spread_cost = tester.spread * tester.contract_size * current_lot
                    comm_cost = tester.commission * tester.contract_size * current_lot
                    days_held_d = bars_held / 96.0
                    swap_cost = 0.0
                    if days_held_d >= 1.0:
                        pv = abs(entry_price * tester.contract_size * current_lot)
                        swap_cost = pv * Settings.SWAP_RATE_ANNUAL / 365.0 * days_held_d
                        swap_costs_total += swap_cost
                    net_pnl = raw_pnl - spread_cost - comm_cost - swap_cost
                    tester.balance += net_pnl
                    trades.append(net_pnl)
                    position = 0.0
                    current_lot = 0.0

                # Open short
                current_lot = Settings.MAX_POSITION_SIZE
                entry_price = exec_price - slippage
                position = -1.0
                bars_held = 0
                entry_bar = t
                atr = atrs[t] if t < len(atrs) else atrs[-1]
                stop_loss = entry_price + (atr * Settings.ATR_SL_MULTIPLIER)
                take_profit = entry_price - (atr * Settings.ATR_TP_MULTIPLIER)

        # --- AI Decision (creates pending action with 1-bar delay) ---
        mkt_seq = feature_data[t - Settings.SEQUENCE_LENGTH:t]
        mkt_tensor = torch.FloatTensor(mkt_seq).unsqueeze(0).to(tester.device)

        # Build trade state (same as training environment)
        upnl = 0.0
        if position != 0 and entry_price > 0:
            upnl = position * (cp - entry_price) * SF
        trade_state = np.array([
            position,
            min(bars_held / 48.0, 1.0),
            np.clip(upnl / 100.0, -1, 1)
        ], dtype=np.float32)
        ts_tensor = torch.FloatTensor(trade_state).unsqueeze(0).to(tester.device)

        with torch.no_grad():
            q_values = tester.model(mkt_tensor, ts_tensor)
            action_idx = q_values.argmax(dim=1).item()

        action_counts[action_idx] += 1

        # Queue action (only if it would change something)
        if action_idx == ACT_BUY and position <= 0:
            pending_action = ACT_BUY
        elif action_idx == ACT_SELL and position >= 0:
            pending_action = ACT_SELL
        elif action_idx == ACT_CLOSE and position != 0:
            pending_action = ACT_CLOSE
        # HOLD or redundant actions: no pending

        # --- SL / TP CHECK on next bar ---
        next_high = highs[t + 1]
        next_low = lows[t + 1]

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
                bars_held = 0
                pending_action = None
            elif next_low <= stop_loss:
                raw_pnl = (stop_loss - entry_price) * tester.contract_size * current_lot
                spread_cost = tester.spread * tester.contract_size * current_lot
                net_pnl = raw_pnl - spread_cost
                tester.balance += net_pnl
                trades.append(net_pnl)
                sl_hits += 1
                position = 0.0
                current_lot = 0.0
                bars_held = 0
                pending_action = None
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
                bars_held = 0
                pending_action = None
            elif next_high >= stop_loss:
                raw_pnl = (entry_price - stop_loss) * tester.contract_size * current_lot
                spread_cost = tester.spread * tester.contract_size * current_lot
                net_pnl = raw_pnl - spread_cost
                tester.balance += net_pnl
                trades.append(net_pnl)
                sl_hits += 1
                position = 0.0
                current_lot = 0.0
                bars_held = 0
                pending_action = None

        if position != 0:
            bars_held += 1

        tester.equity_curve.append(tester.balance)

    # --- Results ---
    print("\n--- ACTION DISTRIBUTION ---")
    total_bars = sum(action_counts.values())
    for idx, label in enumerate(ACTION_NAMES):
        count = action_counts.get(idx, 0)
        pct = count / total_bars * 100 if total_bars > 0 else 0
        print(f"  {label:>6}: {count:>6} ({pct:>5.1f}%)")

    print(f"\n--- EXIT BREAKDOWN ---")
    signal_exits = len(trades) - tp_hits - sl_hits
    print(f"  TP Hits:      {tp_hits}")
    print(f"  SL Hits:      {sl_hits}")
    print(f"  Signal Exits: {signal_exits}")
    print(f"  Swap Costs:   ${swap_costs_total:.4f}")

    tester.print_stats(trades)
    tester.plot_equity()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest trading model")
    parser.add_argument("--pair", type=str, default=None, help="Symbol to backtest")
    parser.add_argument("--oos", action="store_true", help="Out-of-sample: test on data BEFORE training window")
    parser.add_argument("--train-bars", type=int, default=400000, help="Number of M5 bars used for training")
    args = parser.parse_args()

    if not mt5.initialize(
        path=Settings.MT5_PATH,
        login=Settings.MT5_LOGIN,
        password=Settings.MT5_PASSWORD,
        server=Settings.MT5_SERVER
    ):
        logger.error(f"MT5 init failed.")
        exit()

    if args.pair:
        user_symbol = args.pair.upper()
    else:
        print(f"Available pairs: {Settings.PAIRS}")
        user_symbol = input(f"Enter symbol to backtest (Default: EURUSD): ").strip().upper() or "EURUSD"

    if user_symbol not in Settings.PAIR_CONFIGS:
        print(f"Symbol {user_symbol} not configured!")
        exit()

    try:
        run_backtest(user_symbol, oos=args.oos, train_bars=args.train_bars)
    except KeyboardInterrupt:
        print("\nBacktest interrupted.")
    finally:
        mt5.shutdown()
