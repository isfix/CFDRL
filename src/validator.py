# src/validator.py
"""
Phase 26: Walk-Forward Validator with Probability of Backtest Overfitting (PBO).

Based on Da Costa & Gebbie (2020) and Bailey et al. CSCV methodology.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from itertools import combinations
from config import Settings
from src.brain import QNetwork
from src import data_factory
from src.utils import logger


def compute_sharpe(returns, annualize_factor=1.0):
    """Compute Sharpe ratio from array of returns."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return (np.mean(returns) / np.std(returns)) * np.sqrt(annualize_factor)


def simulate_returns(model, feature_data, close_prices, device):
    """
    Simulate per-bar returns using model actions on given data segment.
    Returns array of per-bar PnL values.
    """
    model.eval()
    returns = []
    seq_len = Settings.SEQUENCE_LENGTH
    
    profile_scale = 1.0  # Simplified — can be enhanced per pair
    
    with torch.no_grad():
        for t in range(seq_len, len(feature_data) - 1):
            state = torch.FloatTensor(
                feature_data[t - seq_len : t]
            ).unsqueeze(0).to(device)
            
            q = model(state)
            action_idx = torch.argmax(q).item()
            position = Settings.ACTION_MAP[action_idx]
            
            price_change = close_prices[t] - close_prices[t-1]
            bar_return = position * price_change * profile_scale
            returns.append(bar_return)
    
    return np.array(returns)


def compute_pbo(model_path, df, n_splits=8, symbol='EURUSD'):
    """
    Compute Probability of Backtest Overfitting (PBO) using CSCV.
    
    Method:
    1. Split data into n_splits equal sub-samples (S1, S2, ..., Sn)
    2. For each combination of n_splits/2 sub-samples:
       a. Use combination as "in-sample" (IS)
       b. Use remaining as "out-of-sample" (OOS)
       c. Compute IS and OOS Sharpe ratios
    3. PBO = fraction of combinations where OOS Sharpe <= 0
    
    Args:
        model_path: Path to the .pth model file
        df: DataFrame with features already prepared
        n_splits: Number of sub-samples (must be even)
        symbol: Trading pair
    
    Returns:
        pbo: float [0, 1] — probability of overfitting
        results: dict with detailed statistics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = QNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Prepare data
    feature_data = df[Settings.FEATURES].values
    close_prices = df['close'].values
    
    # Split into n_splits sub-samples
    total_len = len(feature_data)
    split_size = total_len // n_splits
    
    # Compute returns for each sub-sample
    subsample_returns = []
    for i in range(n_splits):
        start = i * split_size
        end = start + split_size
        
        if end + Settings.SEQUENCE_LENGTH > total_len:
            end = total_len
        
        seg_features = feature_data[start:end]
        seg_prices = close_prices[start:end]
        
        if len(seg_features) < Settings.SEQUENCE_LENGTH + 2:
            subsample_returns.append(np.array([0.0]))
            continue
        
        returns = simulate_returns(model, seg_features, seg_prices, device)
        subsample_returns.append(returns)
    
    # CSCV: Enumerate all combinations of n_splits/2
    half = n_splits // 2
    all_indices = list(range(n_splits))
    combos = list(combinations(all_indices, half))
    
    # Limit combinations for very large n_splits
    if len(combos) > 100:
        rng = np.random.default_rng(42)
        combo_indices = rng.choice(len(combos), size=100, replace=False)
        combos = [combos[i] for i in combo_indices]
    
    oos_negative_count = 0
    total_combos = len(combos)
    is_sharpes = []
    oos_sharpes = []
    
    for is_indices in combos:
        oos_indices = tuple(i for i in all_indices if i not in is_indices)
        
        # Aggregate returns for IS and OOS
        is_returns = np.concatenate([subsample_returns[i] for i in is_indices])
        oos_returns = np.concatenate([subsample_returns[i] for i in oos_indices])
        
        # P28 FIX: Trade-only returns for Sharpe (exclude zero-position bars)
        is_trade_returns = is_returns[is_returns != 0]
        oos_trade_returns = oos_returns[oos_returns != 0]
        is_sharpe = compute_sharpe(is_trade_returns)
        oos_sharpe = compute_sharpe(oos_trade_returns)
        
        is_sharpes.append(is_sharpe)
        oos_sharpes.append(oos_sharpe)
        
        if oos_sharpe <= 0:
            oos_negative_count += 1
    
    pbo = oos_negative_count / total_combos if total_combos > 0 else 1.0
    
    results = {
        'pbo': pbo,
        'n_combinations': total_combos,
        'oos_negative_count': oos_negative_count,
        'avg_is_sharpe': np.mean(is_sharpes),
        'avg_oos_sharpe': np.mean(oos_sharpes),
        'min_oos_sharpe': np.min(oos_sharpes) if oos_sharpes else 0,
        'max_oos_sharpe': np.max(oos_sharpes) if oos_sharpes else 0,
    }
    
    return pbo, results


if __name__ == "__main__":
    """
    Standalone PBO assessment.
    Usage: python src/validator.py --pair EURUSD
    """
    import MetaTrader5 as mt5
    import argparse
    
    parser = argparse.ArgumentParser(description="PBO Validator")
    parser.add_argument("--pair", type=str, required=True, help="Symbol (e.g., EURUSD)")
    parser.add_argument("--splits", type=int, default=8, help="Number of sub-samples (default: 8)")
    args = parser.parse_args()
    
    symbol = args.pair.upper()
    model_path = f"models/{symbol}_brain.pth"
    
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found!")
        exit()
    
    if not mt5.initialize(
        path=Settings.MT5_PATH,
        login=Settings.MT5_LOGIN,
        password=Settings.MT5_PASSWORD,
        server=Settings.MT5_SERVER
    ):
        print(f"MT5 init failed.")
        exit()
    
    print(f"Fetching data for {symbol}...")
    df = data_factory.fetch_data(symbol, Settings.TRAIN_DATA_BARS)
    df = data_factory.prepare_features(df)
    
    print(f"Running PBO assessment with {args.splits} splits on {len(df)} bars...")
    pbo, results = compute_pbo(model_path, df, n_splits=args.splits, symbol=symbol)
    
    print("\n--- PBO ASSESSMENT ---")
    print(f"Probability of Backtest Overfitting: {pbo:.2%}")
    print(f"Combinations tested: {results['n_combinations']}")
    print(f"OOS Negative Sharpe: {results['oos_negative_count']}")
    print(f"Avg IS Sharpe:  {results['avg_is_sharpe']:.4f}")
    print(f"Avg OOS Sharpe: {results['avg_oos_sharpe']:.4f}")
    print(f"OOS Sharpe Range: [{results['min_oos_sharpe']:.4f}, {results['max_oos_sharpe']:.4f}]")
    
    if pbo < 0.30:
        print("\n✅ LOW overfitting risk. Model appears robust.")
    elif pbo < 0.50:
        print("\n⚠️ MODERATE overfitting risk. Consider more data or simpler model.")
    else:
        print("\n❌ HIGH overfitting risk. Model is likely overfit to training data.")
    
    mt5.shutdown()
