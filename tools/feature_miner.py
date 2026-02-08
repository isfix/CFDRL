import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm

# Mock Settings just for data loading if needed, or import from config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Settings
from src import data_factory

def generate_rich_features(df):
    """
    Generates a 'Universe' of features based on Diffs and Ratios of OHLCV
    Expanded for Phase 17 (Plan 2)
    """
    df = df.copy()
    
    # 1. Standard Mid-Price
    df['mid_price'] = (df['high'] + df['low']) / 2.0
    
    # Base columns to permute
    base_cols = ['open', 'high', 'low', 'close', 'mid_price', 'tick_volume']
    
    # Extended Lags (Fibonacci-like)
    lags = [1, 2, 3, 5, 8, 13, 21, 34]
    
    print("Generating Extended Rich Features (~2000+)...")
    
    # 1. Inter-Column Relations at Lag 0 (e.g. High/Low, Close-Open)
    # Already doing this for all pairs
    for i, col1 in enumerate(base_cols):
        for col2 in base_cols[i+1:]:
            # Spread: A - B
            df[f'spread_{col1}_{col2}'] = df[col1] - df[col2]
            # Ratio: A / B
            # Avoid div by zero
            df[f'rel_{col1}_{col2}'] = df[col1] / (df[col2] + 1e-9)

    for lag in tqdm(lags):
        for col1 in base_cols:
            # Shifted value
            shifted = df[col1].shift(lag)
            
            # temporal_diff_A_lagN -> A(t) - A(t-N)
            df[f'diff_{col1}_lag{lag}'] = df[col1] - shifted
            
            # temporal_ratio_A_lagN -> A(t) / A(t-N)
            df[f'ratio_{col1}_lag{lag}'] = df[col1] / (shifted + 1e-9)
            
            # Inter-column Temporal (Complex)
            # diff_A_B_lagN -> A(t) - B(t-N)
            # This captures: "Close today vs High 3 days ago"
            for col2 in base_cols:
                shifted2 = df[col2].shift(lag)
                df[f'diff_{col1}_{col2}_lag{lag}'] = df[col1] - shifted2
                # Ratio version? Maybe too many features. Let's stick to Diffs for cross-temporal.

    # Clean NaNs
    df.dropna(inplace=True)
    
    # Create Target (Mid-Price Direction)
    df['target'] = np.where(df['mid_price'].shift(-1) > df['mid_price'], 1, 0)
    
    return df

def select_top_features(df, top_n=50):
    """
    Uses ExtraTreesClassifier to find the most predictive features.
    """
    print(f"Training Tree Classifier to select Top {top_n} features...")
    
    exclude_cols = ['time', 'target', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"Input Feature Count: {len(feature_cols)}")
    
    X = df[feature_cols]
    y = df['target']
    
    # Train Forest
    forest = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    forest.fit(X, y)
    
    # Get Importances
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print Top Features
    print("\n--- TOP FEATURES ---")
    selected_features = []
    for i in range(top_n):
        idx = indices[i]
        feat_name = feature_cols[idx]
        score = importances[idx]
        print(f"{i+1}. {feat_name} ({score:.6f})")
        selected_features.append(feat_name)
        
    # Plot
    plt.figure(figsize=(12, 8))
    plt.title(f"Top {top_n} Feature Importances")
    plt.bar(range(top_n), importances[indices[:top_n]], align="center")
    plt.xticks(range(top_n), [feature_cols[i] for i in indices[:top_n]], rotation=90)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("\nSaved plot to feature_importance.png")
    
    return selected_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="EURUSD")
    args = parser.parse_args()
    
    # Fetch Data
    # Use existing data factory but we might need more bars for mining
    print(f"Fetching data for {args.symbol}...")
    df = data_factory.fetch_data(args.symbol, 50000)
    
    if df.empty:
        print("No data found.")
        exit()
        
    # Generate Universe
    df_rich = generate_rich_features(df)
    
    # Select Best
    top_features = select_top_features(df_rich)
    
    print("\n--- SUGGESTED CONFIG UPDATE ---")
    print(f"FEATURES = {top_features}")
    
    with open("final_features.txt", "w", encoding="utf-8") as f:
        f.write(str(top_features))
    print("Saved features to final_features.txt")
