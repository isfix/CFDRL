import sys
import os
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime
import pytz

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Settings
from src import data_factory
from src.utils import logger

def create_tuning_dataset(symbol="EURUSD"):
    print(f"--- Creating Tuning Dataset for {symbol} ---")
    
    if not mt5.initialize(path=Settings.MT5_PATH, login=Settings.MT5_LOGIN, password=Settings.MT5_PASSWORD, server=Settings.MT5_SERVER):
        print("MT5 Init Failed")
        return

    # Define Time Zones (MT5 is usually UTC+2/3, we assume UTC here for simplicity or local)
    timezone = pytz.timezone("Etc/UTC")
    
    # ... (Date Logic) ...
    
    # Uptrend Sample: Oct 2023 - Dec 2023 (Gold Rally)
    start_up = datetime(2023, 10, 1, tzinfo=timezone)
    end_up = datetime(2023, 12, 1, tzinfo=timezone)
    
    # Downtrend Sample: May 2023 - Oct 2023 (Correction)
    start_down = datetime(2023, 5, 1, tzinfo=timezone)
    end_down = datetime(2023, 7, 1, tzinfo=timezone)
    
    # Chop/Ranging Sample: Jan 2023 - Feb 2023
    start_chop = datetime(2023, 1, 15, tzinfo=timezone)
    end_chop = datetime(2023, 2, 15, tzinfo=timezone)
    
    datasets = []
    
    for name, s, e in [("Uptrend", start_up, end_up), ("Downtrend", start_down, end_down), ("Chop", start_chop, end_chop)]:
        print(f"Fetching {name} ({s.date()} to {e.date()})...")
        rates = mt5.copy_rates_range(symbol, Settings.TIMEFRAME, s, e)
        if rates is None or len(rates) == 0:
            print(f"Warning: No data for {name}. Checking available range...")
            continue
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        print(f"  Got {len(df)} bars.")
        datasets.append(df)
        
    if not datasets:
        print("CRITICAL: No data fetched. Check MT5 connection or Symbol name.")
        return

    # Stitch
    final_df = pd.concat(datasets)
    final_df.sort_index(inplace=True)
    
    # Drop Duplicates
    final_df = final_df[~final_df.index.duplicated(keep='first')]
    
    # Save to tests/data/
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, f"{symbol}_Tuning.csv")
    final_df.to_csv(output_path)
    print(f"\nSUCCESS: Saved {len(final_df)} bars to {output_path}")
    print("Use this file for Fast Prototyping (Optuna/Sanity Checks).")

if __name__ == "__main__":
    create_tuning_dataset()
