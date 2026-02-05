import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Settings
from datetime import datetime, timedelta

def fetch_data_chunked(symbol: str, num_bars: int) -> pd.DataFrame:
    """
    Fetches historical M5 data using chunked backward fetching.
    Identical logic to src/data_factory.py but standalone for extraction.
    """
    if not mt5.initialize(path=Settings.MT5_PATH, login=Settings.MT5_LOGIN, password=Settings.MT5_PASSWORD, server=Settings.MT5_SERVER):
        print(f"MT5 Init Failed: {mt5.last_error()}")
        return pd.DataFrame()

    all_rates = []
    total_fetched = 0
    
    current_date = datetime.now() + timedelta(minutes=Settings.TIMEFRAME * 2) 
    chunk_size_days = 30
    
    print(f"Fetching ~{num_bars} bars for {symbol}...")
    
    while total_fetched < num_bars:
        start_date = current_date - timedelta(days=chunk_size_days)
        rates = mt5.copy_rates_range(symbol, Settings.TIMEFRAME, start_date, current_date)
        
        if rates is None or len(rates) == 0:
            print(f"  No data between {start_date} and {current_date}")
            break
            
        all_rates.insert(0, rates)
        total_fetched += len(rates)
        current_date = start_date
        
        if len(rates) < 10: break
            
    if not all_rates:
        return pd.DataFrame()
        
    rates = np.concatenate(all_rates)
    if len(rates) > num_bars:
         rates = rates[-num_bars:]
         
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[~df.index.duplicated(keep='last')]
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract MT5 data to CSV.')
    parser.add_argument('--bars', type=int, help='Number of bars to fetch')
    parser.add_argument('--symbol', type=str, help='Specific symbol to fetch (optional)')
    
    args = parser.parse_args()
    
    # Determine number of bars
    if args.bars:
        num_bars = args.bars
    else:
        # Interactive Prompt
        print(f"Default bars from config: {Settings.TRAIN_DATA_BARS}")
        user_input = input("Enter number of bars to fetch (Press Enter for default): ")
        if user_input.strip():
            try:
                num_bars = int(user_input)
            except ValueError:
                print("Invalid number. Using default.")
                num_bars = Settings.TRAIN_DATA_BARS
        else:
            num_bars = Settings.TRAIN_DATA_BARS

    target_pairs = [args.symbol] if args.symbol else Settings.PAIRS

    if not os.path.exists("notebook/data"):
        os.makedirs("notebook/data", exist_ok=True)
        
    for symbol in target_pairs:
        print(f"\nProcessing {symbol}...")
        df = fetch_data_chunked(symbol, num_bars)
        
        if not df.empty:
            path = f"notebook/data/{symbol}.csv"
            df.to_csv(path)
            print(f"Saved {len(df)} rows to {path}")
        else:
            print(f"Failed to fetch {symbol}")

    mt5.shutdown()
