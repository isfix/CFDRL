import MetaTrader5 as mt5
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path so we can import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Settings

def test_fetch():
    print("Initializing MT5...")
    if not mt5.initialize(
        path=Settings.MT5_PATH,
        login=Settings.MT5_LOGIN,
        password=Settings.MT5_PASSWORD,
        server=Settings.MT5_SERVER
    ):
        print(f"MT5 init failed: {mt5.last_error()}")
        return

    print("MT5 Initialized Successfully.")
    
    symbol = "XAUUSD"
    count = 100
    
    print(f"\n--- Testing fetch for {symbol} with {count} bars ---")
    rates = mt5.copy_rates_from_pos(symbol, Settings.TIMEFRAME, 0, count)
    
    if rates is None:
        print(f"Fetch failed: {mt5.last_error()}")
    else:
        print(f"Success! Fetched {len(rates)} bars.")
        print(rates[:2])
        
    print(f"\nMax bars in chart: {mt5.terminal_info().maxbars}")
    
    # Try 100,000
    print(f"\n--- Testing fetch for {symbol} with 100,000 bars ---")
    rates_100k = mt5.copy_rates_from_pos(symbol, Settings.TIMEFRAME, 0, 100000)
    if rates_100k is None:
        print(f"100k fetch failed: {mt5.last_error()}")
    else:
        print(f"Success! Fetched {len(rates_100k)} bars.")

    print(f"\n--- Testing Chunked Fetch (Target: 150,000 bars) ---")
    
    all_rates = []
    current_date = datetime.now()
    total_fetched = 0
    target = 150000
    
    from datetime import timedelta
    
    # Loop back in 30 day chunks
    while total_fetched < target:
        start_date = current_date - timedelta(days=30)
        print(f"Fetching {start_date} to {current_date}...")
        
        rates = mt5.copy_rates_range(symbol, Settings.TIMEFRAME, start_date, current_date)
        
        if rates is None or len(rates) == 0:
            print("No data received for this chunk.")
            # If we fail to get data, maybe we reached the end of available history?
            # Or assume we need to go further back?
            # If 0 bars, break to avoid infinite loop
            break
            
        print(f"  Got {len(rates)} bars.")
        
        # Prepend to list (since we go backwards)
        # Note: copy_rates_range returns oldest first. 
        # So 'rates' is [Jan 1 ... Jan 30].
        # We want [Dec 1... Dec 30] + [Jan 1... Jan 30].
        all_rates.insert(0, rates)
        
        total_fetched += len(rates)
        current_date = start_date # Move window back
        
        if len(rates) < 100: # Heuristic for "No more data"
             break
             
    if all_rates:
        import numpy as np
        full_data = np.concatenate(all_rates)
        # Unique check? mt5 range is inclusive/exclusive? 
        # range is [date_from, date_to). 
        # So we should be safe mostly, but overlap might occur on the exact second.
        print(f"Total accumulated bars: {len(full_data)}")
    else:
        print("Failed to accumulate any data.")

    mt5.shutdown()

if __name__ == "__main__":
    test_fetch()
