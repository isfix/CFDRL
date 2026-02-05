import sys
import time
from unittest.mock import MagicMock
import pandas as pd
import numpy as np

# Mock MetaTrader5 before importing other modules
mock_mt5 = MagicMock()
mock_mt5.TIMEFRAME_M5 = 5
mock_mt5.ORDER_TYPE_BUY = 0
mock_mt5.ORDER_TYPE_SELL = 1
sys.modules["MetaTrader5"] = mock_mt5

# Add src to path if needed (it should be fine if running from root)
sys.path.append(".")

from src import live_manager, data_factory
from config import Settings

# Mock data_factory.fetch_data
def mocked_fetch_data(symbol, num_bars):
    time.sleep(0.1) # Simulate network delay
    # Return a simple DataFrame
    dates = pd.date_range(end=pd.Timestamp.now(), periods=num_bars, freq='5min')
    df = pd.DataFrame({
        'open': np.random.rand(num_bars),
        'high': np.random.rand(num_bars),
        'low': np.random.rand(num_bars),
        'close': np.random.rand(num_bars),
        'tick_volume': np.random.randint(1, 100, num_bars),
        'spread': np.random.randint(1, 10, num_bars),
        'real_volume': np.random.randint(1, 100, num_bars)
    }, index=dates)
    return df

data_factory.fetch_data = mocked_fetch_data

# Initialize market state
def setup_market_state():
    live_manager.market_state.clear()
    for symbol in Settings.PAIRS:
        # Initial population - using the mocked fetch_data (which sleeps) so this will be slow too
        # but we don't measure setup time
        live_manager.market_state[symbol] = mocked_fetch_data(symbol, Settings.INIT_DATA_BARS)

def run_benchmark():
    print("Setting up market state...")
    setup_market_state()

    print(f"Running benchmark on {len(Settings.PAIRS)} pairs: {Settings.PAIRS}")
    start_time = time.time()
    live_manager.update_market_state()
    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    run_benchmark()
