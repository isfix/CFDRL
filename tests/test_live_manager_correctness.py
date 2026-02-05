import unittest
from unittest.mock import MagicMock, patch
import sys
import pandas as pd
import numpy as np

# Mock MetaTrader5
mock_mt5 = MagicMock()
mock_mt5.TIMEFRAME_M5 = 5
mock_mt5.ORDER_TYPE_BUY = 0
mock_mt5.ORDER_TYPE_SELL = 1
sys.modules["MetaTrader5"] = mock_mt5

# Add src to path
sys.path.append(".")

from src import live_manager
from config import Settings

class TestLiveManager(unittest.TestCase):
    def setUp(self):
        # Reset market state
        live_manager.market_state.clear()

    def test_update_market_state_correctness(self):
        # 1. Setup initial state
        symbol = Settings.PAIRS[0]

        # Create initial dataframe (timestamps 0, 5, 10, 15)
        dates_initial = pd.date_range(start='2024-01-01 10:00', periods=4, freq='5min')
        df_initial = pd.DataFrame({
            'close': [100, 101, 102, 103]
        }, index=dates_initial)

        live_manager.market_state[symbol] = df_initial

        # 2. Mock fetch_data to return new data (timestamps 15, 20)
        # Note: Timestamp 15 overlaps with initial data but has new value (e.g. updated candle)
        dates_new = pd.date_range(start='2024-01-01 10:15', periods=2, freq='5min')
        df_new = pd.DataFrame({
            'close': [103.5, 104] # 103 updated to 103.5, 104 is new
        }, index=dates_new)

        # We need to mock data_factory.fetch_data.
        # Since live_manager calls it, we should patch it where it is used or imported.
        # live_manager imports data_factory, so we patch src.live_manager.data_factory.fetch_data

        with patch('src.live_manager.data_factory.fetch_data') as mock_fetch:
            mock_fetch.return_value = df_new

            # 3. Run update
            live_manager.update_market_state()

            # 4. Verify result
            # Expected: 0, 5, 10 from initial, 15 (updated), 20 (new)
            updated_df = live_manager.market_state[symbol]

            # Check length
            self.assertEqual(len(updated_df), 5)

            # Check values
            # The logic is:
            # mask = ~current_df.index.isin(new_data.index)
            # combined = pd.concat([current_df[mask], new_data])
            # So index 15 in current_df should be replaced by index 15 in new_data

            self.assertEqual(updated_df.loc[dates_initial[3]]['close'], 103.5) # The overlap should be updated
            self.assertEqual(updated_df.loc[dates_new[1]]['close'], 104)     # The new one
            self.assertEqual(updated_df.loc[dates_initial[2]]['close'], 102) # Old one preserved

    def test_update_market_state_concurrency(self):
        # Test that multiple pairs are updated
        # Setup state for all pairs
        for symbol in Settings.PAIRS:
            live_manager.market_state[symbol] = pd.DataFrame({'close': [100]}, index=[pd.Timestamp('2024-01-01')])

        with patch('src.live_manager.data_factory.fetch_data') as mock_fetch:
            # Return distinct data for each call is hard with return_value, using side_effect
            def side_effect(symbol, bars):
                return pd.DataFrame({'close': [200]}, index=[pd.Timestamp('2024-01-02')])

            mock_fetch.side_effect = side_effect

            live_manager.update_market_state()

            for symbol in Settings.PAIRS:
                self.assertTrue(pd.Timestamp('2024-01-02') in live_manager.market_state[symbol].index)
                self.assertEqual(len(live_manager.market_state[symbol]), 2)

if __name__ == '__main__':
    unittest.main()
