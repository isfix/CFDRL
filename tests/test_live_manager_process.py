import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock MetaTrader5
mock_mt5 = MagicMock()
mock_mt5.TIMEFRAME_M5 = 5
mock_mt5.ORDER_TYPE_BUY = 0
mock_mt5.ORDER_TYPE_SELL = 1
sys.modules["MetaTrader5"] = mock_mt5

# Mock Torch
mock_torch = MagicMock()
sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = MagicMock()

# Add src to path
sys.path.append(".")

# Now import live_manager
# It will import torch, which is now mocked
from src import live_manager

class TestLiveManagerProcess(unittest.TestCase):

    @patch("src.live_manager.get_signal")
    @patch("src.live_manager.execute_trade")
    @patch("src.live_manager.logger")
    def test_process_pair_buy(self, mock_logger, mock_execute, mock_get_signal):
        # Setup
        symbol = "TESTUSD"
        mock_get_signal.return_value = (1, "features") # Buy signal

        # Action
        live_manager.process_pair(symbol)

        # Verify
        mock_get_signal.assert_called_once_with(symbol)
        mock_execute.assert_called_once_with(symbol, 1, df_features="features")

        # Verify logging
        # We expect one call for the signal
        found_log = False
        for call in mock_logger.info.call_args_list:
            if "TESTUSD Signal: BUY" in call[0][0]:
                found_log = True
                break
        self.assertTrue(found_log, "Should log BUY signal")

    @patch("src.live_manager.get_signal")
    @patch("src.live_manager.execute_trade")
    @patch("src.live_manager.logger")
    def test_process_pair_exception(self, mock_logger, mock_execute, mock_get_signal):
        # Setup
        symbol = "TESTUSD"
        mock_get_signal.side_effect = Exception("Inference Failed")

        # Action
        live_manager.process_pair(symbol)

        # Verify
        mock_logger.error.assert_called()
        args, _ = mock_logger.error.call_args
        self.assertIn("Error processing TESTUSD", args[0])
        self.assertIn("Inference Failed", args[0])

if __name__ == "__main__":
    unittest.main()
