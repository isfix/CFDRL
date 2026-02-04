# Deep Multi-Pair Forex Trading System

## Project Overview
This project processes a Multi-Pair Forex Trading Bot using Python, MetaTrader5, and PyTorch. It utilizes an LSTM-based Reinforcement Learning (DQN) strategy for M5 Intraday trading. The system is designed to handle multiple currency pairs simultaneously, training individual models for each.

## Setup

1.  **Prerequisites:**
    *   Python 3.10+
    *   MetaTrader 5 Terminal installed and running.
    *   "Algo Trading" enabled in MT5.
    *   "Allow WebRequest" enabled in MT5 Options.

2.  **Installation:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration
The `config.py` file contains all the settings for the bot. Key configurations include:
*   **MT5 Connection:** Login credentials and path to the terminal.
*   **PAIRS:** List of symbols to trade (e.g., `['XAUUSD', 'EURUSD']`).
*   **AI Model:** Architecture details for the LSTM network (Hidden Dim, Layers, etc.).
*   **Training:** Hyperparameters like Epochs, Learning Rate, and Epsilon for RL.
*   **Risk Management:** ATR multiplier for Stop Loss and Spread filters.

## Usage

### Training (The Weekend Job)
To train the models for all configured pairs:
```bash
python src/trainer.py
```
This will download historical data, train the models, and save them to the `models/` directory.

### Live Trading (The Weekday Job)
To start the live trading bot:
```bash
python src/live_manager.py
```
The bot will load the trained models, connect to MT5, and start trading based on the strategy.

## Folder Structure
*   `config.py`: Main configuration file.
*   `requirements.txt`: Python dependencies.
*   `src/`: Source code directory.
    *   `data_factory.py`: Handles data downloading and feature engineering.
    *   `brain.py`: Defines the LSTM Q-Network architecture.
    *   `trainer.py`: Script for training the RL models.
    *   `live_manager.py`: The live trading daemon.
    *   `utils.py`: Helper functions and logging.
*   `models/`: Stores the trained PyTorch model weights (`.pth` files).
*   `data/`: Temporary storage for downloaded data (if needed).
