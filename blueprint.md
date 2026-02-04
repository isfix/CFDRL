# Master Blueprint: Deep Multi-Pair System

## Phase 1: Project Skeleton & The Master Control File

**Objective:** To create a clean, organized project structure and a central configuration file that allows you to control the entire system from one place. This is the key to "easier usage."

### 1.1 Project Directory Structure
**Prompt for your AI Coder:**
> I am building a Multi-Pair Forex Trading Bot using Python, MetaTrader5, and PyTorch. The strategy is M5 Intraday using LSTM and Reinforcement Learning.
>
> Please create the following directory structure:
>
> ```text
> /
> │
> ├── config.py              # The Master Control File
> ├── requirements.txt       # Python dependencies
> │
> ├── models/                # Storage for trained AI brains (.pth files)
> │
> ├── data/                  # Temporary storage for training data CSVs
> │
> └── src/                   # All source code
>     ├── __init__.py
>     ├── data_factory.py    # Handles data download & feature engineering
>     ├── brain.py           # The LSTM Neural Network class architecture
>     ├── trainer.py         # The Reinforcement Learning training engine
>     ├── live_manager.py    # The live trading bot daemon
>     └── utils.py           # Helper functions (e.g., logging)
> ```

### 1.2 The Master Configuration (`config.py`)
This file is the "cockpit" of your entire operation. Changing a value here should change the behavior of both the trainer and the live bot without touching any other code.

**Prompt for your AI Coder:**
> Create the `config.py` file. This will be the central configuration for the entire project. It must contain a single class `Settings`.
>
> ```python
> # config.py
> import MetaTrader5 as mt5
>
> class Settings:
>     # --- MT5 Connection ---
>     # (Fill these with your actual credentials)
>     MT5_LOGIN = 12345678
>     MT5_PASSWORD = "your_password"
>     MT5_SERVER = "Your_Broker_Server"
>     MT5_PATH = "C:\\Program Files\\MetaTrader 5\\terminal64.exe" # Adjust path if needed
>
>     # --- TRADING PAIRS ---
>     # The list of symbols the bot will train and trade.
>     # To add a pair, just add its name here.
>     PAIRS = ['XAUUSD', 'EURUSD', 'GBPUSD']
>
>     # --- CORE TRADING PARAMETERS ---
>     TIMEFRAME = mt5.TIMEFRAME_M5
>     MAGIC_NUMBER_BASE = 202400  # Base magic number. Bot will add index (e.g., XAUUSD=202400, EURUSD=202401)
>
>     # --- AI MODEL ARCHITECTURE ---
>     # These define the structure of the LSTM brain.
>     SEQUENCE_LENGTH = 60  # Lookback window: 60 candles = 5 hours on M5
>     FEATURES = ['log_ret', 'dist_ema', 'rsi', 'volatility', 'hour']
>     INPUT_DIM = len(FEATURES)
>     HIDDEN_DIM = 128
>     NUM_LAYERS = 2
>     DROPOUT = 0.2
>     OUTPUT_DIM = 3  # 3 Actions: 0=Hold, 1=Buy, 2=Sell
>
>     # --- TRAINING HYPERPARAMETERS ---
>     # Controls how the AI learns.
>     EPOCHS = 20
>     BATCH_SIZE = 64
>     LEARNING_RATE = 0.001
>     GAMMA = 0.99  # Discount factor for future rewards in RL
>     EPSILON_START = 1.0  # Exploration rate (starts at 100%)
>     EPSILON_DECAY = 0.995 # Decay rate per epoch
>     EPSILON_MIN = 0.01   # Minimum exploration rate
>     REPLAY_MEMORY_SIZE = 10000 # How many experiences to store for training
>
>     # --- DATA PARAMETERS ---
>     TRAIN_DATA_BARS = 50000 # Number of bars to download for training
>     INIT_DATA_BARS = 200    # Number of bars to fetch on bot startup
>
>     # --- RISK MANAGEMENT ---
>     # Controls how trades are managed.
>     ATR_PERIOD = 14
>     ATR_SL_MULTIPLIER = 2.5 # Stop Loss = 2.5 * ATR
>     SPREAD_FILTER_POINTS = 30 # For XAUUSD, 30 points = $0.30. Don't trade if spread is wider.
> ```

---

## Phase 2: The Data Factory (`src/data_factory.py`)

**Objective:** To create a robust, reusable module that can fetch and process data for any financial instrument, turning raw price data into the specific features our AI needs to see.

**Prompt for your AI Coder:**
> Create the `src/data_factory.py` file. This script will handle all data operations.
>
> **Requirements:**
> 1.  Import `MetaTrader5`, `pandas`, `pandas_ta`, `numpy`, and the `Settings` class from `config`.
> 2.  Create a function `fetch_data(symbol: str, num_bars: int) -> pd.DataFrame`:
>     - It must connect to MT5 using credentials from the `Settings` class.
>     - It should download `num_bars` of M5 data for the specified `symbol`.
>     - It must handle connection errors gracefully (e.g., print an error and return an empty DataFrame).
>     - Convert the result to a pandas DataFrame with a datetime index.
> 3.  Create a function `prepare_features(df: pd.DataFrame) -> pd.DataFrame`:
>     - This function takes a raw DataFrame and adds the feature columns defined in `Settings.FEATURES`.
>     - **Log Returns:** `np.log(df['close'] / df['close'].shift(1))`
>     - **EMA Distance:** `(df['close'] - ta.ema(df['close'], length=50)) / df['close']`
>     - **RSI:** `ta.rsi(df['close'], length=14) / 100.0` (to scale it between 0 and 1).
>     - **Volatility:** `ta.atr(df['high'], df['low'], df['close'], length=Settings.ATR_PERIOD) / df['close']`
>     - **Time Context:** `df.index.hour / 23.0`
>     - After adding features, it must **drop all rows with NaN values**.
>     - It should return the processed DataFrame.

---

## Phase 3: The LSTM Brain (`src/brain.py`)

**Objective:** To define the core AI architecture using PyTorch. This is a generic blueprint for the Q-Network that will be instantiated separately for each trading pair.

**Prompt for your AI Coder:**
> Create the `src/brain.py` file. This defines the PyTorch Neural Network.
>
> **Requirements:**
> 1.  Import `torch` and `torch.nn`.
> 2.  Import the `Settings` class from `config`.
> 3.  Create a class `QNetwork(nn.Module)`.
> 4.  The `__init__` method should use parameters directly from the `Settings` class:
>     - `input_dim=Settings.INPUT_DIM`
>     - `hidden_dim=Settings.HIDDEN_DIM`
>     - `num_layers=Settings.NUM_LAYERS`
>     - `dropout=Settings.DROPOUT`
>     - `output_dim=Settings.OUTPUT_DIM`
> 5.  Define the layers within `__init__`:
>     - `self.lstm`: An `nn.LSTM` layer using the parameters above. Ensure `batch_first=True`.
>     - `self.fc`: A `nn.Sequential` block containing:
>       - `nn.Linear(hidden_dim, hidden_dim // 2)`
>       - `nn.ReLU()`
>       - `nn.Linear(hidden_dim // 2, output_dim)`
> 6.  Define the `forward(self, x)` method:
>     - It takes an input tensor `x` with shape `(batch_size, sequence_length, input_dim)`.
>     - Pass `x` through the LSTM layer.
>     - Extract **only the output of the final time step** from the LSTM's output tensor.
>     - Pass this final time step's output through the fully connected `self.fc` block.
>     - Return the final Q-values tensor.

Here is **Part 2 of the Master Blueprint**.

This section covers the "Brain Training" engine and the "Live Trading" daemon. These are the most complex parts of the system, so the prompts are designed to ensure your AI coder handles the logic correctly.

---

## Phase 4: The Training Engine (`src/trainer.py`)

**Objective:** To create a script that iterates through every pair in your config, downloads historical data, and trains a specific "Brain" for that pair using Reinforcement Learning (DQN).

**Prompt for your AI Coder:**
> Create the `src/trainer.py` file. This script is responsible for training the AI models.
>
> **Requirements:**
> 1.  **Imports:** `torch`, `torch.optim`, `torch.nn`, `numpy`, `random`, `tqdm` (for progress bars), `src.data_factory`, `src.brain`, and `config.Settings`.
>
> 2.  **Class `Trainer`:** Create a class to handle the training logic.
>     - **`__init__`**: Initialize the `QNetwork`, the `optimizer` (Adam), and the `Loss Function` (MSELoss). Set the device to GPU if available.
>     - **`train_model(symbol)`**: The main function to train a specific pair.
>
> 3.  **Training Logic (The DQN Loop):**
>     - **Step A:** Use `data_factory.fetch_data` and `prepare_features` to get 50,000 bars of training data.
>     - **Step B:** Convert the DataFrame into a "Sliding Window" dataset. We need sequences of length `Settings.SEQUENCE_LENGTH`.
>     - **Step C:** Iterate through `Settings.EPOCHS`. Inside each epoch, iterate through the historical data.
>     - **Step D (The Action):** Implement Epsilon-Greedy strategy.
>         - Generate a random number. If < Epsilon, choose random action (0, 1, 2).
>         - Else, pass the current sequence to the `QNetwork` and choose the action with the highest Q-value.
>     - **Step E (The Reward):**
>         - Calculate the price difference between the *next* candle and the *current* candle.
>         - **Buy Reward:** `(Next_Close - Current_Close) - Spread_Cost`.
>         - **Sell Reward:** `(Current_Close - Next_Close) - Spread_Cost`.
>         - **Hold Reward:** 0 (or a tiny penalty like -0.0001 to discourage laziness).
>     - **Step F (The Learning):**
>         - Calculate Target Q: `Reward + (Gamma * Max_Q_Next_State)`.
>         - Calculate Loss between `Predicted_Q` and `Target_Q`.
>         - Perform Backpropagation: `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`.
>
> 4.  **Saving:**
>     - After the epochs are finished, save the model weights to `models/{symbol}_brain.pth`.
>
> 5.  **Main Execution Block:**
>     - In `if __name__ == "__main__":`, loop through `Settings.PAIRS`.
>     - For each pair, instantiate the `Trainer` and run `train_model(pair)`.
>     - Ensure MT5 is initialized before starting and shutdown after finishing.

---

## Phase 5: The Live Multi-Pair Bot (`src/live_manager.py`)

**Objective:** To create the daemon that runs 24/5. It must efficiently manage data for multiple pairs, load the correct "Brain" for each, and execute trades via MT5.

**Prompt for your AI Coder:**
> Create the `src/live_manager.py` file. This is the live trading bot.
>
> **Requirements:**
> 1.  **Imports:** `MetaTrader5`, `torch`, `pandas`, `time`, `datetime`, `src.data_factory`, `src.brain`, and `config.Settings`.
>
> 2.  **Global State Dictionaries:**
>     - `active_models = {}`: To store the loaded PyTorch models (Key=Symbol, Value=Model).
>     - `market_state = {}`: To store the DataFrame window for each symbol (Key=Symbol, Value=DataFrame).
>
> 3.  **Helper Functions:**
>     - **`load_models()`**: Loop through `Settings.PAIRS`. Initialize a `QNetwork`, load the weights from `models/{symbol}_brain.pth`, set to `.eval()` mode, and store in `active_models`.
>     - **`init_market_state()`**: Loop through `Settings.PAIRS`. Fetch the last 500 bars using `data_factory` to initialize the `market_state` dictionary.
>     - **`update_market_state()`**: The efficiency hack.
>       - Loop through pairs. Fetch ONLY the last 2 bars from MT5.
>       - Append the new closed candle to the dataframe in `market_state`.
>       - Drop the oldest row to keep the window size constant.
>       - Re-calculate features (RSI, ATR) on the updated dataframe.
>
> 4.  **Trading Logic Functions:**
>     - **`get_signal(symbol)`**: Extract the last `SEQUENCE_LENGTH` rows from `market_state[symbol]`, convert to Tensor, pass to `active_models[symbol]`, and return the Action (0, 1, or 2).
>     - **`execute_trade(symbol, signal)`**:
>       - Check `mt5.positions_get(symbol=symbol)`.
>       - **Logic:**
>         - If Signal is BUY and no Long position: Open Buy. Calculate SL using `ATR * Settings.ATR_SL_MULTIPLIER`.
>         - If Signal is SELL and no Short position: Open Sell. Calculate SL using `ATR`.
>         - If Signal opposes current position: Close the position.
>       - **Hard Rule:** If `datetime.now().hour >= 20`, close all positions and return.
>
> 5.  **The Main Loop:**
>     - Initialize MT5.
>     - Call `load_models()` and `init_market_state()`.
>     - Start a `while True` loop:
>       - Sleep for 1 second.
>       - Check if a new M5 candle has just closed (compare current time minutes % 5 == 0).
>       - If New Candle:
>         - Call `update_market_state()`.
>         - Loop through `Settings.PAIRS`:
>           - Get Signal.
>           - Execute Trade.
>           - Print logs (Time, Symbol, Action, Price).

---

## Phase 6: How to Run This System

Once your AI Coder has generated the files, follow these steps to launch your hedge fund.

**Step 1: Preparation**
1.  Open your **MetaTrader 5 Terminal**.
2.  Enable "Algo Trading" in the toolbar.
3.  Go to **Tools > Options > Expert Advisors** and check "Allow WebRequest" (just in case, though the Python API uses IPC).
4.  Ensure the symbols in your `config.py` (e.g., XAUUSD, EURUSD) are visible in the MT5 Market Watch.

**Step 2: Training (The Weekend Job)**
Run this command in your terminal. It will take time (10-30 minutes depending on your GPU/CPU) to download data and train the brains.
```bash
python src/trainer.py
```
*Check the `/models` folder. You should see `XAUUSD_brain.pth`, `EURUSD_brain.pth`, etc.*

**Step 3: Trading (The Weekday Job)**
Run this command to start the bot.
```bash
python src/live_manager.py
```
*The console should print "Models Loaded" and then start printing "Heartbeat" messages every 5 minutes.*

---

