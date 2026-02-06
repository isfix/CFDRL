# Deep Multi-Pair Forex Trading Bot (RL-DQN)

A production-grade Reinforcement Learning trading system for MetaTrader 5, designed to train and trade multiple assets (Forex & Commodities) simultaneously using a configuration-driven architecture.

## 🚀 Key Features
- **Deep Q-Network (DQN)**: LSTM-based neural network for time-series decision making.
- **Independent Training Profiles**: 
  - **EURUSD**: Optimized for Pips (1 Pip = $10 Reward). Includes "London/NY Session" Time Filters.
  - **XAUUSD**: Optimized for Dollars ($1 Move = $1 Reward).
- **Asset Personality Engine**:
  - **Pip Normalization**: Solves the "Decimal Dust" problem by standardizing rewards across different asset classes.
  - **Bollinger Squeeze Filter**: Automatically avoids "Dead Markets" (Low Volatility).
- **Robust Live Execution**:
  - **Auto-Flip Logic**: Instant reversal on signal change (No 5-minute delay).
  - **Broker Safety**: Uses `ORDER_FILLING_FOK` to ensure trade acceptance.

---

## 📂 Project Structure
```
.
├── config.py             # CONTROL CENTER: Manage Pairs, Risk, & Profiles
├── requirements.txt      # Python Dependencies
├── src/
│   ├── trainer.py        # AI Training Engine (CLI-driven)
│   ├── backtester.py     # Local Backtesting Engine (Strict Logic)
│   ├── live_manager.py   # Live Trading Bot (MT5 Connector)
│   ├── data_factory.py   # Feature Engineering (RSI, EMA, Bollinger)
│   └── brain.py          # PyTorch DQN Model (LSTM)
└── notebook/
    └── colab_training.ipynb # Cloud-based Training (Google Colab)
```

---

## 🛠️ Setup & Installation

1. **Prerequisites**:
   - Python 3.8+
   - MetaTrader 5 Terminal (Logged in, Algo Trading Enabled).
   - **Important**: In MT5, go to `Tools > Options > Expert Advisors` and allow **"WebRequest"** (if needed) and **"Allow Automated Trading"**.

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure**:
   - Open `config.py` and set your `MT5_LOGIN`, `MT5_PASSWORD`, and `MT5_SERVER`.
   - Adjust `PAIR_CONFIGS` if you need to change spread/commission simulation.

---

## 🖥️ Usage

### 1. Training (The Brain)
You must train the model for each pair independently. The reward function adapts automatically based on the pair.

```bash
# Train Gold
python src/trainer.py --pair XAUUSD

# Train Euro
python src/trainer.py --pair EURUSD
```

*Models are saved to `models/{SYMBOL}_brain.pth`.*

### 2. Backtesting (Verification)
Run the strictly validated backtester to see how the model performs on unseen data.

```bash
python src/backtester.py
```
*Follow the interactive prompts to select the pair.*

### 3. Live Trading (Production)
Run the live manager. It connects to MT5, monitors all configured pairs in parallel, and executes trades.

```bash
python src/live_manager.py
```

---

## ☁️ Google Colab Training
For faster training using Cloud GPUs:
1. Upload `notebook/colab_training.ipynb` to Google Colab.
2. Upload your MT5 CSV data to Google Drive (`/content/drive/MyDrive/data/{SYMBOL}.csv`).
3. Run the notebook. It includes the same **Pip Normalization** and **Config** logic as the local codebase.
4. Download the trained `.pth` models and place them in your local `models/` folder.

---

## ⚠️ Critical Logic Notes (Fixed)
- **Time Filter**: EURUSD trades **ONLY** between 08:00 and 17:00 (Broker Server Time). XAUUSD trades 24/5.
- **Execution**: The bot now executes `FOK` (Fill or Kill) orders. If your broker does not support this, edit `src/live_manager.py` to use `IOC`.
- **Causality**: The backtester strictly executes at the Open of candle `t` using data from `t-1`. No peeking!
