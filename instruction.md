STANDART OPERATING PROCEDURE (SOP)

Here is the final piece of the puzzle: **The Documentation & Operational Guide**.

---

### Part 1: The `requirements.txt` (Dependencies)
First, you need to tell your AI to generate the list of libraries required to run this.

**Prompt for AI:**
> "Create a `requirements.txt` file for this project. It must include:
> *   `MetaTrader5` (For broker connection)
> *   `torch` (For the LSTM brain)
> *   `pandas` and `numpy` (For data math)
> *   `pandas_ta` (For technical indicators)
> *   `tqdm` (For progress bars during training)
> *   `schedule` (For timing tasks, optional but good practice)"

---

### Part 2: The `README.md` (Project Documentation)
Ask your AI to generate this file so you have a reference inside your code folder.

**Prompt for AI:**
> "Create a detailed `README.md` file. It should explain:
> 1.  **Project Overview:** A Multi-Pair M5 Intraday Bot using LSTM-DQN.
> 2.  **Setup:** How to install requirements (`pip install -r requirements.txt`).
> 3.  **Configuration:** Explanation of key variables in `config.py`.
> 4.  **Usage:**
>     - Command to run training: `python src/trainer.py`
>     - Command to run live bot: `python src/live_manager.py`
> 5.  **Folder Structure:** Description of `models/`, `data/`, and `src/`."

---

### Part 3: Your Standard Operating Procedure (SOP)

This is for **YOU** (the human). Save this text in a notepad. This is how you actually manage the hedge fund you just built.

#### **Phase A: Initial Setup (Do this once)**
1.  **Install Python:** Ensure Python 3.10+ is installed.
2.  **Install MT5:** Download MetaTrader 5 from your broker (IC Markets, Exness, etc.).
3.  **Login:** Log into your **DEMO** account on the MT5 Desktop app.
4.  **Enable Algo Trading:**
    *   Click the "Algo Trading" button in the top toolbar (Make it Green).
    *   Go to **Tools -> Options -> Expert Advisors**.
    *   Check **"Allow algorithmic trading"**.
    *   Check **"Allow DLL imports"** (sometimes needed by Python API).
5.  **Install Libraries:** Open your terminal (Command Prompt) in the project folder and run:
    ```bash
    pip install -r requirements.txt
    ```

#### **Phase B: The Weekend Routine (Training)**
*Do this every Saturday or Sunday when the market is closed.*

1.  **Check Config:** Open `config.py`. Do you want to add a new pair? Change `PAIRS = ['XAUUSD', 'EURUSD']`.
2.  **Run Trainer:**
    ```bash
    python src/trainer.py
    ```
3.  **Wait:** Watch the progress bars. It will download 50,000 candles and train the brain.
4.  **Verify:** Go to the `models/` folder. Ensure you see new files like `XAUUSD_brain.pth` with a current timestamp.
    *   *If the files are old or missing, the training failed. Check the error logs.*

#### **Phase C: The Weekday Routine (Live Trading)**
*Do this on Monday morning (or Sunday night).*

1.  **Open MT5:** Ensure the terminal is running and logged in.
2.  **Start the Bot:**
    ```bash
    python src/live_manager.py
    ```
3.  **Verify Startup:**
    *   Console should say: `Connected to MT5`.
    *   Console should say: `Loaded XAUUSD model`.
    *   Console should say: `Loaded EURUSD model`.
    *   Console should say: `Waiting for next M5 candle...`
4.  **Monitor:**
    *   You do **not** need to stare at it.
    *   Check the console every few hours. You should see a log entry every 5 minutes (e.g., `10:05:00 - XAUUSD - Signal: HOLD`).

#### **Phase D: Troubleshooting (When things break)**

*   **Error: "IPC Connection Failed"**
    *   *Fix:* Your MT5 terminal is closed. Open it.
*   **Error: "Symbol not found"**
    *   *Fix:* The symbol name in `config.py` (e.g., "GOLD") doesn't match your broker's name (e.g., "XAUUSD.m"). Check your Market Watch in MT5.
*   **Error: "Empty DataFrame"**
    *   *Fix:* You haven't downloaded history. Scroll back on the MT5 charts to force it to download data, or run the Trainer again.
*   **Bot isn't taking trades?**
    *   *Reason 1:* It's night time (after 20:00).
    *   *Reason 2:* The spread is too high (Volatility filter).
    *   *Reason 3:* The AI thinks "HOLD" is the best option (This is good! It's filtering bad trades).

---

### Part 4: The "Panic Button" Protocol

Since this is an automated system, you need a plan for when it goes crazy.

1.  **Stop the Python Script:** Click the terminal window and press `Ctrl + C`. This kills the brain.
2.  **Close Positions:** Go to MT5 Terminal -> Toolbox -> Trade tab. Manually close any open positions.
3.  **Analyze:** Look at the logs. Did the AI buy 100 times in 1 minute?
    *   *Fix:* Increase `Settings.SPREAD_FILTER_POINTS` or check if your logic for "Check Existing Position" is broken.
