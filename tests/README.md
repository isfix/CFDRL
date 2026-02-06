# Fast Prototyping Suite
**Stop "Training & Praying". Start Engineering.**

This directory contains a suite of tools designed to speed up your Reinforcement Learning workflow by **20x**. Instead of waiting 5 hours for a full training run to complete, use these scripts to validate your logic and tune your parameters in minutes.

## The Workflow

### 1. Create the "Gym from Hell" (Data Generation)
**Script:** `create_tuning_data.py`
**Goal:** Create a small but difficult dataset (~30k bars) that contains Uptrends, Downtrends, and Chop.
**Why:** If your model can't survive this "Fitness Test", it has no chance on the full history.
**Usage:**
```bash
python tests/create_tuning_data.py
```
*Output:* `tests/data/{SYMBOL}_Tuning.csv`

---

### 2. The 5-Minute Sanity Check (Overfit Test)
**Script:** `sanity_check.py`
**Goal:** Prove that your Neural Network is capable of learning **at all**.
**How:** It tries to memorize a tiny slice of data (2000 bars).
**Success Criteria:** Loss should drop to near 0.00000.
**Usage:**
```bash
python tests/sanity_check.py
```
*   **Result < 0.0001:** PASSED. Your Model Code and Input Features are valid.
*   **Result stays High:** FAILED. You have a bug in `brain.py` or your inputs are garbage.

---

### 3. Static Reward Debugger (Math Check)
**Script:** `reward_debugger.py`
**Goal:** Verify that your Reward Function is **mathematically profitable**.
**How:** It iterates through the Tuning Dataset and calculates the sum of rewards for a "Perfect Strategy".
**Success Criteria:** Total Reward must be significantly POSITIVE.
**Usage:**
```bash
python tests/reward_debugger.py
```
*   **Verdict PASSED:** Massive potential. Proceed to AI training.
*   **Verdict IMPOSSIBLE:** Your Spread/Commissions are too high relative to Volatility. **Do not train AI.** Fix `config.py` scaling first.

---

### 4. Automated Hyperparameter Tuning (Optuna)
**Script:** `optimize_hyperparams.py`
**Goal:** Find the best `LEARNING_RATE`, `GAMMA`, and `BATCH_SIZE` automatically.
**How:** Uses Bayesian Optimization to hunt for parameters. It **PRUNES** (kills) bad trials early to save time.
**Usage:**
```bash
python tests/optimize_hyperparams.py
```
*   Runs 20 Trials.
*   Prints the **Best Params** at the end.
*   **Copy these params into `config.py`.**

---

## Directory Structure
*   `data/`: Stores the generated CSV files (e.g., `EURUSD_Tuning.csv`).
*   `*.py`: The tool scripts.

## FAQ
**Q: I get "File not found"?**
A: Run `python tests/create_tuning_data.py` first.

**Q: Reward Debugger says "IMPOSSIBLE"?**
A: Check `config.py`. Increase `scaling_factor` (e.g., to 10000 for Forex) or reduce `commission` to 0.0. The cost of trading must be < the average volatility of a candle.

**Q: Sanity Check won't converge?**
A: Your model might be too simple, or your Learning Rate is too high/low. Try changing the LR in the script or check `src/brain.py`.
