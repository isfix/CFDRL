graph TD
    %% =========================================================
    %% STYLING & LEGEND
    %% =========================================================
    classDef config fill:#000,stroke:#fff,stroke-width:2px,color:#fff;
    classDef mt5 fill:#ff9900,stroke:#333,stroke-width:2px,color:#fff;
    classDef py fill:#3776ab,stroke:#333,stroke-width:2px,color:#fff;
    classDef data fill:#f9f,stroke:#333,stroke-width:2px,color:#000;
    classDef logic fill:#e1f5fe,stroke:#333,stroke-width:1px,color:#000;
    classDef storage fill:#fff3e0,stroke:#333,stroke-width:2px,color:#000;

    %% =========================================================
    %% GLOBAL CONFIGURATION (The Brain Stem)
    %% =========================================================
    Config[("config.py<br/>Global Settings<br/>PAIRS = ['XAUUSD', 'EURUSD']<br/>TIMEFRAME = M5<br/>SEQ_LEN = 60")]:::config

    %% =========================================================
    %% PIPELINE A: MANUAL TRAINING (Offline / Weekly)
    %% =========================================================
    subgraph Training_Pipeline ["Phase A: Multi-Pair Training Pipeline (Run Once/Week)"]
        direction TB
        
        %% Initialization
        T_Start((Start)) --> T_Connect["Connect to MT5"]:::py
        T_Connect --> T_Loop_Start{"Loop: For Symbol in PAIRS"}:::logic
        
        %% Data Acquisition
        T_Loop_Start --> T_Download["Download History<br/>(50,000 M5 Bars)"]:::mt5
        
        %% Feature Engineering Factory
        subgraph Data_Factory ["Data Factory (src/data_factory.py)"]
            direction TB
            DF_Raw[("Raw OHLCV")]:::data
            DF_Calc["Feature Engineering:<br/>1. Log Returns (Momentum)<br/>2. Dist to EMA50 (Trend)<br/>3. RSI (Oscillator)<br/>4. ATR/Close (Volatility)<br/>5. Hour/23 (Time Context)"]:::logic
            DF_Clean["Drop NaNs & Normalize"]:::logic
            DF_Tensor["Convert to PyTorch Tensor<br/>Shape: (N, 60, 5)"]:::data
            
            DF_Raw --> DF_Calc
            DF_Calc --> DF_Clean
            DF_Clean --> DF_Tensor
        end
        
        T_Download --> DF_Raw
        
        %% The Training Loop
        subgraph RL_Training ["RL Training Loop (src/trainer.py)"]
            direction TB
            RL_Init["Init Q-Network (LSTM)"]:::py
            RL_Iter["Iterate Batch"]:::logic
            RL_Pred["Forward Pass (Prediction)"]:::py
            RL_Reward["Calc Reward:<br/>(Price_Diff - Spread)"]:::logic
            RL_Backprop["Backprop & Optimizer Step"]:::py
            
            RL_Init --> RL_Iter
            RL_Iter --> RL_Pred
            RL_Pred --> RL_Reward
            RL_Reward --> RL_Backprop
            RL_Backprop -- "Next Batch" --> RL_Iter
        end
        
        DF_Tensor --> RL_Init
        
        %% Saving
        RL_Backprop -- "Epochs Done" --> T_Save["Save Weights to File:<br/>models/{SYMBOL}_brain.pth"]:::storage
        T_Save -- "Next Symbol" --> T_Loop_Start
    end
    
    Config -.-> T_Loop_Start

    %% =========================================================
    %% PIPELINE B: LIVE TRADING (24/5 Daemon)
    %% =========================================================
    subgraph Live_Execution ["Phase B: Live Multi-Pair Bot (src/live_manager.py)"]
        direction TB
        
        %% Startup
        L_Start((Start)) --> L_Init["Init MT5 Connection"]:::py
        L_Init --> L_LoadModels["Load Models into RAM Dict<br/>{'XAUUSD': Model_A, 'EURUSD': Model_B}"]:::py
        L_LoadModels --> L_InitData["Fetch Last 100 Bars<br/>Initialize 'State' Windows"]:::mt5
        
        T_Save -.-> L_LoadModels
        
        %% The Heartbeat
        subgraph Heartbeat ["The 5-Minute Heartbeat"]
            direction TB
            L_Timer{"Is New M5 Candle?"}:::logic
            L_PairLoop{"Loop: For Symbol in PAIRS"}:::logic
            
            %% Efficient Data Update
            L_FetchTiny["Fetch Last 2 Bars ONLY"]:::mt5
            L_UpdateState["Update Sliding Window<br/>(Drop Oldest, Add Newest)"]:::logic
            L_Recalc["Update Indicators (Last Row)"]:::logic
            
            %% Inference
            L_Select["Select Model from Dict<br/>model = models[symbol]"]:::logic
            L_Predict["LSTM Inference<br/>Input: Last 60 Bars"]:::py
            L_Output["Output Q-Values<br/>[Hold, Buy, Sell]"]:::data
            
            %% Logic & Risk
            L_TimeCheck{"Time > 20:00?"}:::logic
            L_PosCheck{"Check Existing Position"}:::mt5
            L_Risk["Calc Dynamic Stop Loss<br/>SL = Price +/- (ATR * 2.5)"]:::logic
            
            %% Execution
            L_Exec["Send Order (MT5)"]:::mt5
            L_CloseAll["Force Close All"]:::mt5
            
            %% Flow
            L_Timer -- No --> L_Timer
            L_Timer -- Yes --> L_PairLoop
            L_PairLoop --> L_FetchTiny
            L_FetchTiny --> L_UpdateState
            L_UpdateState --> L_Recalc
            L_Recalc --> L_Select
            L_Select --> L_Predict
            L_Predict --> L_Output
            L_Output --> L_TimeCheck
            
            L_TimeCheck -- Yes --> L_CloseAll
            L_TimeCheck -- No --> L_PosCheck
            L_PosCheck -- "Signal Matches" --> L_PairLoop
            L_PosCheck -- "New Signal" --> L_Risk
            L_Risk --> L_Exec
            L_Exec --> L_PairLoop
        end
        
        L_InitData --> L_Timer
    end

    Config -.-> L_PairLoop