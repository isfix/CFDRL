# src/brain.py
import torch
import torch.nn as nn
from config import Settings

class QNetwork(nn.Module):
    def __init__(self, input_dim=Settings.INPUT_DIM, 
                 hidden_dim=Settings.HIDDEN_DIM, 
                 num_layers=Settings.NUM_LAYERS, 
                 dropout=Settings.DROPOUT, 
                 output_dim=Settings.OUTPUT_DIM):
        super(QNetwork, self).__init__()
        
        # LSTM Layer
        # batch_first=True means input shape is (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully Connected Output Layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, input_dim)
        """
        # Pass through LSTM
        # out shape: (batch_size, seq_len, hidden_dim)
        # _ (hidden states) are ignored
        lstm_out, _ = self.lstm(x)
        
        # We only care about the output of the FINAL time step for prediction
        # last_step_out shape: (batch_size, hidden_dim)
        last_step_out = lstm_out[:, -1, :]
        
        # Pass through Fully Connected layers to get Q-Values
        q_values = self.fc(last_step_out)
        
        return q_values
