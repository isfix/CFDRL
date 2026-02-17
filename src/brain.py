# src/brain.py
"""
Position-Aware QNetwork.

Accepts trade state (position, bars_held, unrealized_pnl) as additional input.
Concatenates trade state with LSTM output before FC layers.
"""
import torch
import torch.nn as nn
from collections import OrderedDict
from config import Settings

TRADE_STATE_DIM = 3  # [current_position, bars_held_normalized, unrealized_pnl]


class QNetwork(nn.Module):
    def __init__(self, input_dim=Settings.INPUT_DIM,
                 encoder_dim=Settings.ENCODER_DIM,
                 hidden_dim=Settings.HIDDEN_DIM,
                 num_layers=Settings.NUM_LAYERS,
                 dropout=Settings.DROPOUT,
                 output_dim=Settings.OUTPUT_DIM,
                 trade_state_dim=TRADE_STATE_DIM):
        super(QNetwork, self).__init__()
        
        self.trade_state_dim = trade_state_dim
        
        # Feature Encoder: compress INPUT_DIM -> ENCODER_DIM per timestep
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, encoder_dim),
            nn.ReLU()
        )
        
        # LSTM operates on compressed features
        self.lstm = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # FC Output — takes LSTM output + trade state
        fc_input_dim = hidden_dim + trade_state_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x, trade_state=None):
        """
        x shape: (batch_size, sequence_length, input_dim)
        trade_state shape: (batch_size, trade_state_dim)  — [position, bars_held, unrealized_pnl]
        """
        batch_size, seq_len, _ = x.size()
        
        # Apply feature encoder to each timestep
        x_flat = x.reshape(batch_size * seq_len, -1)
        encoded = self.feature_encoder(x_flat)
        encoded = encoded.reshape(batch_size, seq_len, -1)
        
        # LSTM on compressed features
        lstm_out, _ = self.lstm(encoded)
        last_step = lstm_out[:, -1, :]
        
        # LayerNorm
        normalized = self.layer_norm(last_step)
        
        # Concatenate trade state if provided
        if trade_state is not None:
            combined = torch.cat([normalized, trade_state], dim=1)
        else:
            # Zero trade state (for compatibility with old usage)
            zeros = torch.zeros(batch_size, self.trade_state_dim, device=x.device)
            combined = torch.cat([normalized, zeros], dim=1)
        
        return self.fc(combined)

    @staticmethod
    def load_with_compat(path, device='cpu'):
        """Load model weights with auto-detected dimensions and key compatibility."""
        state_dict = torch.load(path, map_location=device, weights_only=True)
        
        # Remap old sae_encoder.* keys -> feature_encoder.*
        new_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace('sae_encoder.', 'feature_encoder.')
            new_dict[new_key] = v
        
        # Auto-detect dimensions from checkpoint shapes
        encoder_dim = new_dict['feature_encoder.2.weight'].shape[0]
        hidden_dim = new_dict['layer_norm.weight'].shape[0]
        output_dim = new_dict['fc.3.weight'].shape[0]
        input_dim = new_dict['feature_encoder.0.weight'].shape[1]
        num_layers = sum(1 for k in new_dict if 'lstm.weight_ih_l' in k)
        
        # Detect trade_state_dim from FC input
        fc_input = new_dict['fc.0.weight'].shape[1]
        trade_state_dim = fc_input - hidden_dim
        if trade_state_dim < 0:
            trade_state_dim = TRADE_STATE_DIM
        
        model = QNetwork(
            input_dim=input_dim, encoder_dim=encoder_dim,
            hidden_dim=hidden_dim, num_layers=num_layers,
            output_dim=output_dim, trade_state_dim=trade_state_dim
        ).to(device)
        model.load_state_dict(new_dict, strict=False)
        return model
