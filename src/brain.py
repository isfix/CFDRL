# src/brain.py
"""
Phase 28: QNetwork with Feature Encoder.

Changes from Phase 27:
- Renamed sae_encoder -> feature_encoder (not a true autoencoder)
- Added weight compatibility shim for loading old sae_encoder.* keys
"""
import torch
import torch.nn as nn
from collections import OrderedDict
from config import Settings

class QNetwork(nn.Module):
    def __init__(self, input_dim=Settings.INPUT_DIM,
                 encoder_dim=Settings.ENCODER_DIM,
                 hidden_dim=Settings.HIDDEN_DIM, 
                 num_layers=Settings.NUM_LAYERS, 
                 dropout=Settings.DROPOUT, 
                 output_dim=Settings.OUTPUT_DIM):
        super(QNetwork, self).__init__()
        
        # Feature Encoder: compress INPUT_DIM -> ENCODER_DIM per timestep
        # Da Costa & Gebbie found 5-25 features optimal
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
        
        # FC Output
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        """x shape: (batch_size, sequence_length, input_dim)"""
        batch_size, seq_len, _ = x.size()
        
        # Apply feature encoder to each timestep
        x_flat = x.reshape(batch_size * seq_len, -1)
        encoded = self.feature_encoder(x_flat)
        encoded = encoded.reshape(batch_size, seq_len, -1)
        
        # LSTM on compressed features
        lstm_out, _ = self.lstm(encoded)
        last_step = lstm_out[:, -1, :]
        
        # LayerNorm + FC
        normalized = self.layer_norm(last_step)
        return self.fc(normalized)

    @staticmethod
    def load_with_compat(path, device='cpu'):
        """Load model weights with backward compatibility for old sae_encoder.* keys."""
        model = QNetwork().to(device)
        state_dict = torch.load(path, map_location=device)
        
        # Remap old sae_encoder.* keys -> feature_encoder.*
        new_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace('sae_encoder.', 'feature_encoder.')
            new_dict[new_key] = v
        
        model.load_state_dict(new_dict)
        return model
