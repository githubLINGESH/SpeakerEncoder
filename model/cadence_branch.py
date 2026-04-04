import torch
import torch.nn as nn
import torch.nn.functional as F

class CadenceEncoder(nn.Module):
    """
    Temporal encoder for capturing rhythm, pitch patterns, and speaking style
    This branch learns cultural speaking patterns (Chennai vs Madurai Tamil)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_dim = config.CADENCE_EMBED_DIM
        self.input_dim = config.CADENCE_INPUT_DIM
        self.hidden_dim = config.CADENCE_HIDDEN_DIM
        self.num_layers = config.CADENCE_NUM_LAYERS
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.CADENCE_DROPOUT
        )
        
        # Self-attention for feature aggregation
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.CADENCE_DROPOUT),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim)
        )
        
    def extract_pitch_features(self, log_mel, pitch):
        """
        Extract pitch-related features for cadence
        Args:
            log_mel: (batch, frames, mel_bins)
            pitch: (batch, frames) - F0 values
        Returns:
            (batch, frames, features) - combined features
        """
        # Compute delta features (rate of change)
        delta_pitch = torch.diff(pitch, dim=1, prepend=pitch[:, :1])
        
        # Compute energy from log-mel
        energy = log_mel.mean(dim=-1)
        delta_energy = torch.diff(energy, dim=1, prepend=energy[:, :1])
        
        # Combine all features
        features = torch.cat([
            log_mel[:, :, :40],  # Reduce mel bins for efficiency
            pitch.unsqueeze(-1),
            delta_pitch.unsqueeze(-1),
            energy.unsqueeze(-1),
            delta_energy.unsqueeze(-1)
        ], dim=-1)
        
        return features
    
    def forward(self, log_mel, pitch):
        """
        Args:
            log_mel: (batch, frames, mel_bins) - log-mel spectrogram
            pitch: (batch, frames) - F0 values (0 for unvoiced)
        Returns:
            (batch, embedding_dim) - cadence embedding
        """
        # Extract combined features
        features = self.extract_pitch_features(log_mel, pitch)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Self-attention pooling
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights.squeeze(-1), dim=1)
        attended = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        
        # Project to final embedding
        embedding = self.projection(attended)
        
        return embedding