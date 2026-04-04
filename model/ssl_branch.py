import torch
import torch.nn as nn
from transformers import HubertModel, Wav2Vec2FeatureExtractor

class SSLBranch(nn.Module):
    """
    SSL-based branch using pretrained HuBERT
    Provides multilingual phonetic awareness
    """
    def __init__(self, config, freeze_ssl=True):
        super().__init__()
        self.config = config
        self.embedding_dim = config.SSL_EMBED_DIM
        
        # Load HuBERT model only (no tokenizer needed)
        self.ssl_model = HubertModel.from_pretrained(config.SSL_MODEL_NAME)
        
        # Feature extractor for preprocessing
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            config.SSL_MODEL_NAME
        )
        
        # Freeze SSL if specified
        if freeze_ssl:
            for param in self.ssl_model.parameters():
                param.requires_grad = False
        
        # Get hidden size from model config
        self.hidden_size = self.ssl_model.config.hidden_size  # 768 for HuBERT base
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim)
        )
        
    def forward(self, audio):
        """
        Args:
            audio: (batch, samples) - raw audio at 16kHz
        Returns:
            (batch, embedding_dim) - SSL embedding
        """
        # Ensure audio is in the right range
        if audio.abs().max() > 1.0:
            audio = audio / audio.abs().max()
        
        # Get SSL features
        with torch.no_grad():
            # Forward through HuBERT
            outputs = self.ssl_model(audio, output_hidden_states=True)
            
            # Use the specified layer (default: 9 for HuBERT base)
            # Hidden states: [layer0 (embedding), layer1, ..., layer12]
            layer_idx = self.config.SSL_LAYER
            features = outputs.hidden_states[layer_idx]  # (batch, frames, hidden_size)
            
            # Mean pooling over time
            features = features.mean(dim=1)  # (batch, hidden_size)
        
        # Project to embedding dimension
        embedding = self.projection(features)
        
        return embedding