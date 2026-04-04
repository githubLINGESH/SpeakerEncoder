import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    """
    Attention-based fusion of timbre, cadence, and SSL embeddings
    Dynamically weights each branch based on input characteristics
    """
    def __init__(self, config):
        super().__init__()
        self.timbre_dim = config.TIMBRE_EMBED_DIM
        self.cadence_dim = config.CADENCE_EMBED_DIM
        self.ssl_dim = config.SSL_EMBED_DIM
        self.output_dim = config.FUSION_EMBED_DIM
        
        total_dim = self.timbre_dim + self.cadence_dim + self.ssl_dim
        
        # Branch-specific weights
        self.branch_weights = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )
        
        # Final projection (input is concat + weighted = 384 + 128 = 512)
        self.projection = nn.Sequential(
            nn.Linear(total_dim + self.timbre_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(config.FUSION_DROPOUT),
            nn.Linear(self.output_dim, self.output_dim)
        )
        
    def forward(self, timbre_emb, cadence_emb, ssl_emb):
        """
        Args:
            timbre_emb: (batch, timbre_dim)
            cadence_emb: (batch, cadence_dim)
            ssl_emb: (batch, ssl_dim)
        Returns:
            (batch, output_dim) - fused embedding
        """
        # Concatenate all embeddings
        concat = torch.cat([timbre_emb, cadence_emb, ssl_emb], dim=1)
        
        # Branch-specific weighting
        weights = self.branch_weights(concat)
        
        # Weighted combination
        weighted = (
            timbre_emb * weights[:, 0:1] +
            cadence_emb * weights[:, 1:2] +
            ssl_emb * weights[:, 2:3]
        )
        
        # Concatenate weighted result with raw concatenation
        combined = torch.cat([concat, weighted], dim=1)
        
        # Final projection
        fused = self.projection(combined)
        
        return fused

class SimpleConcatFusion(nn.Module):
    """Simpler concatenation-based fusion"""
    def __init__(self, config):
        super().__init__()
        self.timbre_dim = config.TIMBRE_EMBED_DIM
        self.cadence_dim = config.CADENCE_EMBED_DIM
        self.ssl_dim = config.SSL_EMBED_DIM
        self.output_dim = config.FUSION_EMBED_DIM
        
        self.projection = nn.Sequential(
            nn.Linear(self.timbre_dim + self.cadence_dim + self.ssl_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(config.FUSION_DROPOUT),
            nn.Linear(self.output_dim, self.output_dim)
        )
        
    def forward(self, timbre_emb, cadence_emb, ssl_emb):
        concat = torch.cat([timbre_emb, cadence_emb, ssl_emb], dim=1)
        return self.projection(concat)