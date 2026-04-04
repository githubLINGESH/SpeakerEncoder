import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from model.timbre_branch import ECAPA_TDNN
from model.cadence_branch import CadenceEncoder
from model.ssl_branch import SSLBranch
from model.fusion import AttentionFusion, SimpleConcatFusion

class MultilingualSpeakerEncoder(nn.Module):
    """
    Complete speaker encoder with multi-branch architecture
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Feature extraction
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_mels=80,
            n_fft=400,
            hop_length=160,
            win_length=400
        )
        
        # Pitch extraction (simplified)
        self.pitch_transform = torchaudio.transforms.PitchShift(
            sample_rate=config.SAMPLE_RATE,
            n_steps=0
        )
        
        # Branches
        self.timbre_branch = ECAPA_TDNN(config)
        self.cadence_branch = CadenceEncoder(config)
        self.ssl_branch = SSLBranch(config, freeze_ssl=True)
        
        # Fusion
        if config.FUSION_TYPE == "attention":
            self.fusion = AttentionFusion(config)
        else:
            self.fusion = SimpleConcatFusion(config)
        
        # Final embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(config.FUSION_EMBED_DIM, config.FINAL_EMBED_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.FINAL_EMBED_DIM, config.FINAL_EMBED_DIM)
        )
        
    def extract_features(self, audio):
        """
        Extract log-mel and pitch features
        Args:
            audio: (batch, samples) - raw audio
        Returns:
            log_mel: (batch, frames, 80)
            pitch: (batch, frames)
        """
        # Log-mel spectrogram
        mel = self.mel_transform(audio)
        log_mel = torch.log(mel + 1e-6)
        log_mel = log_mel.transpose(1, 2)  # (batch, frames, mel_bins)
        
        # Simple pitch estimation using harmonic-to-noise ratio approximation
        # For simplicity, we'll use zero pitch; in production, use CREPE or similar
        pitch = torch.zeros(log_mel.size(0), log_mel.size(1), device=audio.device)
        
        return log_mel, pitch
    
    def forward(self, audio):
        """
        Forward pass through all branches
        Args:
            audio: (batch, samples) - raw audio at 16kHz
        Returns:
            (batch, embedding_dim) - normalized speaker embedding
        """
        # Extract features
        log_mel, pitch = self.extract_features(audio)
        
        # Branch-specific embeddings
        timbre_emb = self.timbre_branch(log_mel)
        cadence_emb = self.cadence_branch(log_mel, pitch)
        ssl_emb = self.ssl_branch(audio)
        
        # Fusion
        fused = self.fusion(timbre_emb, cadence_emb, ssl_emb)
        
        # Final embedding
        embedding = self.embedding_head(fused)
        
        # L2 normalization
        if self.config.USE_L2_NORM:
            embedding = F.normalize(embedding, dim=1)
        
        return embedding