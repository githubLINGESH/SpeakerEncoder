import torch
import torchaudio
import random
import numpy as np

class AudioAugmentations:
    """On-the-fly audio augmentations for speaker encoder training."""
    
    def __init__(self, config):
        self.config = config
        self.noise_snr_min = config.NOISE_SNR_MIN
        self.noise_snr_max = config.NOISE_SNR_MAX
        self.reverb_prob = config.REVERB_PROB
        self.speed_prob = config.SPEED_PERTURB_PROB
        self.speed_factors = config.SPEED_FACTORS
    
    def __call__(self, audio):
        """Apply augmentations to audio tensor."""
        audio = audio.clone()
        
        # Speed perturbation
        if random.random() < self.speed_prob:
            factor = random.choice(self.speed_factors)
            audio = self._speed_perturb(audio, factor)
        
        # Add noise
        if self.noise_snr_min > 0:
            audio = self._add_noise(audio)
        
        # Reverb (simplified - can be enhanced with actual RIRs)
        if random.random() < self.reverb_prob:
            audio = self._add_reverb(audio)
        
        return audio
    
    def _speed_perturb(self, audio, factor):
        """Change playback speed."""
        # Simple resampling
        indices = torch.arange(0, len(audio), factor)
        indices = indices[indices < len(audio)].long()
        return audio[indices]
    
    def _add_noise(self, audio):
        """Add random noise with specified SNR."""
        # Generate random noise
        noise = torch.randn_like(audio) * 0.01
        
        # Calculate target SNR
        snr_db = random.uniform(self.noise_snr_min, self.noise_snr_max)
        snr = 10 ** (snr_db / 20)
        
        # Scale noise to achieve target SNR
        audio_power = torch.mean(audio ** 2)
        noise_power = torch.mean(noise ** 2)
        noise = noise * torch.sqrt(audio_power / (noise_power * snr ** 2))
        
        return audio + noise
    
    def _add_reverb(self, audio):
        """Simple reverb simulation (impulse response placeholder)."""
        # Simple delay effect
        delay_len = 160  # 10ms at 16kHz
        if len(audio) > delay_len:
            delayed = torch.zeros_like(audio)
            delayed[delay_len:] = audio[:-delay_len] * 0.3
            return audio + delayed
        return audio