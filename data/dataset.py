import torch
import numpy as np
import librosa
import random
from torch.utils.data import Dataset
from utils.audio_utils import random_segment

class RandomSegmentDataset(Dataset):
    """
    Dataset that returns random segments of variable length.
    CRITICAL: This enables few-shot voice cloning capability.
    """
    
    def __init__(self, metadata_df, config, augmentations=None):
        """
        Args:
            metadata_df: DataFrame with audio_path and speaker_id columns
            config: Configuration object
            augmentations: Optional augmentation pipeline
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.config = config
        self.sample_rate = config.SAMPLE_RATE
        self.min_dur = config.MIN_SEGMENT_DURATION
        self.max_dur = config.MAX_SEGMENT_DURATION
        self.augmentations = augmentations
        
        # Cache audio paths for faster access
        self.audio_paths = self.metadata['audio_path'].values
        self.speaker_ids = self.metadata['speaker_id'].values
        
        # Build speaker to indices mapping
        self.speaker_to_indices = {}
        for idx, spk in enumerate(self.speaker_ids):
            self.speaker_to_indices.setdefault(spk, []).append(idx)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """Returns (audio_segment, speaker_id)"""
        audio_path = self.audio_paths[idx]
        speaker_id = self.speaker_ids[idx]
        
        # Load full audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Extract random segment
        segment = random_segment(audio, self.sample_rate, self.min_dur, self.max_dur)
        
        # Convert to tensor
        segment = torch.from_numpy(segment).float()
        
        # Apply augmentations if enabled
        if self.augmentations is not None and self.config.USE_AUGMENTATION:
            segment = self.augmentations(segment)
        
        return {
            'audio': segment,
            'speaker_id': speaker_id,
            'duration': len(segment) / self.sample_rate
        }
    
    def get_speaker_samples(self, speaker_id, num_samples):
        """Get multiple samples from the same speaker."""
        indices = self.speaker_to_indices[speaker_id]
        if len(indices) < num_samples:
            # Sample with replacement if not enough samples
            chosen = random.choices(indices, k=num_samples)
        else:
            chosen = random.sample(indices, num_samples)
        
        return [self[idx] for idx in chosen]