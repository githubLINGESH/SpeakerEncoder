import random
import torch
from torch.utils.data import Sampler
from typing import List

class BalancedBatchSampler(Sampler):
    """
    Samples batches where each batch contains:
    - num_speakers different speakers
    - utterances_per_speaker utterances from each speaker
    
    This ensures the model sees balanced representation across speakers.
    """
    
    def __init__(self, dataset, speakers: List[str], num_speakers: int, utterances_per_speaker: int):
        """
        Args:
            dataset: The dataset instance
            speakers: List of speaker IDs for each sample in the dataset
            num_speakers: Number of speakers per batch
            utterances_per_speaker: Number of utterances per speaker per batch
        """
        self.dataset = dataset
        self.speakers = speakers
        self.num_speakers = num_speakers
        self.utt_per_spk = utterances_per_speaker
        
        # Build mapping from speaker to list of indices
        self.speaker_to_indices = {}
        for idx, spk in enumerate(speakers):
            self.speaker_to_indices.setdefault(spk, []).append(idx)
        
        # Only keep speakers with enough utterances
        self.valid_speakers = [
            spk for spk, indices in self.speaker_to_indices.items()
            if len(indices) >= utterances_per_speaker
        ]
        
        if len(self.valid_speakers) < num_speakers:
            raise ValueError(
                f"Not enough speakers with at least {utterances_per_speaker} utterances. "
                f"Found {len(self.valid_speakers)}, need {num_speakers}."
            )
        
        # Calculate number of batches
        self.num_batches = len(self.valid_speakers) // num_speakers
    
    def __iter__(self):
        """Yield indices for each batch."""
        # Shuffle speakers each epoch
        shuffled_speakers = self.valid_speakers.copy()
        random.shuffle(shuffled_speakers)
        
        batch_indices = []
        for i in range(0, len(shuffled_speakers), self.num_speakers):
            if i + self.num_speakers > len(shuffled_speakers):
                continue
            
            batch_speakers = shuffled_speakers[i:i + self.num_speakers]
            
            for spk in batch_speakers:
                indices = self.speaker_to_indices[spk]
                # Sample utterances_per_speaker indices (without replacement if possible)
                if len(indices) >= self.utt_per_spk:
                    chosen = random.sample(indices, self.utt_per_spk)
                else:
                    chosen = random.choices(indices, k=self.utt_per_spk)
                batch_indices.extend(chosen)
            
            # Yield batch indices
            yield batch_indices
            batch_indices = []
    
    def __len__(self):
        return self.num_batches