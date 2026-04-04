import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import librosa

def collate_fn(batch):
    audios = [item['audio'] for item in batch]
    speaker_ids = [item['speaker_id'] for item in batch]
    durations = [item['duration'] for item in batch]
    
    # Pad audios to the same length (max length in batch)
    max_len = max(audio.shape[0] for audio in audios)
    padded_audios = []
    for audio in audios:
        if audio.shape[0] < max_len:
            pad_len = max_len - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, pad_len))
        padded_audios.append(audio)
    
    audios = torch.stack(padded_audios)
    
    return {
        'audio': audios,
        'speaker_id': speaker_ids,
        'duration': durations
    }

class SpeakerBalancedDataset(Dataset):
    """
    Dataset that returns (audio, speaker_id) pairs
    """
    def __init__(self, metadata_df, config, augmentations=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.config = config
        self.augmentations = augmentations
        self.sample_rate = config.SAMPLE_RATE
        
        # Build speaker to indices mapping
        self.speaker_to_indices = {}
        for idx, row in self.metadata.iterrows():
            spk = row['speaker_id']
            if spk not in self.speaker_to_indices:
                self.speaker_to_indices[spk] = []
            self.speaker_to_indices[spk].append(idx)
        
        self.speakers = list(self.speaker_to_indices.keys())
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Retry loading with fallback
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # local_path = "D:/Zero_shotVoiceClone\\"
                # collab_path = "/content/drive/MyDrive/"
                # row__ = row['audio_path'].replace(local_path, collab_path)
                # row_collab_path = row__.replace("\\", "/")
                #print(row_collab_path)
                audio, _ = librosa.load(row['audio_path'], sr=self.sample_rate, mono=True)
                break  # Success, exit retry loop
            except Exception as e:
                print(f"Attempt {attempt+1} failed for {row['audio_path']}: {e}")
                if attempt == max_retries - 1:
                    # Last attempt failed, return silence
                    print(f"Using silence for {row['audio_path']}")
                    audio = np.zeros(int(self.sample_rate * 2))
                else:
                    import time
                    time.sleep(1)
        
        # Random segment (1-5 seconds)
        duration = len(audio) / self.sample_rate
        target_dur = random.uniform(self.config.MIN_SEGMENT_DURATION, 
                                    self.config.MAX_SEGMENT_DURATION)
        
        if duration > target_dur:
            start = random.randint(0, int((duration - target_dur) * self.sample_rate))
            samples = int(target_dur * self.sample_rate)
            audio = audio[start:start + samples]
        else:
            # Pad if too short
            target_len = int(target_dur * self.sample_rate)
            if len(audio) < target_len:
                padded = np.zeros(target_len)
                padded[:len(audio)] = audio
                audio = padded
        
        # Convert to tensor
        audio = torch.from_numpy(audio).float()
        
        # Apply augmentations
        if self.augmentations is not None and random.random() < self.config.AUGMENTATION_PROB:
            audio = self.augmentations(audio)
        
        return {
            'audio': audio,
            'speaker_id': row['speaker_id'],
            'duration': len(audio) / self.sample_rate
        }
    
    def get_speaker_batch(self, num_speakers, utterances_per_speaker):
        """
        Create a balanced batch with num_speakers speakers and K utterances each
        """
        selected_speakers = random.sample(self.speakers, num_speakers)
        batch = []
        
        for spk in selected_speakers:
            indices = self.speaker_to_indices[spk]
            # Sample K utterances (with replacement if needed)
            if len(indices) >= utterances_per_speaker:
                chosen = random.sample(indices, utterances_per_speaker)
            else:
                chosen = random.choices(indices, k=utterances_per_speaker)
            batch.extend(chosen)
        
        return [self[idx] for idx in batch]

def create_dataloader(metadata_df, config, augmentations=None, shuffle=True):
    """
    Create a DataLoader with balanced batch sampling
    """
    dataset = SpeakerBalancedDataset(metadata_df, config, augmentations)
    
    # Create balanced batch sampler
    class BalancedBatchSampler(torch.utils.data.Sampler):
        def __init__(self, dataset, config):
            self.dataset = dataset
            self.num_speakers = config.NUM_SPEAKERS_PER_BATCH
            self.utt_per_spk = config.UTTERANCES_PER_SPEAKER
            self.speakers = dataset.speakers
            
        def __iter__(self):
            # Shuffle speakers
            speakers = self.speakers.copy()
            random.shuffle(speakers)
            
            for i in range(0, len(speakers), self.num_speakers):
                if i + self.num_speakers <= len(speakers):
                    batch_speakers = speakers[i:i + self.num_speakers]
                    batch_indices = []
                    for spk in batch_speakers:
                        indices = self.dataset.speaker_to_indices[spk]
                        if len(indices) >= self.utt_per_spk:
                            chosen = random.sample(indices, self.utt_per_spk)
                        else:
                            chosen = random.choices(indices, k=self.utt_per_spk)
                        batch_indices.extend(chosen)
                    yield batch_indices
        
        def __len__(self):
            return len(self.speakers) // self.num_speakers
    
    sampler = BalancedBatchSampler(dataset, config)
    
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=False,  # Set to False for CPU training
        collate_fn=collate_fn
    )
    
    return dataloader