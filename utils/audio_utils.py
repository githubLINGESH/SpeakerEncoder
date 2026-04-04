import os
import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment
from typing import Tuple, Optional

def load_audio(file_path: str, target_sr: int = 16000, mono: bool = True) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Load audio file and resample to target sample rate.
    Returns (audio_array, sample_rate) or (None, None) on error.
    """
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None, None
        
        audio, sr = librosa.load(file_path, sr=target_sr, mono=mono)
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def save_audio(file_path: str, audio: np.ndarray, sample_rate: int = 16000):
    """Save audio to file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        sf.write(file_path, audio, sample_rate)
    except Exception as e:
        print(f"Error saving {file_path}: {e}")

def convert_audio(input_path: str, output_path: str, target_sr: int = 16000, mono: bool = True):
    """Convert audio to target format (WAV, 16kHz, mono)."""
    try:
        if not os.path.exists(input_path):
            print(f"Input file not found: {input_path}")
            return False
        
        audio = AudioSegment.from_file(input_path)
        if mono:
            audio = audio.set_channels(1)
        audio = audio.set_frame_rate(target_sr)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Error converting {input_path} to {output_path}: {e}")
        return False

def compute_energy(audio: np.ndarray) -> float:
    """Compute RMS energy of audio."""
    try:
        return np.sqrt(np.mean(audio.astype(np.float32)**2))
    except:
        return 0.0

def is_noisy(audio: np.ndarray, threshold: float = 0.01) -> bool:
    """Simple energy-based noise detection."""
    return compute_energy(audio) < threshold

def estimate_snr(audio: np.ndarray, noise_floor: Optional[np.ndarray] = None) -> float:
    """Estimate signal-to-noise ratio."""
    try:
        if noise_floor is None:
            # Use first 0.5 seconds as noise estimate
            noise_len = min(8000, len(audio))
            noise_floor = audio[:noise_len]
        
        signal_energy = np.mean(audio ** 2)
        noise_energy = np.mean(noise_floor ** 2)
        
        if noise_energy == 0:
            return float('inf')
        
        snr = 10 * np.log10(signal_energy / noise_energy)
        return snr
    except:
        return 0.0

def random_segment(audio: np.ndarray, sample_rate: int, 
                   min_dur: float, max_dur: float) -> np.ndarray:
    """
    Extract a random segment of length between min_dur and max_dur seconds.
    """
    try:
        total_duration = len(audio) / sample_rate
        
        if total_duration < min_dur:
            # Pad with silence if too short
            target_len = int(min_dur * sample_rate)
            padded = np.zeros(target_len)
            padded[:len(audio)] = audio
            return padded
        
        # Random duration
        seg_dur = np.random.uniform(min_dur, min(max_dur, total_duration))
        seg_len = int(seg_dur * sample_rate)
        
        # Random start
        max_start = len(audio) - seg_len
        start = np.random.randint(0, max_start + 1)
        
        return audio[start:start+seg_len]
    except Exception as e:
        print(f"Error in random_segment: {e}")
        return audio[:int(min_dur * sample_rate)] if len(audio) > 0 else np.zeros(int(min_dur * sample_rate))