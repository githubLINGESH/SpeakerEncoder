"""
Unit tests for data processing components
"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for required dependencies
numpy_available = True
torch_available = True
try:
    import numpy as np
except ImportError:
    numpy_available = False

try:
    import torch
except ImportError:
    torch_available = False

config_available = True
try:
    from config import SAMPLE_RATE, MIN_SEGMENT_DURATION, MAX_SEGMENT_DURATION
except ImportError:
    config_available = False


@pytest.fixture
def sample_audio_array():
    """Create sample audio numpy array"""
    if not numpy_available:
        pytest.skip("NumPy not available")
    
    duration = 5  # seconds
    try:
        num_samples = SAMPLE_RATE * duration
    except NameError:
        num_samples = 16000 * duration
    
    return np.random.randn(num_samples).astype(np.float32)


@pytest.mark.skipif(not config_available, reason="Config not available")
class TestDataProcessing:
    """Test data processing components"""
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_audio_configuration(self):
        """Test audio configuration parameters"""
        assert SAMPLE_RATE == 16000
        assert MIN_SEGMENT_DURATION == 1.0
        assert MAX_SEGMENT_DURATION == 5.0
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_segment_duration_validity(self):
        """Test segment duration constraints"""
        assert MIN_SEGMENT_DURATION < MAX_SEGMENT_DURATION
        assert MIN_SEGMENT_DURATION > 0
        assert MAX_SEGMENT_DURATION > 0
    
    @pytest.mark.unit
    @pytest.mark.data
    @pytest.mark.skipif(not numpy_available, reason="NumPy not available")
    def test_audio_array_conversion(self, sample_audio_array):
        """Test audio array type and shape"""
        assert isinstance(sample_audio_array, np.ndarray)
        assert sample_audio_array.dtype in [np.float32, np.float64]
        assert len(sample_audio_array.shape) == 1
    
    @pytest.mark.unit
    @pytest.mark.data
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_audio_to_tensor_conversion(self, sample_audio_array):
        """Test conversion from numpy array to torch tensor"""
        tensor = torch.from_numpy(sample_audio_array)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == len(sample_audio_array)
    
    @pytest.mark.unit
    @pytest.mark.data
    @pytest.mark.skipif(not numpy_available, reason="NumPy not available")
    def test_audio_normalization(self, sample_audio_array):
        """Test audio normalization"""
        # Simple mean normalization
        mean = np.mean(sample_audio_array)
        std = np.std(sample_audio_array)
        
        normalized = (sample_audio_array - mean) / (std + 1e-6)
        
        # Check normalization worked
        assert np.abs(np.mean(normalized)) < 0.1
        assert np.abs(np.std(normalized) - 1.0) < 0.2
    
    @pytest.mark.unit
    @pytest.mark.data
    @pytest.mark.skipif(not numpy_available, reason="NumPy not available")
    def test_audio_segment_duration_calculation(self, sample_audio_array):
        """Test calculation of audio duration"""
        try:
            duration_seconds = len(sample_audio_array) / SAMPLE_RATE
            assert duration_seconds == 5.0
            
            # Test that it's within valid segment range
            assert duration_seconds >= MIN_SEGMENT_DURATION
            assert duration_seconds <= MAX_SEGMENT_DURATION
        except NameError:
            pytest.skip("Config constants not available")
    
    @pytest.mark.unit
    @pytest.mark.data
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_batch_audio_creation(self):
        """Test creation of batched audio"""
        batch_size = 4
        duration = 2  # seconds
        try:
            num_samples = SAMPLE_RATE * duration
        except NameError:
            num_samples = 16000 * duration
        
        batch = torch.randn(batch_size, num_samples)
        
        assert batch.shape[0] == batch_size
        assert batch.shape[1] == num_samples

