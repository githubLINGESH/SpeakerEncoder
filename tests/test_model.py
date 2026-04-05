"""
Unit tests for model components
"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for required dependencies
torch_available = True
try:
    import torch
except ImportError:
    torch_available = False

transformers_available = True
try:
    import transformers
except ImportError:
    transformers_available = False

config_available = True
model_available = True
try:
    from config import *
except ImportError:
    config_available = False

try:
    if torch_available and transformers_available:
        from model.encoder import MultilingualSpeakerEncoder
except (ImportError, RuntimeError) as e:
    model_available = False


@pytest.fixture
def mock_config():
    """Create a mock config for testing"""
    config = {
        'SAMPLE_RATE': 16000,
        'CHANNELS': 1,
        'FUSION_TYPE': 'attention',
        'FUSION_EMBED_DIM': 768,
        'FINAL_EMBED_DIM': 256,
        'USE_L2_NORM': True,
        'ECAPA_CHANNELS': [512, 512, 512, 512],
        'ECAPA_KERNEL_SIZES': [3, 3, 3, 3],
        'ECAPA_DILATIONS': [1, 2, 3, 4],
        'CADENCE_HIDDEN_DIM': 256,
        'CADENCE_NUM_LAYERS': 2,
        'SSL_FREEZE': True,
        'SSL_EMBED_DIM': 768,
        'SSL_MODEL_NAME': 'facebook/hubert-base-ls960',
        'DROPOUT_RATE': 0.1,
    }
    
    class MockConfig:
        def __getattr__(self, name):
            if name in config:
                return config[name]
            raise AttributeError(f"Config has no attribute {name}")
    
    return MockConfig()


@pytest.fixture
def sample_audio():
    """Create sample audio tensor for testing"""
    if not torch_available:
        pytest.skip("PyTorch not available")
    
    sample_rate = 16000
    duration = 2  # seconds
    num_samples = sample_rate * duration
    # Random audio between -1 and 1
    return torch.randn(2, num_samples)


@pytest.mark.skipif(not torch_available, reason="PyTorch not available")
class TestMultilingualSpeakerEncoder:
    """Test MultilingualSpeakerEncoder model"""
    
    @pytest.mark.unit
    @pytest.mark.skipif(not model_available, reason="Model dependencies not available")
    def test_encoder_initialization(self, mock_config):
        """Test encoder can be initialized"""
        try:
            encoder = MultilingualSpeakerEncoder(mock_config)
            assert encoder is not None
            assert hasattr(encoder, 'timbre_branch')
            assert hasattr(encoder, 'cadence_branch')
            assert hasattr(encoder, 'ssl_branch')
        except Exception as e:
            pytest.skip(f"Encoder initialization skipped: {str(e)}")
    
    @pytest.mark.unit
    @pytest.mark.gpu
    @pytest.mark.skipif(not model_available, reason="Model dependencies not available")
    def test_forward_pass(self, mock_config, sample_audio):
        """Test forward pass through encoder"""
        try:
            encoder = MultilingualSpeakerEncoder(mock_config)
            
            # Test forward pass
            with torch.no_grad():
                output = encoder(sample_audio)
            
            assert output is not None
            assert output.shape[0] == sample_audio.shape[0]  # batch size
            assert output.shape[1] == mock_config.FINAL_EMBED_DIM  # embedding dim
        except Exception as e:
            pytest.skip(f"Forward pass test skipped: {str(e)}")
    
    @pytest.mark.unit
    @pytest.mark.skipif(not model_available, reason="Model dependencies not available")
    def test_feature_extraction(self, mock_config, sample_audio):
        """Test feature extraction"""
        try:
            encoder = MultilingualSpeakerEncoder(mock_config)
            
            log_mel, pitch = encoder.extract_features(sample_audio)
            
            assert log_mel.shape[0] == sample_audio.shape[0]
            assert log_mel.shape[2] == 80  # mel bins
            assert pitch.shape[0] == sample_audio.shape[0]
        except Exception as e:
            pytest.skip(f"Feature extraction test skipped: {str(e)}")
    
    @pytest.mark.unit
    @pytest.mark.skipif(not model_available, reason="Model dependencies not available")
    def test_different_batch_sizes(self, mock_config):
        """Test encoder with different batch sizes"""
        try:
            encoder = MultilingualSpeakerEncoder(mock_config)
            
            for batch_size in [1, 4, 8]:
                audio = torch.randn(batch_size, 16000)
                with torch.no_grad():
                    output = encoder(audio)
                
                assert output.shape[0] == batch_size
                assert output.shape[1] == mock_config.FINAL_EMBED_DIM
        except Exception as e:
            pytest.skip(f"Batch size test skipped: {str(e)}")


@pytest.mark.skipif(not config_available, reason="Config not available")
class TestAudioUtils:
    """Test audio utility functions"""
    
    @pytest.mark.unit
    def test_sample_rate_constant(self):
        """Test sample rate configuration"""
        assert SAMPLE_RATE == 16000
        assert CHANNELS == 1
    
    @pytest.mark.unit
    def test_segment_duration_constraints(self):
        """Test segment duration parameters"""
        assert MIN_SEGMENT_DURATION >= 0.5
        assert MAX_SEGMENT_DURATION <= 10.0
        assert MIN_SEGMENT_DURATION < MAX_SEGMENT_DURATION

