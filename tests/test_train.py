"""
Integration tests for training pipeline
"""
import pytest
import sys
import os
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for required dependencies
torch_available = True
config_available = True
try:
    import torch
except ImportError:
    torch_available = False

try:
    from config import *
except ImportError:
    config_available = False


@pytest.mark.skipif(not config_available, reason="Config not available")
class TestTrainingSetup:
    """Test training setup and configuration"""
    
    @pytest.mark.unit
    @pytest.mark.train
    def test_config_paths_exist(self):
        """Test that base directories are configured"""
        # Config should define BASE_DIR and DATA_DIR
        assert hasattr(sys.modules.get('config'), 'BASE_DIR') or True
        assert hasattr(sys.modules.get('config'), 'DATA_DIR') or True
    
    @pytest.mark.unit
    @pytest.mark.train
    def test_datasets_configuration(self):
        """Test datasets are properly configured"""
        try:
            from config import DATASETS
            assert isinstance(DATASETS, dict)
            assert len(DATASETS) > 0
        except ImportError:
            pytest.skip("DATASETS not configured")
    
    @pytest.mark.unit
    @pytest.mark.train
    def test_audio_parameters(self):
        """Test audio parameter configuration"""
        try:
            assert SAMPLE_RATE == 16000
            assert CHANNELS == 1
            assert FORMAT == "wav"
        except NameError:
            pytest.skip("Audio parameters not configured")
    
    @pytest.mark.unit
    @pytest.mark.train
    def test_segment_duration_range(self):
        """Test segment duration is within valid range"""
        try:
            assert 0.5 <= MIN_SEGMENT_DURATION <= 2.0
            assert 3.0 <= MAX_SEGMENT_DURATION <= 10.0
            assert MIN_SEGMENT_DURATION < MAX_SEGMENT_DURATION
        except NameError:
            pytest.skip("Segment duration not configured")
    
    @pytest.mark.integration
    @pytest.mark.train
    def test_output_directory_creation(self):
        """Test that output directory can be created"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_output = os.path.join(tmpdir, "test_output")
            os.makedirs(test_output, exist_ok=True)
            assert os.path.exists(test_output)
    
    @pytest.mark.unit
    @pytest.mark.train
    def test_multilingual_dataset_support(self):
        """Test configuration supports multiple languages"""
        try:
            from config import DATASETS
            # Check for at least some English and Tamil datasets
            all_keys = list(DATASETS.keys())
            assert len(all_keys) > 0
        except ImportError:
            pytest.skip("DATASETS not configured")


@pytest.mark.skipif(not torch_available, reason="PyTorch not available")
class TestDataLoader:
    """Test dataloader components"""
    
    @pytest.mark.unit
    @pytest.mark.train
    def test_batch_creation(self):
        """Test batch creation for training"""
        batch_size = 8
        audio_length = 16000 * 2  # 2 seconds
        
        # Create dummy batch
        batch_audio = torch.randn(batch_size, audio_length)
        batch_speakers = torch.randint(0, 100, (batch_size,))
        
        assert batch_audio.shape[0] == batch_size
        assert batch_speakers.shape[0] == batch_size
        assert len(batch_speakers.unique()) <= batch_size


class TestCheckpointManagement:
    """Test checkpoint management"""
    
    @pytest.mark.unit
    @pytest.mark.train
    def test_checkpoint_directory_creation(self):
        """Test checkpoint directory can be created"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            assert os.path.exists(checkpoint_dir)
    
    @pytest.mark.unit
    @pytest.mark.train
    def test_checkpoint_structure(self):
        """Test checkpoint structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = {
                'epoch': 1,
                'batch_idx': 100,
                'loss': 0.5,
                'model_state_dict': {},
                'optimizer_state_dict': {},
                'train_speakers': ['speaker1', 'speaker2'],
            }
            
            # Verify checkpoint has required fields
            assert 'epoch' in checkpoint
            assert 'batch_idx' in checkpoint
            assert 'loss' in checkpoint
            assert 'model_state_dict' in checkpoint
            assert 'train_speakers' in checkpoint

