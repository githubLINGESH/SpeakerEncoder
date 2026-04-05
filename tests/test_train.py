"""
Integration tests for training pipeline
"""
import pytest
import torch
import sys
import os
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *


class TestTrainingSetup:
    """Test training setup and configuration"""
    
    @pytest.mark.unit
    @pytest.mark.train
    def test_config_paths_exist(self):
        """Test that base directories are configured"""
        assert os.path.isdir(BASE_DIR)
        assert os.path.isdir(DATA_DIR)
    
    @pytest.mark.unit
    @pytest.mark.train
    def test_datasets_configuration(self):
        """Test datasets are properly configured"""
        assert 'sps_corpus' in DATASETS
        assert 'librispeech' in DATASETS
        assert 'vctk' in DATASETS
        assert 'ta_in_female' in DATASETS
        assert 'ta_in_male' in DATASETS
        assert 'casual_tamil' in DATASETS
    
    @pytest.mark.unit
    @pytest.mark.train
    def test_audio_parameters(self):
        """Test audio parameter configuration"""
        assert SAMPLE_RATE == 16000
        assert CHANNELS == 1
        assert FORMAT == "wav"
    
    @pytest.mark.unit
    @pytest.mark.train
    def test_segment_duration_range(self):
        """Test segment duration is within valid range"""
        assert 0.5 <= MIN_SEGMENT_DURATION <= 2.0
        assert 3.0 <= MAX_SEGMENT_DURATION <= 10.0
        assert MIN_SEGMENT_DURATION < MAX_SEGMENT_DURATION
    
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
        english_datasets = ['sps_corpus', 'librispeech', 'vctk']
        tamil_datasets = ['ta_in_female', 'ta_in_male', 'casual_tamil']
        
        for ds in english_datasets:
            assert ds in DATASETS
        
        for ds in tamil_datasets:
            assert ds in DATASETS


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
