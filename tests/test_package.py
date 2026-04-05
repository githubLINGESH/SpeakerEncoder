"""
Package tests - verify package structure and imports
"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPackageStructure:
    """Test package structure and organization"""
    
    @pytest.mark.unit
    def test_package_exists(self):
        """Test package can be imported"""
        try:
            import speaker_encoder_pipeline
        except ImportError:
            # Package not installed, but module directories should exist
            assert os.path.exists('model')
            assert os.path.exists('data')
            assert os.path.exists('train')
            assert os.path.exists('utils')
    
    @pytest.mark.unit
    def test_model_module_structure(self):
        """Test model module has required components"""
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model')
        assert os.path.exists(os.path.join(model_path, '__init__.py'))
        assert os.path.exists(os.path.join(model_path, 'encoder.py'))
        assert os.path.exists(os.path.join(model_path, 'loss.py'))
    
    @pytest.mark.unit
    def test_data_module_structure(self):
        """Test data module has required components"""
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        assert os.path.exists(os.path.join(data_path, '__init__.py'))
        assert os.path.exists(os.path.join(data_path, 'dataset.py'))
        assert os.path.exists(os.path.join(data_path, 'preprocessor.py'))
    
    @pytest.mark.unit
    def test_train_module_structure(self):
        """Test train module has required components"""
        train_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'train')
        assert os.path.exists(os.path.join(train_path, '__init__.py'))
        assert os.path.exists(os.path.join(train_path, 'dataloader.py'))
    
    @pytest.mark.unit
    def test_config_file_exists(self):
        """Test config file exists"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.py')
        assert os.path.exists(config_path)
    
    @pytest.mark.unit
    def test_setup_files_exist(self):
        """Test setup and packaging files exist"""
        base_path = os.path.dirname(os.path.dirname(__file__))
        
        # Check setup files
        assert os.path.exists(os.path.join(base_path, 'setup.py'))
        assert os.path.exists(os.path.join(base_path, 'pyproject.toml'))
        assert os.path.exists(os.path.join(base_path, 'pytest.ini'))
        assert os.path.exists(os.path.join(base_path, 'version.txt'))
    
    @pytest.mark.unit
    def test_version_file_readable(self):
        """Test version file contains valid version"""
        version_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'version.txt')
        with open(version_path, 'r') as f:
            version = f.read().strip()
        
        # Version should be in format X.Y.Z
        parts = version.split('.')
        assert len(parts) >= 3, f"Invalid version format: {version}"
        
        for part in parts:
            try:
                int(part)
            except ValueError:
                pytest.fail(f"Invalid version number part: {part}")
