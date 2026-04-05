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
        """Test package directories exist"""
        base_path = os.path.dirname(os.path.dirname(__file__))
        assert os.path.exists(os.path.join(base_path, 'model'))
        assert os.path.exists(os.path.join(base_path, 'data'))
        assert os.path.exists(os.path.join(base_path, 'train'))
        assert os.path.exists(os.path.join(base_path, 'utils'))
    
    @pytest.mark.unit
    def test_model_module_structure(self):
        """Test model module has required components"""
        base_path = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base_path, 'model')
        
        required_files = ['__init__.py', 'encoder.py', 'loss.py']
        for file in required_files:
            file_path = os.path.join(model_path, file)
            assert os.path.exists(file_path), f"Missing {file} in model module"
    
    @pytest.mark.unit
    def test_data_module_structure(self):
        """Test data module has required components"""
        base_path = os.path.dirname(os.path.dirname(__file__))
        data_path = os.path.join(base_path, 'data')
        
        required_files = ['__init__.py', 'dataset.py', 'preprocessor.py']
        for file in required_files:
            file_path = os.path.join(data_path, file)
            assert os.path.exists(file_path), f"Missing {file} in data module"
    
    @pytest.mark.unit
    def test_train_module_structure(self):
        """Test train module has required components"""
        base_path = os.path.dirname(os.path.dirname(__file__))
        train_path = os.path.join(base_path, 'train')
        
        required_files = ['__init__.py', 'dataloader.py']
        for file in required_files:
            file_path = os.path.join(train_path, file)
            assert os.path.exists(file_path), f"Missing {file} in train module"
    
    @pytest.mark.unit
    def test_config_file_exists(self):
        """Test config file exists"""
        base_path = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(base_path, 'config.py')
        assert os.path.exists(config_path), "config.py not found"
    
    @pytest.mark.unit
    def test_setup_files_exist(self):
        """Test setup and packaging files exist"""
        base_path = os.path.dirname(os.path.dirname(__file__))
        
        required_files = {
            'setup.py': 'setup.py',
            'pyproject.toml': 'pyproject.toml',
            'pytest.ini': 'pytest.ini',
            'version.txt': 'version.txt',
        }
        
        for file, name in required_files.items():
            file_path = os.path.join(base_path, file)
            assert os.path.exists(file_path), f"Missing {name}"
    
    @pytest.mark.unit
    def test_version_file_readable(self):
        """Test version file contains valid version"""
        base_path = os.path.dirname(os.path.dirname(__file__))
        version_path = os.path.join(base_path, 'version.txt')
        
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
    
    @pytest.mark.unit
    def test_readme_exists(self):
        """Test README file exists"""
        base_path = os.path.dirname(os.path.dirname(__file__))
        readme_path = os.path.join(base_path, 'README.md')
        assert os.path.exists(readme_path), "README.md not found"
    
    @pytest.mark.unit
    def test_license_exists(self):
        """Test LICENSE file exists"""
        base_path = os.path.dirname(os.path.dirname(__file__))
        license_path = os.path.join(base_path, 'LICENSE')
        assert os.path.exists(license_path), "LICENSE file not found"

