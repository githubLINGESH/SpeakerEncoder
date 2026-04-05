"""
Tests for Speaker Encoder Pipeline Package
"""
import pytest
import sys
import os
import warnings

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress specific warnings during testing
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def pytest_configure(config):
    """Configure pytest with custom settings"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "model: marks tests as model-specific"
    )
    config.addinivalue_line(
        "markers", "data: marks tests as data processing tests"
    )
    config.addinivalue_line(
        "markers", "train: marks tests as training tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test items during collection to add markers based on module"""
    for item in items:
        # Add markers based on test file location
        if "test_model" in str(item.fspath):
            item.add_marker(pytest.mark.model)
        elif "test_data" in str(item.fspath):
            item.add_marker(pytest.mark.data)
        elif "test_train" in str(item.fspath):
            item.add_marker(pytest.mark.train)

