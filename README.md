# Speaker Encoder Pipeline

A multilingual speaker encoder with multi-branch architecture for few-shot voice synthesis. This package provides a robust speaker encoding system that captures timbre, cadence, and self-supervised learning features.

## Features

- **Multi-branch Architecture**: Combines timbre, cadence, and SSL features
- **Multilingual Support**: English and Tamil language support
- **Few-shot Learning**: Efficient speaker encoding with minimal audio samples
- **Modular Design**: Easy to extend and customize
- **CI/CD Ready**: Fully configured with GitHub Actions

## Installation

### From PyPI (when published)

```bash
pip install speaker-encoder-pipeline
```

### From Source

```bash
git clone https://github.com/yourusername/Zero_shotVoiceClone.git
cd speaker_encoder_pipeline

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from model.encoder import MultilingualSpeakerEncoder
from config import SAMPLE_RATE
import torch

# Initialize model
encoder = MultilingualSpeakerEncoder(config)

# Prepare audio (mono, 16kHz)
audio = torch.randn(batch_size, SAMPLE_RATE * duration)

# Get speaker embedding
embedding = encoder(audio)
print(embedding.shape)  # (batch_size, embedding_dim)
```

### Command-line Interface

```bash
# Prepare data
speaker-encoder-prepare --data-dir ./data

# Train model
speaker-encoder-train --epochs 100 --batch-size 32

# Validate speakers
speaker-encoder-validate --model-path ./models/best_model.pt

# Evaluate performance
speaker-encoder-evaluate --test-dir ./test_data
```

## Development Setup

### Installation with Dev Dependencies

```bash
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_model.py

# Run specific test
pytest tests/test_model.py::TestMultilingualSpeakerEncoder::test_forward_pass

# Run with coverage report
pytest --cov=model --cov=data --cov=train --cov=utils --cov-report=html

# Run only unit tests
pytest -m unit

# Run without slow tests
pytest -m "not slow"

# Run in parallel (faster)
pytest -n auto
```

### Code Quality

```bash
# Format code with black
black model/ data/ train/ utils/

# Check imports with isort
isort model/ data/ train/ utils/

# Lint with flake8
flake8 model/ data/ train/ utils/

# Type checking with mypy
mypy model/ data/ train/ utils/
```

## Package Structure

```
speaker_encoder_pipeline/
├── model/
│   ├── encoder.py          # Main encoder architecture
│   ├── timbre_branch.py    # Timbre extraction
│   ├── cadence_branch.py   # Cadence extraction
│   ├── ssl_branch.py       # SSL features
│   ├── fusion.py           # Feature fusion
│   └── loss.py             # Loss functions
├── data/
│   ├── dataset.py          # Dataset classes
│   ├── preprocessor.py     # Audio preprocessing
│   ├── augmentations.py    # Data augmentation
│   ├── sampler.py          # Sampling strategies
│   └── __init__.py
├── train/
│   ├── dataloader.py       # Data loading
│   └── __init__.py
├── utils/
│   ├── audio_utils.py      # Audio utilities
│   └── __init__.py
├── tests/
│   ├── test_model.py       # Model tests
│   ├── test_data.py        # Data tests
│   ├── test_train.py       # Training tests
│   ├── test_package.py     # Package structure tests
│   └── conftest.py         # Pytest configuration
├── config.py               # Configuration
├── setup.py                # Package setup
├── pyproject.toml          # Modern Python packaging
├── pytest.ini              # Pytest configuration
├── requirements.txt        # Core dependencies
├── requirements-dev.txt    # Development dependencies
└── README.md               # This file
```

## Configuration

Edit `config.py` to customize:

- Audio parameters (sample rate, channels, format)
- Dataset paths and sources
- Model architecture parameters
- Training hyperparameters

## Publishing to PyPI

### 1. Update Version

```bash
# Update version in version.txt
echo "0.2.0" > version.txt

# Commit changes
git add version.txt
git commit -m "Bump version to 0.2.0"
```

### 2. Create a Git Tag

```bash
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0
```

### 3. GitHub Actions will automatically:
- Run all tests on multiple Python versions and OSes
- Build distribution packages (wheel and sdist)
- Publish to PyPI (requires `PYPI_API_TOKEN` secret in GitHub)
- Create a GitHub Release

### 4. Configure PyPI Token (one-time setup)

In your GitHub repository:
1. Go to **Settings → Secrets and variables → Actions**
2. Add a new secret: `PYPI_API_TOKEN`
3. Generate token at https://pypi.org/manage/account/token/

## Testing

The project uses pytest with the following markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests
- `@pytest.mark.gpu` - GPU-required tests
- `@pytest.mark.model` - Model-specific tests
- `@pytest.mark.data` - Data processing tests
- `@pytest.mark.train` - Training tests

## CI/CD Workflow

The GitHub Actions workflow (`python-package.yml`):

1. **Test Job**: Runs on every push and pull request
   - Tests on Ubuntu, Windows, and macOS
   - Python 3.9, 3.10, 3.11
   - Linting with flake8
   - Code formatting with black
   - Import sorting with isort
   - Coverage reports to Codecov

2. **Build and Publish Job**: Runs on version tags
   - Builds distribution packages
   - Validates with twine
   - Publishes to PyPI
   - Creates GitHub Release

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and write tests
4. Ensure all tests pass: `pytest`
5. Format code: `black` and `isort`
6. Commit and push to your fork
7. Create a Pull Request

## License

MIT License - see LICENSE file for details

## Citation

If you use this package in your research, please cite:

```bibtex
@software{speaker_encoder_2025,
  title={Speaker Encoder Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/Zero_shotVoiceClone}
}
```

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review test cases for usage examples

## Changelog

### Version 0.1.1 (Initial Release)
- Multi-branch speaker encoder
- Multilingual support (English, Tamil)
- Complete training pipeline
- CI/CD with GitHub Actions
- PyPI packaging support
