# Test Troubleshooting Guide

## Common Test Errors and Fixes

### Error 1: ModuleNotFoundError: No module named 'transformers'

**Cause:** The `transformers` library is not installed.

**Solution:**
```bash
# Install transformers
pip install transformers

# Or install all dependencies
pip install -r requirements.txt
```

### Error 2: ModuleNotFoundError: No module named 'torch'

**Cause:** PyTorch is not installed.

**Solution:**
```bash
# Install PyTorch and torchaudio
pip install torch torchaudio

# Or install all dependencies
pip install -r requirements.txt
```

### Error 3: Tests Skip or Pass Without Running

**Cause:** When optional dependencies are missing, tests gracefully skip instead of failing.

**This is expected behavior!** The test suite is designed to:
- Skip tests when dependencies are unavailable
- Show which tests were skipped
- Report PASS as long as skip conditions are met

**Solutions:**

1. **Install all dependencies to run all tests:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Check which tests were skipped:**
   ```bash
   pytest tests/ -v -rs  # -rs shows skipped test reasons
   ```

---

## Running Tests

### Basic Test Run

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with short output
pytest tests/ -q
```

### Running Specific Tests

```bash
# Run package structure tests only (no dependencies required)
pytest tests/test_package.py -v

# Run unit tests
pytest tests/ -m unit -v

# Run specific test class
pytest tests/test_model.py::TestMultilingualSpeakerEncoder -v

# Run tests and show why they were skipped
pytest tests/ -v -rs
```

### Coverage Report

```bash
# Generate coverage
pytest tests/ --cov=model --cov=data --cov=train --cov=utils --cov-report=html

# View report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
xdg-open htmlcov/index.html  # Linux
```

---

## Installing Dependencies

### Minimum Dependencies (for package structure tests)
```bash
pip install pytest pytest-cov
```

### Core Dependencies (for model tests)
```bash
pip install -r requirements.txt
```

This includes:
- torch
- torchaudio
- transformers
- librosa
- numpy
- pandas
- scikit-learn
- etc.

### Full Development Setup
```bash
# Install with development dependencies
pip install -e ".[dev]"

# This includes testing, linting, and building tools
```

---

## GitHub Actions Test Failure

### If tests fail in GitHub Actions but pass locally:

1. **Check the specific error message:**
   - Go to GitHub repository
   - Click **Actions** tab
   - Click the failing workflow run
   - Expand the failed job logs
   - Look for the error message

2. **Common reasons:**
   - Dependency version mismatch
   - OS-specific issues (Windows vs Linux vs macOS)
   - Python version incompatibility
   - Timeout during large download

3. **Solutions:**
   - Update `requirements.txt` with compatible versions
   - Add Python version pin in `pyproject.toml`
   - Skip long-running tests in CI with `@pytest.mark.slow`
   - Increase timeout in workflow

### Update requirements.txt for compatibility

```bash
# Freeze current working versions
pip freeze > requirements-frozen.txt

# Or manually specify versions in requirements.txt
torch==2.0.1
torchaudio==2.0.2
transformers==4.35.0
```

---

## Test Markers

Tests are organized with markers. Run specific test categories:

```bash
# Unit tests only
pytest tests/ -m unit

# Integration tests only
pytest tests/ -m integration

# Exclude slow tests
pytest tests/ -m "not slow"

# GPU tests only
pytest tests/ -m gpu

# Data processing tests
pytest tests/ -m data

# Model tests
pytest tests/ -m model

# Training tests
pytest tests/ -m train
```

---

## Dependency Check

To see what's installed:

```bash
# List installed packages
pip list | grep -E "torch|transformers|librosa|numpy"

# Check specific package version
pip show torch

# Check if package is importable
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
```

---

## Debug Test Failures

### Run with full traceback:

```bash
pytest tests/ -v --tb=long
```

### Run with print statements visible:

```bash
pytest tests/ -v -s
```

### Run single test with debugging:

```bash
pytest tests/test_model.py::TestMultilingualSpeakerEncoder::test_encoder_initialization -v -s
```

### Write debug info to file:

```bash
pytest tests/ -v --tb=short > test_results.txt 2>&1
```

---

## Expected Test Results

### With all dependencies installed:
- Most tests should PASS
- Some GPU tests might SKIP (if no GPU available)
- Coverage should be >80%

### With minimal dependencies:
- Package structure tests PASS
- Model tests SKIP (dependencies unavailable)
- This is expected behavior

### Example output:
```
tests/test_package.py::TestPackageStructure::test_package_exists PASSED [ 10%]
tests/test_package.py::TestPackageStructure::test_version_file_readable PASSED [ 20%]
tests/test_model.py::TestMultilingualSpeakerEncoder::test_encoder_initialization SKIPPED [ 30%]
tests/test_data.py::TestDataProcessing::test_audio_configuration PASSED [ 40%]

==== 6 passed, 1 skipped in 1.23s ====
```

---

## GitHub Actions CI/CD

### The workflow tests on:
- **OS:** Ubuntu, Windows, macOS
- **Python:** 3.9, 3.10, 3.11
- **Total combinations:** 9 test runs per commit

### What the workflow does:
1. ✅ Installs dependencies
2. ✅ Runs linting (flake8)
3. ✅ Checks formatting (black, isort)
4. ✅ Runs tests (pytest)
5. ✅ Generates coverage
6. ✅ Uploads to Codecov

---

## Quick Fixes

### Fix: "pytest not found"
```bash
pip install pytest pytest-cov
```

### Fix: "ImportError in test collection"
```bash
pip install -r requirements.txt
```

### Fix: "Tests run forever / timeout"
```bash
# Run with timeout
pytest tests/ --timeout=60

# Or increase GitHub Actions timeout in workflow
```

### Fix: "Coverage not generated"
```bash
pip install pytest-cov
pytest tests/ --cov=model --cov-report=term-missing
```

---

## Support

For additional help:
1. Read [PUBLISHING.md](PUBLISHING.md) for publishing guide
2. Read [README.md](README.md) for project overview
3. Check test file docstrings for test details
4. Open a GitHub issue for CI/CD problems
