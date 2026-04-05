# Publishing & CI/CD Guide for Speaker Encoder Pipeline

## Overview

This document explains how to publish your Speaker Encoder Pipeline as a Python package to PyPI using GitHub Actions with version-based releases.

## Prerequisites

1. **GitHub Account**: Repository with Actions enabled
2. **PyPI Account**: https://pypi.org (Free tier available)
3. **Version Control**: Git initialized and GitHub repo set up

## Step-by-Step Setup

### Step 1: Set Up PyPI Token

1. Create a PyPI account at https://pypi.org/account/register/
2. Go to https://pypi.org/manage/account/token/
3. Click "Add API token"
4. Create a token for the entire PyPI account
5. Copy the token (you won't see it again!)

### Step 2: Configure GitHub Secrets

1. Go to your GitHub repository
2. Navigate to **Settings → Secrets and variables → Actions**
3. Click **New repository secret**
4. Name: `PYPI_API_TOKEN`
5. Value: Paste your PyPI token
6. Click **Add secret**

### Step 3: Verify Project Files

Ensure these files exist in your repository:

✅ `setup.py` - Package configuration
✅ `pyproject.toml` - Modern build system config
✅ `pytest.ini` - Test configuration
✅ `version.txt` - Current version
✅ `requirements.txt` - Core dependencies
✅ `requirements-dev.txt` - Development dependencies
✅ `README.md` - Package documentation
✅ `LICENSE` - License file (MIT included)
✅ `MANIFEST.in` - Package contents specification
✅ `.github/workflows/python-package.yml` - CI/CD workflow
✅ `tests/` - Test suite directory

### Step 4: Update Your Information

Edit the following files with your information:

#### setup.py
```python
author='Your Name',
author_email='your.email@example.com',
url='https://github.com/yourusername/Zero_shotVoiceClone',
```

#### pyproject.toml
```toml
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]

[project.urls]
Homepage = "https://github.com/yourusername/Zero_shotVoiceClone"
```

#### README.md
- Replace `yourusername` with your GitHub username
- Update any project-specific information

## Publishing Process

### For Development Releases (Manual Testing)

```bash
# 1. Install build tools
pip install -r requirements-dev.txt

# 2. Build distribution
python -m build

# 3. Check distribution
twine check dist/*

# 4. Test publish to TestPyPI (optional)
twine upload --repository testpypi dist/*

# 5. Install from TestPyPI to test
pip install --index-url https://test.pypi.org/simple/ speaker-encoder-pipeline
```

### For Production Releases (Automated via GitHub Actions)

#### Option A: Publishing from Main Branch (Automatic)

When you push to the main branch, the workflow:
- ✅ Runs all tests
- ✅ Checks code quality
- ✅ Builds distribution
- ❌ Does NOT publish (requires tag)

#### Option B: Publishing with Version Tags (Recommended)

1. **Update Version Number**
   ```bash
   # Edit version.txt
   echo "0.1.0" > version.txt
   ```

2. **Commit the Version Change**
   ```bash
   git add version.txt
   git commit -m "Bump version to 0.1.0"
   git push origin main
   ```

3. **Wait for Tests to Pass**
   - GitHub Actions runs all tests
   - Check Actions tab in your repository
   - All tests must pass before proceeding

4. **Create a Git Tag**
   ```bash
   # Create lightweight tag
   git tag v0.1.0

   # OR create annotated tag with message
   git tag -a v0.1.0 -m "Release version 0.1.0"

   # Push tag to GitHub
   git push origin v0.1.0
   ```

5. **GitHub Actions Automatically**
   - ✅ Runs all tests one more time
   - ✅ Builds distribution packages
   - ✅ Validates with twine
   - ✅ Publishes to PyPI
   - ✅ Creates GitHub Release with artifacts

6. **Verify Publication**
   ```bash
   # Check PyPI (takes ~5 minutes to appear)
   pip install speaker-encoder-pipeline

   # Or visit: https://pypi.org/project/speaker-encoder-pipeline/
   ```

## Versioning Strategy

### Semantic Versioning (MAJOR.MINOR.PATCH)

Example: `0.1.0`

- **MAJOR** (0): Incompatible API changes
- **MINOR** (1): New features, backward compatible
- **PATCH** (0): Bug fixes, backward compatible

### Version Progression Example

```
0.1.0  → Initial release
0.1.1  → Bug fix
0.2.0  → New features
1.0.0  → First stable release
1.0.1  → Bug fix
1.1.0  → New features
2.0.0  → Major breaking changes
```

### Update version.txt Before Each Release

```bash
echo "0.2.0" > version.txt
git add version.txt
git commit -m "Bump version to 0.2.0"
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin main
git push origin v0.2.0
```

## Workflow Triggers

### Test on Every Push/PR
```yaml
on:
  push:
    branches: [ "main", "develop" ]
  pull_request:
    branches: [ "main", "develop" ]
```

### Publish Only on Version Tags
```yaml
if: startsWith(github.ref, 'refs/tags/v')
```

The workflow file automatically:
- Runs tests on Ubuntu, Windows, and macOS
- Tests on Python 3.9, 3.10, 3.11
- Checks code quality
- Publishes to PyPI only when a tag is pushed

## Monitoring Releases

### Check Workflow Status
1. Go to your GitHub repository
2. Click **Actions** tab
3. See workflow runs and their status

### Track Package on PyPI
1. Visit: `https://pypi.org/project/speaker-encoder-pipeline/`
2. See version history and download stats

### Download Stats
```bash
pip install pypistats
pypistats overall speaker-encoder-pipeline --last-month
```

## Troubleshooting

### Build Fails on GitHub Actions

**Check logs:**
1. Go to Actions tab
2. Click failing workflow run
3. Expand job logs
4. Look for error messages

**Common issues:**
- Missing dependencies: Add to `requirements.txt`
- Import errors: Check `__init__.py` files
- Test failures: Run locally with `pytest`

### PyPI Token Issues

**Error: "Invalid authentication"**
- Verify token is correct in GitHub Secrets
- Ensure token hasn't expired
- Regenerate token if needed

**Error: "Project already exists"**
- You can update existing versions
- Just push a new tag with updated code

### Package Not Installing

```bash
# Clear cache
pip cache purge

# Reinstall
pip install --no-cache-dir speaker-encoder-pipeline
```

## Testing Locally Before Publishing

```bash
# 1. Install in development mode
pip install -e ".[dev]"

# 2. Run all tests
pytest -v

# 3. Check code quality
black --check model/ data/ train/ utils/
flake8 model/ data/ train/ utils/
isort --check-only model/ data/ train/ utils/

# 4. Build locally
python -m build

# 5. Check distribution
twine check dist/*
```

## CI/CD Features

✅ **Multi-OS Testing**: Ubuntu, Windows, macOS
✅ **Multi-Python**: 3.9, 3.10, 3.11
✅ **Code Quality**: Linting, formatting checks
✅ **Test Coverage**: Automatic coverage reports
✅ **Auto-publish**: Manual, version-based publishing
✅ **GitHub Releases**: Automatic release notes
✅ **Codecov Integration**: Optional coverage tracking

## Best Practices

1. **Always run tests locally before pushing**
   ```bash
   pytest && black model/ data/ train/ utils/
   ```

2. **Write meaningful commit messages**
   ```bash
   git commit -m "Add feature: speaker verification"
   ```

3. **Keep version.txt updated**
   - Update before each release
   - Commit with clear message

4. **Test published package**
   ```bash
   pip install speaker-encoder-pipeline
   python -c "import model; print('Success!')"
   ```

5. **Monitor workflows**
   - Don't push multiple times quickly
   - Wait for workflow to complete
   - Review any failures immediately

## Additional Resources

- [PyPI Documentation](https://packaging.python.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Semantic Versioning](https://semver.org/)
- [Python Packaging Guide](https://packaging.python.org/tutorials/packaging-projects/)

## Quick Reference Commands

```bash
# Build package locally
python -m build

# Check package
twine check dist/*

# Test publish
twine upload --repository testpypi dist/*

# Real publish (requires PyPI token)
twine upload dist/*

# Create version tag
git tag -a v0.1.0 -m "Release 0.1.0"
git push origin v0.1.0

# View published package
pip install speaker-encoder-pipeline
```

## Support

For issues:
1. Check GitHub Actions logs
2. Run tests locally
3. Open GitHub issue
4. Check PyPI project page
