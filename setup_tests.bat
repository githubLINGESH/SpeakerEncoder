@echo off
REM Speaker Encoder Pipeline - Quick Fix for GitHub Actions Tests (Windows)

echo ===================================
echo Speaker Encoder - Quick Setup
echo ===================================
echo.

REM Step 1: Install all requirements
echo [Step 1] Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo. Error installing requirements
    exit /b 1
)
echo. OK - Dependencies installed
echo.

REM Step 2: Install optional/missing packages  
echo [Step 2] Installing optional dependencies...
pip install transformers soundfile pydantic
echo. OK - Optional dependencies installed
echo.

REM Step 3: Install development tools
echo [Step 3] Installing development tools...
pip install pytest pytest-cov black isort flake8
echo. OK - Development tools installed
echo.

REM Step 4: Run tests
echo [Step 4] Running tests...
pytest tests/ -v --tb=short
echo.

REM Step 5: Show summary
echo ===================================
echo Test run complete!
echo ===================================
echo.
echo Next steps:
echo 1. Check test results above
echo 2. If tests pass, push to GitHub
echo 3. GitHub Actions will automatically test and publish
echo.
echo To run tests locally:
echo   pytest tests/ -v
echo.
echo To generate coverage report:
echo   pytest tests/ --cov=model --cov=data --cov-report=html
echo.
