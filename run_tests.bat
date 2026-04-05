@echo off
REM Quick test runner script for Windows
REM Speaker Encoder Pipeline - Test Suite

setlocal enabledelayedexpansion

echo ===================================
echo Speaker Encoder Pipeline - Test Suite
echo ===================================

REM Check if pytest is installed
where pytest >nul 2>nul
if errorlevel 1 (
    echo Installing pytest and dependencies...
    pip install -r requirements-dev.txt
)

REM Parse arguments
set "VERBOSE="
set "COVERAGE="
set "PARALLEL="
set "MARKERS="
set "SHOW_HELP=0"

for %%a in (%*) do (
    if "%%a"=="-v" set "VERBOSE=-v"
    if "%%a"=="--verbose" set "VERBOSE=-v"
    if "%%a"=="-c" set "COVERAGE=--cov=model --cov=data --cov=train --cov=utils --cov-report=html --cov-report=term-missing"
    if "%%a"=="--coverage" set "COVERAGE=--cov=model --cov=data --cov=train --cov=utils --cov-report=html --cov-report=term-missing"
    if "%%a"=="-u" set "MARKERS=-m unit"
    if "%%a"=="--unit" set "MARKERS=-m unit"
    if "%%a"=="-i" set "MARKERS=-m integration"
    if "%%a"=="--integration" set "MARKERS=-m integration"
    if "%%a"=="-h" set "SHOW_HELP=1"
    if "%%a"=="--help" set "SHOW_HELP=1"
)

if %SHOW_HELP%==1 (
    echo Usage: run_tests.bat [OPTIONS]
    echo.
    echo Options:
    echo   -v, --verbose        Verbose output
    echo   -c, --coverage       Generate coverage report
    echo   -u, --unit           Run only unit tests
    echo   -i, --integration    Run only integration tests
    echo   -h, --help           Show this help message
    echo.
    echo Examples:
    echo   run_tests.bat                    # Run all tests
    echo   run_tests.bat -v -c              # Verbose with coverage
    echo   run_tests.bat -u                 # Unit tests
    endlocal
    exit /b 0
)

REM Run linting checks
echo.
echo [1/4] Running code quality checks...

where flake8 >nul 2>nul
if not errorlevel 1 (
    echo   - Checking with flake8...
    flake8 model\ data\ train\ utils\ --count --select=E9,F63,F7,F82 --show-source --statistics
) else (
    echo   - flake8 not installed, skipping...
)

REM Run pytest
echo.
echo [2/4] Running pytest...

set "PYTEST_CMD=pytest tests/ %VERBOSE% %COVERAGE% %MARKERS%"
echo   Command: !PYTEST_CMD!
echo.

!PYTEST_CMD!
if errorlevel 1 (
    echo.
    echo [3/4] Some tests failed
    endlocal
    exit /b 1
)

echo.
echo [3/4] All tests passed!

REM Summary
echo.
echo [4/4] Test Summary
echo   - Test framework: pytest
echo   - Test directory: tests/
echo   - Coverage targets: model, data, train, utils

if exist htmlcov\index.html (
    echo   - Coverage report: htmlcov/index.html
)

echo.
echo === Test Suite Complete ===

endlocal
