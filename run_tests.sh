#!/bin/bash
# Quick test runner script for Speaker Encoder Pipeline

set -e

echo "==================================="
echo "Speaker Encoder Pipeline - Test Suite"
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${YELLOW}Installing pytest and dependencies...${NC}"
    pip install -r requirements-dev.txt
fi

# Parse arguments
VERBOSE=""
COVERAGE=""
PARALLEL=""
MARKERS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -c|--coverage)
            COVERAGE="--cov=model --cov=data --cov=train --cov=utils --cov-report=html --cov-report=term-missing"
            shift
            ;;
        -p|--parallel)
            PARALLEL="-n auto"
            shift
            ;;
        -u|--unit)
            MARKERS="-m unit"
            shift
            ;;
        -i|--integration)
            MARKERS="-m integration"
            shift
            ;;
        --no-slow)
            MARKERS="-m 'not slow'"
            shift
            ;;
        -h|--help)
            echo "Usage: ./run_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose        Verbose output"
            echo "  -c, --coverage       Generate coverage report"
            echo "  -p, --parallel       Run tests in parallel"
            echo "  -u, --unit           Run only unit tests"
            echo "  -i, --integration    Run only integration tests"
            echo "  --no-slow            Skip slow tests"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_tests.sh                    # Run all tests"
            echo "  ./run_tests.sh -v -c              # Verbose with coverage"
            echo "  ./run_tests.sh -u -p              # Unit tests in parallel"
            echo "  ./run_tests.sh -i --coverage      # Integration tests with coverage"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run linting checks
echo ""
echo -e "${YELLOW}[1/4] Running code quality checks...${NC}"

if command -v flake8 &> /dev/null; then
    echo "  - Checking with flake8..."
    flake8 model/ data/ train/ utils/ --count --select=E9,F63,F7,F82 --show-source --statistics || true
else
    echo "  - flake8 not installed, skipping..."
fi

if command -v black &> /dev/null; then
    echo "  - Checking formatting with black..."
    black --check model/ data/ train/ utils/ || true
else
    echo "  - black not installed, skipping..."
fi

# Run pytest
echo ""
echo -e "${YELLOW}[2/4] Running pytest...${NC}"

# Build pytest command
PYTEST_CMD="pytest tests/ $VERBOSE $COVERAGE $PARALLEL $MARKERS"

echo "  Command: $PYTEST_CMD"
echo ""

if eval $PYTEST_CMD; then
    echo ""
    echo -e "${GREEN}[3/4] All tests passed! ✓${NC}"
else
    echo ""
    echo -e "${RED}[3/4] Some tests failed ✗${NC}"
    exit 1
fi

# Summary
echo ""
echo -e "${YELLOW}[4/4] Test Summary${NC}"
echo "  - Test framework: pytest"
echo "  - Test directory: tests/"
echo "  - Coverage targets: model, data, train, utils"

if [ -f "htmlcov/index.html" ]; then
    echo "  - Coverage report: htmlcov/index.html"
fi

echo ""
echo -e "${GREEN}=== Test Suite Complete ===${NC}"
