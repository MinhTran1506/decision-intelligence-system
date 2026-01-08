#!/bin/bash

# Decision Intelligence Studio - Setup Script
# This script sets up the complete environment for the demo

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║         DECISION INTELLIGENCE STUDIO - SETUP SCRIPT              ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}➜ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check Python version
print_info "Checking Python version..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

if ! $PYTHON_CMD -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"; then
    print_error "Python 3.10 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi
print_success "Python version: $PYTHON_VERSION"

# Check if in WSL
if grep -qi microsoft /proc/version; then
    print_success "Running in WSL"
else
    print_warning "Not running in WSL. Some features may not work as expected."
fi

# Create directory structure
print_info "Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/outputs
mkdir -p models
mkdir -p logs
mkdir -p src/utils
mkdir -p src/data
mkdir -p src/causal
mkdir -p src/api
mkdir -p src/ui
mkdir -p notebooks
mkdir -p tests
print_success "Directory structure created"

# Create __init__.py files
print_info "Creating Python package structure..."
touch src/__init__.py
touch src/utils/__init__.py
touch src/data/__init__.py
touch src/causal/__init__.py
touch src/api/__init__.py
touch tests/__init__.py
print_success "Package structure created"

# Create virtual environment
print_info "Creating virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
# Detect OS and use appropriate activation path
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    venv/Scripts/activate
else
    source venv/bin/activate
fi
print_success "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    python.exe -m pip install --upgrade pip > /dev/null 2>&1
else
    pip install --upgrade pip > /dev/null 2>&1
fi
print_success "Pip upgraded"

# Install dependencies
print_info "Installing Python dependencies (this may take a few minutes)..."
echo ""
echo "Installing core dependencies..."
pip install --quiet pandas numpy pyarrow scikit-learn scipy joblib 2>&1 | grep -v "already satisfied" || true

echo "Installing causal inference libraries..."
pip install --quiet dowhy econml 2>&1 | grep -v "already satisfied" || true

echo "Installing API framework..."
pip install --quiet fastapi uvicorn[standard] pydantic python-multipart 2>&1 | grep -v "already satisfied" || true

echo "Installing visualization and UI..."
pip install --quiet matplotlib seaborn plotly streamlit 2>&1 | grep -v "already satisfied" || true

echo "Installing utilities..."
pip install --quiet loguru click tqdm pyyaml python-dotenv duckdb requests 2>&1 | grep -v "already satisfied" || true

echo "Installing testing framework..."
pip install --quiet pytest pytest-cov 2>&1 | grep -v "already satisfied" || true

print_success "All dependencies installed"

# Create a simple test script
print_info "Creating test script..."
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""Test if all dependencies are installed correctly"""

def test_imports():
    """Test critical imports"""
    print("Testing imports...")
    
    try:
        import pandas
        print("✓ pandas")
    except ImportError as e:
        print(f"✗ pandas: {e}")
        return False
    
    try:
        import numpy
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn")
    except ImportError as e:
        print(f"✗ scikit-learn: {e}")
        return False
    
    try:
        import dowhy
        print("✓ dowhy")
    except ImportError as e:
        print(f"✗ dowhy: {e}")
        return False
    
    try:
        import econml
        print("✓ econml")
    except ImportError as e:
        print(f"✗ econml: {e}")
        return False
    
    try:
        import fastapi
        print("✓ fastapi")
    except ImportError as e:
        print(f"✗ fastapi: {e}")
        return False
    
    try:
        import uvicorn
        print("✓ uvicorn")
    except ImportError as e:
        print(f"✗ uvicorn: {e}")
        return False
    
    print("\n✓ All critical imports successful!")
    return True

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)
EOF

chmod +x test_installation.py

# Test installation
print_info "Testing installation..."
$PYTHON_CMD test_installation.py
if [ $? -eq 0 ]; then
    print_success "Installation test passed"
else
    print_error "Installation test failed"
    exit 1
fi

# Print summary
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                      SETUP COMPLETE!                             ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
print_success "Environment is ready!"
echo ""
echo "Next steps:"
echo "  1. Make sure you have all Python scripts in their respective directories"
echo "  2. Activate the environment:"
echo "     $ source venv/bin/activate"
echo ""
echo "  3. Run the pipeline:"
echo "     $ python run_pipeline.py"
echo ""
echo "  4. Start the API (in a new terminal):"
echo "     $ source venv/bin/activate"
echo "     $ python -m src.api.main"
echo ""
echo "  5. Open the dashboard:"
echo "     $ python -m http.server 8080 --directory src/ui"
echo "     Then visit: http://localhost:8080"
echo ""
print_info "For more information, see README.md"
echo ""