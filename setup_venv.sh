#!/bin/bash
# Setup script to create and configure local virtual environment

set -e  # Exit on error

echo "=========================================="
echo "Setting up local virtual environment"
echo "=========================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.7 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $(python3 --version)"

# Create virtual environment
VENV_NAME="venv"
if [ -d "$VENV_NAME" ]; then
    echo "Virtual environment '$VENV_NAME' already exists."
    read -p "Do you want to remove it and create a new one? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_NAME"
    else
        echo "Keeping existing virtual environment."
        exit 0
    fi
fi

echo "Creating virtual environment: $VENV_NAME"
python3 -m venv "$VENV_NAME"

echo "Activating virtual environment..."
source "$VENV_NAME/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✓ Virtual environment setup complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "To test the installation, run:"
echo "  python test.py"
echo ""

