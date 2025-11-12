#!/bin/bash

# Script to install Neptune into the virtual environment on TACC
# Usage: bash install_neptune.sh

echo "=========================================="
echo "Installing Neptune into virtual environment"
echo "=========================================="

# Load required modules
module reset
module load gcc/13.2.0
module load cuda/12.8
module load python3/3.11.8

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if venv is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment activated: $VIRTUAL_ENV"
else
    echo "Error: Virtual environment not activated!"
    exit 1
fi

# Display Python version and location
echo "Python version: $(python --version)"
echo "Python location: $(which python)"
echo ""

# Upgrade pip first
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install Neptune
echo "Installing Neptune..."
python -m pip install neptune

# Verify installation
echo ""
echo "Verifying Neptune installation..."
python -c "import neptune; print(f'Neptune version: {neptune.__version__}')"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Neptune installed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Error: Neptune installation verification failed!"
    echo "=========================================="
    exit 1
fi

# Deactivate virtual environment
deactivate

echo "Installation complete!"

